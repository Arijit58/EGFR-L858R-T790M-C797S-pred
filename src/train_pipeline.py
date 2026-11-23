"""
End-to-end training, 5-fold stratified cross-validation, OECD external validation,
Y-randomization, Applicability Domain calibration, and figure generation for the
EGFR C797S ChemBERTa-2 LLM QSAR pipeline.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

transformers.logging.set_verbosity_error()

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.chemistry import (
    DESCRIPTOR_NAMES,
    compute_chiral_ecfp4_array,
    compute_physchem_vector,
    enumerate_smiles,
    standardize_smiles,
)

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PRETRAINED_DIR = ROOT_DIR / "models" / "chemberta_pretrained"
PROD_DIR = ROOT_DIR / "models" / "production"
REPORT_DIR = ROOT_DIR / "reports"
FIG_DIR = REPORT_DIR / "figures"

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SmilesDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: np.ndarray | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def curate_dataset(csv_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        raw_smi = str(row["SMILES"]).strip()
        label = int(round(float(row["FLOAT"])))
        comp_id = int(row["No."])
        rec = standardize_smiles(raw_smi, uncharge=False)
        if rec is None:
            raise ValueError(f"Invalid SMILES in {split_name} No.={comp_id}: {raw_smi}")
        records.append(
            {
                "No": comp_id,
                "raw_smiles": raw_smi,
                "canonical_smiles": rec.canonical_smiles,
                "inchikey": rec.inchikey,
                "murcko_scaffold": rec.murcko_smiles,
                "label": label,
                "mol": rec.mol,
            }
        )
    return pd.DataFrame(records)


@torch.no_grad()
def extract_chemberta_embeddings(
    smiles_list: List[str],
    tokenizer,
    encoder_model: nn.Module,
    n_tta: int = 5,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract L2-normalized 384-d mean-pooled ChemBERTa-2 embeddings averaged over
    n_tta canonical + randomized SMILES strings per molecule.
    """
    encoder_model.eval()
    all_variants: List[str] = []
    mol_slices: List[Tuple[int, int]] = []
    for i, smi in enumerate(smiles_list):
        vars_i = enumerate_smiles(smi, n_variants=n_tta, seed=SEED + i)
        start = len(all_variants)
        all_variants.extend(vars_i)
        mol_slices.append((start, len(all_variants)))

    enc = tokenizer(
        all_variants,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    seq_embs = []
    for start in range(0, len(all_variants), batch_size):
        end = min(start + batch_size, len(all_variants))
        input_ids = enc["input_ids"][start:end]
        attention_mask = enc["attention_mask"][start:end]
        outputs = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)
        mask_exp = attention_mask.unsqueeze(-1).float()
        mean_pool = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        cls_pool = hidden[:, 0, :]
        comb = 0.5 * mean_pool + 0.5 * cls_pool
        seq_embs.append(comb.cpu().numpy())

    seq_embs_arr = np.vstack(seq_embs)
    mol_embs = np.zeros((len(smiles_list), seq_embs_arr.shape[1]), dtype=np.float32)
    for idx, (s, e) in enumerate(mol_slices):
        v = seq_embs_arr[s:e].mean(axis=0)
        norm = np.linalg.norm(v) + 1e-12
        mol_embs[idx] = v / norm
    return mol_embs


def finetune_chemberta_classifier(
    train_smiles: List[str],
    train_labels: np.ndarray,
    pretrained_dir: Path,
    tokenizer,
    n_aug: int = 4,
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 6e-5,
    seed: int = SEED,
) -> nn.Module:
    """
    Fine-tune ChemBERTa-77M-MTR end-to-end on CPU using SMILES enumeration
    augmentation and class-weighted cross-entropy loss.
    """
    seed_everything(seed)
    aug_smiles: List[str] = []
    aug_labels: List[int] = []
    for i, (smi, lbl) in enumerate(zip(train_smiles, train_labels)):
        variants = enumerate_smiles(smi, n_variants=n_aug, seed=seed + i)
        for v in variants:
            aug_smiles.append(v)
            aug_labels.append(int(lbl))

    enc = tokenizer(
        aug_smiles,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    ds = SmilesDataset(enc, np.array(aug_labels, dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    config = AutoConfig.from_pretrained(pretrained_dir, num_labels=2)
    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.15
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_dir, config=config, ignore_mismatched_sizes=True
    )
    model.train()

    # Inverse class frequency weighting for EGFR C797S ~63.5% active / 36.5% inactive split
    n_neg = max(int((train_labels == 0).sum()), 1)
    n_pos = max(int((train_labels == 1).sum()), 1)
    total = n_neg + n_pos
    class_weights = torch.tensor([total / (2.0 * n_neg), total / (2.0 * n_pos)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    total_steps = max(len(loader) * epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-6)

    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = criterion(out.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model.eval()
    return model


@torch.no_grad()
def predict_chemberta_proba(
    smiles_list: List[str],
    tokenizer,
    clf_model: nn.Module,
    n_tta: int = 5,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict P(Active=1) with SMILES Test-Time Augmentation."""
    clf_model.eval()
    all_variants: List[str] = []
    mol_slices: List[Tuple[int, int]] = []
    for i, smi in enumerate(smiles_list):
        vars_i = enumerate_smiles(smi, n_variants=n_tta, seed=SEED + 1000 + i)
        s = len(all_variants)
        all_variants.extend(vars_i)
        mol_slices.append((s, len(all_variants)))

    enc = tokenizer(
        all_variants,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    probs_seq = []
    for start in range(0, len(all_variants), batch_size):
        end = min(start + batch_size, len(all_variants))
        out = clf_model(
            input_ids=enc["input_ids"][start:end],
            attention_mask=enc["attention_mask"][start:end],
        )
        p = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        probs_seq.append(p)
    probs_arr = np.concatenate(probs_seq)

    mol_probs = np.zeros(len(smiles_list), dtype=np.float64)
    for idx, (s, e) in enumerate(mol_slices):
        mol_probs[idx] = float(probs_arr[s:e].mean())
    return mol_probs


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find probability cutoff maximizing MCC on Training OOF predictions only."""
    best_thr = 0.50
    best_mcc = -1.0
    for thr in np.linspace(0.30, 0.75, 91):
        preds = (y_prob >= thr).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
    return round(best_thr, 3)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "ROC_AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "PR_AUC": round(float(average_precision_score(y_true, y_prob)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "Balanced_Accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Sensitivity": round(float(sens), 4),
        "Specificity": round(float(spec), 4),
        "F1": round(float(f1_score(y_true, y_pred)), 4),
        "Brier_Score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "Threshold": round(float(threshold), 3),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def build_structural_classifier(seed: int = SEED):
    """Chirality-aware ECFP4 + Physicochemical ensemble structural head."""
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=seed + 7,
        n_jobs=-1,
    )
    return rf, et


def build_frozen_llm_head(seed: int = SEED) -> Pipeline:
    """Classifier on top of frozen 384-d ChemBERTa-77M-MTR embeddings."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.5,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_hybrid_feature_head(seed: int = SEED) -> Pipeline:
    """Regularized neural/linear classifier over concatenated [ChemBERTa-384d + Chiral ECFP4 + Physchem]."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 64),
                    activation="relu",
                    alpha=0.25,
                    learning_rate_init=8e-4,
                    max_iter=350,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def generate_figures(
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_probs: Dict[str, np.ndarray],
    cv_summary: Dict[str, Dict[str, float]],
    test_summary: Dict[str, Dict[str, float]],
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    ad_test_sim: np.ndarray,
    ad_cutoff: float,
    yrand_aucs: List[float],
    yrand_mccs: List[float],
    true_cv_auc: float,
    true_cv_mcc: float,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    palette = {
        "ECFP4 + RF (Baseline)": "#64748b",
        "ChemBERTa-2 (Frozen)": "#0ea5e9",
        "ChemBERTa-2 (Fine-Tuned LLM)": "#8b5cf6",
        "Hybrid ChemBERTa-2 + Chiral ECFP4": "#10b981",
    }

    # 1. ROC and Precision-Recall Curves on External Test Set
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=220)
    ax_roc, ax_pr = axes
    for name, probs in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_v = test_summary[name]["ROC_AUC"]
        lw = 2.6 if "Hybrid" in name or "Fine-Tuned" in name else 1.8
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_v:.3f})", color=palette[name], lw=lw)

        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_v = test_summary[name]["PR_AUC"]
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {pr_v:.3f})", color=palette[name], lw=lw)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.2)
    ax_roc.set_title("External Test Set ROC Curves (N = 148)", fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
    ax_roc.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax_roc.grid(alpha=0.2)

    baseline_pos = float(y_test.mean())
    ax_pr.axhline(baseline_pos, color="k", linestyle="--", alpha=0.4, lw=1.2, label=f"No-Skill ({baseline_pos:.2f})")
    ax_pr.set_title("External Test Set Precision-Recall Curves", fontweight="bold")
    ax_pr.set_xlabel("Recall (Sensitivity)")
    ax_pr.set_ylabel("Precision (PPV)")
    ax_pr.legend(loc="lower left", frameon=True, fontsize=8.5)
    ax_pr.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr_curves.png")
    plt.close(fig)

    # 2. Confusion Matrix (Best Model) + Metric Comparison Bar Chart
    best_name = "Hybrid ChemBERTa-2 + Chiral ECFP4"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=220)
    ax_bar, ax_cm = axes

    metrics_to_plot = ["ROC_AUC", "MCC", "Balanced_Accuracy", "Sensitivity", "Specificity"]
    x = np.arange(len(metrics_to_plot))
    width = 0.19
    for idx, (name, color) in enumerate(palette.items()):
        vals = [test_summary[name][m] for m in metrics_to_plot]
        ax_bar.bar(x + (idx - 1.5) * width, vals, width, label=name, color=color, alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["ROC-AUC", "MCC", "Bal. Acc", "Sensitivity", "Specificity"])
    ax_bar.set_ylim(0, 1.08)
    ax_bar.set_ylabel("Score on External Test Set")
    ax_bar.set_title("Benchmark Across Classical, LLM & Hybrid Models", fontweight="bold")
    ax_bar.legend(fontsize=8, loc="lower right")
    ax_bar.grid(axis="y", alpha=0.25)

    m_best = test_summary[best_name]
    cm = np.array([[m_best["TN"], m_best["FP"]], [m_best["FN"], m_best["TP"]]])
    im = ax_cm.imshow(cm, cmap="Greens")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Predicted Inactive (0)", "Predicted Active (1)"])
    ax_cm.set_yticklabels(["True Inactive (0)", "True Active (1)"])
    ax_cm.set_title(f"{best_name}\nExternal Test Confusion Matrix (N = 148)", fontweight="bold")
    for r in range(2):
        for c in range(2):
            ax_cm.text(
                c,
                r,
                f"{cm[r, c]}\n({cm[r, c] / len(y_test) * 100:.1f}%)",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if cm[r, c] > cm.max() * 0.55 else "#0f172a",
            )
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_and_metrics.png")
    plt.close(fig)

    # 3. Applicability Domain & Y-Randomization Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)
    ax_ad, ax_yr = axes

    correct_mask = ((test_probs[best_name] >= m_best["Threshold"]).astype(int) == y_test)
    ax_ad.scatter(
        ad_test_sim[correct_mask],
        np.abs(test_probs[best_name][correct_mask] - 0.5) * 2.0,
        c="#10b981",
        alpha=0.8,
        edgecolors="white",
        s=55,
        label=f"Correct Prediction ({correct_mask.sum()})",
    )
    ax_ad.scatter(
        ad_test_sim[~correct_mask],
        np.abs(test_probs[best_name][~correct_mask] - 0.5) * 2.0,
        c="#ef4444",
        marker="X",
        s=70,
        label=f"Misclassified ({(~correct_mask).sum()})",
    )
    ax_ad.axvline(ad_cutoff, color="#f59e0b", linestyle="--", lw=2.0, label=f"AD Threshold = {ad_cutoff:.3f}")
    ax_ad.set_xlabel("Top-5 k-NN ChemBERTa-2 Cosine Similarity to Training Set")
    ax_ad.set_ylabel("Model Confidence |2P - 1|")
    ax_ad.set_title("Applicability Domain (AD) Reliability Analysis", fontweight="bold")
    ax_ad.legend(fontsize=8.5)
    ax_ad.grid(alpha=0.2)

    ax_yr.scatter(range(1, len(yrand_aucs) + 1), yrand_aucs, color="#94a3b8", s=50, label="Y-Scrambled ROC-AUC")
    ax_yr.scatter(range(1, len(yrand_mccs) + 1), yrand_mccs, color="#cbd5e1", marker="s", s=45, label="Y-Scrambled MCC")
    ax_yr.axhline(true_cv_auc, color="#10b981", lw=2.3, label=f"True 5-Fold CV ROC-AUC ({true_cv_auc:.3f})")
    ax_yr.axhline(true_cv_mcc, color="#8b5cf6", lw=2.3, linestyle="-.", label=f"True 5-Fold CV MCC ({true_cv_mcc:.3f})")
    ax_yr.axhline(0.5, color="k", linestyle=":", alpha=0.4)
    ax_yr.set_xlabel("Permutation Run Index")
    ax_yr.set_ylabel("5-Fold Cross-Validation Metric")
    ax_yr.set_ylim(-0.25, 1.05)
    ax_yr.set_title("OECD Y-Randomization Robustness Test (20 Runs)", fontweight="bold")
    ax_yr.legend(fontsize=8.2, loc="center right")
    ax_yr.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "applicability_domain_and_yrand.png")
    plt.close(fig)

    # 4. Chemical Space PCA of Fine-Tuned ChemBERTa-2 Embeddings
    pca = PCA(n_components=2, random_state=SEED)
    all_emb = np.vstack([train_emb, test_emb])
    coords = pca.fit_transform(all_emb)
    tr_c = coords[: len(train_emb)]
    te_c = coords[len(train_emb) :]

    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=220)
    ax.scatter(
        tr_c[y_train == 0, 0], tr_c[y_train == 0, 1],
        c="#f87171", alpha=0.55, s=42, marker="o", label="Train Inactive (0)"
    )
    ax.scatter(
        tr_c[y_train == 1, 0], tr_c[y_train == 1, 1],
        c="#34d399", alpha=0.55, s=42, marker="o", label="Train Active (1)"
    )
    ax.scatter(
        te_c[y_test == 0, 0], te_c[y_test == 0, 1],
        c="#dc2626", edgecolors="black", linewidth=0.6, s=58, marker="^", label="Test Inactive (0)"
    )
    ax.scatter(
        te_c[y_test == 1, 0], te_c[y_test == 1, 1],
        c="#059669", edgecolors="black", linewidth=0.6, s=58, marker="^", label="Test Active (1)"
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("EGFR C797S Chemical Space in Fine-Tuned ChemBERTa-2 Latent Space", fontweight="bold")
    ax.legend(frameon=True, fontsize=8.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chemical_space_pca.png")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

    PROD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Curating and standardizing Train.csv & Test.csv (preserving stereochemistry)...")
    train_df = curate_dataset(DATA_DIR / "Train.csv", "Train")
    test_df = curate_dataset(DATA_DIR / "Test.csv", "Test")

    overlap_inchi = set(train_df["inchikey"]).intersection(set(test_df["inchikey"]))
    dup_train = int(train_df["inchikey"].duplicated().sum())
    dup_test = int(test_df["inchikey"].duplicated().sum())
    print(
        f"  Train N={len(train_df)} (Active={int(train_df['label'].sum())}, Inactive={int((train_df['label']==0).sum())}) | "
        f"Test N={len(test_df)} (Active={int(test_df['label'].sum())}, Inactive={int((test_df['label']==0).sum())})"
    )
    print(f"  Exact Stereo-InChIKey Train/Test overlap: {len(overlap_inchi)} | Internal dupes: Train={dup_train}, Test={dup_test}")

    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)
    train_smiles = train_df["canonical_smiles"].tolist()
    test_smiles = test_df["canonical_smiles"].tolist()

    print("[2/6] Computing Chirality-aware ECFP4 (2048-bit) + Physicochemical descriptors...")
    X_ecfp_train = np.vstack([compute_chiral_ecfp4_array(m) for m in train_df["mol"]])
    X_ecfp_test = np.vstack([compute_chiral_ecfp4_array(m) for m in test_df["mol"]])
    X_phys_train = np.vstack([compute_physchem_vector(m) for m in train_df["mol"]])
    X_phys_test = np.vstack([compute_physchem_vector(m) for m in test_df["mol"]])

    phys_scaler = StandardScaler()
    X_phys_train_sc = phys_scaler.fit_transform(X_phys_train)
    X_phys_test_sc = phys_scaler.transform(X_phys_test)

    X_struct_train = np.hstack([X_ecfp_train, X_phys_train_sc])
    X_struct_test = np.hstack([X_ecfp_test, X_phys_test_sc])

    print("[3/6] Extracting pretrained ChemBERTa-77M-MTR embeddings (5x SMILES TTA)...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_DIR)
    base_encoder = AutoModel.from_pretrained(PRETRAINED_DIR)
    X_llm_frozen_train = extract_chemberta_embeddings(train_smiles, tokenizer, base_encoder, n_tta=5)
    X_llm_frozen_test = extract_chemberta_embeddings(test_smiles, tokenizer, base_encoder, n_tta=5)

    print("[4/6] Running 5-Fold Stratified Cross-Validation (including End-to-End ChemBERTa-2 fine-tuning)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_rf = np.zeros(len(train_df), dtype=np.float64)
    oof_struct_ens = np.zeros(len(train_df), dtype=np.float64)
    oof_llm_frozen = np.zeros(len(train_df), dtype=np.float64)
    oof_llm_ft = np.zeros(len(train_df), dtype=np.float64)
    oof_hybrid_mlp = np.zeros(len(train_df), dtype=np.float64)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_struct_train, y_train), start=1):
        f_t0 = time.time()
        # 1. Classical Baseline (RF + ExtraTrees structural ensemble)
        rf_f, et_f = build_structural_classifier(seed=SEED + fold_idx)
        rf_f.fit(X_struct_train[tr_idx], y_train[tr_idx])
        et_f.fit(X_struct_train[tr_idx], y_train[tr_idx])
        oof_rf[va_idx] = rf_f.predict_proba(X_struct_train[va_idx])[:, 1]
        oof_struct_ens[va_idx] = 0.5 * oof_rf[va_idx] + 0.5 * et_f.predict_proba(X_struct_train[va_idx])[:, 1]

        # 2. Frozen ChemBERTa-2 + Logistic Regression
        frozen_clf = build_frozen_llm_head(seed=SEED + fold_idx)
        frozen_clf.fit(X_llm_frozen_train[tr_idx], y_train[tr_idx])
        oof_llm_frozen[va_idx] = frozen_clf.predict_proba(X_llm_frozen_train[va_idx])[:, 1]

        # 3. End-to-End Fine-Tuned ChemBERTa-2 (with 4x SMILES enumeration + 5x TTA)
        tr_smi_fold = [train_smiles[i] for i in tr_idx]
        va_smi_fold = [train_smiles[i] for i in va_idx]
        ft_model_fold = finetune_chemberta_classifier(
            tr_smi_fold,
            y_train[tr_idx],
            PRETRAINED_DIR,
            tokenizer,
            n_aug=4,
            epochs=11,
            batch_size=32,
            lr=6e-5,
            seed=SEED + fold_idx * 13,
        )
        oof_llm_ft[va_idx] = predict_chemberta_proba(va_smi_fold, tokenizer, ft_model_fold, n_tta=5)

        # 4. Hybrid Multimodal MLP over [ChemBERTa-384d + Chiral ECFP4 + Physchem]
        X_comb_tr = np.hstack([X_llm_frozen_train[tr_idx] * 4.0, X_ecfp_train[tr_idx], X_phys_train_sc[tr_idx]])
        X_comb_va = np.hstack([X_llm_frozen_train[va_idx] * 4.0, X_ecfp_train[va_idx], X_phys_train_sc[va_idx]])
        hyb_mlp = build_hybrid_feature_head(seed=SEED + fold_idx)
        hyb_mlp.fit(X_comb_tr, y_train[tr_idx])
        oof_hybrid_mlp[va_idx] = hyb_mlp.predict_proba(X_comb_va)[:, 1]

        fold_auc = roc_auc_score(y_train[va_idx], oof_llm_ft[va_idx])
        print(f"  Fold {fold_idx}/5 done in {time.time() - f_t0:.1f}s | Fine-Tuned ChemBERTa-2 Val AUC = {fold_auc:.4f}")

    # Meta-stacker learned strictly on OOF predictions of Train.csv (combines Fine-Tuned ChemBERTa-2 + Hybrid MLP + Chiral Structural Ensemble)
    meta_X_oof = np.column_stack([oof_llm_ft, oof_hybrid_mlp, oof_struct_ens])
    oof_hybrid_final = np.zeros(len(train_df), dtype=np.float64)
    for tr_idx, va_idx in skf.split(meta_X_oof, y_train):
        meta_fold = LogisticRegression(C=1.0, class_weight="balanced", random_state=SEED)
        meta_fold.fit(meta_X_oof[tr_idx], y_train[tr_idx])
        oof_hybrid_final[va_idx] = meta_fold.predict_proba(meta_X_oof[va_idx])[:, 1]

    meta_final = LogisticRegression(C=1.0, class_weight="balanced", random_state=SEED)
    meta_final.fit(meta_X_oof, y_train)

    oof_dict = {
        "ECFP4 + RF (Baseline)": oof_rf,
        "ChemBERTa-2 (Frozen)": oof_llm_frozen,
        "ChemBERTa-2 (Fine-Tuned LLM)": oof_llm_ft,
        "Hybrid ChemBERTa-2 + Chiral ECFP4": oof_hybrid_final,
    }

    thresholds: Dict[str, float] = {}
    cv_summary: Dict[str, Dict[str, float]] = {}
    for name, oof_p in oof_dict.items():
        thr = find_optimal_threshold(y_train, oof_p)
        thresholds[name] = thr
        cv_summary[name] = compute_metrics(y_train, oof_p, threshold=thr)

    print("\n[5/6] Training final production models on full Train.csv (N=222) & evaluating on locked Test.csv (N=148)...")
    rf_prod, et_prod = build_structural_classifier(seed=SEED)
    rf_prod.fit(X_struct_train, y_train)
    et_prod.fit(X_struct_train, y_train)

    frozen_prod = build_frozen_llm_head(seed=SEED)
    frozen_prod.fit(X_llm_frozen_train, y_train)

    ft_prod = finetune_chemberta_classifier(
        train_smiles,
        y_train,
        PRETRAINED_DIR,
        tokenizer,
        n_aug=5,
        epochs=12,
        batch_size=32,
        lr=6e-5,
        seed=SEED,
    )

    X_comb_train_full = np.hstack([X_llm_frozen_train * 4.0, X_ecfp_train, X_phys_train_sc])
    X_comb_test_full = np.hstack([X_llm_frozen_test * 4.0, X_ecfp_test, X_phys_test_sc])
    hyb_mlp_prod = build_hybrid_feature_head(seed=SEED)
    hyb_mlp_prod.fit(X_comb_train_full, y_train)

    test_p_rf = rf_prod.predict_proba(X_struct_test)[:, 1]
    test_p_struct_ens = 0.5 * test_p_rf + 0.5 * et_prod.predict_proba(X_struct_test)[:, 1]
    test_p_frozen = frozen_prod.predict_proba(X_llm_frozen_test)[:, 1]
    test_p_ft = predict_chemberta_proba(test_smiles, tokenizer, ft_prod, n_tta=5)
    test_p_hyb_mlp = hyb_mlp_prod.predict_proba(X_comb_test_full)[:, 1]

    meta_X_test = np.column_stack([test_p_ft, test_p_hyb_mlp, test_p_struct_ens])
    test_p_hybrid = meta_final.predict_proba(meta_X_test)[:, 1]

    test_probs = {
        "ECFP4 + RF (Baseline)": test_p_rf,
        "ChemBERTa-2 (Frozen)": test_p_frozen,
        "ChemBERTa-2 (Fine-Tuned LLM)": test_p_ft,
        "Hybrid ChemBERTa-2 + Chiral ECFP4": test_p_hybrid,
    }

    test_summary: Dict[str, Dict[str, float]] = {}
    for name, tp in test_probs.items():
        test_summary[name] = compute_metrics(y_test, tp, threshold=thresholds[name])

    # Extract Fine-Tuned ChemBERTa-2 backbone embeddings for Applicability Domain & Chemical Space visualization
    ft_encoder = ft_prod.roberta
    train_ft_emb = extract_chemberta_embeddings(train_smiles, tokenizer, ft_encoder, n_tta=3)
    test_ft_emb = extract_chemberta_embeddings(test_smiles, tokenizer, ft_encoder, n_tta=3)

    # Applicability Domain: top-5 k-NN cosine similarity in fine-tuned ChemBERTa-2 embedding space
    sim_train_mat = cosine_similarity(train_ft_emb, train_ft_emb)
    np.fill_diagonal(sim_train_mat, -1.0)
    train_top5_sim = np.sort(sim_train_mat, axis=1)[:, -5:].mean(axis=1)
    ad_mean = float(train_top5_sim.mean())
    ad_std = float(train_top5_sim.std())
    ad_cutoff = round(ad_mean - 2.0 * ad_std, 4)  # 95% lower one-sided AD boundary
    ad_warn_cutoff = round(ad_mean - 1.2 * ad_std, 4)

    sim_test_mat = cosine_similarity(test_ft_emb, train_ft_emb)
    test_top5_sim = np.sort(sim_test_mat, axis=1)[:, -5:].mean(axis=1)
    in_ad_mask = test_top5_sim >= ad_cutoff
    in_ad_metrics = compute_metrics(
        y_test[in_ad_mask],
        test_p_hybrid[in_ad_mask],
        threshold=thresholds["Hybrid ChemBERTa-2 + Chiral ECFP4"],
    )

    print("[6/6] Running OECD Y-Randomization (20 permutations) & saving figures + production bundle...")
    yrand_aucs: List[float] = []
    yrand_mccs: List[float] = []
    rng = np.random.RandomState(SEED)
    for perm_i in range(20):
        y_perm = rng.permutation(y_train)
        oof_perm = np.zeros(len(y_train))
        for tr_idx, va_idx in skf.split(X_comb_train_full, y_perm):
            clf_p = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=perm_i)
            clf_p.fit(X_comb_train_full[tr_idx], y_perm[tr_idx])
            oof_perm[va_idx] = clf_p.predict_proba(X_comb_train_full[va_idx])[:, 1]
        yrand_aucs.append(float(roc_auc_score(y_perm, oof_perm)))
        yrand_mccs.append(float(matthews_corrcoef(y_perm, (oof_perm >= 0.5).astype(int))))

    true_cv_auc = cv_summary["Hybrid ChemBERTa-2 + Chiral ECFP4"]["ROC_AUC"]
    true_cv_mcc = cv_summary["Hybrid ChemBERTa-2 + Chiral ECFP4"]["MCC"]
    # Calculate cRp^2 analog on centered AUC (R = 2*AUC - 1)
    r_true = max(2.0 * true_cv_auc - 1.0, 0.0)
    r_rand_mean = float(np.mean([abs(2.0 * a - 1.0) for a in yrand_aucs]))
    crp2 = round(float((r_true ** 2) * np.sqrt(max(r_true ** 2 - r_rand_mean ** 2, 0.0))), 4)

    generate_figures(
        y_train=y_train,
        y_test=y_test,
        test_probs=test_probs,
        cv_summary=cv_summary,
        test_summary=test_summary,
        train_emb=train_ft_emb,
        test_emb=test_ft_emb,
        ad_test_sim=test_top5_sim,
        ad_cutoff=ad_cutoff,
        yrand_aucs=yrand_aucs,
        yrand_mccs=yrand_mccs,
        true_cv_auc=true_cv_auc,
        true_cv_mcc=true_cv_mcc,
    )

    # Save Fine-Tuned ChemBERTa-2 HuggingFace checkpoint + classical/hybrid heads + reference library
    ft_dir = PROD_DIR / "chemberta_finetuned"
    ft_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(ft_dir)
    ft_prod.save_pretrained(ft_dir)

    bundle = {
        "phys_scaler": phys_scaler,
        "rf_prod": rf_prod,
        "et_prod": et_prod,
        "frozen_prod": frozen_prod,
        "hyb_mlp_prod": hyb_mlp_prod,
        "meta_final": meta_final,
        "thresholds": thresholds,
        "meta_weights": meta_final.coef_[0].tolist(),
        "ad_config": {
            "mean_top5_cosine": round(ad_mean, 4),
            "std_top5_cosine": round(ad_std, 4),
            "cutoff_strict": ad_cutoff,
            "cutoff_warn": ad_warn_cutoff,
        },
        "reference_train": {
            "No": train_df["No"].tolist(),
            "canonical_smiles": train_smiles,
            "label": y_train.tolist(),
            "ft_embeddings": train_ft_emb,
            "ecfp4": X_ecfp_train,
        },
    }
    joblib.dump(bundle, PROD_DIR / "hybrid_bundle.joblib", compress=3)

    # Save detailed external test predictions CSV
    pred_df = pd.DataFrame(
        {
            "No": test_df["No"],
            "Canonical_SMILES": test_smiles,
            "True_Label": y_test,
            "Prob_ChemBERTa_FineTuned": np.round(test_p_ft, 4),
            "Prob_Hybrid_LLM_ECFP4": np.round(test_p_hybrid, 4),
            "Predicted_Label": (test_p_hybrid >= thresholds["Hybrid ChemBERTa-2 + Chiral ECFP4"]).astype(int),
            "AD_Cosine_Sim": np.round(test_top5_sim, 4),
            "In_Applicability_Domain": in_ad_mask,
        }
    )
    pred_df.to_csv(REPORT_DIR / "external_test_predictions.csv", index=False)

    full_report = {
        "dataset": {
            "train_total": int(len(train_df)),
            "train_active": int(y_train.sum()),
            "train_inactive": int((y_train == 0).sum()),
            "test_total": int(len(test_df)),
            "test_active": int(y_test.sum()),
            "test_inactive": int((y_test == 0).sum()),
            "train_test_inchikey_overlap": int(len(overlap_inchi)),
            "unique_murcko_scaffolds_train": int(train_df["murcko_scaffold"].nunique()),
            "unique_murcko_scaffolds_test": int(test_df["murcko_scaffold"].nunique()),
        },
        "cv_5fold_oof": cv_summary,
        "external_test": test_summary,
        "applicability_domain": {
            "cutoff_cosine_top5": ad_cutoff,
            "warn_cosine_top5": ad_warn_cutoff,
            "test_coverage_pct": round(float(in_ad_mask.mean() * 100.0), 2),
            "test_in_domain_metrics": in_ad_metrics,
        },
        "y_randomization": {
            "n_permutations": 20,
            "mean_scrambled_auc": round(float(np.mean(yrand_aucs)), 4),
            "std_scrambled_auc": round(float(np.std(yrand_aucs)), 4),
            "mean_scrambled_mcc": round(float(np.mean(yrand_mccs)), 4),
            "cRp2": crp2,
        },
        "training_wall_time_sec": round(time.time() - t0, 2),
    }
    with open(REPORT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n================ BENCHMARK SUMMARY ================")
    print("5-Fold Stratified CV (Train, N=222):")
    for k, v in cv_summary.items():
        print(f"  {k:<36} | ROC-AUC={v['ROC_AUC']:.4f} | MCC={v['MCC']:.4f} | BalAcc={v['Balanced_Accuracy']:.4f}")
    print("\nExternal Validation (Test, N=148):")
    for k, v in test_summary.items():
        print(
            f"  {k:<36} | ROC-AUC={v['ROC_AUC']:.4f} | PR-AUC={v['PR_AUC']:.4f} | "
            f"MCC={v['MCC']:.4f} | SE={v['Sensitivity']:.4f} | SP={v['Specificity']:.4f}"
        )
    print(f"\nY-Randomization cRp^2 = {crp2:.4f} (threshold > 0.5) | AD Coverage = {in_ad_mask.mean()*100:.1f}%")
    print(f"Total CPU pipeline time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
