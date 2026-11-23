"""
Production CPU inference engine for the EGFR C797S NCE Screening Web App.
Provides millisecond hybrid LLM predictions, Applicability Domain (AD) assessment,
combined ChemBERTa-2 Input*Gradient + Structural atom explainability maps,
medicinal chemistry profiling, and nearest-neighbor SAR analog retrieval.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

transformers.logging.set_verbosity_error()

from rdkit import Chem
from src.chemistry import (
    compute_chiral_ecfp4_array,
    compute_nce_profile,
    compute_physchem_vector,
    enumerate_smiles,
    render_mol_svg,
    standardize_smiles,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PRETRAINED_DIR = ROOT_DIR / "models" / "chemberta_pretrained"
PROD_DIR = ROOT_DIR / "models" / "production"
REPORT_DIR = ROOT_DIR / "reports"

# Regex matching SMILES organic subset atoms (Br, Cl, B, C, N, O, P, S, F, I, c, n, o, p, s) or bracketed atoms [...]
_SMILES_ATOM_RE = re.compile(r"(\[[^\]]+\]|Br|Cl|[BCNOPSFIbcnops])")


def map_smiles_atoms_to_spans(smiles: str, expected_atoms: int) -> Optional[List[Tuple[int, int]]]:
    """Map each atom index (0..N-1) in MolFromSmiles(smiles) to its character span in `smiles`."""
    spans = [(m.start(), m.end()) for m in _SMILES_ATOM_RE.finditer(smiles)]
    if len(spans) == expected_atoms:
        return spans
    return None


class EGFRC797SPredictor:
    def __init__(self) -> None:
        ft_dir = PROD_DIR / "chemberta_finetuned"
        self.tokenizer = AutoTokenizer.from_pretrained(ft_dir)
        self.ft_model = AutoModelForSequenceClassification.from_pretrained(ft_dir)
        self.ft_model.eval()

        pretrained_src = str(PRETRAINED_DIR) if (PRETRAINED_DIR / "config.json").exists() else "DeepChem/ChemBERTa-77M-MTR"
        self.pretrained_encoder = AutoModel.from_pretrained(pretrained_src)
        self.pretrained_encoder.eval()

        joblib_path = PROD_DIR / "hybrid_bundle.joblib"
        npz_path = PROD_DIR / "portable_state.npz"
        meta_path = PROD_DIR / "portable_meta.json"

        loaded_joblib = False
        if joblib_path.exists():
            try:
                bundle = joblib.load(joblib_path)
                self.phys_scaler = bundle["phys_scaler"]
                self.rf_prod = bundle["rf_prod"]
                self.et_prod = bundle["et_prod"]
                self.hyb_mlp_prod = bundle["hyb_mlp_prod"]
                self.meta_final = bundle["meta_final"]
                self.threshold = float(bundle["thresholds"]["Hybrid ChemBERTa-2 + Chiral ECFP4"])
                self.thresholds_all = bundle["thresholds"]
                self.ad_config = bundle["ad_config"]
                self.ref = bundle["reference_train"]
                # Verify sklearn tree compatibility in current runtime
                _ = self.rf_prod.predict_proba(np.zeros((1, 2060), dtype=np.float32))
                loaded_joblib = True
            except Exception:
                loaded_joblib = False

        if not loaded_joblib:
            if not (npz_path.exists() and meta_path.exists()):
                raise RuntimeError("Neither compatible hybrid_bundle.joblib nor portable_state.npz found.")
            from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            arrs = np.load(npz_path)
            pmeta = json.loads(meta_path.read_text(encoding="utf-8"))
            X_ecfp = arrs["X_ecfp_train"]
            X_phys = arrs["X_phys_train"]
            X_llm = arrs["X_llm_frozen_train"]
            y_tr = arrs["y_train"]

            self.phys_scaler = StandardScaler().fit(X_phys)
            X_phys_sc = self.phys_scaler.transform(X_phys)
            X_struct = np.hstack([X_ecfp, X_phys_sc])

            self.rf_prod = RandomForestClassifier(
                n_estimators=500, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ).fit(X_struct, y_tr)
            self.et_prod = ExtraTreesClassifier(
                n_estimators=500, max_features="sqrt", class_weight="balanced_subsample", random_state=49, n_jobs=-1
            ).fit(X_struct, y_tr)

            X_comb = np.hstack([X_llm * 4.0, X_ecfp, X_phys_sc])
            self.hyb_mlp_prod = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(256, 64), alpha=0.25, learning_rate_init=8e-4, max_iter=350, random_state=42)),
                ]
            ).fit(X_comb, y_tr)

            self.meta_final = LogisticRegression()
            self.meta_final.classes_ = np.array([0, 1], dtype=np.int64)
            self.meta_final.coef_ = arrs["meta_coef"]
            self.meta_final.intercept_ = arrs["meta_intercept"]

            self.threshold = float(pmeta["thresholds"]["Hybrid ChemBERTa-2 + Chiral ECFP4"])
            self.thresholds_all = pmeta["thresholds"]
            self.ad_config = pmeta["ad_config"]
            self.ref = {
                "No": pmeta["No"],
                "canonical_smiles": pmeta["canonical_smiles"],
                "label": pmeta["label"],
                "ft_embeddings": arrs["ft_embeddings"],
                "ecfp4": X_ecfp,
            }

        metrics_path = REPORT_DIR / "metrics_summary.json"
        self.metrics_summary = (
            json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        )
        # Precompute training top-5 cosine similarity distribution for live AD plotting
        tr_sim_mat = cosine_similarity(self.ref["ft_embeddings"], self.ref["ft_embeddings"])
        np.fill_diagonal(tr_sim_mat, -1.0)
        self.train_top5_sims = np.sort(tr_sim_mat, axis=1)[:, -5:].mean(axis=1)

    @torch.no_grad()
    def _encode_smiles(self, smiles_list: List[str], encoder, n_tta: int = 5) -> np.ndarray:
        all_vars: List[str] = []
        slices: List[Tuple[int, int]] = []
        for i, smi in enumerate(smiles_list):
            v = enumerate_smiles(smi, n_variants=n_tta, seed=42 + i)
            s = len(all_vars)
            all_vars.extend(v)
            slices.append((s, len(all_vars)))

        enc = self.tokenizer(
            all_vars, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        comb = 0.5 * mean_pool + 0.5 * hidden[:, 0, :]
        arr = comb.cpu().numpy()

        res = np.zeros((len(smiles_list), arr.shape[1]), dtype=np.float32)
        for i, (s, e) in enumerate(slices):
            vec = arr[s:e].mean(axis=0)
            res[i] = vec / (np.linalg.norm(vec) + 1e-12)
        return res

    @torch.no_grad()
    def _predict_ft_proba(self, smiles_list: List[str], n_tta: int = 5) -> np.ndarray:
        all_vars: List[str] = []
        slices: List[Tuple[int, int]] = []
        for i, smi in enumerate(smiles_list):
            v = enumerate_smiles(smi, n_variants=n_tta, seed=1042 + i)
            s = len(all_vars)
            all_vars.extend(v)
            slices.append((s, len(all_vars)))

        enc = self.tokenizer(
            all_vars, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        logits = self.ft_model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        ).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        out = np.zeros(len(smiles_list), dtype=np.float64)
        for i, (s, e) in enumerate(slices):
            out[i] = float(probs[s:e].mean())
        return out

    def _compute_atom_attributions(
        self,
        mol: Chem.Mol,
        can_smiles: str,
        llm_frozen_vec: np.ndarray,
        phys_sc_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Combine (1) Fine-Tuned ChemBERTa-2 Input x Gradient token attributions
        mapped onto atoms and (2) Chiral Morgan atom-deletion sensitivity.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return np.zeros(0, dtype=np.float32)

        # Part A: ChemBERTa-2 Input x Gradient attribution
        llm_atom_scores = np.zeros(n_atoms, dtype=np.float64)
        spans = map_smiles_atoms_to_spans(can_smiles, n_atoms)
        try:
            enc = self.tokenizer(
                can_smiles,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=256,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            embed_layer = self.ft_model.roberta.embeddings.word_embeddings
            inputs_embeds = embed_layer(input_ids).detach().requires_grad_(True)
            self.ft_model.zero_grad()
            out = self.ft_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            margin = out.logits[0, 1] - out.logits[0, 0]
            margin.backward()

            grad = inputs_embeds.grad[0]  # (L, H)
            token_attr = (grad * inputs_embeds.detach()[0]).sum(dim=-1).cpu().numpy()

            if spans is not None:
                for tok_idx, (t_start, t_end) in enumerate(offsets):
                    if t_end <= t_start:
                        continue
                    for atom_idx, (a_start, a_end) in enumerate(spans):
                        if max(t_start, a_start) < min(t_end, a_end):
                            llm_atom_scores[atom_idx] += float(token_attr[tok_idx])
        except Exception:
            pass

        # Part B: Atom-masked Morgan environment delta
        base_ecfp = compute_chiral_ecfp4_array(mol)
        from rdkit.Chem import AllChem
        bit_info: Dict[int, List[Tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048, useChirality=True, bitInfo=bit_info
        )
        atom_to_bits: List[List[int]] = [[] for _ in range(n_atoms)]
        for bit_id, envs in bit_info.items():
            for center_atom, radius in envs:
                if radius == 0:
                    atom_to_bits[center_atom].append(bit_id)
                else:
                    env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
                    atoms_in_env = {center_atom}
                    for bidx in env_bonds:
                        b = mol.GetBondWithIdx(bidx)
                        atoms_in_env.add(b.GetBeginAtomIdx())
                        atoms_in_env.add(b.GetEndAtomIdx())
                    for a_i in atoms_in_env:
                        atom_to_bits[a_i].append(bit_id)

        perturbed_ecfp = np.repeat(base_ecfp[None, :], n_atoms + 1, axis=0)
        for a_i in range(n_atoms):
            if atom_to_bits[a_i]:
                perturbed_ecfp[a_i + 1, atom_to_bits[a_i]] = 0.0

        phys_rep = np.repeat(phys_sc_vec[None, :], n_atoms + 1, axis=0)
        llm_rep = np.repeat((llm_frozen_vec * 4.0)[None, :], n_atoms + 1, axis=0)

        struct_mat = np.hstack([perturbed_ecfp, phys_rep])
        comb_mat = np.hstack([llm_rep, perturbed_ecfp, phys_rep])

        p_struct = 0.5 * self.rf_prod.predict_proba(struct_mat)[:, 1] + 0.5 * self.et_prod.predict_proba(struct_mat)[:, 1]
        p_mlp = self.hyb_mlp_prod.predict_proba(comb_mat)[:, 1]
        score_vec = 0.5 * p_struct + 0.5 * p_mlp
        fp_atom_scores = score_vec[0] - score_vec[1:]

        # Normalize and fuse LLM + Structural atom attributions
        def _norm(v: np.ndarray) -> np.ndarray:
            s = float(np.max(np.abs(v)))
            return v / s if s > 1e-8 else v

        fused = 0.45 * _norm(llm_atom_scores) + 0.55 * _norm(fp_atom_scores)
        return fused.astype(np.float32)

    def _retrieve_nearest_analogs(
        self, ft_vec: np.ndarray, ecfp_vec: np.ndarray, top_k: int = 3
    ) -> Tuple[float, float, str, List[Dict[str, Any]]]:
        cos_sims = cosine_similarity(ft_vec.reshape(1, -1), self.ref["ft_embeddings"])[0]
        top5_cos = float(np.sort(cos_sims)[-5:].mean())

        ref_ecfp = self.ref["ecfp4"]
        inter = np.dot(ref_ecfp, ecfp_vec)
        union = ref_ecfp.sum(axis=1) + ecfp_vec.sum() - inter
        tanimotos = np.divide(inter, np.maximum(union, 1e-9))
        max_tanimoto = float(np.max(tanimotos))

        if top5_cos >= self.ad_config["cutoff_warn"] or max_tanimoto >= 0.55:
            ad_status = "Inside AD (High Reliability)"
            ad_level = "high"
        elif top5_cos >= self.ad_config["cutoff_strict"] or max_tanimoto >= 0.38:
            ad_status = "Inside AD — Borderline (Moderate Reliability)"
            ad_level = "moderate"
        else:
            ad_status = "Outside AD (Out-of-Domain — Extrapolation Warning)"
            ad_level = "low"

        # Rank analogs by combined Tanimoto + ChemBERTa similarity
        combined_sim = 0.6 * tanimotos + 0.4 * np.clip(cos_sims, 0.0, 1.0)
        best_idx = np.argsort(combined_sim)[::-1][:top_k]

        analogs = []
        for idx in best_idx:
            smi_ref = self.ref["canonical_smiles"][int(idx)]
            mol_ref = Chem.MolFromSmiles(smi_ref)
            svg_ref = render_mol_svg(mol_ref, width=260, height=180) if mol_ref else ""
            lbl = int(self.ref["label"][int(idx)])
            analogs.append(
                {
                    "No": int(self.ref["No"][int(idx)]),
                    "smiles": smi_ref,
                    "label": lbl,
                    "activity_text": "Active (1)" if lbl == 1 else "Inactive (0)",
                    "tanimoto": round(float(tanimotos[int(idx)]), 3),
                    "llm_cosine": round(float(cos_sims[int(idx)]), 3),
                    "svg": svg_ref,
                }
            )
        return round(top5_cos, 4), round(max_tanimoto, 4), f"{ad_level}|{ad_status}", analogs

    def predict_single(self, smiles: str) -> Dict[str, Any]:
        rec = standardize_smiles(smiles, uncharge=False)
        if rec is None:
            return {"valid": False, "error": "Invalid SMILES string. Could not parse chemical structure."}

        ecfp = compute_chiral_ecfp4_array(rec.mol)
        phys = compute_physchem_vector(rec.mol)
        phys_sc = self.phys_scaler.transform(phys.reshape(1, -1))[0]
        X_struct = np.hstack([ecfp, phys_sc]).reshape(1, -1)

        llm_frozen = self._encode_smiles([rec.canonical_smiles], self.pretrained_encoder, n_tta=5)[0]
        ft_emb = self._encode_smiles([rec.canonical_smiles], self.ft_model.roberta, n_tta=3)[0]

        p_ft = float(self._predict_ft_proba([rec.canonical_smiles], n_tta=5)[0])
        p_struct = float(
            0.5 * self.rf_prod.predict_proba(X_struct)[0, 1]
            + 0.5 * self.et_prod.predict_proba(X_struct)[0, 1]
        )
        X_comb = np.hstack([llm_frozen * 4.0, ecfp, phys_sc]).reshape(1, -1)
        p_hyb_mlp = float(self.hyb_mlp_prod.predict_proba(X_comb)[0, 1])

        meta_in = np.array([[p_ft, p_hyb_mlp, p_struct]])
        p_final = float(self.meta_final.predict_proba(meta_in)[0, 1])
        pred_label = int(p_final >= self.threshold)

        atom_weights = self._compute_atom_attributions(rec.mol, rec.canonical_smiles, llm_frozen, phys_sc)
        plain_svg = render_mol_svg(rec.mol, atom_weights=None, width=420, height=280)
        explain_svg = render_mol_svg(rec.mol, atom_weights=atom_weights, width=420, height=280)

        top5_cos, max_tan, ad_tag, analogs = self._retrieve_nearest_analogs(ft_emb, ecfp, top_k=3)
        ad_level, ad_status = ad_tag.split("|", 1)
        ad_mean = float(self.ad_config.get("mean_top5_cosine", 0.868))
        ad_std = max(float(self.ad_config.get("std_top5_cosine", 0.0531)), 1e-6)
        z_dist = round((ad_mean - top5_cos) / ad_std, 3)
        in_ad = ad_level in ("high", "moderate")

        thr_ft = float(self.thresholds_all.get("ChemBERTa-2 (Fine-Tuned LLM)", 0.625))
        thr_ecfp = float(self.thresholds_all.get("ECFP4 + RF (Baseline)", 0.59))
        lbl_ft = int(p_ft >= thr_ft)
        lbl_struct = int(p_struct >= thr_ecfp)
        consensus_votes = pred_label + lbl_ft + lbl_struct

        nce_prof = compute_nce_profile(rec.mol)

        return {
            "valid": True,
            "input_smiles": smiles.strip(),
            "canonical_smiles": rec.canonical_smiles,
            "inchikey": rec.inchikey,
            "murcko_scaffold": rec.murcko_smiles,
            "predicted_label": pred_label,
            "prediction_text": "Active (EGFR C797S Inhibitor)" if pred_label == 1 else "Inactive",
            "probability": round(p_final, 4),
            "threshold": self.threshold,
            "consensus_active_votes": f"{consensus_votes}/3",
            "model_decisions": [
                {
                    "model": "Hybrid ChemBERTa-2 + Chiral ECFP4 (Primary)",
                    "probability": round(p_final, 4),
                    "cutoff": round(self.threshold, 3),
                    "prediction": "Active (1)" if pred_label == 1 else "Inactive (0)",
                },
                {
                    "model": "ChemBERTa-2 (End-to-End Fine-Tuned LLM)",
                    "probability": round(p_ft, 4),
                    "cutoff": round(thr_ft, 3),
                    "prediction": "Active (1)" if lbl_ft == 1 else "Inactive (0)",
                },
                {
                    "model": "Chiral ECFP4 + RF Structural Ensemble",
                    "probability": round(p_struct, 4),
                    "cutoff": round(thr_ecfp, 3),
                    "prediction": "Active (1)" if lbl_struct == 1 else "Inactive (0)",
                },
            ],
            "component_scores": {
                "chemberta_finetuned_llm": round(p_ft, 4),
                "multimodal_llm_mlp": round(p_hyb_mlp, 4),
                "chiral_ecfp4_ensemble": round(p_struct, 4),
            },
            "applicability_domain": {
                "in_domain": in_ad,
                "level": ad_level,
                "status": ad_status,
                "top5_llm_cosine": top5_cos,
                "strict_cutoff": self.ad_config["cutoff_strict"],
                "warn_cutoff": self.ad_config["cutoff_warn"],
                "z_distance": z_dist,
                "max_tanimoto": max_tan,
                "nearest_train_no": analogs[0]["No"] if analogs else None,
            },
            "profile": nce_prof,
            "atom_weights": atom_weights.tolist(),
            "plain_svg": plain_svg,
            "explain_svg": explain_svg,
            "nearest_analogs": analogs,
        }

    def predict_batch(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Fast vectorized batch screening for CSV uploads of NCE libraries."""
        results: List[Dict[str, Any]] = []
        valid_indices: List[int] = []
        valid_recs = []

        for idx, row in enumerate(items):
            name = str(row.get("id") or row.get("name") or idx + 1)
            smi = str(row.get("smiles") or "").strip()
            rec = standardize_smiles(smi, uncharge=False)
            if rec is None:
                results.append(
                    {
                        "ID": name,
                        "Input_SMILES": smi,
                        "Valid": False,
                        "Prediction": "Invalid SMILES",
                        "Predicted_Class": None,
                        "Probability_Active": None,
                        "ChemBERTa_LLM_Score": None,
                        "ECFP4_Score": None,
                        "Consensus_Active_Votes": "N/A",
                        "In_AD": False,
                        "AD_Status": "Invalid Structure",
                        "AD_LLM_Cosine_Sim": None,
                        "AD_Z_Distance": None,
                        "Max_Tanimoto_Train": None,
                        "Nearest_Train_No": None,
                        "MW": None,
                        "cLogP": None,
                        "TPSA": None,
                        "QED": None,
                        "Lipinski_Pass": None,
                        "Canonical_SMILES": "",
                    }
                )
            else:
                valid_indices.append(idx)
                valid_recs.append((name, smi, rec))
                results.append({})

        if not valid_recs:
            return results

        can_list = [r[2].canonical_smiles for r in valid_recs]
        ecfp_mat = np.vstack([compute_chiral_ecfp4_array(r[2].mol) for r in valid_recs])
        phys_mat = np.vstack([compute_physchem_vector(r[2].mol) for r in valid_recs])
        phys_sc_mat = self.phys_scaler.transform(phys_mat)
        X_struct = np.hstack([ecfp_mat, phys_sc_mat])

        llm_frozen_mat = self._encode_smiles(can_list, self.pretrained_encoder, n_tta=3)
        ft_emb_mat = self._encode_smiles(can_list, self.ft_model.roberta, n_tta=3)

        p_ft = self._predict_ft_proba(can_list, n_tta=3)
        p_struct = 0.5 * self.rf_prod.predict_proba(X_struct)[:, 1] + 0.5 * self.et_prod.predict_proba(X_struct)[:, 1]
        X_comb = np.hstack([llm_frozen_mat * 4.0, ecfp_mat, phys_sc_mat])
        p_mlp = self.hyb_mlp_prod.predict_proba(X_comb)[:, 1]

        meta_in = np.column_stack([p_ft, p_mlp, p_struct])
        p_final = self.meta_final.predict_proba(meta_in)[:, 1]

        cos_mat = cosine_similarity(ft_emb_mat, self.ref["ft_embeddings"])
        top5_cos_arr = np.sort(cos_mat, axis=1)[:, -5:].mean(axis=1)

        ref_ecfp = self.ref["ecfp4"]
        inter = np.dot(ecfp_mat, ref_ecfp.T)
        union = ecfp_mat.sum(axis=1, keepdims=True) + ref_ecfp.sum(axis=1)[None, :] - inter
        tanimoto_mat = np.divide(inter, np.maximum(union, 1e-9))
        max_tan_arr = np.max(tanimoto_mat, axis=1)
        nearest_idx_arr = np.argmax(tanimoto_mat, axis=1)

        ad_mean = float(self.ad_config.get("mean_top5_cosine", 0.868))
        ad_std = max(float(self.ad_config.get("std_top5_cosine", 0.0531)), 1e-6)
        thr_ft = float(self.thresholds_all.get("ChemBERTa-2 (Fine-Tuned LLM)", 0.625))
        thr_ecfp = float(self.thresholds_all.get("ECFP4 + RF (Baseline)", 0.59))

        for k, out_idx in enumerate(valid_indices):
            name, raw_smi, rec = valid_recs[k]
            prof = compute_nce_profile(rec.mol)
            pf = float(p_final[k])
            pft_k = float(p_ft[k])
            pst_k = float(p_struct[k])
            lbl = int(pf >= self.threshold)
            votes = lbl + int(pft_k >= thr_ft) + int(pst_k >= thr_ecfp)

            t5c = float(top5_cos_arr[k])
            mt = float(max_tan_arr[k])
            z_d = round((ad_mean - t5c) / ad_std, 3)
            nearest_no = int(self.ref["No"][int(nearest_idx_arr[k])])

            if t5c >= self.ad_config["cutoff_warn"] or mt >= 0.55:
                ad_st = "Inside AD (High)"
                in_ad = True
            elif t5c >= self.ad_config["cutoff_strict"] or mt >= 0.38:
                ad_st = "Inside AD (Borderline)"
                in_ad = True
            else:
                ad_st = "Outside AD (Extrapolation)"
                in_ad = False

            results[out_idx] = {
                "ID": name,
                "Input_SMILES": raw_smi,
                "Valid": True,
                "Prediction": "Active" if lbl == 1 else "Inactive",
                "Predicted_Class": lbl,
                "Probability_Active": round(pf, 4),
                "ChemBERTa_LLM_Score": round(pft_k, 4),
                "ECFP4_Score": round(pst_k, 4),
                "Consensus_Active_Votes": f"{votes}/3",
                "In_AD": in_ad,
                "AD_Status": ad_st,
                "AD_LLM_Cosine_Sim": round(t5c, 4),
                "AD_Z_Distance": z_d,
                "Max_Tanimoto_Train": round(mt, 3),
                "Nearest_Train_No": nearest_no,
                "MW": prof["MW"],
                "cLogP": prof["cLogP"],
                "TPSA": prof["TPSA"],
                "QED": prof["QED"],
                "Lipinski_Pass": prof["lipinski_pass"],
                "Canonical_SMILES": rec.canonical_smiles,
            }
        return results
