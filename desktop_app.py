"""
Standalone Native Windows Desktop GUI Tool for EGFR C797S NCE Activity &
OECD Applicability Domain Prediction.

Developed by: Mr. Arijit Bhattacharya and Dr. Om Silakari,
Department of Pharmaceutical Sciences and Drug Research,
Punjabi University, Patiala, Punjab, India.
"""
from __future__ import annotations

import io
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from rdkit import Chem

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chemistry import render_mol_png_bytes
from src.predictor import EGFRC797SPredictor

CREDIT_TEXT = (
    "Developed by: Mr. Arijit Bhattacharya and Dr. Om Silakari, "
    "Department of Pharmaceutical Sciences and Drug Research, "
    "Punjabi University, Patiala, Punjab, India"
)

PRESETS: Dict[str, str] = {
    "1. Brigatinib-like P(=O)Me2 NCE [Active | Inside AD]": "CP(=O)(C)c1ccccc1Nc1nc(Nc2cc(C)c(N3CCC(N4CCN(C)CC4)CC3)cc2OC)ncc1Br",
    "2. N-SO2Et Indole Rescue Cliff [Active | Inside AD]": "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(S(=O)(=O)CC)c4ccccc34)n2)cc2cnn(CCN(C)C)c12",
    "3. N-Me Indole Acrylamide Cliff [Inactive | Inside AD]": "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)cc2cnn(CCN(C)C)c12",
    "4. Chiral [C@H] Enantiomer [Active | Inside AD]": "C([C@H](C)NCc1c(nn(c1)c1nc(ncc1)Nc1c(cc(c(c1)C)N1CC[C@H](CC1)N1CCN(CC1)C)OC)c1ccccc1)O",
    "5. Chiral [C@@H] Enantiomer [Inactive | Inside AD]": "C([C@@H](C)NCc1c(nn(c1)c1nc(ncc1)Nc1c(cc(c(c1)C)N1CC[C@H](CC1)N1CCN(CC1)C)OC)c1ccccc1)O",
    "6. Sulfonamide-Aminothiazole NCE [Active | Inside AD]": "CCS(=O)(=O)Nc1nc(-c2ccc(F)cc2)c(-c2ccnc(Nc3ccc4n(CCN(C)C)ccc4c3)n2)s1",
    "7. Unrelated Steroid-like Scaffold [Outside AD Demo]": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
}


def make_ad_plot_pil(pred_obj: EGFRC797SPredictor, nce_cos: float, width_px: int = 440, height_px: int = 230) -> Image.Image:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    tr_sims = pred_obj.train_top5_sims
    strict_cut = pred_obj.ad_config["cutoff_strict"]
    warn_cut = pred_obj.ad_config["cutoff_warn"]

    ax.axvspan(0.40, strict_cut, color="#fee2e2", alpha=0.6, label="Outside AD")
    ax.axvspan(strict_cut, warn_cut, color="#fef3c7", alpha=0.6, label="Borderline")
    ax.axvspan(warn_cut, 1.02, color="#dcfce7", alpha=0.5, label="Inside AD")
    ax.hist(tr_sims, bins=20, density=True, color="#64748b", alpha=0.38, edgecolor="white")
    ax.axvline(strict_cut, color="#dc2626", linestyle="--", lw=1.5, label=f"Cutoff ({strict_cut:.3f})")

    x_nce = float(np.clip(nce_cos, 0.42, 1.0))
    ax.axvline(x_nce, color="#3730a3", lw=2.2)
    ax.scatter([x_nce], [4.2], color="#4f46e5", edgecolors="white", s=130, zorder=6, marker="*", label=f"NCE ({nce_cos:.3f})")

    ax.set_xlim(max(0.45, min(float(tr_sims.min()) - 0.08, x_nce - 0.05)), 1.01)
    ax.set_xlabel("ChemBERTa-2 Top-5 Cosine Similarity", fontsize=8)
    ax.set_title("OECD Applicability Domain Position", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    ax.legend(loc="upper left", fontsize=6.8, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


class StandaloneEGFRApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EGFR-C797S NCE Predictor v1.0 | Molecular Modeling Lab, Punjabi University, Patiala")
        self.geometry("1360x840")
        self.minsize(1120, 720)
        self.configure(bg="#f1f5f9")

        ico_path = ROOT_DIR / "assets" / "mml_icon.ico"
        if ico_path.exists():
            try:
                self.iconbitmap(str(ico_path))
            except Exception:
                pass

        self.predictor: Optional[EGFRC797SPredictor] = None
        self.last_single_result: Optional[Dict[str, Any]] = None
        self.batch_df: Optional[pd.DataFrame] = None
        self._img_refs: Dict[str, ImageTk.PhotoImage] = {}

        self._setup_styles()
        self._build_header()
        self._build_footer()
        self._build_notebook()

        # Load model asynchronously so window opens instantaneously
        threading.Thread(target=self._init_model_bg, daemon=True).start()

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background="#f1f5f9", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(16, 8))
        style.map("TNotebook.Tab", background=[("selected", "#1e3a8a")], foreground=[("selected", "white")])
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#e2e8f0", foreground="#0f172a")
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=26)

    def _load_logo(self, filename: str, target_h: int) -> Optional[ImageTk.PhotoImage]:
        p = ROOT_DIR / "assets" / filename
        if not p.exists():
            return None
        im = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        comp = Image.alpha_composite(bg, im).convert("RGB")
        ratio = target_h / float(comp.height)
        target_w = max(1, int(comp.width * ratio))
        resized = comp.resize((target_w, target_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        self._img_refs[f"logo_{filename}_{target_h}"] = photo
        return photo

    def _build_header(self) -> None:
        header = tk.Frame(self, bg="#ffffff", bd=0, highlightthickness=2, highlightbackground="#1e3a8a")
        header.pack(side=tk.TOP, fill=tk.X)

        inner = tk.Frame(header, bg="#ffffff", padx=16, pady=8)
        inner.pack(fill=tk.X)

        # Left Logo: Punjabi University, Patiala
        pu_photo = self._load_logo("punjabi_university_logo.png", target_h=68)
        if pu_photo is not None:
            tk.Label(inner, image=pu_photo, bg="#ffffff").pack(side=tk.LEFT, padx=(0, 16))

        # Right Logo: MML Molecular Modeling Lab
        mml_photo = self._load_logo("mml_logo.png", target_h=78)
        if mml_photo is not None:
            tk.Label(inner, image=mml_photo, bg="#ffffff").pack(side=tk.RIGHT, padx=(16, 0))

        # Center Institutional Title & Author Credits
        center = tk.Frame(inner, bg="#ffffff")
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            center,
            text="EGFR-C797S NCE Activity & Applicability Domain Predictor",
            font=("Segoe UI", 15, "bold"),
            fg="#0f172a",
            bg="#ffffff",
        ).pack(anchor="center")

        tk.Label(
            center,
            text="Hybrid Fine-Tuned ChemBERTa-2 Chemical Language Model & Chirality-Aware QSAR Virtual Screening Suite",
            font=("Segoe UI", 9, "italic"),
            fg="#475569",
            bg="#ffffff",
        ).pack(anchor="center", pady=(1, 3))

        tk.Label(
            center,
            text=CREDIT_TEXT,
            font=("Segoe UI", 9, "bold"),
            fg="#1e3a8a",
            bg="#eff6ff",
            padx=12,
            pady=3,
        ).pack(anchor="center")

    def _build_footer(self) -> None:
        footer = tk.Frame(self, bg="#0f172a", padx=14, pady=6)
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="⏳ Initializing ChemBERTa-2 LLM & Applicability Domain Engine on CPU...")
        tk.Label(
            footer,
            textvariable=self.status_var,
            font=("Segoe UI", 8, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
        ).pack(side=tk.LEFT)

        tk.Label(
            footer,
            text=CREDIT_TEXT,
            font=("Segoe UI", 8),
            fg="#e2e8f0",
            bg="#0f172a",
        ).pack(side=tk.RIGHT)

    def _build_notebook(self) -> None:
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.tab_single = tk.Frame(self.nb, bg="#f8fafc")
        self.tab_batch = tk.Frame(self.nb, bg="#f8fafc")
        self.tab_val = tk.Frame(self.nb, bg="#f8fafc")

        self.nb.add(self.tab_single, text="  1. Single NCE Prediction & Applicability Domain  ")
        self.nb.add(self.tab_batch, text="  2. Batch CSV Virtual Screening (Active/Inactive + AD)  ")
        self.nb.add(self.tab_val, text="  3. Internal & External Validation Report  ")

        self._build_single_tab()
        self._build_batch_tab()
        self._build_val_tab()

    # ------------------------------------------------------------------
    # TAB 1: SINGLE NCE PREDICTION
    # ------------------------------------------------------------------
    def _build_single_tab(self) -> None:
        left = tk.Frame(self.tab_single, bg="#ffffff", width=410, padx=14, pady=12, highlightthickness=1, highlightbackground="#cbd5e1")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 6), pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="Input New Chemical Entity (NCE)", font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#0f172a").pack(anchor="w")
        tk.Label(left, text="Load Benchmark / Activity-Cliff / AD Preset:", font=("Segoe UI", 8, "bold"), bg="#ffffff", fg="#475569").pack(anchor="w", pady=(8, 2))

        self.preset_var = tk.StringVar(value=list(PRESETS.keys())[0])
        preset_cb = ttk.Combobox(left, textvariable=self.preset_var, values=list(PRESETS.keys()), state="readonly", font=("Segoe UI", 9))
        preset_cb.pack(fill=tk.X, pady=(0, 8))
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_change)

        tk.Label(left, text="NCE SMILES String (Isomeric SMILES @/@@ Supported):", font=("Segoe UI", 8, "bold"), bg="#ffffff", fg="#475569").pack(anchor="w")
        self.smiles_txt = tk.Text(left, height=4, font=("Consolas", 9), wrap=tk.CHAR, bd=1, relief=tk.SOLID)
        self.smiles_txt.pack(fill=tk.X, pady=(2, 8))
        self.smiles_txt.insert("1.0", list(PRESETS.values())[0])

        btn_row = tk.Frame(left, bg="#ffffff")
        btn_row.pack(fill=tk.X, pady=(0, 10))
        self.pred_btn = tk.Button(
            btn_row,
            text="Predict Activity & Applicability Domain",
            font=("Segoe UI", 9, "bold"),
            bg="#1e3a8a",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief=tk.FLAT,
            pady=7,
            cursor="hand2",
            command=self.run_single_prediction,
        )
        self.pred_btn.pack(fill=tk.X)

        exp_row = tk.Frame(left, bg="#ffffff")
        exp_row.pack(fill=tk.X, pady=(4, 10))
        tk.Button(
            exp_row, text="Clear SMILES", font=("Segoe UI", 8, "bold"), bg="#e2e8f0", relief=tk.FLAT, padx=10, pady=4,
            command=lambda: self.smiles_txt.delete("1.0", tk.END)
        ).pack(side=tk.LEFT)
        tk.Button(
            exp_row, text="Save NCE Report (.txt)", font=("Segoe UI", 8, "bold"), bg="#e0e7ff", fg="#1e3a8a", relief=tk.FLAT, padx=10, pady=4,
            command=self.export_single_report
        ).pack(side=tk.RIGHT)

        tk.Label(left, text="Multi-Model Consensus Breakdown", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#0f172a").pack(anchor="w", pady=(6, 4))
        self.consensus_tree = ttk.Treeview(left, columns=("Model", "Call", "Prob", "Cut"), show="headings", height=4)
        self.consensus_tree.heading("Model", text="Model")
        self.consensus_tree.heading("Call", text="Class")
        self.consensus_tree.heading("Prob", text="P(Act)")
        self.consensus_tree.heading("Cut", text="Cutoff")
        self.consensus_tree.column("Model", width=175)
        self.consensus_tree.column("Call", width=75, anchor="center")
        self.consensus_tree.column("Prob", width=60, anchor="center")
        self.consensus_tree.column("Cut", width=55, anchor="center")
        self.consensus_tree.pack(fill=tk.X)

        tk.Label(left, text="Physicochemical & Drug-Likeness Profile", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#0f172a").pack(anchor="w", pady=(10, 4))
        self.phys_lbl = tk.Label(left, text="--", font=("Consolas", 8), bg="#f8fafc", fg="#1e293b", justify=tk.LEFT, anchor="nw", padx=8, pady=8, relief=tk.SOLID, bd=1)
        self.phys_lbl.pack(fill=tk.BOTH, expand=True)

        # Right Results Panel
        right = tk.Frame(self.tab_single, bg="#f8fafc")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 8), pady=8)

        # Top Verdict & AD Banners
        banners = tk.Frame(right, bg="#f8fafc")
        banners.pack(fill=tk.X)

        self.verdict_frame = tk.Frame(banners, bg="#ecfdf5", padx=14, pady=10, highlightthickness=2, highlightbackground="#10b981")
        self.verdict_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.verdict_title = tk.Label(self.verdict_frame, text="READY TO SCREEN", font=("Segoe UI", 14, "bold"), bg="#ecfdf5", fg="#065f46")
        self.verdict_title.pack(anchor="w")
        self.verdict_sub = tk.Label(self.verdict_frame, text="Enter SMILES and click Predict", font=("Segoe UI", 9, "bold"), bg="#ecfdf5", fg="#047857")
        self.verdict_sub.pack(anchor="w")

        self.ad_frame = tk.Frame(banners, bg="#f0fdf4", padx=14, pady=10, highlightthickness=2, highlightbackground="#86efac")
        self.ad_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.ad_title = tk.Label(self.ad_frame, text="APPLICABILITY DOMAIN (OECD)", font=("Segoe UI", 12, "bold"), bg="#f0fdf4", fg="#14532d")
        self.ad_title.pack(anchor="w")
        self.ad_sub = tk.Label(self.ad_frame, text="Cosine Sim: -- | Z-Distance: -- | Tanimoto: --", font=("Segoe UI", 8, "bold"), bg="#f0fdf4", fg="#166534")
        self.ad_sub.pack(anchor="w")

        # Visual Cards Row (Explainability PNG + Structure PNG)
        vis_row = tk.Frame(right, bg="#f8fafc")
        vis_row.pack(fill=tk.BOTH, expand=True, pady=(8, 6))

        c_exp = tk.Frame(vis_row, bg="#ffffff", padx=8, pady=6, highlightthickness=1, highlightbackground="#cbd5e1")
        c_exp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        tk.Label(c_exp, text="Atom Pharmacophore Attribution (Green = Pro-Active | Red = Detrimental)", font=("Segoe UI", 8, "bold"), bg="#ffffff", fg="#334155").pack()
        self.explain_img_lbl = tk.Label(c_exp, bg="#ffffff")
        self.explain_img_lbl.pack(fill=tk.BOTH, expand=True)

        c_str = tk.Frame(vis_row, bg="#ffffff", padx=8, pady=6, highlightthickness=1, highlightbackground="#cbd5e1")
        c_str.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(c_str, text="Standardized 2D Structure & Live Applicability Domain Plot", font=("Segoe UI", 8, "bold"), bg="#ffffff", fg="#334155").pack()
        self.ad_plot_lbl = tk.Label(c_str, bg="#ffffff")
        self.ad_plot_lbl.pack(fill=tk.BOTH, expand=True)

        # Bottom Nearest Analogs Strip
        analog_box = tk.Frame(right, bg="#ffffff", padx=12, pady=8, highlightthickness=1, highlightbackground="#cbd5e1")
        analog_box.pack(fill=tk.X)
        self.analog_lbl = tk.Label(
            analog_box,
            text="Closest Experimental Analogs in EGFR C797S Training Set: --",
            font=("Segoe UI", 8, "bold"),
            bg="#ffffff",
            fg="#1e293b",
            justify=tk.LEFT,
        )
        self.analog_lbl.pack(anchor="w")

    # ------------------------------------------------------------------
    # TAB 2: BATCH CSV VIRTUAL SCREENING
    # ------------------------------------------------------------------
    def _build_batch_tab(self) -> None:
        top_bar = tk.Frame(self.tab_batch, bg="#ffffff", padx=14, pady=10, highlightthickness=1, highlightbackground="#cbd5e1")
        top_bar.pack(fill=tk.X, padx=10, pady=8)

        tk.Button(
            top_bar, text="📂 Browse & Screen CSV Library...", font=("Segoe UI", 9, "bold"),
            bg="#1e3a8a", fg="white", relief=tk.FLAT, padx=14, pady=6, cursor="hand2",
            command=self.run_csv_file_dialog
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            top_bar, text="🧪 Load & Screen Sample 8-NCE Library", font=("Segoe UI", 9, "bold"),
            bg="#059669", fg="white", relief=tk.FLAT, padx=14, pady=6, cursor="hand2",
            command=self.run_sample_batch
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            top_bar, text="💾 Export Batch Predictions + AD to CSV...", font=("Segoe UI", 9, "bold"),
            bg="#4f46e5", fg="white", relief=tk.FLAT, padx=14, pady=6, cursor="hand2",
            command=self.save_batch_csv_dialog
        ).pack(side=tk.RIGHT)

        self.batch_summary_var = tk.StringVar(value="Upload any CSV file containing a 'SMILES' column to screen New Chemical Entities on CPU.")
        tk.Label(top_bar, textvariable=self.batch_summary_var, font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#334155").pack(side=tk.LEFT, padx=14)

        table_frame = tk.Frame(self.tab_batch, bg="#ffffff", highlightthickness=1, highlightbackground="#cbd5e1")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        cols = (
            "ID", "Prediction", "Probability_Active", "ChemBERTa_LLM_Score", "ECFP4_Score",
            "Consensus_Active_Votes", "AD_Status", "AD_LLM_Cosine_Sim", "AD_Z_Distance",
            "Max_Tanimoto_Train", "Nearest_Train_No", "MW", "cLogP", "TPSA", "QED", "Lipinski_Pass"
        )
        self.batch_tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        headers = {
            "ID": ("Compound ID", 150),
            "Prediction": ("Prediction", 80),
            "Probability_Active": ("P(Active)", 75),
            "ChemBERTa_LLM_Score": ("LLM Prob", 75),
            "ECFP4_Score": ("ECFP4 Prob", 80),
            "Consensus_Active_Votes": ("Votes", 55),
            "AD_Status": ("Applicability Domain", 165),
            "AD_LLM_Cosine_Sim": ("Cos Sim", 70),
            "AD_Z_Distance": ("Z-Dist", 65),
            "Max_Tanimoto_Train": ("Max Tan", 70),
            "Nearest_Train_No": ("Near No.", 65),
            "MW": ("MW", 65),
            "cLogP": ("cLogP", 58),
            "TPSA": ("TPSA", 58),
            "QED": ("QED", 55),
            "Lipinski_Pass": ("Lipinski", 65),
        }
        for c, (label, w) in headers.items():
            self.batch_tree.heading(c, text=label)
            self.batch_tree.column(c, width=w, anchor="center" if c != "ID" and c != "AD_Status" else "w")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.batch_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.batch_tree.xview)
        self.batch_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.batch_tree.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # TAB 3: MODEL VALIDATION & PLOTS
    # ------------------------------------------------------------------
    def _build_val_tab(self) -> None:
        left_v = tk.Frame(self.tab_val, bg="#ffffff", padx=14, pady=12, highlightthickness=1, highlightbackground="#cbd5e1")
        left_v.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=8)

        tk.Label(left_v, text="OECD Internal (5-Fold CV, N=222) & External Validation (Test, N=148)", font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#0f172a").pack(anchor="w", pady=(0, 6))

        vcols = ("Split", "Model", "TP", "FP", "TN", "FN", "SE", "SP", "ACC", "AUC", "MCC")
        self.val_tree = ttk.Treeview(left_v, columns=vcols, show="headings", height=11)
        widths = {"Split": 75, "Model": 215, "TP": 38, "FP": 38, "TN": 38, "FN": 38, "SE": 58, "SP": 58, "ACC": 58, "AUC": 58, "MCC": 58}
        for c in vcols:
            self.val_tree.heading(c, text=c)
            self.val_tree.column(c, width=widths[c], anchor="w" if c in ("Split", "Model") else "center")
        self.val_tree.pack(fill=tk.X, pady=(0, 10))

        about_card = tk.Frame(left_v, bg="#eff6ff", padx=12, pady=10, highlightthickness=1, highlightbackground="#bfdbfe")
        about_card.pack(fill=tk.X, pady=(6, 0))
        tk.Label(
            about_card,
            text=(
                "OECD Robustness & Applicability Domain Summary:\n"
                "• Stereo-InChIKey Overlap (Train vs. Test): 0 exact duplicates\n"
                "• 20-Run Y-Randomization Mean Scrambled AUC = 0.5001 ± 0.0420 | cRp² = 0.5424 (> 0.50 Pass)\n"
                "• Strict Applicability Domain Threshold (Top-5 ChemBERTa Cosine): 0.7618 (External Coverage = 97.3%)\n\n"
                f"{CREDIT_TEXT}"
            ),
            font=("Segoe UI", 8, "bold"),
            bg="#eff6ff",
            fg="#1e3a8a",
            justify=tk.LEFT,
        ).pack(anchor="w")

        right_v = tk.Frame(self.tab_val, bg="#ffffff", padx=12, pady=12, highlightthickness=1, highlightbackground="#cbd5e1")
        right_v.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=8)

        fig_bar = tk.Frame(right_v, bg="#ffffff")
        fig_bar.pack(fill=tk.X, pady=(0, 6))
        tk.Label(fig_bar, text="Select Diagnostic Plot:", font=("Segoe UI", 9, "bold"), bg="#ffffff").pack(side=tk.LEFT, padx=(0, 8))
        self.fig_map = {
            "External ROC & PR Curves": "roc_pr_curves.png",
            "Benchmark & Confusion Matrix": "confusion_and_metrics.png",
            "Applicability Domain & Y-Randomization": "applicability_domain_and_yrand.png",
            "ChemBERTa-2 Chemical Space PCA": "chemical_space_pca.png",
        }
        self.fig_choice = tk.StringVar(value="External ROC & PR Curves")
        fig_cb = ttk.Combobox(fig_bar, textvariable=self.fig_choice, values=list(self.fig_map.keys()), state="readonly", width=36)
        fig_cb.pack(side=tk.LEFT)
        fig_cb.bind("<<ComboboxSelected>>", lambda _e: self._show_validation_figure())

        self.val_fig_lbl = tk.Label(right_v, bg="#ffffff")
        self.val_fig_lbl.pack(fill=tk.BOTH, expand=True)
        self._show_validation_figure()

    # ------------------------------------------------------------------
    # LOGIC & CALLBACKS
    # ------------------------------------------------------------------
    def _init_model_bg(self) -> None:
        try:
            pred = EGFRC797SPredictor()
            self.predictor = pred
            self.after(0, self._on_model_ready)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Model Initialization Error", str(exc)))

    def _on_model_ready(self) -> None:
        self.status_var.set("✅ Ready | Offline CPU Inference Active (ChemBERTa-77M-MTR + Chiral ECFP4 + OECD AD)")
        self._populate_validation_table()
        self.run_single_prediction()

    def _populate_validation_table(self) -> None:
        if not self.predictor or not self.predictor.metrics_summary:
            return
        m = self.predictor.metrics_summary
        for item in self.val_tree.get_children():
            self.val_tree.delete(item)
        for name, v in m.get("cv_5fold_oof", {}).items():
            self.val_tree.insert("", tk.END, values=("Internal CV", name, v["TP"], v["FP"], v["TN"], v["FN"], v["Sensitivity"], v["Specificity"], v["Accuracy"], v["ROC_AUC"], v["MCC"]))
        for name, v in m.get("external_test", {}).items():
            self.val_tree.insert("", tk.END, values=("External Test", name, v["TP"], v["FP"], v["TN"], v["FN"], v["Sensitivity"], v["Specificity"], v["Accuracy"], v["ROC_AUC"], v["MCC"]))
        in_ad = m.get("applicability_domain", {}).get("test_in_domain_metrics")
        if in_ad:
            self.val_tree.insert("", tk.END, values=("Ext (In-AD)", "Hybrid (Inside AD, N=144)", in_ad["TP"], in_ad["FP"], in_ad["TN"], in_ad["FN"], in_ad["Sensitivity"], in_ad["Specificity"], in_ad["Accuracy"], in_ad["ROC_AUC"], in_ad["MCC"]))

    def _show_validation_figure(self) -> None:
        fname = self.fig_map.get(self.fig_choice.get(), "roc_pr_curves.png")
        fpath = ROOT_DIR / "reports" / "figures" / fname
        if not fpath.exists():
            return
        im = Image.open(fpath)
        im.thumbnail((560, 420), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(im)
        self._img_refs["val_fig"] = photo
        self.val_fig_lbl.configure(image=photo)

    def _on_preset_change(self, _event=None) -> None:
        smi = PRESETS.get(self.preset_var.get(), "")
        if smi:
            self.smiles_txt.delete("1.0", tk.END)
            self.smiles_txt.insert("1.0", smi)
            self.run_single_prediction()

    def run_single_prediction(self) -> None:
        if self.predictor is None:
            messagebox.showinfo("Loading", "Model is still warming up on CPU. Please wait a few seconds.")
            return
        smi = self.smiles_txt.get("1.0", tk.END).strip()
        if not smi:
            messagebox.showwarning("Input Required", "Please enter a SMILES string.")
            return

        res = self.predictor.predict_single(smi)
        if not res.get("valid", False):
            messagebox.showerror("Invalid Chemical Structure", res.get("error", "Could not parse SMILES."))
            return

        self.last_single_result = res
        is_act = res["predicted_label"] == 1
        prob_pct = res["probability"] * 100.0

        # Update Activity Banner
        bg_v = "#ecfdf5" if is_act else "#fef2f2"
        bd_v = "#10b981" if is_act else "#ef4444"
        fg_v = "#065f46" if is_act else "#991b1b"
        self.verdict_frame.configure(bg=bg_v, highlightbackground=bd_v)
        self.verdict_title.configure(
            text=f"{'ACTIVE (Class 1)' if is_act else 'INACTIVE (Class 0)'} — {prob_pct:.1f}%",
            bg=bg_v, fg=fg_v
        )
        self.verdict_sub.configure(
            text=f"Consensus: {res['consensus_active_votes']} Active Votes | Cutoff: {res['threshold']*100:.1f}%",
            bg=bg_v, fg=fg_v
        )

        # Update AD Banner
        ad = res["applicability_domain"]
        ad_colors = {
            "high": ("#f0fdf4", "#22c55e", "#14532d"),
            "moderate": ("#fffbeb", "#f59e0b", "#78350f"),
            "low": ("#fef2f2", "#ef4444", "#7f1d1d"),
        }
        bg_a, bd_a, fg_a = ad_colors.get(ad["level"], ad_colors["high"])
        self.ad_frame.configure(bg=bg_a, highlightbackground=bd_a)
        self.ad_title.configure(
            text=("INSIDE AD" if ad["in_domain"] else "OUTSIDE AD") + f" — {ad['status']}",
            bg=bg_a, fg=fg_a
        )
        self.ad_sub.configure(
            text=f"LLM Cosine: {ad['top5_llm_cosine']:.4f} (Cut: {ad['strict_cutoff']}) | Z: {ad['z_distance']} | Max Tanimoto: {ad['max_tanimoto']:.3f}",
            bg=bg_a, fg=fg_a
        )

        # Update Consensus Tree
        for it in self.consensus_tree.get_children():
            self.consensus_tree.delete(it)
        for row in res["model_decisions"]:
            short_m = row["model"].replace(" (Primary)", "").replace(" (End-to-End Fine-Tuned LLM)", " Fine-Tuned").replace(" Structural Ensemble", "")
            self.consensus_tree.insert("", tk.END, values=(short_m, row["prediction"], f"{row['probability']*100:.1f}%", f"{row['cutoff']*100:.1f}%"))

        # Update Physicochemical Profile
        p = res["profile"]
        motifs = ", ".join(p["detected_motifs"]) if p["detected_motifs"] else "None (Novel scaffold)"
        phys_str = (
            f"Formula    : {p['formula']}    | MW    : {p['MW']} Da\n"
            f"cLogP      : {p['cLogP']}          | TPSA  : {p['TPSA']} Å²\n"
            f"HBD / HBA  : {p['HBD']} / {p['HBA']}        | QED   : {p['QED']}\n"
            f"RotBonds   : {p['RotBonds']}            | HetCyc: {p['HeteroCycles']}\n"
            f"Lipinski   : {'PASS' if p['lipinski_pass'] else 'FAIL'} ({p['lipinski_violations']} viol.) | Veber : {'PASS' if p['veber_pass'] else 'FAIL'}\n"
            f"Motifs     : {motifs[:55]}"
        )
        self.phys_lbl.configure(text=phys_str)

        # Render High-DPI Explainability PNG & AD Plot
        mol = Chem.MolFromSmiles(res["canonical_smiles"])
        aw = np.array(res["atom_weights"], dtype=np.float32)
        png_exp = render_mol_png_bytes(mol, atom_weights=aw, width=430, height=270)
        im_exp = Image.open(io.BytesIO(png_exp))
        ph_exp = ImageTk.PhotoImage(im_exp)
        self._img_refs["exp_mol"] = ph_exp
        self.explain_img_lbl.configure(image=ph_exp)

        im_ad = make_ad_plot_pil(self.predictor, ad["top5_llm_cosine"], width_px=430, height_px=255)
        ph_ad = ImageTk.PhotoImage(im_ad)
        self._img_refs["ad_plot"] = ph_ad
        self.ad_plot_lbl.configure(image=ph_ad)

        analog_parts = [
            f"Train No. {a['No']} [{a['activity_text']} | Tan={a['tanimoto']}, LLM={a['llm_cosine']}]"
            for a in res["nearest_analogs"]
        ]
        self.analog_lbl.configure(text="Top-3 Nearest Training Analogs:   " + "   •   ".join(analog_parts))

    def export_single_report(self) -> None:
        if not self.last_single_result:
            return
        d = self.last_single_result
        ad = d["applicability_domain"]
        p = d["profile"]
        save_path = filedialog.asksaveasfilename(
            title="Save Single NCE Report",
            defaultextension=".txt",
            initialfile="EGFR_C797S_NCE_Report.txt",
            filetypes=[("Text Report", "*.txt")],
        )
        if not save_path:
            return
        report_lines = [
            "==========================================================================",
            "EGFR-C797S NCE ACTIVITY & APPLICABILITY DOMAIN REPORT",
            CREDIT_TEXT,
            "==========================================================================",
            f"Input SMILES         : {d['input_smiles']}",
            f"Canonical SMILES     : {d['canonical_smiles']}",
            f"InChIKey             : {d['inchikey']}",
            f"Predicted Activity   : {d['prediction_text']} (Class = {d['predicted_label']})",
            f"Hybrid Probability   : {d['probability']:.4f} (Cutoff = {d['threshold']})",
            f"Model Consensus      : {d['consensus_active_votes']} Active Votes",
            "--------------------------------------------------------------------------",
            "OECD APPLICABILITY DOMAIN (AD)",
            f"Inside AD            : {'YES' if ad['in_domain'] else 'NO (Extrapolation Warning)'}",
            f"AD Status            : {ad['status']}",
            f"ChemBERTa Cosine Sim : {ad['top5_llm_cosine']:.4f} (Strict Cutoff = {ad['strict_cutoff']})",
            f"AD Z-Distance        : {ad['z_distance']} (Pass <= 2.0)",
            f"Max ECFP4 Tanimoto   : {ad['max_tanimoto']:.3f} (Closest Train Compound = No. {ad['nearest_train_no']})",
            "--------------------------------------------------------------------------",
            "MEDICINAL CHEMISTRY PROFILE",
            f"Formula: {p['formula']} | MW: {p['MW']} Da | cLogP: {p['cLogP']} | TPSA: {p['TPSA']} A^2 | QED: {p['QED']}",
            f"Lipinski Ro5: {'PASS' if p['lipinski_pass'] else 'FAIL'} | Veber: {'PASS' if p['veber_pass'] else 'FAIL'}",
            "==========================================================================",
        ]
        Path(save_path).write_text("\n".join(report_lines), encoding="utf-8")
        messagebox.showinfo("Saved", f"Saved NCE report to:\n{save_path}")

    def run_sample_batch(self) -> None:
        sample_path = ROOT_DIR / "sample_nce_input.csv"
        if sample_path.exists():
            self._process_batch_df(pd.read_csv(sample_path))

    def run_csv_file_dialog(self) -> None:
        fpath = filedialog.askopenfilename(title="Select NCE CSV File", filetypes=[("CSV Files", "*.csv")])
        if not fpath:
            return
        try:
            df_in = pd.read_csv(fpath)
            self._process_batch_df(df_in)
        except Exception as exc:
            messagebox.showerror("CSV Read Error", str(exc))

    def _process_batch_df(self, df_in: pd.DataFrame) -> None:
        if self.predictor is None:
            messagebox.showinfo("Loading", "Model is still warming up on CPU.")
            return
        smi_col = next((c for c in df_in.columns if c.strip().lower() in ("smiles", "canonical_smiles", "smi")), None)
        id_col = next((c for c in df_in.columns if c.strip().lower() in ("id", "no", "no.", "name", "compound_id")), None)
        if smi_col is None:
            messagebox.showerror("Missing SMILES Column", f"CSV must contain a 'SMILES' column.\nFound: {list(df_in.columns)}")
            return

        items: List[Dict[str, str]] = []
        for idx, row in df_in.head(1000).iterrows():
            sv = str(row[smi_col]).strip()
            if sv and sv.lower() != "nan":
                cid = str(row[id_col]).strip() if id_col else f"NCE-{idx+1}"
                items.append({"id": cid, "smiles": sv})

        self.status_var.set(f"⏳ Screening {len(items)} NCEs on CPU...")
        self.update_idletasks()

        results = self.predictor.predict_batch(items)
        self.batch_df = pd.DataFrame(results)

        for it in self.batch_tree.get_children():
            self.batch_tree.delete(it)

        cols = self.batch_tree["columns"]
        for _, r in self.batch_df.iterrows():
            vals = [r.get(c, "") for c in cols]
            self.batch_tree.insert("", tk.END, values=vals)

        n_act = int((self.batch_df["Predicted_Class"] == 1).sum())
        n_ad = int((self.batch_df["In_AD"] == True).sum())
        self.batch_summary_var.set(f"Screened {len(self.batch_df)} NCEs  |  Active: {n_act}  |  Inside AD: {n_ad}")
        self.status_var.set(f"✅ Batch screening complete ({len(self.batch_df)} NCEs evaluated).")

    def save_batch_csv_dialog(self) -> None:
        if self.batch_df is None or self.batch_df.empty:
            messagebox.showwarning("No Results", "Run a batch screening first.")
            return
        save_p = filedialog.asksaveasfilename(
            title="Export Batch Predictions with Applicability Domain",
            defaultextension=".csv",
            initialfile="EGFR_C797S_Batch_Predictions_with_AD.csv",
            filetypes=[("CSV File", "*.csv")],
        )
        if save_p:
            self.batch_df.to_csv(save_p, index=False)
            messagebox.showinfo("Export Successful", f"Saved {len(self.batch_df)} predictions to:\n{save_p}")


if __name__ == "__main__":
    app = StandaloneEGFRApp()
    app.mainloop()
