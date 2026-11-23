"""
Streamlit Web Application for EGFR C797S Mutant NCE Activity & Applicability Domain Prediction
powered by Fine-Tuned ChemBERTa-2 (77M-MTR) + Chirality-Aware ECFP4 Hybrid Ensemble.

Run locally:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predictor import EGFRC797SPredictor

try:
    from streamlit_ketcher import st_ketcher
    HAS_KETCHER = True
except Exception:
    HAS_KETCHER = False

CREDIT_LINE = (
    "Developed by: Mr. Arijit Bhattacharya and Dr. Om Silakari, "
    "Department of Pharmaceutical Sciences and Drug Research, "
    "Punjabi University, Patiala, Punjab, India"
)


def img_to_b64(rel_path: str) -> str:
    p = ROOT_DIR / rel_path
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode("utf-8")


PU_LOGO_B64 = img_to_b64("assets/punjabi_university_logo.png")
MML_LOGO_B64 = img_to_b64("assets/mml_logo.png")

st.set_page_config(
    page_title="EGFR C797S NCE Activity & Applicability Domain Predictor | Punjabi University",
    page_icon=str(ROOT_DIR / "assets" / "mml_logo.png") if (ROOT_DIR / "assets" / "mml_logo.png").exists() else "🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #312e81 100%);
        color: white;
        padding: 1.35rem 1.75rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
    }
    .hero-box h1 { margin: 0; font-size: 1.6rem; color: white; }
    .hero-box p { margin: 0.3rem 0 0; color: #cbd5e1; font-size: 0.92rem; }
    .verdict-active {
        background: #ecfdf5; border: 2px solid #10b981; color: #065f46;
        padding: 1.15rem 1.35rem; border-radius: 12px; margin-bottom: 1rem;
    }
    .verdict-inactive {
        background: #fef2f2; border: 2px solid #ef4444; color: #991b1b;
        padding: 1.15rem 1.35rem; border-radius: 12px; margin-bottom: 1rem;
    }
    .ad-card-high {
        background: #f0fdf4; border: 1.5px solid #86efac; color: #14532d;
        padding: 1rem 1.2rem; border-radius: 10px; margin-bottom: 1rem;
    }
    .ad-card-moderate {
        background: #fffbeb; border: 1.5px solid #fcd34d; color: #78350f;
        padding: 1rem 1.2rem; border-radius: 10px; margin-bottom: 1rem;
    }
    .ad-card-low {
        background: #fef2f2; border: 1.5px solid #fca5a5; color: #7f1d1d;
        padding: 1rem 1.2rem; border-radius: 10px; margin-bottom: 1rem;
    }
    .svg-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 10px; text-align: center;
    }
    .badge-pill {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 700; margin-right: 6px;
    }
    .b-high { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
    .b-mod  { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
    .b-low  { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Fine-Tuned ChemBERTa-2 LLM & Applicability Domain Reference...")
def load_predictor() -> EGFRC797SPredictor:
    return EGFRC797SPredictor()


def svg_to_html(svg_text: str, max_width: int = 420) -> str:
    b64 = base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")
    return (
        f'<div class="svg-card">'
        f'<img src="data:image/svg+xml;base64,{b64}" style="max-width:{max_width}px; width:100%; height:auto;" />'
        f"</div>"
    )


def render_live_ad_plot(pred_obj: EGFRC797SPredictor, nce_cos: float, nce_prob: float) -> bytes:
    """Plot the queried NCE directly onto the ChemBERTa-2 Applicability Domain space."""
    fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=180)
    tr_sims = pred_obj.train_top5_sims
    strict_cut = pred_obj.ad_config["cutoff_strict"]
    warn_cut = pred_obj.ad_config["cutoff_warn"]

    ax.axvspan(0.40, strict_cut, color="#fee2e2", alpha=0.55, label="Outside AD (Extrapolation)")
    ax.axvspan(strict_cut, warn_cut, color="#fef3c7", alpha=0.55, label="Borderline AD")
    ax.axvspan(warn_cut, 1.02, color="#dcfce7", alpha=0.45, label="Inside AD (High Reliability)")

    ax.hist(tr_sims, bins=22, density=True, color="#64748b", alpha=0.35, edgecolor="white", label="Training Set Density (N=222)")
    ax.axvline(strict_cut, color="#dc2626", linestyle="--", lw=1.6, label=f"Strict Cutoff ({strict_cut:.3f})")

    x_nce = float(np.clip(nce_cos, 0.42, 1.0))
    ax.axvline(x_nce, color="#4f46e5", lw=2.4)
    ax.scatter([x_nce], [4.5], color="#4f46e5", edgecolors="white", s=150, zorder=6, marker="*", label=f"Queried NCE ({nce_cos:.3f})")

    ax.set_xlim(max(0.45, min(float(tr_sims.min()) - 0.08, x_nce - 0.05)), 1.01)
    ax.set_xlabel("Top-5 k-NN ChemBERTa-2 Cosine Similarity to Training Set", fontsize=8.5)
    ax.set_ylabel("Density", fontsize=8.5)
    ax.set_title("Queried NCE Position in Applicability Domain (OECD Principle 3)", fontsize=9.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=6.8, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


PRESETS: Dict[str, str] = {
    "Custom / Manual NCE Input": "",
    "1. Brigatinib-like P(=O)Me2 NCE [Active | Inside AD]": "CP(=O)(C)c1ccccc1Nc1nc(Nc2cc(C)c(N3CCC(N4CCN(C)CC4)CC3)cc2OC)ncc1Br",
    "2. N-SO2Et Indole Rescue Cliff [Active | Inside AD]": "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(S(=O)(=O)CC)c4ccccc34)n2)cc2cnn(CCN(C)C)c12",
    "3. N-Me Indole Acrylamide Cliff [Inactive | Inside AD]": "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)cc2cnn(CCN(C)C)c12",
    "4. Chiral [C@H] Enantiomer [Active | Inside AD]": "C([C@H](C)NCc1c(nn(c1)c1nc(ncc1)Nc1c(cc(c(c1)C)N1CC[C@H](CC1)N1CCN(CC1)C)OC)c1ccccc1)O",
    "5. Chiral [C@@H] Enantiomer [Inactive | Inside AD]": "C([C@@H](C)NCc1c(nn(c1)c1nc(ncc1)Nc1c(cc(c(c1)C)N1CC[C@H](CC1)N1CCN(CC1)C)OC)c1ccccc1)O",
    "6. Sulfonamide-Aminothiazole NCE [Active | Inside AD]": "CCS(=O)(=O)Nc1nc(-c2ccc(F)cc2)c(-c2ccnc(Nc3ccc4n(CCN(C)C)ccc4c3)n2)s1",
    "7. Unrelated Steroid-like Scaffold [Outside AD Demo]": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
}

DEMO_BATCH = "\n".join(
    [
        "NCE-Brigatinib-Analog, CP(=O)(C)c1ccccc1Nc1nc(Nc2cc(CC)c(N3CCC(N4CCN(C)CC4)CC3)cc2OC)ncc1Br",
        "NCE-SulfonylIndole-Rescue, C=CC(=O)Nc1cc(Nc2nccc(-c3cn(S(=O)(=O)CC)c4ccccc34)n2)cc2cnn(CCN(C)C)c12",
        "Ctrl-NMeIndole-Inactive, C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)cc2cnn(CCN(C)C)c12",
        "Chiral-Isomer-Active, C([C@H](C)NCc1c(nn(c1)c1nc(ncc1)Nc1c(cc(c(c1)C)N1CC[C@H](CC1)N1CCN(CC1)C)OC)c1ccccc1)O",
        "Chiral-Isomer-Inactive, C([C@@H](C)NCc1c(nn(c1)c1nc(ncc1)Nc1c(cc(c(c1)C)N1CC[C@H](CC1)N1CCN(CC1)C)OC)c1ccccc1)O",
        "NCE-ThiazoleSulfonamide, CCS(=O)(=O)Nc1nc(-c2ccc(F)cc2)c(-c2ccnc(Nc3ccc(N4CCN(C)CC4)cc3OC)n2)s1",
        "Novel-OutOfDomain-Check, CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    ]
)

predictor = load_predictor()
metrics_data = predictor.metrics_summary

# Sidebar
with st.sidebar:
    if MML_LOGO_B64:
        st.markdown(
            f'<div style="text-align:center; background:#ffffff; padding:10px; border-radius:10px; border:1px solid #e2e8f0; margin-bottom:10px;">'
            f'<img src="data:image/png;base64,{MML_LOGO_B64}" style="max-width:155px; width:100%; height:auto;" />'
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("### 🧬 EGFR C797S QSAR-LLM")
    st.caption("Predicts **Active (`1`) vs Inactive (`0`)** & **OECD Applicability Domain** for New Chemical Entities")
    st.markdown(
        """
        - **LLM Backbone:** `DeepChem/ChemBERTa-77M-MTR` (Fine-Tuned + 5× SMILES TTA)
        - **Structural Fusion:** `2048-bit Chiral ECFP4` + RDKit 2D Descriptors
        - **AD Criterion:** Top-5 Cosine Similarity ($D_{\\text{strict}} = 0.7618$, $Z \\le 2.0$) & Chiral Tanimoto
        """
    )
    if metrics_data and "external_test" in metrics_data:
        hyb_m = metrics_data["external_test"]["Hybrid ChemBERTa-2 + Chiral ECFP4"]
        in_ad_m = metrics_data["applicability_domain"]["test_in_domain_metrics"]
        c1, c2 = st.columns(2)
        c1.metric("In-AD ROC-AUC", f"{in_ad_m['ROC_AUC']:.3f}")
        c2.metric("In-AD MCC", f"{in_ad_m['MCC']:.3f}")
        c3, c4 = st.columns(2)
        c3.metric("Sensitivity", f"{hyb_m['Sensitivity']*100:.1f}%")
        c4.metric("AD Coverage", f"{metrics_data['applicability_domain']['test_coverage_pct']}%")
        st.caption(f"Y-Randomization $cR_p^2$ = **{metrics_data['y_randomization']['cRp2']}** (> 0.50 Pass)")
    st.divider()
    st.caption(f"**{CREDIT_LINE}**")

# Institutional Masthead with Both Official Logos
pu_img_html = (
    f'<img src="data:image/png;base64,{PU_LOGO_B64}" style="height:72px; width:auto; object-fit:contain;" alt="Punjabi University Patiala" />'
    if PU_LOGO_B64 else ""
)
mml_img_html = (
    f'<img src="data:image/png;base64,{MML_LOGO_B64}" style="height:86px; width:auto; object-fit:contain;" alt="Molecular Modeling Lab" />'
    if MML_LOGO_B64 else ""
)

st.markdown(
    f"""
    <div style="background:#ffffff; border:2px solid #1e3a8a; border-radius:14px; padding:14px 22px; margin-bottom:1.2rem; box-shadow:0 3px 10px rgba(15,23,42,0.06);">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:18px; flex-wrap:wrap;">
            <div style="flex:0 0 auto;">{pu_img_html}</div>
            <div style="flex:1 1 380px; text-align:center;">
                <div style="font-size:1.52rem; font-weight:800; color:#0f172a; line-height:1.25;">
                    EGFR<sup>C797S</sup> NCE Activity &amp; Applicability Domain Studio
                </div>
                <div style="font-size:0.88rem; color:#475569; margin-top:3px;">
                    Hybrid Fine-Tuned ChemBERTa-2 Chemical Language Model &amp; Chirality-Aware QSAR Screening Tool
                </div>
                <div style="display:inline-block; background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a8a; font-size:0.83rem; font-weight:700; padding:5px 14px; border-radius:999px; margin-top:8px;">
                    {CREDIT_LINE}
                </div>
            </div>
            <div style="flex:0 0 auto;">{mml_img_html}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_val = st.tabs(
    [
        "🧪 1. Single NCE Prediction & Applicability Domain",
        "📂 2. Batch NCE CSV Screening (Active/Inactive + AD)",
        "📊 3. Internal & External Validation Report",
    ]
)

# ==============================================================================
# TAB 1: SINGLE NCE PREDICTOR & APPLICABILITY DOMAIN
# ==============================================================================
with tab_single:
    col_in, col_out = st.columns([1.02, 1.48], gap="large")

    with col_in:
        st.subheader("1. Enter or Draw NCE Structure")
        preset_choice = st.selectbox(
            "Select Benchmark NCE or Activity/AD Preset:",
            options=list(PRESETS.keys()),
            index=1,
        )
        default_smi = PRESETS[preset_choice] or "CP(=O)(C)c1ccccc1Nc1nc(Nc2cc(C)c(N3CCC(N4CCN(C)CC4)CC3)cc2OC)ncc1Br"

        use_sketcher = False
        if HAS_KETCHER:
            use_sketcher = st.toggle("Open Interactive 2D Molecule Sketcher (Ketcher)", value=False)

        if use_sketcher and HAS_KETCHER:
            st.caption("Draw or modify your NCE and click **Apply** in the sketcher:")
            drawn_smi = st_ketcher(default_smi, height=380)
            smiles_query = st.text_area("Query NCE SMILES (Isomeric SMILES):", value=drawn_smi or default_smi, height=90)
        else:
            smiles_query = st.text_area(
                "Paste NCE SMILES String (Preserves @/@@ & E/Z Stereochemistry):",
                value=default_smi,
                height=105,
            )

        st.button("🔍 Predict Activity & Check Applicability Domain", type="primary", use_container_width=True)

        # Model Consensus Table directly below input
        if smiles_query.strip():
            res = predictor.predict_single(smiles_query.strip())
            if res.get("valid", False):
                st.markdown("#### Multi-Model Consensus Breakdown")
                df_dec = pd.DataFrame(res["model_decisions"]).rename(
                    columns={
                        "model": "Model",
                        "prediction": "Class Call",
                        "probability": "P(Active)",
                        "cutoff": "Cutoff",
                    }
                )
                st.dataframe(df_dec, use_container_width=True, hide_index=True)

    with col_out:
        if smiles_query.strip():
            if not res.get("valid", False):
                st.error(res.get("error", "Invalid SMILES string."))
            else:
                is_active = res["predicted_label"] == 1
                verdict_cls = "verdict-active" if is_active else "verdict-inactive"
                binary_str = "ACTIVE (Class 1)" if is_active else "INACTIVE (Class 0)"
                prob_pct = res["probability"] * 100.0

                # 1. Activity Verdict Banner
                st.markdown(
                    f"""
                    <div class="{verdict_cls}">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                            <div>
                                <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;">EGFR C797S Activity Prediction</div>
                                <div style="font-size:1.55rem; font-weight:900;">{binary_str}</div>
                                <div style="font-size:0.85rem; margin-top:2px;">
                                    Model Consensus: <b>{res['consensus_active_votes']} Active Votes</b> &bull; Decision Cutoff: {res['threshold']*100:.1f}%
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-size:0.78rem; font-weight:700;">Predicted Probability</div>
                                <div style="font-size:2.1rem; font-weight:900;">{prob_pct:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 2. Dedicated Applicability Domain (AD) Verdict Card + Live AD Plot
                ad = res["applicability_domain"]
                ad_card_cls = f"ad-card-{ad['level']}"
                in_ad_label = "✅ INSIDE APPLICABILITY DOMAIN" if ad["in_domain"] else "⚠️ OUTSIDE APPLICABILITY DOMAIN"
                st.markdown(
                    f"""
                    <div class="{ad_card_cls}">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                            <div>
                                <div style="font-size:0.76rem; font-weight:700; text-transform:uppercase;">OECD Principle 3 — Applicability Domain</div>
                                <div style="font-size:1.18rem; font-weight:800;">{in_ad_label} — {ad['status']}</div>
                            </div>
                            <div style="font-size:0.85rem; font-weight:700;">
                                Standardized AD Distance (Z): {ad['z_distance']} (Pass &le; 2.0)
                            </div>
                        </div>
                        <div style="font-size:0.84rem; margin-top:6px;">
                            <b>ChemBERTa-2 Top-5 Cosine Similarity:</b> <code>{ad['top5_llm_cosine']:.4f}</code> (Strict Cutoff: <code>{ad['strict_cutoff']:.4f}</code>) &nbsp;|&nbsp;
                            <b>Max Chiral ECFP4 Tanimoto:</b> <code>{ad['max_tanimoto']:.3f}</code> (Closest Analog: <b>Train No. {ad['nearest_train_no']}</b>)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                ad_png = render_live_ad_plot(predictor, ad["top5_llm_cosine"], res["probability"])
                st.image(ad_png, use_container_width=True)

                # 3. 2D SVG Explainability & Standardized Depiction
                st.markdown("#### Atom-Level Pharmacophore Attribution & Standardized 2D Structure")
                v1, v2 = st.columns(2)
                with v1:
                    st.caption("🟢 **Emerald** = Favors EGFR C797S Activity | 🔴 **Coral** = Reduces Activity")
                    st.markdown(svg_to_html(res["explain_svg"]), unsafe_allow_html=True)
                with v2:
                    st.caption("Standardized Canonical Isomeric 2D Depiction")
                    st.markdown(svg_to_html(res["plain_svg"]), unsafe_allow_html=True)
                st.code(res["canonical_smiles"], language="text")

    # Full-width Medicinal Chemistry & Nearest Analogs section
    if smiles_query.strip() and res.get("valid", False):
        st.divider()
        p = res["profile"]
        lip_cls = "b-high" if p["lipinski_pass"] else "b-low"
        veb_cls = "b-high" if p["veber_pass"] else "b-mod"
        st.markdown(
            f"#### NCE Drug-Likeness & Physicochemical Profile &nbsp; "
            f'<span class="badge-pill {lip_cls}">Lipinski Ro5: {"PASS" if p["lipinski_pass"] else "FAIL"} ({p["lipinski_violations"]} viol.)</span>'
            f'<span class="badge-pill {veb_cls}">Veber Oral Rules: {"PASS" if p["veber_pass"] else "FAIL"}</span>',
            unsafe_allow_html=True,
        )
        m_cols = st.columns(5)
        m_cols[0].metric("Formula", p["formula"])
        m_cols[1].metric("MW (Da)", p["MW"])
        m_cols[2].metric("cLogP", p["cLogP"])
        m_cols[3].metric("TPSA (Å²)", p["TPSA"])
        m_cols[4].metric("QED Drug-Likeness", p["QED"])

        m_cols2 = st.columns(5)
        m_cols2[0].metric("HBD / HBA", f"{p['HBD']} / {p['HBA']}")
        m_cols2[1].metric("Rotatable Bonds", p["RotBonds"])
        m_cols2[2].metric("Heterocycles", p["HeteroCycles"])
        m_cols2[3].metric("Aromatic Rings", p["ArRings"])
        m_cols2[4].metric("Fraction Csp3", p["Fsp3"])

        motifs_str = " • ".join(p["detected_motifs"]) if p["detected_motifs"] else "Novel / unlisted chemotype (no canonical reference SMARTS alert triggered)"
        st.info(f"**Detected EGFR / Heterocycle Motifs:** {motifs_str}")

        st.markdown("#### Top-3 Nearest Experimental Analogs in `Train.csv` (SAR & AD Evidence)")
        a_cols = st.columns(3)
        for idx, analog in enumerate(res["nearest_analogs"]):
            with a_cols[idx]:
                act_cls = "b-high" if analog["label"] == 1 else "b-low"
                st.markdown(
                    f"**Train No. {analog['No']}** &nbsp; <span class='badge-pill {act_cls}'>{analog['activity_text']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(svg_to_html(analog["svg"], max_width=260), unsafe_allow_html=True)
                st.caption(f"Tanimoto: **{analog['tanimoto']}** | ChemBERTa Sim: **{analog['llm_cosine']}**")

# ==============================================================================
# TAB 2: BATCH VIRTUAL SCREENING
# ==============================================================================
with tab_batch:
    st.subheader("Batch NCE Virtual Screening (Active/Inactive + Applicability Domain)")
    bcol1, bcol2 = st.columns(2, gap="large")

    batch_items: List[Dict[str, str]] = []

    with bcol1:
        st.markdown("**Option A: Upload CSV Library of NCEs**")
        st.caption("Upload any `.csv` containing a `SMILES` column (and optional `ID` / `No.` / `Name` column).")
        sample_csv_path = ROOT_DIR / "sample_nce_input.csv"
        if sample_csv_path.exists():
            st.download_button(
                "📄 Download Sample NCE Input Template (.csv)",
                data=sample_csv_path.read_bytes(),
                file_name="sample_nce_input.csv",
                mime="text/csv",
            )
        up_file = st.file_uploader("Choose CSV file", type=["csv"])
        run_csv_btn = st.button("🚀 Predict Activity & AD for Uploaded CSV", type="primary", use_container_width=True)

    with bcol2:
        st.markdown("**Option B: Paste Multi-NCE SMILES List**")
        st.caption("Format: `ID, SMILES` per line (or one `SMILES` per line).")
        batch_text = st.text_area("Paste NCE library:", value=DEMO_BATCH, height=165)
        run_txt_btn = st.button("🚀 Predict Activity & AD for Pasted List", use_container_width=True)

    if run_csv_btn and up_file is not None:
        df_up = pd.read_csv(up_file)
        smi_col = next((c for c in df_up.columns if c.strip().lower() in ("smiles", "canonical_smiles", "smi")), None)
        id_col = next((c for c in df_up.columns if c.strip().lower() in ("id", "no", "no.", "name", "compound_id")), None)
        if smi_col is None:
            st.error(f"Could not find a `SMILES` column in uploaded CSV. Found columns: {list(df_up.columns)}")
        else:
            for i, row in df_up.head(1000).iterrows():
                s_val = str(row[smi_col]).strip()
                if s_val and s_val.lower() != "nan":
                    cid = str(row[id_col]).strip() if id_col else f"NCE-{i+1}"
                    batch_items.append({"id": cid, "smiles": s_val})

    elif run_txt_btn and batch_text.strip():
        for i, line in enumerate(batch_text.strip().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if "," in line:
                cid, smi = line.split(",", 1)
                batch_items.append({"id": cid.strip(), "smiles": smi.strip()})
            else:
                batch_items.append({"id": f"NCE-{i}", "smiles": line})

    if batch_items:
        with st.spinner(f"Evaluating {len(batch_items)} NCEs for EGFR C797S Activity & Applicability Domain..."):
            batch_res = predictor.predict_batch(batch_items)
            df_out = pd.DataFrame(batch_res)

        n_act = int((df_out["Predicted_Class"] == 1).sum())
        n_indom = int((df_out["In_AD"] == True).sum())
        n_high_conf_hits = int(((df_out["Predicted_Class"] == 1) & (df_out["In_AD"] == True)).sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total NCEs Screened", len(df_out))
        k2.metric("Predicted Active (Class 1)", f"{n_act} ({n_act/len(df_out)*100:.1f}%)")
        k3.metric("Inside Applicability Domain", f"{n_indom} ({n_indom/len(df_out)*100:.1f}%)")
        k4.metric("High-Confidence In-AD Hits", n_high_conf_hits)

        st.dataframe(df_out, use_container_width=True, hide_index=True)
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Complete Predictions + Applicability Domain Table (.csv)",
            data=csv_bytes,
            file_name="EGFR_C797S_NCE_Predictions_with_AD.csv",
            mime="text/csv",
            type="primary",
        )

# ==============================================================================
# TAB 3: INTERNAL & EXTERNAL VALIDATION REPORT
# ==============================================================================
with tab_val:
    st.subheader("Internal (5-Fold CV) & External Validation Metrics")
    if metrics_data and "external_test" in metrics_data:
        st.markdown("##### Internal Validation (5-Fold Stratified Cross-Validation on `Train.csv`, `N = 222`)")
        rows_cv = []
        for mname, cvals in metrics_data["cv_5fold_oof"].items():
            rows_cv.append(
                {
                    "Model": mname,
                    "TP": cvals["TP"],
                    "FP": cvals["FP"],
                    "TN": cvals["TN"],
                    "FN": cvals["FN"],
                    "Sensitivity": cvals["Sensitivity"],
                    "Specificity": cvals["Specificity"],
                    "Accuracy": cvals["Accuracy"],
                    "ROC-AUC": cvals["ROC_AUC"],
                    "MCC": cvals["MCC"],
                }
            )
        st.dataframe(pd.DataFrame(rows_cv), use_container_width=True, hide_index=True)

        st.markdown("##### External Validation (Locked `Test.csv`, `N = 148`)")
        rows_ext = []
        for mname, tvals in metrics_data["external_test"].items():
            rows_ext.append(
                {
                    "Model": mname,
                    "TP": tvals["TP"],
                    "FP": tvals["FP"],
                    "TN": tvals["TN"],
                    "FN": tvals["FN"],
                    "Sensitivity": tvals["Sensitivity"],
                    "Specificity": tvals["Specificity"],
                    "Accuracy": tvals["Accuracy"],
                    "ROC-AUC": tvals["ROC_AUC"],
                    "MCC": tvals["MCC"],
                }
            )
        in_ad_v = metrics_data["applicability_domain"]["test_in_domain_metrics"]
        rows_ext.append(
            {
                "Model": "Hybrid ChemBERTa-2 + Chiral ECFP4 (Inside AD, N=144)",
                "TP": in_ad_v["TP"],
                "FP": in_ad_v["FP"],
                "TN": in_ad_v["TN"],
                "FN": in_ad_v["FN"],
                "Sensitivity": in_ad_v["Sensitivity"],
                "Specificity": in_ad_v["Specificity"],
                "Accuracy": in_ad_v["Accuracy"],
                "ROC-AUC": in_ad_v["ROC_AUC"],
                "MCC": in_ad_v["MCC"],
            }
        )
        st.dataframe(pd.DataFrame(rows_ext), use_container_width=True, hide_index=True)

        dcol1, dcol2 = st.columns(2)
        ext_csv_path = ROOT_DIR / "reports" / "external_test_predictions.csv"
        met_json_path = ROOT_DIR / "reports" / "metrics_summary.json"
        if ext_csv_path.exists():
            dcol1.download_button(
                "⬇️ Download External Test Predictions + AD (.csv)",
                data=ext_csv_path.read_bytes(),
                file_name="external_test_predictions.csv",
                mime="text/csv",
            )
        if met_json_path.exists():
            dcol2.download_button(
                "⬇️ Download Full Validation Metrics (.json)",
                data=met_json_path.read_bytes(),
                file_name="metrics_summary.json",
                mime="application/json",
            )

    fig_dir = ROOT_DIR / "reports" / "figures"
    f1, f2 = st.columns(2)
    if (fig_dir / "roc_pr_curves.png").exists():
        f1.image(str(fig_dir / "roc_pr_curves.png"), caption="External Test Set ROC & Precision-Recall Curves", use_container_width=True)
    if (fig_dir / "confusion_and_metrics.png").exists():
        f2.image(str(fig_dir / "confusion_and_metrics.png"), caption="Benchmark Comparison & External Confusion Matrix", use_container_width=True)

    f3, f4 = st.columns(2)
    if (fig_dir / "applicability_domain_and_yrand.png").exists():
        f3.image(str(fig_dir / "applicability_domain_and_yrand.png"), caption="Applicability Domain Reliability & 20-Run Y-Randomization", use_container_width=True)
    if (fig_dir / "chemical_space_pca.png").exists():
        f4.image(str(fig_dir / "chemical_space_pca.png"), caption="Fine-Tuned ChemBERTa-2 Latent Chemical Space PCA", use_container_width=True)

st.markdown(
    f"""
    <div style="margin-top:2.2rem; padding:14px 20px; background:#0f172a; color:#f8fafc; border-radius:10px; text-align:center; font-size:0.86rem; font-weight:600;">
        {CREDIT_LINE}
    </div>
    """,
    unsafe_allow_html=True,
)
