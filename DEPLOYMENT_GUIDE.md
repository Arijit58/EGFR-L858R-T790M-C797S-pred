# Step-by-Step Deployment Guide: EGFR C797S NCE Activity & Applicability Domain Web App

All files in `streamlit_cloud_bundle/` are self-contained and ready to deploy on **Streamlit Community Cloud**, **Hugging Face Spaces**, or a **local laptop CPU**.

---

## 1. Complete File Manifest (What Each File Does)

| File / Folder | Purpose |
| :--- | :--- |
| `streamlit_app.py` | **Main Web App UI** — Predicts `Active (1)` vs `Inactive (0)`, plots & quantifies **OECD Applicability Domain (AD)**, displays atom-level pharmacophore attributions, drug-likeness, and batch CSV screening. |
| `predict_nce.py` | **Command-Line NCE Predictor** — Screen single SMILES or batch CSV files directly from terminal (`python predict_nce.py --csv sample_nce_input.csv --out results.csv`). |
| `sample_nce_input.csv` | **Sample NCE Library Template** — Ready-to-upload CSV demonstrating Active, Inactive, Chiral Cliff, and Out-of-Domain NCEs. |
| `requirements.txt` | **Python Dependencies** — Pre-configured with `--extra-index-url https://download.pytorch.org/whl/cpu` for fast, lightweight CPU installation. |
| `packages.txt` | **Linux System Libraries** — Installs `libxrender1` & `libxext6` on Streamlit Cloud for RDKit 2D vector drawing. |
| `runtime.txt` | **Python Version Pin** — Pins `python-3.11` on Streamlit Community Cloud for 100% binary wheel compatibility. |
| `.streamlit/config.toml` | **Streamlit Theme & Server Config** — Sets clean scientific UI colors and 50 MB CSV upload limit. |
| `Dockerfile` | **Container Deployment** — Optional 1-click container setup for Hugging Face Spaces, Render, or Docker. |
| `src/chemistry.py` | **Chemoinformatics Engine** — RDKit salt-stripping, canonical isomeric SMILES (`@/@@`), chiral `ECFP4`, Lipinski/Veber/QED profiling, and SVG rendering. |
| `src/predictor.py` | **Hybrid LLM & AD Engine** — Combines fine-tuned `ChemBERTa-77M-MTR` + `Chiral ECFP4`, computes AD Cosine/Z-score/Tanimoto metrics, and `Input × Gradient` atom explainability. |
| `models/production/` | **Trained Checkpoints** — Fine-tuned `ChemBERTa-2` (`model.safetensors`, `13.1 MB`) + cross-version NumPy AD & ensemble state (`portable_state.npz` & `portable_meta.json`). |
| `reports/` | **Validation Results** — Full internal/external metrics (`metrics_summary.json`), external test predictions (`external_test_predictions.csv`), and ROC/PR/AD/PCA figures. |

---

## 2. Deploy Free on Streamlit Community Cloud (Recommended)

1. Sign in to **GitHub** (https://github.com) and create a **New Repository** (e.g., `egfr-c797s-nce-predictor`).
2. Upload **all files and folders inside `streamlit_cloud_bundle/`** to the root of that GitHub repository:
   ```powershell
   cd "d:\Users\ariji\OneDrive\Desktop\USB drive backup\Nitrogen containing heterocycle derivatives review\QSAR model\EGFR_C797S_LLM_Streamlit\streamlit_cloud_bundle"
   git init
   git add .
   git commit -m "Initial commit: EGFR C797S ChemBERTa-2 NCE & AD Web App"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/egfr-c797s-nce-predictor.git
   git push -u origin main
   ```
   *(Every file is `< 14 MB`, so standard GitHub push or web upload works without Git LFS.)*
3. Go to **https://share.streamlit.io** and click **Create app**.
4. Select your repository, branch `main`, and set **Main file path** to:
   ```text
   streamlit_app.py
   ```
5. Click **Deploy!** Within ~2–3 minutes, you will have a public shareable URL for screening NCEs.

---

## 3. Run Locally on Your Laptop (CPU)

```powershell
cd "d:\Users\ariji\OneDrive\Desktop\USB drive backup\Nitrogen containing heterocycle derivatives review\QSAR model\EGFR_C797S_LLM_Streamlit\streamlit_cloud_bundle"
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\streamlit run streamlit_app.py
```

### Run Batch NCE + Applicability Domain Screening via CLI
```powershell
.\.venv\Scripts\python predict_nce.py --csv sample_nce_input.csv --out my_nce_results_with_AD.csv
```
