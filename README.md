# EGFR C797S ChemBERTa-2 LLM QSAR & Streamlit NCE Web App

A CPU-optimized Chemical Language Model (LLM) QSAR pipeline and Streamlit web application for predicting biological activity (`1 = Active`, `0 = Inactive`) of nitrogen-containing heterocyclic New Chemical Entities (NCEs) against the **EGFR C797S** mutant.

## Deploy to Streamlit Community Cloud (Free)
1. Create a new GitHub repository and upload the contents of **`streamlit_cloud_bundle/`** (all files are `< 14 MB`, so standard GitHub web upload or `git push` works without Git LFS).
2. Go to **https://share.streamlit.io** and click **Create app**.
3. Select your repository, branch (`main`), and set **Main file path** to:
   ```text
   streamlit_app.py
   ```
4. Click **Deploy!** (`runtime.txt`, `packages.txt`, and `requirements.txt` automatically configure Python 3.12, RDKit X11 libraries, and lightweight CPU PyTorch).

## Run Locally on Windows CPU

### Streamlit App
```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```
*(Or double-click `run_streamlit.bat` inside the main workspace folder.)*

### FastAPI Alternative App
Double-click `run_webapp.bat` or run:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Folder Structure
```text
├── streamlit_app.py                         # Main Streamlit entrypoint (Single NCE, Batch CSV, OECD Results)
├── requirements.txt                         # CPU-optimized dependencies for Streamlit Cloud
├── packages.txt                             # Linux apt packages for RDKit 2D SVG rendering
├── runtime.txt                              # Pins Python 3.12 on Streamlit Cloud
├── .streamlit/config.toml                   # UI theme & upload config
├── data/
│   ├── Train.csv                            # Training set (N = 222: 141 Active, 81 Inactive)
│   └── Test.csv                             # External test set (N = 148: 99 Active, 49 Inactive)
├── models/
│   ├── chemberta_pretrained/                # Cached DeepChem/ChemBERTa-77M-MTR backbone (~13 MB)
│   └── production/
│       ├── chemberta_finetuned/             # Fine-tuned EGFR C797S ChemBERTa-2 weights (~13 MB)
│       ├── portable_state.npz               # Cross-version NumPy model state & AD embeddings
│       └── portable_meta.json               # Thresholds, AD cutoffs, and reference library metadata
├── reports/
│   ├── metrics_summary.json                 # Complete 5-fold CV, External Test, AD & Y-randomization metrics
│   ├── external_test_predictions.csv        # Compound-by-compound predictions on Test.csv (N = 148)
│   └── figures/                             # High-res ROC/PR, Confusion Matrix, AD/Y-Rand & PCA plots
└── src/
    ├── chemistry.py                         # RDKit standardization, chiral ECFP4, NCE profile & SVG attribution
    ├── predictor.py                         # Hybrid CPU inference, Input*Gradient attribution & analog search
    └── train_pipeline.py                    # Reproducible end-to-end training & OECD validation script
```
