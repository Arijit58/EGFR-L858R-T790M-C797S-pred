"""
FastAPI Web Application for EGFR C797S Mutant NCE Activity Prediction
powered by Fine-Tuned ChemBERTa-2 (77M-MTR) Hybrid Chemical Language Model.

Launch locally on CPU:
    .venv\\Scripts\\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.predictor import EGFRC797SPredictor

ROOT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT_DIR / "templates"
FIG_DIR = ROOT_DIR / "reports" / "figures"
ASSETS_DIR = ROOT_DIR / "assets"

app = FastAPI(
    title="EGFR C797S ChemBERTa-2 NCE Predictor | Punjabi University, Patiala",
    description="Standalone CPU LLM-QSAR Tool developed by Mr. Arijit Bhattacharya and Dr. Om Silakari, Punjabi University, Patiala.",
    version="1.0.0",
)

if FIG_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIG_DIR)), name="figures")
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_PREDICTOR: Optional[EGFRC797SPredictor] = None


def get_predictor() -> EGFRC797SPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = EGFRC797SPredictor()
    return _PREDICTOR


class SinglePredictRequest(BaseModel):
    smiles: str


class BatchItem(BaseModel):
    id: Optional[str] = None
    smiles: str


class BatchPredictRequest(BaseModel):
    items: List[BatchItem]


@app.on_event("startup")
def preload_model() -> None:
    prod_bundle = ROOT_DIR / "models" / "production" / "hybrid_bundle.joblib"
    if prod_bundle.exists():
        get_predictor()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    pred = get_predictor()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"metrics": pred.metrics_summary},
    )


@app.get("/api/metrics")
async def api_metrics():
    pred = get_predictor()
    return JSONResponse(pred.metrics_summary)


@app.post("/api/predict")
async def api_predict_single(req: SinglePredictRequest):
    pred = get_predictor()
    result = pred.predict_single(req.smiles)
    if not result.get("valid", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Invalid SMILES"))
    return JSONResponse(result)


@app.post("/api/batch")
async def api_predict_batch(req: BatchPredictRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="No molecules provided.")
    if len(req.items) > 1000:
        raise HTTPException(status_code=400, detail="Batch size capped at 1,000 NCEs per request on CPU.")
    pred = get_predictor()
    payload = [{"id": it.id or str(i + 1), "smiles": it.smiles} for i, it in enumerate(req.items)]
    results = pred.predict_batch(payload)
    return JSONResponse({"count": len(results), "results": results})


@app.post("/api/batch_csv")
async def api_predict_csv(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty or missing headers.")

    # Auto-detect SMILES and ID columns case-insensitively
    smiles_col = None
    id_col = None
    for col in reader.fieldnames:
        cl = col.strip().lower()
        if cl in ("smiles", "canonical_smiles", "smi", "structure"):
            smiles_col = col
        elif cl in ("id", "no", "no.", "name", "compound_id", "mol_id", "title"):
            id_col = col
    if smiles_col is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not find a 'SMILES' column in CSV headers: {reader.fieldnames}",
        )

    rows: List[Dict[str, str]] = []
    for idx, r in enumerate(reader, start=1):
        smi = (r.get(smiles_col) or "").strip()
        if not smi:
            continue
        cid = (r.get(id_col) or str(idx)).strip() if id_col else str(idx)
        rows.append({"id": cid, "smiles": smi})

    if not rows:
        raise HTTPException(status_code=400, detail="No valid SMILES rows found in CSV.")
    if len(rows) > 1000:
        rows = rows[:1000]

    pred = get_predictor()
    results = pred.predict_batch(rows)
    return JSONResponse({"count": len(results), "results": results})
