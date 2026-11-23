"""
Standalone CLI predictor for screening New Chemical Entities (NCEs) for
EGFR C797S Activity (Active=1 / Inactive=0) and OECD Applicability Domain (AD).

Usage examples:
  Single SMILES:
    python predict_nce.py --smiles "CP(=O)(C)c1ccccc1Nc1nc(Nc2ccc(N3CCN(C)CC3)cc2OC)ncc1Cl"

  Batch CSV screening:
    python predict_nce.py --csv sample_nce_input.csv --out nce_predictions_with_AD.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predictor import EGFRC797SPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict EGFR C797S Activity and Applicability Domain for NCEs.")
    parser.add_argument("--smiles", type=str, default=None, help="Single NCE SMILES string to evaluate.")
    parser.add_argument("--csv", type=str, default=None, help="Input CSV file containing a 'SMILES' column.")
    parser.add_argument("--out", type=str, default="nce_predictions_with_AD.csv", help="Output CSV path for batch mode.")
    args = parser.parse_args()

    if not args.smiles and not args.csv:
        parser.print_help()
        sys.exit(1)

    predictor = EGFRC797SPredictor()

    if args.smiles:
        res = predictor.predict_single(args.smiles)
        if not res.get("valid", False):
            print(f"[ERROR] {res.get('error')}")
            sys.exit(1)
        ad = res["applicability_domain"]
        print("\n================ EGFR C797S NCE PREDICTION ================")
        print(f"Input SMILES      : {res['input_smiles']}")
        print(f"Canonical SMILES  : {res['canonical_smiles']}")
        print(f"Prediction        : {res['prediction_text']} (Class = {res['predicted_label']})")
        print(f"Probability       : {res['probability']:.4f} (Decision Cutoff = {res['threshold']})")
        print(f"Model Consensus   : {res['consensus_active_votes']} Active Votes")
        print("---------------- APPLICABILITY DOMAIN (AD) ----------------")
        print(f"Inside AD?        : {'YES' if ad['in_domain'] else 'NO (Extrapolation Warning)'}")
        print(f"AD Status         : {ad['status']}")
        print(f"LLM Cosine Sim    : {ad['top5_llm_cosine']:.4f} (Strict Cutoff = {ad['strict_cutoff']:.4f})")
        print(f"AD Z-Distance     : {ad['z_distance']} (Pass <= 2.0)")
        print(f"Max ECFP4 Tanimoto: {ad['max_tanimoto']:.3f} (Closest Train Compound = No. {ad['nearest_train_no']})")
        print("===========================================================\n")

    if args.csv:
        df_in = pd.read_csv(args.csv)
        smi_col = next((c for c in df_in.columns if c.strip().lower() in ("smiles", "canonical_smiles", "smi")), None)
        id_col = next((c for c in df_in.columns if c.strip().lower() in ("id", "no", "no.", "name", "compound_id")), None)
        if smi_col is None:
            raise ValueError(f"Input CSV must contain a 'SMILES' column. Found: {list(df_in.columns)}")

        items = []
        for idx, row in df_in.iterrows():
            s_val = str(row[smi_col]).strip()
            if s_val and s_val.lower() != "nan":
                cid = str(row[id_col]).strip() if id_col else f"NCE-{idx+1}"
                items.append({"id": cid, "smiles": s_val})

        results = predictor.predict_batch(items)
        df_out = pd.DataFrame(results)
        df_out.to_csv(args.out, index=False)
        n_act = int((df_out["Predicted_Class"] == 1).sum())
        n_ad = int((df_out["In_AD"] == True).sum())
        print(f"Screened {len(df_out)} NCEs -> {n_act} Active | {n_ad} Inside Applicability Domain.")
        print(f"Saved enriched results to: {args.out}")


if __name__ == "__main__":
    main()
