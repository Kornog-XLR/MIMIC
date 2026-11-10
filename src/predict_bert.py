"""Run inference with a finalized BERT model on a data split."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bert_utils import CATEGORIES, predict_probs, select_device
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description="BERT inference for Charlson classification.")
    parser.add_argument("--run", required=True, help="Run name (subdirectory of models/)")
    parser.add_argument("--split", default="test", help="Split to predict on")
    parser.add_argument("--pmh", default="data/interim/pmh.parquet")
    parser.add_argument("--splits-file", default="data/processed/splits.parquet")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    final_dir = f"models/{args.run}/final"
    if not os.path.isdir(final_dir):
        print(f"Erreur : {final_dir} n'existe pas. Lance d'abord finalize_bert.py.")
        sys.exit(1)

    # Load config
    with open(os.path.join(final_dir, "final_config.json")) as f:
        config = json.load(f)
    max_length = config["max_length"]

    # Load thresholds
    with open(os.path.join(final_dir, "thresholds.json")) as f:
        thresholds = json.load(f)

    device = select_device()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(final_dir)
    model = AutoModelForSequenceClassification.from_pretrained(final_dir)

    # Load data
    splits = pd.read_parquet(args.splits_file)
    hadm_ids = set(splits.loc[splits["split"] == args.split, "hadm_id"])
    pmh = pd.read_parquet(args.pmh, columns=["hadm_id", "pmh_text"])
    pmh = pmh[pmh["hadm_id"].isin(hadm_ids)].dropna(subset=["pmh_text"])
    print(f"Split '{args.split}' : {len(pmh)} notes")

    texts = pmh["pmh_text"].tolist()
    all_probs = predict_probs(model, tokenizer, texts, max_length, args.batch_size, device)

    # Build output DataFrame
    result = pd.DataFrame({"hadm_id": pmh["hadm_id"].values})
    for j, cat in enumerate(CATEGORIES):
        result[f"{cat}_prob"] = all_probs[:, j]
        result[cat] = (all_probs[:, j] >= thresholds[cat]).astype(int)

    out_path = f"data/processed/preds_bert_{args.split}_{args.run}.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"Prédictions enregistrées dans {out_path}")


if __name__ == "__main__":
    main()
