"""Inspect disagreements between a method and gold annotations.

Usage (run in YOUR terminal, never executed by the assistant):
    python src/inspect_gold_errors.py --cat chf --method rules --type fn
    python src/inspect_gold_errors.py --cat diab --method bert_v1 --type fp
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.charlson_mapping import CATEGORIES


def _load_preds(method, split="gold"):
    candidates = [
        f"data/processed/preds_{method}_{split}.parquet",
        f"data/processed/preds_bert_{split}_{method}.parquet",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_parquet(path)
    raise FileNotFoundError(f"Prédictions introuvables : {candidates}")


def main():
    parser = argparse.ArgumentParser(description="Inspecter les erreurs gold.")
    parser.add_argument("--cat", required=True, choices=CATEGORIES)
    parser.add_argument("--method", required=True, help="Nom de la méthode (rules, bert_v1…)")
    parser.add_argument("--type", required=True, choices=["fp", "fn"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--gold", default="data/processed/gold_annotations.csv")
    parser.add_argument("--labels", default="data/processed/labels_charlson.parquet")
    parser.add_argument("--pmh", default="data/interim/pmh.parquet")
    args = parser.parse_args()

    gold = pd.read_csv(args.gold)
    preds = _load_preds(args.method)
    icd = pd.read_parquet(args.labels)
    pmh = pd.read_parquet(args.pmh, columns=["hadm_id", "pmh_text"])

    merged = pmh.merge(gold[["hadm_id", args.cat, "commentaire"]], on="hadm_id")
    merged = merged.rename(columns={args.cat: "gold"})
    merged = merged.merge(
        preds[["hadm_id", args.cat]].rename(columns={args.cat: "pred"}),
        on="hadm_id",
    )
    merged = merged.merge(
        icd[["hadm_id", args.cat]].rename(columns={args.cat: "icd"}),
        on="hadm_id",
    )

    if args.type == "fp":
        errors = merged[(merged["pred"] == 1) & (merged["gold"] == 0)]
        label = "FAUX POSITIFS"
    else:
        errors = merged[(merged["pred"] == 0) & (merged["gold"] == 1)]
        label = "FAUX NÉGATIFS"

    print(f"\n{'=' * 60}")
    print(f"{label} pour '{args.cat}' (méthode={args.method})")
    print(f"Total : {len(errors)} erreurs")
    print(f"{'=' * 60}\n")

    for i, (_, row) in enumerate(errors.head(args.n).iterrows()):
        text = row["pmh_text"] if pd.notna(row["pmh_text"]) else "(vide)"
        comment = row["commentaire"] if pd.notna(row["commentaire"]) else ""
        print(f"--- Exemple {i + 1} (hadm_id={row['hadm_id']}) ---")
        print(f"  Prédiction: {int(row['pred'])}  |  Gold: {int(row['gold'])}  |  ICD: {int(row['icd'])}")
        if comment:
            print(f"  Commentaire: {comment}")
        print(f"  PMH: {text}")
        print()


if __name__ == "__main__":
    main()
