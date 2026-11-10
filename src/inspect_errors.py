"""Inspect false positives / false negatives for a given Charlson category.

Usage (run in YOUR terminal, never executed by the assistant):
    python src/inspect_errors.py --cat chf --type fn --n 20
    python src/inspect_errors.py --cat diab --type fp --split val
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.charlson_mapping import CATEGORIES


def highlight(text: str, patterns: list[re.Pattern]) -> str:
    """Surround matched terms with >>> ... <<< markers."""
    for pat in patterns:
        text = pat.sub(lambda m: f">>>{m.group()}<<<", text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Inspect prediction errors.")
    parser.add_argument("--cat", required=True, choices=CATEGORIES, help="Charlson category")
    parser.add_argument("--type", required=True, choices=["fp", "fn"], help="Error type: fp or fn")
    parser.add_argument("--n", type=int, default=10, help="Number of examples to display")
    parser.add_argument("--split", default="val")
    parser.add_argument("--lexicon", default="config/charlson_lexicon.yaml")
    parser.add_argument("--pmh", default="data/interim/pmh.parquet")
    parser.add_argument("--labels", default="data/processed/labels_charlson.parquet")
    parser.add_argument("--preds", default=None, help="Predictions parquet (default: auto from split)")
    parser.add_argument("--splits-file", default="data/processed/splits.parquet")
    args = parser.parse_args()

    if args.preds is None:
        args.preds = f"data/processed/preds_rules_{args.split}.parquet"

    # Load lexicon for highlighting
    with open(args.lexicon) as f:
        lexicon = yaml.safe_load(f)

    cat_terms = lexicon.get(args.cat, [])
    patterns = []
    for term in cat_terms:
        if term.startswith("regex:"):
            patterns.append(re.compile(term[6:], re.IGNORECASE))
        else:
            patterns.append(re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE))

    # Load data
    splits = pd.read_parquet(args.splits_file)
    split_hadm = set(splits.loc[splits["split"] == args.split, "hadm_id"])

    pmh = pd.read_parquet(args.pmh, columns=["hadm_id", "pmh_text"])
    pmh = pmh[pmh["hadm_id"].isin(split_hadm)]

    labels = pd.read_parquet(args.labels)
    labels = labels[labels["hadm_id"].isin(split_hadm)][["hadm_id", args.cat]]

    preds = pd.read_parquet(args.preds)
    preds = preds[["hadm_id", args.cat]]

    merged = pmh.merge(labels, on="hadm_id", suffixes=("", "_true"))
    merged = merged.merge(preds, on="hadm_id", suffixes=("_true", "_pred"))

    # Handle column naming
    true_col = f"{args.cat}_true" if f"{args.cat}_true" in merged.columns else args.cat
    pred_col = f"{args.cat}_pred" if f"{args.cat}_pred" in merged.columns else args.cat

    if args.type == "fp":
        # False positives: predicted 1, true 0
        errors = merged[(merged[pred_col] == 1) & (merged[true_col] == 0)]
        label = "FAUX POSITIFS"
    else:
        # False negatives: predicted 0, true 1
        errors = merged[(merged[pred_col] == 0) & (merged[true_col] == 1)]
        label = "FAUX NÉGATIFS"

    print(f"\n{'='*60}")
    print(f"{label} pour '{args.cat}' (split={args.split})")
    print(f"Total : {len(errors)} erreurs")
    print(f"{'='*60}\n")

    for i, (_, row) in enumerate(errors.head(args.n).iterrows()):
        text = row["pmh_text"] if pd.notna(row["pmh_text"]) else "(vide)"
        highlighted = highlight(text, patterns)
        print(f"--- Exemple {i+1} (hadm_id={row['hadm_id']}) ---")
        print(highlighted)
        print()


if __name__ == "__main__":
    main()
