"""Extract and shuffle gold-set stays for blind annotation."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def prepare_gold(
    splits_path: str,
    pmh_path: str,
    output_path: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the gold annotation file (no ICD labels, blind)."""
    splits = pd.read_parquet(splits_path)
    pmh = pd.read_parquet(pmh_path, columns=["hadm_id", "pmh_text"])

    gold = splits[splits["split"] == "gold"][["hadm_id"]].copy()
    gold = gold.merge(pmh, on="hadm_id", how="left")

    # Shuffle with fixed seed
    gold = gold.sample(frac=1, random_state=seed).reset_index(drop=True)
    gold["ordre"] = gold.index + 1

    result = gold[["hadm_id", "ordre", "pmh_text"]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Gold set : {len(result)} séjours → {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare gold set for annotation.")
    parser.add_argument("--splits", default="data/processed/splits.parquet")
    parser.add_argument("--pmh", default="data/interim/pmh.parquet")
    parser.add_argument("--output", default="data/processed/gold_to_annotate.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_gold(args.splits, args.pmh, args.output, args.seed)


if __name__ == "__main__":
    main()
