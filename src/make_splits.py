"""Build train/val/test/gold splits from PMH + Charlson labels."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.charlson_mapping import CATEGORIES

# Categories considered "rare" for gold enrichment
RARE_CATEGORIES = ["aids", "hp", "msld", "metacanc", "dementia", "rheumd"]


def build_pool(
    pmh: pd.DataFrame,
    labels: pd.DataFrame,
    max_chars: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter and deduplicate to build the eligible pool.

    Returns:
        (pool DataFrame, dict of exclusion counts by reason)
    """
    exclusions: dict[str, int] = {}

    merged = pmh.merge(labels, on=["hadm_id", "subject_id"], how="inner")

    n_total = len(pmh)
    n_after_join = len(merged)
    exclusions["no_labels_match"] = n_total - n_after_join

    mask_pmh = merged["pmh_found"] & (merged["n_chars"] > 0)
    exclusions["pmh_not_found_or_empty"] = int((~mask_pmh).sum())
    merged = merged[mask_pmh]

    mask_icd = merged["has_icd"]
    exclusions["no_icd"] = int((~mask_icd).sum())
    merged = merged[mask_icd]

    mask_long = merged["n_chars"] > max_chars
    exclusions["too_long"] = int(mask_long.sum())
    merged = merged[~mask_long]

    # Deduplicate: keep smallest note_id per hadm_id
    before_dedup = len(merged)
    merged = merged.sort_values("note_id").drop_duplicates(subset="hadm_id", keep="first")
    exclusions["duplicate_hadm_id"] = before_dedup - len(merged)

    return merged, exclusions


def _sample_one_per_patient(
    pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample up to n stays, at most 1 per patient."""
    # Shuffle patients, pick one stay per patient
    patients = pool["subject_id"].unique()
    rng.shuffle(patients)
    patients = patients[:n]

    sub = pool[pool["subject_id"].isin(patients)]
    # Pick one stay per patient (smallest note_id for determinism)
    return sub.sort_values("note_id").drop_duplicates(subset="subject_id", keep="first")


def make_splits(
    pool: pd.DataFrame,
    n_gold: int,
    gold_enrich: int,
    n_test: int,
    n_val: int,
    n_train: int,
    max_per_subject: int,
    min_pos: int,
    seed: int,
) -> pd.DataFrame:
    """Split pool into gold/test/val/train with patient-level separation."""
    rng = np.random.default_rng(seed)
    used_subjects: set = set()
    splits: list[pd.DataFrame] = []

    # --- Gold: random ---
    gold_random = _sample_one_per_patient(pool, n_gold, rng)
    gold_random = gold_random.assign(split="gold", sampling="random")
    used_subjects.update(gold_random["subject_id"].unique())
    splits.append(gold_random)

    # --- Gold: enriched from rare categories ---
    if gold_enrich > 0:
        remaining = pool[~pool["subject_id"].isin(used_subjects)]
        rare_positive = remaining[remaining[RARE_CATEGORIES].any(axis=1)]
        if len(rare_positive) > 0:
            gold_enr = _sample_one_per_patient(rare_positive, gold_enrich, rng)
            gold_enr = gold_enr.assign(split="gold", sampling="enriched")
            used_subjects.update(gold_enr["subject_id"].unique())
            splits.append(gold_enr)

    # --- Test ---
    remaining = pool[~pool["subject_id"].isin(used_subjects)]
    test_df = _sample_one_per_patient(remaining, n_test, rng)
    test_df = test_df.assign(split="test", sampling="random")
    used_subjects.update(test_df["subject_id"].unique())
    splits.append(test_df)

    # --- Val ---
    remaining = pool[~pool["subject_id"].isin(used_subjects)]
    val_df = _sample_one_per_patient(remaining, n_val, rng)
    val_df = val_df.assign(split="val", sampling="random")
    used_subjects.update(val_df["subject_id"].unique())
    splits.append(val_df)

    # --- Train: random base ---
    remaining = pool[~pool["subject_id"].isin(used_subjects)]
    train_patients = remaining["subject_id"].unique().copy()
    rng.shuffle(train_patients)

    # Pick up to max_per_subject stays per patient
    train_base = remaining[remaining["subject_id"].isin(train_patients)]
    train_base = (
        train_base.sort_values("note_id")
        .groupby("subject_id")
        .head(max_per_subject)
    )
    # Limit to n_train
    if len(train_base) > n_train:
        train_base = train_base.sample(n=n_train, random_state=seed)
    train_base = train_base.assign(split="train", sampling="random")
    train_ids = set(train_base["hadm_id"])

    # --- Train: enrichment for rare categories ---
    enriched_parts = []
    rest_of_pool = remaining[~remaining["hadm_id"].isin(train_ids)]

    for cat in CATEGORIES:
        n_pos = int(train_base[cat].sum())
        if n_pos < min_pos:
            needed = min_pos - n_pos
            candidates = rest_of_pool[
                (rest_of_pool[cat] == 1) & (~rest_of_pool["hadm_id"].isin(train_ids))
            ]
            if len(candidates) > 0:
                add = candidates.sample(n=min(needed, len(candidates)), random_state=seed)
                add = add.assign(split="train", sampling="enriched")
                enriched_parts.append(add)
                train_ids.update(add["hadm_id"])

    splits.append(train_base)
    if enriched_parts:
        enriched_all = pd.concat(enriched_parts).drop_duplicates(subset="hadm_id")
        splits.append(enriched_all)

    result = pd.concat(splits, ignore_index=True)

    # --- Assertions ---
    # No subject_id shared between splits
    for s1 in ["gold", "test", "val", "train"]:
        for s2 in ["gold", "test", "val", "train"]:
            if s1 >= s2:
                continue
            subj1 = set(result.loc[result["split"] == s1, "subject_id"])
            subj2 = set(result.loc[result["split"] == s2, "subject_id"])
            overlap = subj1 & subj2
            assert len(overlap) == 0, f"Patient overlap between {s1} and {s2}: {len(overlap)}"

    # No duplicate hadm_id
    assert result["hadm_id"].is_unique, "Duplicate hadm_id in splits"

    return result[["hadm_id", "subject_id", "split", "sampling"]]


def print_stats(
    result: pd.DataFrame,
    pool: pd.DataFrame,
    exclusions: dict[str, int],
    min_pos: int,
    stats_output: str,
):
    """Print and save aggregate statistics."""
    stats_rows = []

    # Pool size
    print(f"Pool éligible       : {len(pool)}")
    stats_rows.append({"metric": "pool_eligible", "value": len(pool)})
    for reason, count in exclusions.items():
        print(f"  exclu ({reason}): {count}")
        stats_rows.append({"metric": f"exclu_{reason}", "value": count})

    # Merge categories for stats
    pool_cats = pool[["hadm_id"] + CATEGORIES]
    result_with_cats = result.merge(pool_cats, on="hadm_id", how="left")

    # Per-split stats
    print(f"\n{'Split':<8} {'N':>7} {'Patients':>9} {'Enrichis':>9}")
    print("-" * 37)
    for split_name in ["gold", "test", "val", "train"]:
        sub = result_with_cats[result_with_cats["split"] == split_name]
        n = len(sub)
        n_pat = sub["subject_id"].nunique()
        n_enr = int((sub["sampling"] == "enriched").sum())
        print(f"{split_name:<8} {n:>7} {n_pat:>9} {n_enr:>9}")
        stats_rows.append({"metric": f"{split_name}_n", "value": n})
        stats_rows.append({"metric": f"{split_name}_patients", "value": n_pat})
        stats_rows.append({"metric": f"{split_name}_enriched", "value": n_enr})

    # Per-split per-category prevalence
    print(f"\n{'Split':<8} {'Catégorie':<12} {'N_pos':>7} {'%':>7}")
    print("-" * 38)
    below_min = []
    for split_name in ["gold", "test", "val", "train"]:
        sub = result_with_cats[result_with_cats["split"] == split_name]
        n = len(sub)
        for cat in CATEGORIES:
            n_pos = int(sub[cat].sum())
            pct = 100.0 * n_pos / n if n else 0
            print(f"{split_name:<8} {cat:<12} {n_pos:>7} {pct:>6.1f}%")
            stats_rows.append({"metric": f"{split_name}_{cat}_n", "value": n_pos})
            stats_rows.append({"metric": f"{split_name}_{cat}_pct", "value": round(pct, 2)})
            if split_name == "train" and n_pos < min_pos:
                below_min.append(cat)

    if below_min:
        print(f"\nCatégories < {min_pos} positifs dans train : {', '.join(below_min)}")
    else:
        print(f"\nToutes les catégories >= {min_pos} positifs dans train.")

    pd.DataFrame(stats_rows).to_csv(stats_output, index=False)
    print(f"\nStats enregistrées dans {stats_output}")


def main():
    parser = argparse.ArgumentParser(description="Build train/val/test/gold splits.")
    parser.add_argument("--pmh", default="data/interim/pmh.parquet")
    parser.add_argument("--labels", default="data/processed/labels_charlson.parquet")
    parser.add_argument("--output", default="data/processed/splits.parquet")
    parser.add_argument("--stats-output", default="outputs/splits_stats.csv")
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gold", type=int, default=200)
    parser.add_argument("--gold-enrich", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--max-per-subject", type=int, default=2)
    parser.add_argument("--min-pos", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)

    pmh = pd.read_parquet(args.pmh)
    labels = pd.read_parquet(args.labels)

    pool, exclusions = build_pool(pmh, labels, args.max_chars)

    result = make_splits(
        pool=pool,
        n_gold=args.n_gold,
        gold_enrich=args.gold_enrich,
        n_test=args.n_test,
        n_val=args.n_val,
        n_train=args.n_train,
        max_per_subject=args.max_per_subject,
        min_pos=args.min_pos,
        seed=args.seed,
    )

    result.to_parquet(args.output, index=False)
    print(f"Splits enregistrés dans {args.output}")

    print_stats(result, pool, exclusions, args.min_pos, args.stats_output)


if __name__ == "__main__":
    main()
