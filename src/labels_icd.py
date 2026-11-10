"""Build Charlson comorbidity labels from ICD codes (MIMIC-IV)."""

import argparse
import os

import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.charlson_mapping import (
    CATEGORIES,
    vectorized_match, vectorized_score,
)


def build_labels(diag: pd.DataFrame, hadm_ids: set) -> pd.DataFrame:
    """Build 17 binary Charlson columns per hadm_id from diagnoses dataframe.

    Uses vectorized prefix matching for performance.

    Args:
        diag: DataFrame with columns hadm_id, icd_code, icd_version.
        hadm_ids: set of hadm_ids to include.
    """
    diag = diag[diag["hadm_id"].isin(hadm_ids)].copy()

    if len(diag) == 0:
        df = pd.DataFrame({"hadm_id": list(hadm_ids)})
        for cat in CATEGORIES:
            df[cat] = 0
        return df

    # Vectorized matching: one bool column per category per code row
    print("Mapping vectorisé des codes ICD…")
    matches = vectorized_match(
        diag["icd_code"].astype(str), diag["icd_version"].astype(int)
    )
    matches["hadm_id"] = diag["hadm_id"].values

    # Aggregate: max per hadm_id (any match → 1)
    agg = matches.groupby("hadm_id")[CATEGORIES].max().astype(int)

    # Add hadm_ids that had no ICD codes at all
    all_hadm = pd.DataFrame({"hadm_id": list(hadm_ids)})
    result = all_hadm.merge(agg, on="hadm_id", how="left")
    result[CATEGORIES] = result[CATEGORIES].fillna(0).astype(int)

    return result


def compute_age_at_admission(
    labels: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.Series:
    """Compute age at admission: anchor_age + (year(admittime) - anchor_year)."""
    pat = patients.set_index("subject_id")[["anchor_age", "anchor_year"]]
    adm_year = (
        pd.to_datetime(admissions.set_index("hadm_id")["admittime"]).dt.year
    )
    anchor_age = labels["subject_id"].map(pat["anchor_age"])
    anchor_year = labels["subject_id"].map(pat["anchor_year"])
    admit_year = labels["hadm_id"].map(adm_year)
    return anchor_age + (admit_year - anchor_year)


def compute_scores(
    labels: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    """Add charlson_icd, charlson_icd_age and age_admission columns (vectorized)."""
    labels = labels.copy()
    ages = compute_age_at_admission(labels, patients, admissions)
    labels["age_admission"] = ages

    scores = vectorized_score(labels, ages)
    labels["charlson_icd"] = scores["charlson_icd"]
    labels["charlson_icd_age"] = scores["charlson_icd_age"]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Build Charlson labels from ICD codes.")
    parser.add_argument("--pmh", default="data/interim/pmh.parquet")
    parser.add_argument("--diagnoses", default="data/raw/diagnoses_icd.csv")
    parser.add_argument("--patients", default="data/raw/patients.csv")
    parser.add_argument("--admissions", default="data/raw/admissions.csv")
    parser.add_argument("--output", default="data/processed/labels_charlson.parquet")
    parser.add_argument("--stats-output", default="outputs/labels_stats.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)

    # Load pmh to get the list of hadm_ids
    pmh = pd.read_parquet(args.pmh, columns=["hadm_id", "subject_id", "pmh_found"])
    hadm_ids = set(pmh["hadm_id"].unique())
    print(f"Séjours dans pmh.parquet : {len(hadm_ids)}")

    # Load diagnoses
    diag = pd.read_csv(args.diagnoses, dtype={"icd_code": str, "icd_version": int})
    diag_filtered = diag[diag["hadm_id"].isin(hadm_ids)]
    print(f"Codes ICD dans le périmètre : {len(diag_filtered)}")

    # ICD version distribution
    hadm_versions = diag_filtered.groupby("hadm_id")["icd_version"].apply(set)
    n_icd9_only = (hadm_versions.apply(lambda s: s == {9})).sum()
    n_icd10_only = (hadm_versions.apply(lambda s: s == {10})).sum()
    n_mixed = (hadm_versions.apply(lambda s: 9 in s and 10 in s)).sum()
    n_with_icd = len(hadm_versions)

    # Build labels
    labels = build_labels(diag_filtered, hadm_ids)

    # Mark hadm_ids without any ICD code
    hadm_with_icd = set(diag_filtered["hadm_id"].unique())
    labels["has_icd"] = labels["hadm_id"].isin(hadm_with_icd)

    # Set categories to NaN for hadm_ids without ICD codes
    no_icd_mask = ~labels["has_icd"]
    labels.loc[no_icd_mask, CATEGORIES] = pd.NA

    # Add subject_id
    sid_map = pmh.set_index("hadm_id")["subject_id"]
    labels["subject_id"] = labels["hadm_id"].map(sid_map)

    # Load patients for age
    patients = pd.read_csv(args.patients, usecols=["subject_id", "anchor_age", "anchor_year"])
    admissions = pd.read_csv(args.admissions, usecols=["hadm_id", "admittime"])

    # Compute scores (only for those with ICD)
    labels_with_icd = labels[labels["has_icd"]].copy()
    labels_no_icd = labels[~labels["has_icd"]].copy()
    labels_no_icd["charlson_icd"] = pd.NA
    labels_no_icd["charlson_icd_age"] = pd.NA
    labels_no_icd["age_admission"] = pd.NA

    if len(labels_with_icd) > 0:
        labels_with_icd = compute_scores(labels_with_icd, patients, admissions)

    labels_final = pd.concat([labels_with_icd, labels_no_icd], ignore_index=True)

    # Reorder columns
    out_cols = ["hadm_id", "subject_id", "has_icd"] + CATEGORIES + ["charlson_icd", "charlson_icd_age", "age_admission"]
    labels_final = labels_final[out_cols]
    labels_final.to_parquet(args.output, index=False)

    # --- Statistics (aggregate only) ---
    pmh_found_map = pmh.set_index("hadm_id")["pmh_found"]

    stats_rows = []

    # General stats
    pct_icd = 100.0 * n_with_icd / len(hadm_ids) if len(hadm_ids) else 0
    stats_rows.append({"metric": "n_sejours_pmh", "value": len(hadm_ids)})
    stats_rows.append({"metric": "n_avec_icd", "value": n_with_icd})
    stats_rows.append({"metric": "pct_avec_icd", "value": round(pct_icd, 2)})
    stats_rows.append({"metric": "n_icd9_only", "value": n_icd9_only})
    stats_rows.append({"metric": "n_icd10_only", "value": n_icd10_only})
    stats_rows.append({"metric": "n_mixed", "value": n_mixed})

    # Category prevalence (on has_icd=True AND pmh_found=True)
    subset = labels_final[
        (labels_final["has_icd"]) & (labels_final["hadm_id"].map(pmh_found_map).fillna(False))
    ]
    n_subset = len(subset)
    stats_rows.append({"metric": "n_subset_icd_and_pmh", "value": n_subset})

    print(f"\n{'Catégorie':<15} {'N':>8} {'%':>8}")
    print("-" * 33)
    for cat in CATEGORIES:
        n_pos = int(subset[cat].sum())
        pct = 100.0 * n_pos / n_subset if n_subset else 0
        print(f"{cat:<15} {n_pos:>8} {pct:>7.1f}%")
        stats_rows.append({"metric": f"n_{cat}", "value": n_pos})
        stats_rows.append({"metric": f"pct_{cat}", "value": round(pct, 2)})

    # Score distribution
    scores = labels_final.loc[labels_final["has_icd"], "charlson_icd"].dropna()
    print(f"\nDistribution charlson_icd (n={len(scores)}):")
    print(f"  Moyenne : {scores.mean():.2f}")
    print(f"  Médiane : {scores.median():.0f}")
    q = scores.quantile([0, 0.25, 0.5, 0.75, 1.0])
    for p, v in q.items():
        print(f"  Q{p:.0%}     : {v:.0f}")
        stats_rows.append({"metric": f"charlson_q{int(p*100)}", "value": int(v)})
    stats_rows.append({"metric": "charlson_mean", "value": round(scores.mean(), 2)})

    # Counts by score value
    print("\nEffectifs par score :")
    score_counts = scores.clip(upper=10).astype(int).value_counts().sort_index()
    for val, cnt in score_counts.items():
        label = f"{val}+" if val == 10 else str(val)
        print(f"  {label:>4} : {cnt}")
        stats_rows.append({"metric": f"charlson_n_{label}", "value": int(cnt)})

    # Age at admission distribution
    ages = labels_final.loc[labels_final["has_icd"], "age_admission"].dropna()
    if len(ages) > 0:
        print(f"\nDistribution age_admission (n={len(ages)}):")
        aq = ages.quantile([0, 0.25, 0.5, 0.75, 1.0])
        for p, v in aq.items():
            label = f"age_q{int(p*100)}"
            print(f"  Q{p:.0%}     : {v:.0f}")
            stats_rows.append({"metric": label, "value": int(v)})

    pd.DataFrame(stats_rows).to_csv(args.stats_output, index=False)
    print(f"\nStats enregistrées dans {args.stats_output}")
    print(f"Labels enregistrés dans {args.output}")


if __name__ == "__main__":
    main()
