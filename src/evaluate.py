"""Final evaluation: classification, Charlson score agreement, method comparison.

Usage:
    python src/evaluate.py --split test --methods rules bert_v1
    python src/evaluate.py --split gold --methods rules bert_v1
"""

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.charlson_mapping import CATEGORIES, vectorized_score

RARE_CATEGORIES = {"aids", "hp", "msld", "metacanc", "dementia", "rheumd"}


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    """Point estimate + 95% bootstrap CI."""
    point = float(metric_fn(y_true, y_pred))
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            scores.append(0.0)
    lo = float(np.percentile(scores, 2.5))
    hi = float(np.percentile(scores, 97.5))
    return point, lo, hi


def bootstrap_ci_prob(y_true, y_prob, metric_fn, n_boot=1000, seed=42):
    """Bootstrap CI for probability metrics (AUROC, AUPRC)."""
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return 0.0, np.nan, np.nan
    point = float(metric_fn(y_true, y_prob))
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            scores.append(float(metric_fn(yt, yp)))
        except Exception:
            continue
    if len(scores) < 10:
        return point, np.nan, np.nan
    return point, float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def bootstrap_macro_f1(ref_matrix, pred_matrix, n_boot=1000, seed=42):
    """Bootstrap macro F1 by resampling rows."""
    n_samples, n_cats = ref_matrix.shape
    f1s = [f1_score(ref_matrix[:, j], pred_matrix[:, j], zero_division=0)
           for j in range(n_cats)]
    point = float(np.mean(f1s))

    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n_samples, n_samples)
        boot_f1s = [f1_score(ref_matrix[idx, j], pred_matrix[idx, j], zero_division=0)
                    for j in range(n_cats)]
        scores.append(float(np.mean(boot_f1s)))
    return point, float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ---------------------------------------------------------------------------
# Classification table
# ---------------------------------------------------------------------------
def classification_table(ref, pred, categories=None, n_boot=1000, seed=42):
    """Per-category P/R/F1 with CI, micro/macro. Returns DataFrame."""
    if categories is None:
        categories = CATEGORIES

    has_probs = any(f"{c}_prob" in pred.columns for c in categories)
    rows = []

    _p = lambda yt, yp: precision_score(yt, yp, zero_division=0)
    _r = lambda yt, yp: recall_score(yt, yp, zero_division=0)
    _f = lambda yt, yp: f1_score(yt, yp, zero_division=0)

    for cat in categories:
        y_true = ref[cat].values.astype(int)
        y_pred = pred[cat].values.astype(int)

        p, plo, phi = bootstrap_ci(y_true, y_pred, _p, n_boot, seed)
        r, rlo, rhi = bootstrap_ci(y_true, y_pred, _r, n_boot, seed)
        f, flo, fhi = bootstrap_ci(y_true, y_pred, _f, n_boot, seed)

        row = {
            "category": cat, "n_true": int(y_true.sum()),
            "precision": round(p, 4), "precision_lo": round(plo, 4), "precision_hi": round(phi, 4),
            "recall": round(r, 4), "recall_lo": round(rlo, 4), "recall_hi": round(rhi, 4),
            "f1": round(f, 4), "f1_lo": round(flo, 4), "f1_hi": round(fhi, 4),
        }

        if has_probs and f"{cat}_prob" in pred.columns:
            y_prob = pred[f"{cat}_prob"].values
            au, aulo, auhi = bootstrap_ci_prob(y_true, y_prob, roc_auc_score, n_boot, seed)
            ap, aplo, aphi = bootstrap_ci_prob(y_true, y_prob, average_precision_score, n_boot, seed)
            row.update({
                "auroc": round(au, 4), "auroc_lo": round(aulo, 4), "auroc_hi": round(auhi, 4),
                "auprc": round(ap, 4), "auprc_lo": round(aplo, 4), "auprc_hi": round(aphi, 4),
            })

        rows.append(row)

    # Micro
    all_true = np.concatenate([ref[c].values for c in categories]).astype(int)
    all_pred = np.concatenate([pred[c].values for c in categories]).astype(int)
    mp, mplo, mphi = bootstrap_ci(all_true, all_pred, _p, n_boot, seed)
    mr, mrlo, mrhi = bootstrap_ci(all_true, all_pred, _r, n_boot, seed)
    mf, mflo, mfhi = bootstrap_ci(all_true, all_pred, _f, n_boot, seed)
    rows.append({
        "category": "micro_avg", "n_true": int(all_true.sum()),
        "precision": round(mp, 4), "precision_lo": round(mplo, 4), "precision_hi": round(mphi, 4),
        "recall": round(mr, 4), "recall_lo": round(mrlo, 4), "recall_hi": round(mrhi, 4),
        "f1": round(mf, 4), "f1_lo": round(mflo, 4), "f1_hi": round(mfhi, 4),
    })

    # Macro F1
    ref_m = ref[list(categories)].values.astype(int)
    pred_m = pred[list(categories)].values.astype(int)
    macro, mlo, mhi = bootstrap_macro_f1(ref_m, pred_m, n_boot, seed)
    rows.append({
        "category": "macro_avg", "n_true": 0,
        "f1": round(macro, 4), "f1_lo": round(mlo, 4), "f1_hi": round(mhi, 4),
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charlson score
# ---------------------------------------------------------------------------
def compute_scores(flags_df):
    """Compute Charlson score from 17 binary columns. Returns Series."""
    return vectorized_score(flags_df[CATEGORIES])["charlson_icd"].astype(int)


def score_to_class(score):
    """Map score to ordinal class: 0, 1-2, 3-4, >=5."""
    if score == 0:
        return 0
    if score <= 2:
        return 1
    if score <= 4:
        return 2
    return 3


def score_agreement(ref_scores, pred_scores):
    """ICC, weighted kappa, exact/within-1, Bland-Altman stats."""
    ref = np.asarray(ref_scores, dtype=float)
    pred = np.asarray(pred_scores, dtype=float)
    n = len(ref)

    # ICC (absolute agreement, two-way random, single measures)
    # Perfect agreement → pingouin returns NaN (zero variance); handle explicitly
    if np.array_equal(ref, pred):
        icc_val, icc_lo, icc_hi = 1.0, 1.0, 1.0
    else:
        try:
            import warnings
            icc_data = pd.DataFrame({
                "targets": list(range(n)) + list(range(n)),
                "raters": ["ref"] * n + ["pred"] * n,
                "ratings": list(ref) + list(pred),
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                icc_result = pg.intraclass_corr(
                    data=icc_data, targets="targets", raters="raters", ratings="ratings",
                )
            row = icc_result[icc_result["Type"] == "ICC2"]
            icc_val = float(row["ICC"].values[0])
            ci = row["CI95%"].values[0]
            icc_lo, icc_hi = float(ci[0]), float(ci[1])
            # Replace NaN from degenerate cases
            if np.isnan(icc_val):
                icc_val, icc_lo, icc_hi = np.nan, np.nan, np.nan
        except Exception:
            icc_val, icc_lo, icc_hi = np.nan, np.nan, np.nan

    # Weighted kappa on score classes
    ref_cls = np.array([score_to_class(s) for s in ref])
    pred_cls = np.array([score_to_class(s) for s in pred])
    try:
        kappa = float(cohen_kappa_score(ref_cls, pred_cls, weights="quadratic"))
    except Exception:
        kappa = np.nan

    # Exact and within ±1
    exact_pct = round(float(np.mean(ref == pred)) * 100, 1)
    within1_pct = round(float(np.mean(np.abs(ref - pred) <= 1)) * 100, 1)

    # Bland-Altman
    diffs = pred - ref
    bias = round(float(np.mean(diffs)), 3)
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    loa_lower = round(bias - 1.96 * std_diff, 3)
    loa_upper = round(bias + 1.96 * std_diff, 3)

    return {
        "icc": round(icc_val, 4), "icc_lo": round(icc_lo, 4), "icc_hi": round(icc_hi, 4),
        "kappa_quad": round(kappa, 4),
        "exact_pct": exact_pct, "within1_pct": within1_pct,
        "bias": bias, "loa_lower": loa_lower, "loa_upper": loa_upper,
    }


# ---------------------------------------------------------------------------
# Method comparison
# ---------------------------------------------------------------------------
def paired_bootstrap_macro_f1(ref_df, pred_a_df, pred_b_df,
                               categories=None, n_boot=1000, seed=42):
    """Bootstrap difference in macro F1 between two methods (A - B)."""
    if categories is None:
        categories = CATEGORIES
    ref_m = ref_df[list(categories)].values.astype(int)
    pred_a = pred_a_df[list(categories)].values.astype(int)
    pred_b = pred_b_df[list(categories)].values.astype(int)
    n_samples, n_cats = ref_m.shape

    def _macro_f1(ref_mat, pred_mat):
        return float(np.mean([
            f1_score(ref_mat[:, j], pred_mat[:, j], zero_division=0)
            for j in range(n_cats)
        ]))

    diff = _macro_f1(ref_m, pred_a) - _macro_f1(ref_m, pred_b)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n_samples, n_samples)
        d = _macro_f1(ref_m[idx], pred_a[idx]) - _macro_f1(ref_m[idx], pred_b[idx])
        diffs.append(d)
    return (round(diff, 4),
            round(float(np.percentile(diffs, 2.5)), 4),
            round(float(np.percentile(diffs, 97.5)), 4))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_bland_altman(ref_scores, pred_scores, method, save_path):
    ref = np.asarray(ref_scores, dtype=float)
    pred = np.asarray(pred_scores, dtype=float)
    means = (ref + pred) / 2
    diffs = pred - ref
    bias = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    loa_lo, loa_hi = bias - 1.96 * std_d, bias + 1.96 * std_d

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(means, diffs, alpha=0.4, s=10)
    ax.axhline(bias, color="red", linestyle="-", label=f"Biais = {bias:.2f}")
    ax.axhline(loa_lo, color="gray", linestyle="--",
               label=f"LOA = [{loa_lo:.2f}, {loa_hi:.2f}]")
    ax.axhline(loa_hi, color="gray", linestyle="--")
    ax.set_xlabel("Moyenne (réf + préd) / 2")
    ax.set_ylabel("Différence (préd − réf)")
    ax.set_title(f"Bland-Altman : {method}")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_f1_bars(all_results, split, save_path):
    """Grouped bar chart of F1 per category and method with CI."""
    methods = list(all_results.keys())
    cats = CATEGORIES

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(cats))
    width = 0.8 / max(len(methods), 1)

    for i, method in enumerate(methods):
        df = all_results[method]
        df_c = df[df["category"].isin(cats)].set_index("category")
        f1s = [float(df_c.loc[c, "f1"]) if c in df_c.index else 0 for c in cats]
        lo = [float(df_c.loc[c, "f1_lo"]) if c in df_c.index and "f1_lo" in df_c.columns else 0 for c in cats]
        hi = [float(df_c.loc[c, "f1_hi"]) if c in df_c.index and "f1_hi" in df_c.columns else 0 for c in cats]
        yerr_lo = [f - l for f, l in zip(f1s, lo)]
        yerr_hi = [h - f for f, h in zip(f1s, hi)]

        ax.bar(x + i * width, f1s, width,
               yerr=[yerr_lo, yerr_hi], capsize=2, label=method, alpha=0.8)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"F1 par catégorie ({split})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _align(ref, pred):
    """Align ref and pred on common hadm_ids."""
    common = set(ref["hadm_id"]) & set(pred["hadm_id"])
    r = ref[ref["hadm_id"].isin(common)].sort_values("hadm_id").reset_index(drop=True)
    p = pred[pred["hadm_id"].isin(common)].sort_values("hadm_id").reset_index(drop=True)
    return r, p


def _load_preds(method, split):
    """Find and load predictions parquet."""
    candidates = [
        f"data/processed/preds_{method}_{split}.parquet",
        f"data/processed/preds_bert_{split}_{method}.parquet",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_parquet(path)
    raise FileNotFoundError(
        f"Prédictions introuvables pour method={method}, split={split}. "
        f"Candidats testés : {candidates}"
    )


def _print_summary(table):
    """Print a compact summary of a classification table."""
    for _, r in table.iterrows():
        f1 = r.get("f1", 0)
        flo = r.get("f1_lo", np.nan)
        fhi = r.get("f1_hi", np.nan)
        ci = f" [{flo:.3f}, {fhi:.3f}]" if pd.notna(flo) else ""
        st = f" ({r['sample_type']})" if "sample_type" in r.index and pd.notna(r.get("sample_type")) and r.get("sample_type") != "all" else ""
        n = f" n={int(r['n_true'])}" if pd.notna(r.get("n_true")) and r.get("n_true", 0) > 0 else ""
        print(f"  {r['category']:<15} F1={f1:.3f}{ci}{n}{st}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Évaluation finale.")
    parser.add_argument("--split", required=True, choices=["test", "gold"])
    parser.add_argument("--methods", nargs="+", required=True,
                        help="Noms des méthodes (ex. rules bert_v1)")
    parser.add_argument("--labels", default="data/processed/labels_charlson.parquet")
    parser.add_argument("--gold", default="data/processed/gold_annotations.csv")
    parser.add_argument("--splits-file", default="data/processed/splits.parquet")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split = args.split
    os.makedirs("outputs/figures", exist_ok=True)

    # --- Load reference ---
    if split == "gold":
        gold = pd.read_csv(args.gold)
        ref = gold[["hadm_id"] + CATEGORIES].copy()
        splits = pd.read_parquet(args.splits_file)
        gold_sampling = splits[splits["split"] == "gold"][["hadm_id", "sampling"]]
        ref = ref.merge(gold_sampling, on="hadm_id", how="left")
        icd = pd.read_parquet(args.labels)
        icd = icd[icd["hadm_id"].isin(ref["hadm_id"])]
    else:
        icd = pd.read_parquet(args.labels)
        splits = pd.read_parquet(args.splits_file)
        split_ids = set(splits.loc[splits["split"] == split, "hadm_id"])
        ref = icd[icd["hadm_id"].isin(split_ids)].copy()

    # --- Load predictions ---
    preds = {}
    for method in args.methods:
        preds[method] = _load_preds(method, split)

    # =========================================================
    # 1. Classification metrics
    # =========================================================
    all_class_results = {}
    all_class_rows = []

    for method in args.methods:
        ref_a, pred_a = _align(ref, preds[method])

        if split == "gold":
            mask_random = ref_a["sampling"] == "random"
            ref_rand = ref_a[mask_random].reset_index(drop=True)
            pred_rand = pred_a[mask_random].reset_index(drop=True)

            non_rare = [c for c in CATEGORIES if c not in RARE_CATEGORIES]
            rare = [c for c in CATEGORIES if c in RARE_CATEGORIES]

            # Non-rare on random
            tbl_nr = classification_table(ref_rand, pred_rand, non_rare, args.n_boot, args.seed)
            tbl_nr = tbl_nr[~tbl_nr["category"].isin(["micro_avg", "macro_avg"])]
            tbl_nr["sample_type"] = "random"

            # Rare on all
            tbl_r = classification_table(ref_a, pred_a, rare, args.n_boot, args.seed)
            tbl_r = tbl_r[~tbl_r["category"].isin(["micro_avg", "macro_avg"])]
            tbl_r["sample_type"] = "random+enriched"

            # Micro/macro on random (all 17 categories)
            tbl_agg = classification_table(ref_rand, pred_rand, CATEGORIES, args.n_boot, args.seed)
            tbl_agg = tbl_agg[tbl_agg["category"].isin(["micro_avg", "macro_avg"])]
            tbl_agg["sample_type"] = "random"

            table = pd.concat([tbl_nr, tbl_r, tbl_agg], ignore_index=True)
        else:
            table = classification_table(ref_a, pred_a, CATEGORIES, args.n_boot, args.seed)
            table["sample_type"] = "all"

        table["method"] = method
        all_class_results[method] = table
        all_class_rows.append(table)

    class_df = pd.concat(all_class_rows, ignore_index=True)
    class_path = f"outputs/eval_{split}_classification.csv"
    class_df.to_csv(class_path, index=False)
    print(f"Classification → {class_path}")

    for method in args.methods:
        print(f"\n--- {method} ---")
        _print_summary(all_class_results[method])

    # =========================================================
    # 2. Charlson score agreement
    # =========================================================
    score_rows = []
    for method in args.methods:
        ref_a, pred_a = _align(ref, preds[method])

        if split == "gold":
            mask = ref_a["sampling"] == "random"
            ref_s, pred_s = ref_a[mask], pred_a[mask]
        else:
            ref_s, pred_s = ref_a, pred_a

        ref_scores = compute_scores(ref_s)
        pred_scores = compute_scores(pred_s)

        agr = score_agreement(ref_scores.values, pred_scores.values)
        agr["method"] = method
        score_rows.append(agr)

        plot_bland_altman(ref_scores.values, pred_scores.values, method,
                          f"outputs/figures/bland_altman_{split}_{method}.png")

    score_df = pd.DataFrame(score_rows)
    score_path = f"outputs/eval_{split}_score.csv"
    score_df.to_csv(score_path, index=False)
    print(f"\nScore de Charlson → {score_path}")
    for _, r in score_df.iterrows():
        print(f"  {r['method']}: ICC={r['icc']:.3f} [{r['icc_lo']:.3f}, {r['icc_hi']:.3f}]"
              f"  κ={r['kappa_quad']:.3f}  exact={r['exact_pct']:.0f}%  ±1={r['within1_pct']:.0f}%"
              f"  biais={r['bias']:.2f}")

    # =========================================================
    # 3. Method comparison
    # =========================================================
    if len(args.methods) > 1:
        comp_rows = []
        for m1, m2 in combinations(args.methods, 2):
            ref_a1, pred_a1 = _align(ref, preds[m1])
            ref_a2, pred_a2 = _align(ref, preds[m2])
            common = set(ref_a1["hadm_id"]) & set(ref_a2["hadm_id"])

            ref_c = ref[ref["hadm_id"].isin(common)].sort_values("hadm_id").reset_index(drop=True)
            p1 = preds[m1][preds[m1]["hadm_id"].isin(common)].sort_values("hadm_id").reset_index(drop=True)
            p2 = preds[m2][preds[m2]["hadm_id"].isin(common)].sort_values("hadm_id").reset_index(drop=True)

            if split == "gold":
                mask = ref_c["sampling"] == "random"
                ref_c = ref_c[mask].reset_index(drop=True)
                p1 = p1[mask.values].reset_index(drop=True)
                p2 = p2[mask.values].reset_index(drop=True)

            diff, lo, hi = paired_bootstrap_macro_f1(ref_c, p1, p2, CATEGORIES, args.n_boot, args.seed)
            comp_rows.append({
                "method_a": m1, "method_b": m2,
                "diff_macro_f1": diff, "diff_lo": lo, "diff_hi": hi,
            })

        comp_df = pd.DataFrame(comp_rows)
        comp_path = f"outputs/eval_{split}_comparisons.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\nComparaisons → {comp_path}")
        for _, r in comp_df.iterrows():
            print(f"  {r['method_a']} − {r['method_b']}: "
                  f"Δ macro F1 = {r['diff_macro_f1']:+.4f} [{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}]")

    # F1 bar chart
    plot_f1_bars(all_class_results, split, f"outputs/figures/f1_by_category_{split}.png")

    # =========================================================
    # 4. Gold-specific: ICD agreement + sensitivity
    # =========================================================
    if split == "gold":
        # ICD vs gold agreement
        icd_a = icd[icd["hadm_id"].isin(ref["hadm_id"])].sort_values("hadm_id").reset_index(drop=True)
        gold_a = ref[ref["hadm_id"].isin(icd_a["hadm_id"])].sort_values("hadm_id").reset_index(drop=True)

        icd_rows = []
        for cat in CATEGORIES:
            y_gold = gold_a[cat].values.astype(int)
            y_icd = icd_a[cat].values.astype(int)
            p = precision_score(y_gold, y_icd, zero_division=0)
            r = recall_score(y_gold, y_icd, zero_division=0)
            f = f1_score(y_gold, y_icd, zero_division=0)
            try:
                k = cohen_kappa_score(y_gold, y_icd)
            except Exception:
                k = np.nan
            icd_rows.append({
                "category": cat, "n_gold": int(y_gold.sum()), "n_icd": int(y_icd.sum()),
                "precision": round(p, 4), "recall": round(r, 4),
                "f1": round(f, 4), "kappa": round(float(k), 4) if pd.notna(k) else np.nan,
            })

        icd_df = pd.DataFrame(icd_rows)
        icd_path = "outputs/eval_gold_icd_agreement.csv"
        icd_df.to_csv(icd_path, index=False)
        print(f"\nAccord ICD vs gold → {icd_path}")
        for _, r in icd_df.iterrows():
            print(f"  {r['category']:<15} F1={r['f1']:.3f}  κ={r['kappa']:.3f}  "
                  f"n_gold={r['n_gold']}  n_icd={r['n_icd']}")

        # Sensitivity: exclude uncertain
        if "incertain" in gold.columns:
            mask_certain = ~gold["incertain"].astype(bool)
            n_uncertain = int((~mask_certain).sum())
            gold_certain = gold[mask_certain]
            ref_certain = gold_certain[["hadm_id"] + CATEGORIES].copy()
            ref_certain = ref_certain.merge(gold_sampling, on="hadm_id", how="left")

            print(f"\n--- Sensibilité (excluant {n_uncertain} incertains) ---")
            sens_rows = []
            for method in args.methods:
                ref_c, pred_c = _align(ref_certain, preds[method])
                mask_r = ref_c["sampling"] == "random"
                ref_r = ref_c[mask_r].reset_index(drop=True)
                pred_r = pred_c[mask_r].reset_index(drop=True)

                tbl = classification_table(ref_r, pred_r, CATEGORIES, args.n_boot, args.seed)
                tbl["method"] = method
                sens_rows.append(tbl)

                macro = tbl[tbl["category"] == "macro_avg"].iloc[0]
                micro = tbl[tbl["category"] == "micro_avg"].iloc[0]
                print(f"  {method}: macro F1={macro['f1']:.3f} [{macro['f1_lo']:.3f}, {macro['f1_hi']:.3f}]"
                      f"  micro F1={micro['f1']:.3f} [{micro['f1_lo']:.3f}, {micro['f1_hi']:.3f}]")

            sens_df = pd.concat(sens_rows, ignore_index=True)
            sens_path = "outputs/eval_gold_sensitivity.csv"
            sens_df.to_csv(sens_path, index=False)

    print("\nTerminé.")


if __name__ == "__main__":
    main()
