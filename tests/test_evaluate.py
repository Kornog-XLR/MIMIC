"""Tests for src/evaluate.py on fictitious data."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.charlson_mapping import CATEGORIES
from src.evaluate import (
    RARE_CATEGORIES,
    bootstrap_ci,
    bootstrap_ci_prob,
    bootstrap_macro_f1,
    classification_table,
    compute_scores,
    paired_bootstrap_macro_f1,
    score_agreement,
    score_to_class,
)


# ---------------------------------------------------------------------------
# Helpers to build test DataFrames
# ---------------------------------------------------------------------------
def _make_df(hadm_ids, cat_values, categories, probs=None):
    """Build a DataFrame with hadm_id + category columns [+ prob columns]."""
    df = pd.DataFrame({"hadm_id": hadm_ids})
    for i, cat in enumerate(categories):
        df[cat] = cat_values[:, i]
    if probs is not None:
        for i, cat in enumerate(categories):
            df[f"{cat}_prob"] = probs[:, i]
    return df


def _zeros_df(hadm_ids, categories=CATEGORIES):
    """DataFrame with all zeros for every category."""
    n = len(hadm_ids)
    return _make_df(hadm_ids, np.zeros((n, len(categories)), dtype=int), categories)


# ---------------------------------------------------------------------------
# Hand-calculated metrics (2 categories, 4 samples)
# ---------------------------------------------------------------------------
class TestHandCalculatedMetrics:
    """
    cat_a: true=[1,1,0,0], pred=[1,0,0,0] → P=1.0, R=0.5, F1=2/3
    cat_b: true=[0,1,1,0], pred=[0,1,1,1] → P=2/3, R=1.0, F1=0.8
    micro: TP=3, FP=1, FN=1 → P=3/4, R=3/4, F1=0.75
    macro F1: (2/3 + 0.8) / 2 ≈ 0.7333
    """

    @pytest.fixture()
    def data(self):
        cats = ["cat_a", "cat_b"]
        ref = _make_df([1, 2, 3, 4], np.array([[1, 0], [1, 1], [0, 1], [0, 0]]), cats)
        pred = _make_df([1, 2, 3, 4], np.array([[1, 0], [0, 1], [0, 1], [0, 1]]), cats)
        return ref, pred, cats

    def test_per_category_f1(self, data):
        ref, pred, cats = data
        table = classification_table(ref, pred, cats, n_boot=100, seed=42)
        row_a = table[table["category"] == "cat_a"].iloc[0]
        row_b = table[table["category"] == "cat_b"].iloc[0]
        assert abs(row_a["f1"] - 2 / 3) < 0.001
        assert abs(row_b["f1"] - 0.8) < 0.001

    def test_per_category_precision_recall(self, data):
        ref, pred, cats = data
        table = classification_table(ref, pred, cats, n_boot=100, seed=42)
        row_a = table[table["category"] == "cat_a"].iloc[0]
        assert abs(row_a["precision"] - 1.0) < 0.001
        assert abs(row_a["recall"] - 0.5) < 0.001
        row_b = table[table["category"] == "cat_b"].iloc[0]
        assert abs(row_b["recall"] - 1.0) < 0.001

    def test_micro_f1(self, data):
        ref, pred, cats = data
        table = classification_table(ref, pred, cats, n_boot=100, seed=42)
        micro = table[table["category"] == "micro_avg"].iloc[0]
        assert abs(micro["f1"] - 0.75) < 0.001

    def test_macro_f1(self, data):
        ref, pred, cats = data
        table = classification_table(ref, pred, cats, n_boot=100, seed=42)
        macro = table[table["category"] == "macro_avg"].iloc[0]
        expected = (2 / 3 + 0.8) / 2
        assert abs(macro["f1"] - expected) < 0.001

    def test_ci_present(self, data):
        ref, pred, cats = data
        table = classification_table(ref, pred, cats, n_boot=100, seed=42)
        row_a = table[table["category"] == "cat_a"].iloc[0]
        assert "f1_lo" in row_a.index
        assert "f1_hi" in row_a.index
        assert row_a["f1_lo"] <= row_a["f1"]
        assert row_a["f1_hi"] >= row_a["f1"]

    def test_output_shape(self, data):
        ref, pred, cats = data
        table = classification_table(ref, pred, cats, n_boot=100, seed=42)
        # 2 categories + micro + macro = 4 rows
        assert len(table) == 4


# ---------------------------------------------------------------------------
# Bootstrap reproducibility
# ---------------------------------------------------------------------------
class TestBootstrapReproducibility:
    def test_same_seed_same_result(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 50).astype(int)
        y_pred = rng.randint(0, 2, 50).astype(int)
        metric_fn = lambda yt, yp: f1_score(yt, yp, zero_division=0)

        _, lo1, hi1 = bootstrap_ci(y_true, y_pred, metric_fn, n_boot=200, seed=123)
        _, lo2, hi2 = bootstrap_ci(y_true, y_pred, metric_fn, n_boot=200, seed=123)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_different_seed_different_result(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 50).astype(int)
        y_pred = rng.randint(0, 2, 50).astype(int)
        metric_fn = lambda yt, yp: f1_score(yt, yp, zero_division=0)

        _, lo1, _ = bootstrap_ci(y_true, y_pred, metric_fn, n_boot=200, seed=1)
        _, lo2, _ = bootstrap_ci(y_true, y_pred, metric_fn, n_boot=200, seed=2)
        # Very unlikely to be identical with different seeds
        assert lo1 != lo2

    def test_macro_f1_bootstrap_reproducible(self):
        rng = np.random.RandomState(0)
        ref_m = rng.randint(0, 2, (30, 3))
        pred_m = rng.randint(0, 2, (30, 3))
        _, lo1, hi1 = bootstrap_macro_f1(ref_m, pred_m, n_boot=200, seed=99)
        _, lo2, hi2 = bootstrap_macro_f1(ref_m, pred_m, n_boot=200, seed=99)
        assert lo1 == lo2
        assert hi1 == hi2


# ---------------------------------------------------------------------------
# Bootstrap CI for probability metrics
# ---------------------------------------------------------------------------
class TestBootstrapCIProb:
    def test_all_zeros_returns_nan_ci(self):
        y_true = np.zeros(10, dtype=int)
        y_prob = np.random.rand(10)
        _, lo, hi = bootstrap_ci_prob(y_true, y_prob, lambda yt, yp: 0.0, n_boot=50, seed=42)
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_normal_case(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 50)
        y_prob = rng.rand(50)
        point, lo, hi = bootstrap_ci_prob(y_true, y_prob,
            lambda yt, yp: float(np.mean(yp[yt == 1] > 0.5)),
            n_boot=100, seed=42)
        assert lo <= point <= hi or np.isnan(lo)


# ---------------------------------------------------------------------------
# Random / enriched separation for gold
# ---------------------------------------------------------------------------
class TestRandomEnrichedSeparation:
    def test_rare_on_enriched_has_more_samples(self):
        """Rare categories should use more samples when enriched are included."""
        n_random, n_enriched = 20, 10
        hadm_all = list(range(n_random + n_enriched))
        hadm_random = list(range(n_random))

        cats = ["rare_cat", "normal_cat"]
        rng = np.random.RandomState(42)

        # All samples have labels
        ref_all = _make_df(hadm_all, rng.randint(0, 2, (30, 2)), cats)
        pred_all = _make_df(hadm_all, rng.randint(0, 2, (30, 2)), cats)

        # Random subset
        ref_rand = ref_all[ref_all["hadm_id"].isin(hadm_random)].reset_index(drop=True)
        pred_rand = pred_all[pred_all["hadm_id"].isin(hadm_random)].reset_index(drop=True)

        # Full table on all
        tbl_all = classification_table(ref_all, pred_all, cats, n_boot=50, seed=42)
        # Table on random only
        tbl_rand = classification_table(ref_rand, pred_rand, cats, n_boot=50, seed=42)

        # Both should have rows for the categories
        assert len(tbl_all[tbl_all["category"] == "rare_cat"]) == 1
        assert len(tbl_rand[tbl_rand["category"] == "rare_cat"]) == 1

    def test_rare_categories_list(self):
        """Check RARE_CATEGORIES is a subset of CATEGORIES."""
        assert RARE_CATEGORIES.issubset(set(CATEGORIES))


# ---------------------------------------------------------------------------
# Charlson score with hierarchy
# ---------------------------------------------------------------------------
class TestCharlsonScoreComputation:
    def test_zero_score(self):
        df = _zeros_df([1, 2, 3])
        scores = compute_scores(df)
        assert (scores == 0).all()

    def test_simple_score(self):
        """mi=1 → score 1."""
        df = _zeros_df([1])
        df.loc[0, "mi"] = 1
        scores = compute_scores(df)
        assert scores.iloc[0] == 1

    def test_hierarchy_diabwc_suppresses_diab(self):
        """diab=1, diabwc=1 → only diabwc counted (weight 2), diab suppressed."""
        df = _zeros_df([1])
        df.loc[0, "diab"] = 1
        df.loc[0, "diabwc"] = 1
        scores = compute_scores(df)
        # diabwc(2) only, diab suppressed
        assert scores.iloc[0] == 2

    def test_hierarchy_metacanc_suppresses_canc(self):
        """canc=1, metacanc=1 → metacanc(6), canc suppressed."""
        df = _zeros_df([1])
        df.loc[0, "canc"] = 1
        df.loc[0, "metacanc"] = 1
        scores = compute_scores(df)
        assert scores.iloc[0] == 6

    def test_hierarchy_msld_suppresses_mld(self):
        """mld=1, msld=1 → msld(3), mld suppressed."""
        df = _zeros_df([1])
        df.loc[0, "mld"] = 1
        df.loc[0, "msld"] = 1
        scores = compute_scores(df)
        assert scores.iloc[0] == 3

    def test_multiple_categories(self):
        """mi=1(1) + chf=1(1) + rend=1(2) = 4."""
        df = _zeros_df([1])
        df.loc[0, "mi"] = 1
        df.loc[0, "chf"] = 1
        df.loc[0, "rend"] = 1
        scores = compute_scores(df)
        assert scores.iloc[0] == 4


# ---------------------------------------------------------------------------
# Score classes
# ---------------------------------------------------------------------------
class TestScoreToClass:
    def test_zero(self):
        assert score_to_class(0) == 0

    def test_one(self):
        assert score_to_class(1) == 1

    def test_two(self):
        assert score_to_class(2) == 1

    def test_three(self):
        assert score_to_class(3) == 2

    def test_four(self):
        assert score_to_class(4) == 2

    def test_five(self):
        assert score_to_class(5) == 3

    def test_ten(self):
        assert score_to_class(10) == 3


# ---------------------------------------------------------------------------
# ICC and kappa on simple cases
# ---------------------------------------------------------------------------
class TestScoreAgreement:
    def test_perfect_agreement(self):
        ref = np.array([0, 1, 2, 3, 4, 5])
        pred = np.array([0, 1, 2, 3, 4, 5])
        agr = score_agreement(ref, pred)
        assert agr["exact_pct"] == 100.0
        assert agr["within1_pct"] == 100.0
        assert agr["bias"] == 0.0
        # ICC should be very high (1.0 or close)
        assert agr["icc"] > 0.99
        # Kappa should be 1.0
        assert agr["kappa_quad"] == 1.0

    def test_constant_bias(self):
        """Constant +1 bias: not perfect but high agreement."""
        ref = np.array([0, 2, 4, 6, 8])
        pred = np.array([1, 3, 5, 7, 9])
        agr = score_agreement(ref, pred)
        assert agr["exact_pct"] == 0.0
        assert agr["within1_pct"] == 100.0
        assert abs(agr["bias"] - 1.0) < 0.001

    def test_poor_agreement(self):
        """Reversed scores: poor agreement metrics."""
        ref = np.array([0, 1, 2, 3, 4])
        pred = np.array([4, 3, 2, 1, 0])
        agr = score_agreement(ref, pred)
        # Exact match should be low (only middle value matches)
        assert agr["exact_pct"] <= 20.1
        # Bias should be ~0 (symmetric reversal)
        assert abs(agr["bias"]) < 0.01

    def test_bland_altman_loa(self):
        ref = np.array([0, 1, 2, 3, 4, 5])
        pred = np.array([0, 1, 2, 3, 4, 5])
        agr = score_agreement(ref, pred)
        assert agr["loa_lower"] == 0.0
        assert agr["loa_upper"] == 0.0


# ---------------------------------------------------------------------------
# Paired bootstrap macro F1 difference
# ---------------------------------------------------------------------------
class TestPairedBootstrap:
    def test_same_method_zero_diff(self):
        cats = ["a", "b"]
        ref = _make_df([1, 2, 3, 4], np.array([[1, 0], [0, 1], [1, 1], [0, 0]]), cats)
        pred = _make_df([1, 2, 3, 4], np.array([[1, 0], [0, 1], [1, 0], [0, 1]]), cats)
        diff, lo, hi = paired_bootstrap_macro_f1(ref, pred, pred, cats, n_boot=100, seed=42)
        assert diff == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_better_method_positive_diff(self):
        cats = ["a", "b"]
        ref = _make_df([1, 2, 3, 4], np.array([[1, 0], [0, 1], [1, 1], [0, 0]]), cats)
        # Method A: perfect
        pred_a = ref.copy()
        # Method B: all zeros
        pred_b = _make_df([1, 2, 3, 4], np.zeros((4, 2), dtype=int), cats)
        diff, lo, hi = paired_bootstrap_macro_f1(ref, pred_a, pred_b, cats, n_boot=100, seed=42)
        assert diff > 0


# ---------------------------------------------------------------------------
# Classification table with probabilities
# ---------------------------------------------------------------------------
class TestClassificationWithProbs:
    def test_auroc_auprc_present(self):
        cats = ["x"]
        ref = _make_df([1, 2, 3, 4, 5, 6], np.array([[1], [1], [0], [0], [1], [0]]), cats)
        probs = np.array([[0.9], [0.8], [0.2], [0.1], [0.7], [0.3]])
        pred = _make_df([1, 2, 3, 4, 5, 6],
                        (probs > 0.5).astype(int), cats, probs=probs)
        table = classification_table(ref, pred, cats, n_boot=50, seed=42)
        row = table[table["category"] == "x"].iloc[0]
        assert "auroc" in row.index
        assert "auprc" in row.index
        assert 0 <= row["auroc"] <= 1
        assert 0 <= row["auprc"] <= 1
