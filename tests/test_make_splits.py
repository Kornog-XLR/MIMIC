"""Tests for src.make_splits."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.charlson_mapping import CATEGORIES
from src.make_splits import build_pool, make_splits


def _make_fake_data(n_patients=400, seed=123):
    """Generate fake PMH + labels data for testing.

    Creates ~500 stays across n_patients patients.
    Some patients have 2 stays.
    """
    rng = np.random.default_rng(seed)

    # Generate patients, some with 2 stays
    subject_ids = []
    hadm_ids = []
    note_ids = []
    hid = 100000
    nid = 200000
    for sid in range(1, n_patients + 1):
        n_stays = 1 if rng.random() < 0.7 else 2
        for _ in range(n_stays):
            subject_ids.append(sid)
            hadm_ids.append(hid)
            note_ids.append(nid)
            hid += 1
            nid += 1

    n = len(hadm_ids)

    # PMH data
    pmh = pd.DataFrame({
        "note_id": note_ids,
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "pmh_text": ["some text"] * n,
        "pmh_found": [True] * n,
        "n_chars": rng.integers(10, 2000, size=n),
    })
    # Make a few pmh_not_found
    pmh.loc[0:4, "pmh_found"] = False
    pmh.loc[0:4, "n_chars"] = 0
    # Make a few very long
    pmh.loc[5:7, "n_chars"] = 5000

    # Labels
    labels_data = {
        "hadm_id": hadm_ids,
        "subject_id": subject_ids,
        "has_icd": [True] * n,
    }
    for cat in CATEGORIES:
        # Low prevalence by default
        labels_data[cat] = rng.choice([0, 1], size=n, p=[0.95, 0.05])

    # Make aids very rare
    labels_data["aids"] = np.zeros(n, dtype=int)
    labels_data["aids"][rng.choice(n, size=3, replace=False)] = 1

    # Make hp rare
    labels_data["hp"] = np.zeros(n, dtype=int)
    labels_data["hp"][rng.choice(n, size=5, replace=False)] = 1

    labels_data["charlson_icd"] = rng.integers(0, 10, size=n)
    labels_data["charlson_icd_age"] = rng.integers(0, 15, size=n)

    # A few without ICD
    labels_data["has_icd"][0] = False
    labels_data["has_icd"][1] = False

    labels = pd.DataFrame(labels_data)

    return pmh, labels


class TestBuildPool:
    def test_excludes_pmh_not_found(self):
        pmh, labels = _make_fake_data()
        pool, excl = build_pool(pmh, labels, max_chars=3000)
        assert excl["pmh_not_found_or_empty"] > 0
        assert pool["pmh_found"].all()

    def test_excludes_no_icd(self):
        pmh, labels = _make_fake_data()
        pool, excl = build_pool(pmh, labels, max_chars=3000)
        assert pool["has_icd"].all()

    def test_excludes_too_long(self):
        pmh, labels = _make_fake_data()
        pool, excl = build_pool(pmh, labels, max_chars=3000)
        assert excl["too_long"] > 0
        assert (pool["n_chars"] <= 3000).all()

    def test_no_duplicate_hadm(self):
        pmh, labels = _make_fake_data()
        pool, _ = build_pool(pmh, labels, max_chars=3000)
        assert pool["hadm_id"].is_unique


class TestMakeSplits:
    @pytest.fixture()
    def pool_and_splits(self):
        pmh, labels = _make_fake_data(n_patients=400, seed=123)
        pool, _ = build_pool(pmh, labels, max_chars=3000)
        result = make_splits(
            pool=pool,
            n_gold=20,
            gold_enrich=5,
            n_test=50,
            n_val=50,
            n_train=200,
            max_per_subject=2,
            min_pos=5,
            seed=42,
        )
        return pool, result

    def test_no_patient_overlap(self, pool_and_splits):
        _, result = pool_and_splits
        splits = ["gold", "test", "val", "train"]
        for i, s1 in enumerate(splits):
            for s2 in splits[i + 1:]:
                subj1 = set(result.loc[result["split"] == s1, "subject_id"])
                subj2 = set(result.loc[result["split"] == s2, "subject_id"])
                assert len(subj1 & subj2) == 0, f"Overlap between {s1} and {s2}"

    def test_no_duplicate_hadm(self, pool_and_splits):
        _, result = pool_and_splits
        assert result["hadm_id"].is_unique

    def test_split_sizes(self, pool_and_splits):
        _, result = pool_and_splits
        counts = result["split"].value_counts()
        assert counts.get("gold", 0) > 0
        assert counts.get("test", 0) > 0
        assert counts.get("val", 0) > 0
        assert counts.get("train", 0) > 0

    def test_gold_has_enriched(self, pool_and_splits):
        _, result = pool_and_splits
        gold = result[result["split"] == "gold"]
        assert (gold["sampling"] == "enriched").any()

    def test_expected_columns(self, pool_and_splits):
        _, result = pool_and_splits
        assert set(result.columns) == {"hadm_id", "subject_id", "split", "sampling"}

    def test_reproducibility(self):
        pmh, labels = _make_fake_data(n_patients=400, seed=123)
        pool, _ = build_pool(pmh, labels, max_chars=3000)

        kwargs = dict(
            pool=pool, n_gold=20, gold_enrich=5,
            n_test=50, n_val=50, n_train=200,
            max_per_subject=2, min_pos=5, seed=42,
        )
        r1 = make_splits(**kwargs)
        r2 = make_splits(**kwargs)
        pd.testing.assert_frame_equal(r1.sort_values("hadm_id").reset_index(drop=True),
                                       r2.sort_values("hadm_id").reset_index(drop=True))

    def test_train_enrichment(self):
        """A very rare category should get enriched in train."""
        pmh, labels = _make_fake_data(n_patients=400, seed=123)
        pool, _ = build_pool(pmh, labels, max_chars=3000)

        result = make_splits(
            pool=pool,
            n_gold=10, gold_enrich=0,
            n_test=20, n_val=20, n_train=100,
            max_per_subject=2,
            min_pos=50,  # Very high threshold to force enrichment
            seed=42,
        )
        train = result[result["split"] == "train"]
        assert (train["sampling"] == "enriched").any()
