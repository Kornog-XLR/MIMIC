"""Tests for prepare_gold and annotation_io."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from src.annotation_io import (
    CATEGORIES,
    ANNOTATION_COLUMNS,
    load_annotations,
    save_annotation,
    find_resume_index,
)
from src.prepare_gold import prepare_gold


class TestPrepareGold:
    @pytest.fixture()
    def fake_data(self, tmp_path):
        splits = pd.DataFrame({
            "hadm_id": [1, 2, 3, 4, 5],
            "subject_id": [10, 20, 30, 40, 50],
            "split": ["gold", "gold", "gold", "train", "test"],
            "sampling": ["random", "random", "enriched", "random", "random"],
        })
        pmh = pd.DataFrame({
            "hadm_id": [1, 2, 3, 4, 5],
            "pmh_text": ["HTN, DM2", "COPD, asthma", "None", "Cancer", "CHF"],
        })
        splits_path = str(tmp_path / "splits.parquet")
        pmh_path = str(tmp_path / "pmh.parquet")
        splits.to_parquet(splits_path, index=False)
        pmh.to_parquet(pmh_path, index=False)
        return splits_path, pmh_path, tmp_path

    def test_only_gold_stays(self, fake_data):
        splits_path, pmh_path, tmp_path = fake_data
        out = str(tmp_path / "gold.parquet")
        result = prepare_gold(splits_path, pmh_path, out, seed=42)
        assert set(result["hadm_id"]) == {1, 2, 3}

    def test_no_labels_columns(self, fake_data):
        splits_path, pmh_path, tmp_path = fake_data
        out = str(tmp_path / "gold.parquet")
        result = prepare_gold(splits_path, pmh_path, out, seed=42)
        assert "sampling" not in result.columns
        assert "split" not in result.columns
        for cat in CATEGORIES:
            assert cat not in result.columns

    def test_has_ordre_column(self, fake_data):
        splits_path, pmh_path, tmp_path = fake_data
        out = str(tmp_path / "gold.parquet")
        result = prepare_gold(splits_path, pmh_path, out, seed=42)
        assert list(result["ordre"]) == [1, 2, 3]

    def test_reproducibility(self, fake_data):
        splits_path, pmh_path, tmp_path = fake_data
        out1 = str(tmp_path / "gold1.parquet")
        out2 = str(tmp_path / "gold2.parquet")
        r1 = prepare_gold(splits_path, pmh_path, out1, seed=42)
        r2 = prepare_gold(splits_path, pmh_path, out2, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_output_columns(self, fake_data):
        splits_path, pmh_path, tmp_path = fake_data
        out = str(tmp_path / "gold.parquet")
        result = prepare_gold(splits_path, pmh_path, out, seed=42)
        assert list(result.columns) == ["hadm_id", "ordre", "pmh_text"]


class TestAnnotationIO:
    def test_load_empty(self, tmp_path):
        path = str(tmp_path / "annotations.csv")
        df = load_annotations(path)
        assert len(df) == 0
        assert list(df.columns) == ANNOTATION_COLUMNS

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "annotations.csv")
        flags = {cat: 0 for cat in CATEGORIES}
        flags["mi"] = 1
        flags["chf"] = 1

        save_annotation(path, hadm_id=1001, flags=flags, incertain=False, commentaire="test")

        df = load_annotations(path)
        assert len(df) == 1
        assert df.iloc[0]["hadm_id"] == 1001
        assert df.iloc[0]["mi"] == 1
        assert df.iloc[0]["chf"] == 1
        assert df.iloc[0]["diab"] == 0
        assert df.iloc[0]["incertain"] == 0

    def test_update_existing(self, tmp_path):
        path = str(tmp_path / "annotations.csv")
        flags = {cat: 0 for cat in CATEGORIES}

        save_annotation(path, hadm_id=1001, flags=flags, incertain=False, commentaire="first")
        flags["mi"] = 1
        save_annotation(path, hadm_id=1001, flags=flags, incertain=True, commentaire="corrected")

        df = load_annotations(path)
        assert len(df) == 1  # No duplicate
        assert df.iloc[0]["mi"] == 1
        assert df.iloc[0]["incertain"] == 1
        assert df.iloc[0]["commentaire"] == "corrected"

    def test_multiple_annotations(self, tmp_path):
        path = str(tmp_path / "annotations.csv")
        flags = {cat: 0 for cat in CATEGORIES}

        save_annotation(path, hadm_id=1001, flags=flags, incertain=False, commentaire="")
        save_annotation(path, hadm_id=1002, flags=flags, incertain=False, commentaire="")
        save_annotation(path, hadm_id=1003, flags=flags, incertain=False, commentaire="")

        df = load_annotations(path)
        assert len(df) == 3


class TestFindResumeIndex:
    def test_no_annotations(self):
        gold = pd.DataFrame({"hadm_id": [1, 2, 3]})
        annotations = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        assert find_resume_index(annotations, gold) == 0

    def test_partial_annotations(self):
        gold = pd.DataFrame({"hadm_id": [1, 2, 3]})
        annotations = pd.DataFrame({"hadm_id": [1]})
        assert find_resume_index(annotations, gold) == 1

    def test_all_done(self):
        gold = pd.DataFrame({"hadm_id": [1, 2, 3]})
        annotations = pd.DataFrame({"hadm_id": [1, 2, 3]})
        assert find_resume_index(annotations, gold) == 3

    def test_gap_in_middle(self):
        gold = pd.DataFrame({"hadm_id": [1, 2, 3, 4]})
        annotations = pd.DataFrame({"hadm_id": [1, 3]})  # 2 skipped
        assert find_resume_index(annotations, gold) == 1  # Returns first unannotated
