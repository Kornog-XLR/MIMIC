"""Tests for Charlson mapping and label building."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from src.charlson_mapping import CATEGORIES, match_code, charlson_score
from src.labels_icd import build_labels, compute_age_at_admission


# ---------------------------------------------------------------------------
# Test match_code: at least one ICD-9 and one ICD-10 per category
# ---------------------------------------------------------------------------
class TestMatchCodeICD9:
    """One ICD-9-CM code per Charlson category."""

    def test_mi(self):
        assert "mi" in match_code("4100", 9)

    def test_chf(self):
        assert "chf" in match_code("4280", 9)

    def test_pvd(self):
        assert "pvd" in match_code("4400", 9)

    def test_cevd(self):
        assert "cevd" in match_code("4310", 9)

    def test_dementia(self):
        assert "dementia" in match_code("2900", 9)

    def test_cpd(self):
        assert "cpd" in match_code("4910", 9)

    def test_rheumd(self):
        assert "rheumd" in match_code("7100", 9)

    def test_pud(self):
        assert "pud" in match_code("5310", 9)

    def test_mld(self):
        assert "mld" in match_code("5710", 9)

    def test_diab(self):
        assert "diab" in match_code("25000", 9)

    def test_diabwc(self):
        assert "diabwc" in match_code("25040", 9)

    def test_hp(self):
        assert "hp" in match_code("3420", 9)

    def test_rend(self):
        assert "rend" in match_code("5851", 9)

    def test_canc(self):
        assert "canc" in match_code("1500", 9)

    def test_msld(self):
        assert "msld" in match_code("4560", 9)

    def test_metacanc(self):
        assert "metacanc" in match_code("1970", 9)

    def test_aids(self):
        assert "aids" in match_code("0420", 9)


class TestMatchCodeICD10:
    """One ICD-10 code per Charlson category."""

    def test_mi(self):
        assert "mi" in match_code("I2101", 10)

    def test_chf(self):
        assert "chf" in match_code("I509", 10)

    def test_pvd(self):
        assert "pvd" in match_code("I702", 10)

    def test_cevd(self):
        assert "cevd" in match_code("I630", 10)

    def test_dementia(self):
        assert "dementia" in match_code("F011", 10)

    def test_cpd(self):
        assert "cpd" in match_code("J449", 10)

    def test_rheumd(self):
        assert "rheumd" in match_code("M059", 10)

    def test_pud(self):
        assert "pud" in match_code("K259", 10)

    def test_mld(self):
        assert "mld" in match_code("K703", 10)

    def test_diab(self):
        assert "diab" in match_code("E119", 10)

    def test_diabwc(self):
        assert "diabwc" in match_code("E112", 10)

    def test_hp(self):
        assert "hp" in match_code("G819", 10)

    def test_rend(self):
        assert "rend" in match_code("N189", 10)

    def test_canc(self):
        assert "canc" in match_code("C189", 10)

    def test_msld(self):
        assert "msld" in match_code("K721", 10)

    def test_metacanc(self):
        assert "metacanc" in match_code("C780", 10)

    def test_aids(self):
        assert "aids" in match_code("B20", 10)


class TestMatchCodeICD10CM:
    """ICD-10-CM additions for MIMIC-IV."""

    def test_diab_e08(self):
        assert "diab" in match_code("E0800", 10)

    def test_diab_e09(self):
        assert "diab" in match_code("E0965", 10)

    def test_diabwc_e08(self):
        assert "diabwc" in match_code("E0822", 10)

    def test_diabwc_e09(self):
        assert "diabwc" in match_code("E0940", 10)

    def test_canc_c4a(self):
        assert "canc" in match_code("C4A0", 10)

    def test_canc_c7a(self):
        assert "canc" in match_code("C7A01", 10)

    def test_metacanc_c7b(self):
        assert "metacanc" in match_code("C7B00", 10)

    def test_rend_i132(self):
        assert "rend" in match_code("I132", 10)


class TestMatchCodeEdgeCases:
    def test_no_match(self):
        assert match_code("Z000", 10) == []

    def test_unknown_code(self):
        assert match_code("XXXXX", 9) == []


# ---------------------------------------------------------------------------
# Test charlson_score with hierarchy
# ---------------------------------------------------------------------------
class TestCharlsonScore:
    def test_zero_score(self):
        flags = {cat: 0 for cat in CATEGORIES}
        s, sa = charlson_score(flags)
        assert s == 0
        assert sa is None

    def test_single_category(self):
        flags = {cat: 0 for cat in CATEGORIES}
        flags["mi"] = 1  # weight 1
        s, _ = charlson_score(flags)
        assert s == 1

    def test_weighted_category(self):
        flags = {cat: 0 for cat in CATEGORIES}
        flags["metacanc"] = 1  # weight 6
        s, _ = charlson_score(flags)
        assert s == 6

    def test_hierarchy_diab(self):
        """diabwc (weight 2) suppresses diab (weight 1)."""
        flags = {cat: 0 for cat in CATEGORIES}
        flags["diab"] = 1
        flags["diabwc"] = 1
        s, _ = charlson_score(flags)
        assert s == 2  # only diabwc counted, not diab+diabwc=3

    def test_hierarchy_liver(self):
        """msld (weight 3) suppresses mld (weight 1)."""
        flags = {cat: 0 for cat in CATEGORIES}
        flags["mld"] = 1
        flags["msld"] = 1
        s, _ = charlson_score(flags)
        assert s == 3

    def test_hierarchy_cancer(self):
        """metacanc (weight 6) suppresses canc (weight 2)."""
        flags = {cat: 0 for cat in CATEGORIES}
        flags["canc"] = 1
        flags["metacanc"] = 1
        s, _ = charlson_score(flags)
        assert s == 6

    def test_no_hierarchy_when_dominant_absent(self):
        """diab alone should NOT be suppressed."""
        flags = {cat: 0 for cat in CATEGORIES}
        flags["diab"] = 1
        s, _ = charlson_score(flags)
        assert s == 1

    def test_age_adjustment_under_50(self):
        flags = {cat: 0 for cat in CATEGORIES}
        flags["mi"] = 1
        s, sa = charlson_score(flags, age=45)
        assert s == 1
        assert sa == 1  # no age points under 50

    def test_age_adjustment_60(self):
        flags = {cat: 0 for cat in CATEGORIES}
        flags["mi"] = 1
        s, sa = charlson_score(flags, age=60)
        assert s == 1
        assert sa == 3  # 1 + 2 age points (decade 50-59=1, 60-69=2 → (60-40)//10=2)

    def test_age_adjustment_90(self):
        flags = {cat: 0 for cat in CATEGORIES}
        s, sa = charlson_score(flags, age=90)
        assert s == 0
        assert sa == 4  # capped at 4


# ---------------------------------------------------------------------------
# Test build_labels with fixture data
# ---------------------------------------------------------------------------
class TestBuildLabels:
    @pytest.fixture()
    def fixture_diag(self):
        """Fake diagnoses: 3 hadm_ids, mix of ICD-9 and ICD-10."""
        return pd.DataFrame({
            "hadm_id": [
                # hadm 1001: MI (ICD-9) + CHF (ICD-10) — mixed stay
                1001, 1001,
                # hadm 1002: diabetes with complications ICD-10
                1002,
                # hadm 1003: cancer ICD-9 + metastatic ICD-10
                1003, 1003,
            ],
            "icd_code": [
                "4100", "I509",
                "E112",
                "1500", "C780",
            ],
            "icd_version": [
                9, 10,
                10,
                9, 10,
            ],
        })

    def test_mixed_stay(self, fixture_diag):
        labels = build_labels(fixture_diag, {1001, 1002, 1003})
        row = labels[labels["hadm_id"] == 1001].iloc[0]
        assert row["mi"] == 1
        assert row["chf"] == 1
        assert row["diab"] == 0

    def test_diabwc_only(self, fixture_diag):
        labels = build_labels(fixture_diag, {1001, 1002, 1003})
        row = labels[labels["hadm_id"] == 1002].iloc[0]
        assert row["diabwc"] == 1
        assert row["diab"] == 0  # raw flags, no hierarchy here

    def test_cancer_and_meta(self, fixture_diag):
        labels = build_labels(fixture_diag, {1001, 1002, 1003})
        row = labels[labels["hadm_id"] == 1003].iloc[0]
        assert row["canc"] == 1
        assert row["metacanc"] == 1  # raw flags, both present

    def test_hadm_without_codes(self, fixture_diag):
        """hadm 9999 has no ICD codes — should have all zeros."""
        labels = build_labels(fixture_diag, {1001, 9999})
        row = labels[labels["hadm_id"] == 9999].iloc[0]
        for cat in CATEGORIES:
            assert row[cat] == 0


class TestAgeAtAdmission:
    """Test age computation: anchor_age + (year(admittime) - anchor_year)."""

    def test_two_stays_different_years(self):
        """Patient 1 admitted in 2150 and 2153, anchor_age=60, anchor_year=2150."""
        labels = pd.DataFrame({
            "hadm_id": [5001, 5002],
            "subject_id": [1, 1],
        })
        patients = pd.DataFrame({
            "subject_id": [1],
            "anchor_age": [60],
            "anchor_year": [2150],
        })
        admissions = pd.DataFrame({
            "hadm_id": [5001, 5002],
            "admittime": ["2150-03-15 10:00:00", "2153-07-20 14:00:00"],
        })
        ages = compute_age_at_admission(labels, patients, admissions)
        assert ages.iloc[0] == 60  # 60 + (2150 - 2150)
        assert ages.iloc[1] == 63  # 60 + (2153 - 2150)

    def test_multiple_patients(self):
        """Two patients with different anchor years."""
        labels = pd.DataFrame({
            "hadm_id": [6001, 6002],
            "subject_id": [10, 20],
        })
        patients = pd.DataFrame({
            "subject_id": [10, 20],
            "anchor_age": [45, 70],
            "anchor_year": [2140, 2160],
        })
        admissions = pd.DataFrame({
            "hadm_id": [6001, 6002],
            "admittime": ["2145-01-01 00:00:00", "2162-06-15 00:00:00"],
        })
        ages = compute_age_at_admission(labels, patients, admissions)
        assert ages.iloc[0] == 50  # 45 + (2145 - 2140)
        assert ages.iloc[1] == 72  # 70 + (2162 - 2160)
