"""Tests for src.rules_baseline on fake PMH texts."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.rules_baseline import (
    load_lexicon,
    compile_lexicon,
    split_segments,
    predict,
)


@pytest.fixture(scope="module")
def compiled():
    """Load and compile the real lexicon for testing."""
    lex = load_lexicon("config/charlson_lexicon.yaml")
    return compile_lexicon(lex)


# ---------------------------------------------------------------------------
# Segment splitting
# ---------------------------------------------------------------------------
class TestSplitSegments:
    def test_single_line_commas(self):
        segs = split_segments("HTN, DM2, COPD")
        assert len(segs) == 3
        assert "HTN" in segs
        assert "DM2" in segs

    def test_multiline(self):
        text = "- HTN\n- DM2\n- COPD"
        segs = split_segments(text)
        assert len(segs) == 3

    def test_numbered_list(self):
        text = "1. HTN\n2. DM2\n3. COPD"
        segs = split_segments(text)
        assert len(segs) == 3
        assert "HTN" in segs

    def test_empty(self):
        assert split_segments("") == []
        assert split_segments("   ") == []

    def test_semicolons(self):
        segs = split_segments("HTN; DM2; COPD")
        assert len(segs) == 3


# ---------------------------------------------------------------------------
# Simple detection
# ---------------------------------------------------------------------------
class TestSimpleDetection:
    def test_detect_chf(self, compiled):
        flags, terms = predict("CHF", compiled)
        assert flags["chf"] == 1

    def test_detect_full_name(self, compiled):
        flags, _ = predict("congestive heart failure", compiled)
        assert flags["chf"] == 1

    def test_detect_copd(self, compiled):
        flags, _ = predict("COPD", compiled)
        assert flags["cpd"] == 1

    def test_detect_dm2(self, compiled):
        flags, _ = predict("DM2", compiled)
        assert flags["diab"] == 1

    def test_detect_ckd(self, compiled):
        flags, _ = predict("CKD stage 4", compiled)
        assert flags["rend"] == 1

    def test_detect_multiple(self, compiled):
        flags, _ = predict("HTN, DM2, COPD, CHF", compiled)
        assert flags["diab"] == 1
        assert flags["cpd"] == 1
        assert flags["chf"] == 1

    def test_no_match(self, compiled):
        flags, _ = predict("HTN, hyperlipidemia", compiled)
        # HTN and HLD are not Charlson categories
        assert all(v == 0 for v in flags.values())


# ---------------------------------------------------------------------------
# Abbreviations
# ---------------------------------------------------------------------------
class TestAbbreviations:
    def test_mi_abbreviation(self, compiled):
        flags, _ = predict("s/p MI", compiled)
        assert flags["mi"] == 1

    def test_ra_abbreviation(self, compiled):
        flags, _ = predict("RA on methotrexate", compiled)
        assert flags["rheumd"] == 1

    def test_esrd(self, compiled):
        flags, _ = predict("ESRD on HD", compiled)
        assert flags["rend"] == 1

    def test_sle(self, compiled):
        flags, _ = predict("SLE", compiled)
        assert flags["rheumd"] == 1


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------
class TestNegation:
    def test_negated_no(self, compiled):
        flags, _ = predict("no history of MI", compiled)
        assert flags["mi"] == 0

    def test_negated_denies(self, compiled):
        flags, _ = predict("denies diabetes", compiled)
        assert flags["diab"] == 0

    def test_negated_negative_for(self, compiled):
        flags, _ = predict("negative for cancer", compiled)
        assert flags["canc"] == 0

    def test_negated_rule_out(self, compiled):
        flags, _ = predict("r/o MI", compiled)
        assert flags["mi"] == 0

    def test_negation_local_scope(self, compiled):
        """Negation on one line should not affect another."""
        text = "- no DM\n- COPD\n- CHF"
        flags, _ = predict(text, compiled)
        assert flags["diab"] == 0  # negated
        assert flags["cpd"] == 1   # not negated
        assert flags["chf"] == 1   # not negated

    def test_negation_local_comma_list(self, compiled):
        """In comma-separated list, negation scope is per item."""
        flags, _ = predict("no MI, DM2, COPD", compiled)
        assert flags["mi"] == 0    # negated
        assert flags["diab"] == 1  # separate item, not negated
        assert flags["cpd"] == 1


# ---------------------------------------------------------------------------
# Family history
# ---------------------------------------------------------------------------
class TestFamilyHistory:
    def test_family_history_excluded(self, compiled):
        flags, _ = predict("family history of cancer", compiled)
        assert flags["canc"] == 0

    def test_mother_excluded(self, compiled):
        flags, _ = predict("mother with DM2", compiled)
        assert flags["diab"] == 0


# ---------------------------------------------------------------------------
# s/p and h/o (should be POSITIVE)
# ---------------------------------------------------------------------------
class TestHistorical:
    def test_sp_positive(self, compiled):
        flags, _ = predict("s/p MI", compiled)
        assert flags["mi"] == 1

    def test_ho_positive(self, compiled):
        flags, _ = predict("h/o stroke", compiled)
        assert flags["cevd"] == 1


# ---------------------------------------------------------------------------
# Diabetes complicated vs uncomplicated
# ---------------------------------------------------------------------------
class TestDiabetesSeverity:
    def test_diab_uncomplicated(self, compiled):
        flags, _ = predict("DM2", compiled)
        assert flags["diab"] == 1
        assert flags["diabwc"] == 0

    def test_diabwc_direct(self, compiled):
        flags, _ = predict("diabetic nephropathy", compiled)
        assert flags["diabwc"] == 1

    def test_diabwc_cooccurrence(self, compiled):
        """DM + complication in same text → diabwc."""
        flags, _ = predict("DM2, neuropathy, retinopathy", compiled)
        assert flags["diab"] == 1
        assert flags["diabwc"] == 1

    def test_diabwc_complicated_by(self, compiled):
        flags, _ = predict("DM2 c/b nephropathy", compiled)
        assert flags["diabwc"] == 1


# ---------------------------------------------------------------------------
# Liver disease severity
# ---------------------------------------------------------------------------
class TestLiverSeverity:
    def test_mild_liver(self, compiled):
        flags, _ = predict("hepatitis C", compiled)
        assert flags["mld"] == 1
        assert flags["msld"] == 0

    def test_severe_liver_direct(self, compiled):
        flags, _ = predict("decompensated cirrhosis", compiled)
        assert flags["msld"] == 1

    def test_severe_liver_cooccurrence(self, compiled):
        """Cirrhosis + ascites → msld."""
        flags, _ = predict("cirrhosis, ascites, variceal bleeding", compiled)
        assert flags["mld"] == 1
        assert flags["msld"] == 1


# ---------------------------------------------------------------------------
# Metastatic cancer
# ---------------------------------------------------------------------------
class TestMetastaticCancer:
    def test_cancer_without_mets(self, compiled):
        flags, _ = predict("breast cancer", compiled)
        assert flags["canc"] == 1
        assert flags["metacanc"] == 0

    def test_metastatic_sets_both(self, compiled):
        flags, _ = predict("metastatic lung cancer", compiled)
        assert flags["canc"] == 1
        assert flags["metacanc"] == 1

    def test_stage_iv(self, compiled):
        flags, _ = predict("colon cancer stage IV", compiled)
        assert flags["canc"] == 1
        assert flags["metacanc"] == 1


# ---------------------------------------------------------------------------
# Bullet list with mixed negation
# ---------------------------------------------------------------------------
class TestMixedList:
    def test_bullet_list_mixed(self, compiled):
        text = (
            "- HTN\n"
            "- no history of MI\n"
            "- DM2\n"
            "- s/p CABG\n"
            "- COPD\n"
            "- family history of cancer"
        )
        flags, _ = predict(text, compiled)
        assert flags["mi"] == 0    # negated
        assert flags["diab"] == 1  # positive
        assert flags["cpd"] == 1   # positive
        assert flags["canc"] == 0  # family history
