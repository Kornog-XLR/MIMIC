"""Tests for src.extract_pmh.extract_pmh."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.extract_pmh import extract_pmh
from tests.fixtures.fake_notes import (
    NOTE_STANDARD,
    NOTE_PMH_SHORT_HEADER,
    NOTE_MED_SURG,
    NOTE_NO_PMH,
    NOTE_PMH_LAST,
    NOTE_NUMBERED_LIST,
    NOTE_PMH_NONE,
    NOTE_PMH_INLINE,
    NOTE_PMH_EMPTY,
    NOTE_MED_AND_SURG,
    NOTE_MEDICAL_HISTORY,
)


class TestExtractPmh:
    def test_standard_format(self):
        result = extract_pmh(NOTE_STANDARD)
        assert result is not None
        assert "Hypertension" in result
        assert "Type 2 diabetes" in result
        assert "Hyperlipidemia" in result
        # Should not contain the next section
        assert "Lives alone" not in result

    def test_pmh_short_header(self):
        result = extract_pmh(NOTE_PMH_SHORT_HEADER)
        assert result is not None
        assert "COPD" in result
        assert "Atrial fibrillation" in result
        assert "Mother" not in result

    def test_medical_surgical_history(self):
        result = extract_pmh(NOTE_MED_SURG)
        assert result is not None
        assert "Appendectomy" in result
        assert "Lisinopril" not in result

    def test_absent_section(self):
        result = extract_pmh(NOTE_NO_PMH)
        assert result is None

    def test_pmh_last_section(self):
        result = extract_pmh(NOTE_PMH_LAST)
        assert result is not None
        assert "Asthma" in result
        assert "Depression" in result

    def test_numbered_list(self):
        result = extract_pmh(NOTE_NUMBERED_LIST)
        assert result is not None
        assert "Coronary artery disease" in result
        assert "Chronic kidney disease" in result
        assert "Vitals" not in result

    def test_pmh_none_value(self):
        result = extract_pmh(NOTE_PMH_NONE)
        assert result is not None
        assert result.strip().lower() == "none"

    def test_pmh_inline(self):
        result = extract_pmh(NOTE_PMH_INLINE)
        assert result is not None
        assert "HTN" in result
        assert "DM2" in result
        assert "Non-smoker" not in result

    def test_pmh_empty_between_headers(self):
        result = extract_pmh(NOTE_PMH_EMPTY)
        assert result is None

    def test_medical_and_surgical(self):
        result = extract_pmh(NOTE_MED_AND_SURG)
        assert result is not None
        assert "Diverticulosis" in result
        assert "observation" not in result

    def test_medical_history_header(self):
        result = extract_pmh(NOTE_MEDICAL_HISTORY)
        assert result is not None
        assert "Migraines" in result
        assert "Negative" not in result

    def test_none_input(self):
        assert extract_pmh(None) is None

    def test_empty_string(self):
        assert extract_pmh("") is None
