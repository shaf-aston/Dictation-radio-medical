"""Tests for PHI de-identification — the safety-critical upload gate."""

from __future__ import annotations

import pytest

from src.cloud.exceptions import PrivacyError
from src.cloud.privacy import DeIdentifier


def test_redacts_patient_name():
    deid = DeIdentifier({"name": "John Smith"})
    out = deid.deidentify_text("Findings for John Smith show a tear.")
    assert "John Smith" not in out
    assert "[REDACTED]" in out


def test_redacts_dates_and_ids():
    deid = DeIdentifier({})
    out = deid.deidentify_text("Study on 12/03/2024 accession AB1234567 complete.")
    assert "12/03/2024" not in out
    assert "AB1234567" not in out
    assert "[DATE]" in out
    assert "[ID]" in out


def test_validate_clean_passes_when_clean():
    deid = DeIdentifier({"name": "Jane Doe"})
    assert deid.validate_clean("the rotator cuff is intact") is True


def test_validate_clean_raises_when_identifier_survives():
    deid = DeIdentifier({"name": "Jane Doe"})
    # A string the de-identifier did not process still contains the name.
    with pytest.raises(PrivacyError):
        deid.validate_clean("report for Jane Doe")


def test_short_identifier_ignored():
    # Two-char-or-less fields are not treated as identifiers (avoids over-redaction).
    deid = DeIdentifier({"id": "X"})
    out = deid.deidentify_text("X marks the lesion")
    assert out == "X marks the lesion"


def test_round_trip_text_then_validate():
    deid = DeIdentifier({"name": "Robert Brown", "accession": "9988776"})
    cleaned = deid.deidentify_text("Robert Brown, accession 9988776, ACL tear.")
    assert deid.validate_clean(cleaned) is True
