"""Tests for PHI de-identification — the safety-critical upload gate."""

from __future__ import annotations

import importlib
import sys

import pytest

from src.medical.deid import DeIdentifier, PrivacyError


@pytest.fixture
def real_soundfile():
    """Yield the real soundfile module, bypassing any test-harness stub.

    Pops the stub (if any) before the test and restores it after so that
    other tests that rely on the stub are unaffected.
    """
    stub = sys.modules.pop("soundfile", None)
    try:
        sf = importlib.import_module("soundfile")
    except Exception:
        if stub is not None:
            sys.modules["soundfile"] = stub
        pytest.skip("real soundfile not installed")
    if not hasattr(sf, "write"):
        if stub is not None:
            sys.modules["soundfile"] = stub
        pytest.skip("soundfile unavailable (stubbed)")
    yield sf
    if stub is not None:
        sys.modules["soundfile"] = stub
    else:
        sys.modules.pop("soundfile", None)


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


def test_deidentify_audio_silences_phi_preserves_findings(tmp_path, real_soundfile):
    """Segments whose transcript names the patient are silenced (±0.5s);
    other audio is untouched. Guards the acoustic side of the upload gate.
    """
    np = pytest.importorskip("numpy")
    sf = real_soundfile

    sr = 16000
    audio = (0.3 * np.sin(2 * np.pi * 220 * np.arange(4 * sr) / sr)).astype("float32")
    wav = tmp_path / "session.wav"
    sf.write(str(wav), audio, sr)

    deid = DeIdentifier({"name": "John Smith"})
    segments = [
        {"start": 0.5, "end": 1.5, "text": "patient John Smith referred"},  # PHI
        {"start": 2.5, "end": 3.5, "text": "wertebra body intact"},          # finding
    ]
    out = deid.deidentify_audio(str(wav), segments, tmp_path / "clips", "sess1")
    assert out is not None

    clip, csr = sf.read(out, dtype="float32")
    phi = clip[int(0.6 * csr):int(1.4 * csr)]          # inside silenced window
    finding = clip[int(2.6 * csr):int(3.4 * csr)]      # preserved
    assert float(np.abs(phi).max()) == 0.0
    assert float(np.abs(finding).max()) > 0.01
