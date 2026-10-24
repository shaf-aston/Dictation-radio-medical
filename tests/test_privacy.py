"""Patient identifiers never leave the device.

Two gates are pinned here: the de-identifier itself, and the one network path
that uses it in the dictation package (the optional Groq cleanup). The second
is tested by replacing the Groq client with a recorder, so the test can read
exactly what would have been sent.
"""

import sys
import types

import pytest

from src.dictation.postprocess import llm_cleanup
from src.medical.deid import DeIdentifier, PrivacyError

PATIENT = {"name": "Jane Doe", "id": "MRN12345678", "dob": "01/02/1960"}


@pytest.mark.parametrize(
    "text, gone",
    [
        ("Patient Jane Doe has a meniscal tear.", "Jane Doe"),
        ("jane doe, left knee.", "jane doe"),
        ("MRN12345678 reviewed.", "MRN12345678"),
        ("Born 01/02/1960.", "01/02/1960"),
        ("Compared with the study of 3/4/2024.", "3/4/2024"),
    ],
)
def test_identifier_is_scrubbed(text, gone):
    assert gone not in DeIdentifier(PATIENT).deidentify_text(text)


def test_clinical_words_survive_scrubbing():
    text = "Oblique tear of the posterior horn of the medial meniscus."
    assert DeIdentifier(PATIENT).deidentify_text(text) == text


def test_validate_clean_refuses_a_surviving_identifier():
    with pytest.raises(PrivacyError):
        DeIdentifier(PATIENT).validate_clean("Jane Doe")


@pytest.fixture
def groq_recorder(monkeypatch):
    """Turn cleanup on and capture what would be sent to Groq."""
    sent = []

    class _Completions:
        def create(self, **kwargs):
            sent.append(kwargs["messages"][-1]["content"])
            message = types.SimpleNamespace(content='{"corrected_text": "cleaned", "changes": []}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _Groq:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=_Groq))
    monkeypatch.setattr(llm_cleanup, "is_enabled", lambda: True)
    monkeypatch.setattr(llm_cleanup, "get_groq_key", lambda: "test-key")
    return sent


def test_cleanup_sends_scrubbed_text_only(groq_recorder):
    llm_cleanup.clean_with_llm("Jane Doe MRN12345678: medial meniscal tear.", PATIENT)
    (payload,) = groq_recorder
    assert "Jane Doe" not in payload
    assert "MRN12345678" not in payload
    assert "meniscal tear" in payload


def test_cleanup_is_off_by_default(monkeypatch, groq_recorder):
    monkeypatch.setattr(llm_cleanup, "is_enabled", lambda: False)
    assert llm_cleanup.clean_with_llm("Some text.", PATIENT) == ("Some text.", [])
    assert groq_recorder == []


def test_cleanup_returns_input_when_scrubbing_fails(monkeypatch, groq_recorder):
    # If an identifier somehow survives, nothing is sent and the report is kept.
    monkeypatch.setattr(DeIdentifier, "pseudonymise", lambda self, text: (text, {}))
    text = "Jane Doe: medial meniscal tear."
    assert llm_cleanup.clean_with_llm(text, PATIENT) == (text, [])
    assert groq_recorder == []


def test_a_patient_called_ted_is_not_mistaken_for_a_leak():
    # "[REDACTED]" contains "ted": the placeholder used to fail every check.
    deid = DeIdentifier({"name": "Ted"})
    assert deid.validate_clean(deid.deidentify_text("Ted has a fracture."))


def test_what_was_scrubbed_is_put_back_after_the_round_trip():
    from src.medical.deid import restore
    text = "Jane Doe, MRN12345678. Compared with 3/4/2024: stable."
    safe, originals = DeIdentifier(PATIENT).pseudonymise(text)
    assert "Jane" not in safe and "3/4/2024" not in safe
    assert restore(safe, originals) == text


@pytest.mark.parametrize("returned", ["Stable.", "[PHI-1] [PHI-1] stable.", "[PHI-1] [PHI-9] stable."])
def test_a_token_dropped_doubled_or_invented_refuses_the_cleanup(returned):
    from src.medical.deid import restore
    assert restore(returned, {"[PHI-1]": "3/4/2024"}) is None


@pytest.mark.parametrize(
    "text, gone",
    [
        ("Compared with the study of March 4, 2024.", "March 4, 2024"),
        ("Accession RX7654321 attached.", "RX7654321"),  # not the patient's own ID
        ("Seen by JD.", "JD"),  # two characters is long enough to be an identifier
    ],
)
def test_generic_identifiers_are_scrubbed(text, gone):
    assert gone not in DeIdentifier({**PATIENT, "name": "JD"}).deidentify_text(text)


def test_a_one_letter_value_is_not_treated_as_an_identifier():
    # Otherwise every "J" in a report would be redacted.
    assert DeIdentifier({"name": "J"}).deidentify_text("J-shaped tear.") == "J-shaped tear."


# -- audio: the training clip silences what was said around a name -----------

SR = 16000


def _tone(tmp_path, channels=1):
    import numpy as np
    import soundfile as sf
    tone = (0.5 * np.sin(np.linspace(0, 2 * np.pi * 440 * 3, 3 * SR))).astype("float32")
    audio = tone if channels == 1 else np.stack([tone, np.zeros_like(tone)], axis=1)
    path = tmp_path / "in.wav"
    sf.write(str(path), audio, SR)
    return str(path)


def _scrub(tmp_path, wav):
    import soundfile as sf
    segments = [
        {"start": 1.0, "end": 1.5, "text": "Jane Doe was seen"},
        {"start": 2.6, "end": 2.8, "text": "no effusion"},
    ]
    out = DeIdentifier(PATIENT).deidentify_audio(wav, segments, tmp_path / "a" / "b", "s1")
    assert out is not None and "s1" in out
    audio, _ = sf.read(out, dtype="float32")
    return audio


@pytest.mark.parametrize("channels", [1, 2])
def test_audio_around_a_spoken_name_is_silenced_and_nothing_else(tmp_path, channels):
    import numpy as np
    audio = _scrub(tmp_path, _tone(tmp_path, channels))
    assert audio.ndim == 1
    # The flagged segment plus half a second either side is silent...
    assert np.all(audio[int(0.5 * SR):int(2.0 * SR)] == 0.0)
    # ...and the speech outside it, including the clean segment, is untouched.
    assert np.abs(audio[:int(0.45 * SR)]).max() > 0.4
    assert np.abs(audio[int(2.05 * SR):]).max() > 0.4


def test_unreadable_audio_gives_no_clip(tmp_path):
    assert DeIdentifier(PATIENT).deidentify_audio(str(tmp_path / "missing.wav"), [], tmp_path, "s1") is None


@pytest.mark.parametrize("date", ["January 2, 2024", "September 14 2023", "Dec 1, 2022", "jul 4, 2021"])
def test_month_name_dates_are_scrubbed(date):
    assert date not in DeIdentifier(PATIENT).deidentify_text(f"Compared with {date}.")
