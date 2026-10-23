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
    monkeypatch.setattr(DeIdentifier, "deidentify_text", lambda self, text: text)
    text = "Jane Doe: medial meniscal tear."
    assert llm_cleanup.clean_with_llm(text, PATIENT) == (text, [])
    assert groq_recorder == []
