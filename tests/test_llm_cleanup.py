"""Tests for the optional Groq report-cleanup stage.

Guards the safety properties: it is off unless explicitly enabled + consented,
PHI is scrubbed before the request, and any failure returns the text unchanged
(cleanup must never lose the radiologist's report).
"""

from __future__ import annotations

import json
import sys
import types

from src.core.settings import Settings
from src.dictation.postprocess import llm_cleanup


def _enable(monkeypatch):
    s = Settings()
    s.set("groq_cleanup_enabled", True)
    s.set("cloud_training_consent", True)
    monkeypatch.setattr(llm_cleanup, "get_groq_key", lambda: "test-key")


def _install_fake_groq(monkeypatch, payload: dict, capture: dict) -> None:
    """Install a fake ``groq`` module whose client records the prompt sent."""
    module = types.ModuleType("groq")

    class _FakeGroq:
        def __init__(self, *a, **k) -> None:
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, **kwargs):
            capture["messages"] = kwargs["messages"]
            msg = types.SimpleNamespace(content=json.dumps(payload))
            choice = types.SimpleNamespace(message=msg)
            return types.SimpleNamespace(choices=[choice])

    module.Groq = _FakeGroq  # type: ignore
    monkeypatch.setitem(sys.modules, "groq", module)


def test_disabled_returns_unchanged(monkeypatch):
    # Default settings: feature off → no-op, no key needed.
    text = "the lungs are clear"
    assert llm_cleanup.clean_with_llm(text) == (text, [])


def test_enabled_applies_correction(monkeypatch):
    _enable(monkeypatch)
    capture: dict = {}
    _install_fake_groq(
        monkeypatch,
        {"corrected_text": "The lungs are clear.", "changes": ['"the" → "The"']},
        capture,
    )
    cleaned, changes = llm_cleanup.clean_with_llm("the lungs are clear")
    assert cleaned == "The lungs are clear."
    assert changes == ['"the" → "The"']


def test_phi_scrubbed_before_send(monkeypatch):
    _enable(monkeypatch)
    capture: dict = {}
    _install_fake_groq(monkeypatch, {"corrected_text": "ok"}, capture)
    llm_cleanup.clean_with_llm(
        "Patient John Smith MRN 12345678 has clear lungs",
        patient_info={"name": "John Smith", "id": "12345678"},
    )
    sent = capture["messages"][-1]["content"]
    assert "John Smith" not in sent
    assert "12345678" not in sent


def test_failure_returns_original(monkeypatch):
    _enable(monkeypatch)

    module = types.ModuleType("groq")

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("network down")

    module.Groq = _Boom  # type: ignore
    monkeypatch.setitem(sys.modules, "groq", module)

    text = "important findings here"
    assert llm_cleanup.clean_with_llm(text) == (text, [])


def test_empty_corrected_text_keeps_original(monkeypatch):
    _enable(monkeypatch)
    capture: dict = {}
    _install_fake_groq(monkeypatch, {"corrected_text": "   "}, capture)
    text = "keep me"
    assert llm_cleanup.clean_with_llm(text) == (text, [])
