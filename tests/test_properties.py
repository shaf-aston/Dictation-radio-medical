"""Rules that must hold for EVERY input, not just the examples we thought of.

Each test states a property and lets Hypothesis generate hundreds of inputs
looking for one that breaks it (and then shrinks it to the smallest such
input). The example tables elsewhere pin known cases; these pin the shape.
"""

import string
import sys
import types

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.dictation.postprocess import llm_cleanup
from src.dictation.postprocess.pipeline import postprocess_transcript
from src.dictation.postprocess.voice_commands import apply_scratch_commands
from src.features.template_fields import (
    find_fields,
    next_field,
    previous_field,
    resolve_choice,
)
from src.medical.deid import DeIdentifier, restore
from src.medical.spelling import apply_uk_spelling

# Report-like text: words, digits, punctuation, brackets and new lines.
_ALPHABET = string.ascii_letters + string.digits + " .,:;/-[]{}()\n"
report_text = st.text(alphabet=_ALPHABET, max_size=300)
words = st.text(alphabet=string.ascii_letters, min_size=1, max_size=12)
# Clinical sentences the pipeline really sees.
SENTENCES = [
    "no pneumothorax", "the medial meniscus is torn", "measures 12 mm",
    "scratch that", "full stop", "new paragraph", "comma", "the lesion characterized",
    "impression", "left knee", "bilateral pleural effusions", "at l4 l5",
]
dictation = st.lists(st.sampled_from(SENTENCES) | words, max_size=25).map(" ".join)
FAST = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# -- template fields ----------------------------------------------------------

@FAST
@given(report_text)
def test_fields_are_where_they_say_and_tab_visits_each_in_order(text):
    fields = find_fields(text)
    for f in fields:
        assert text[f.start:f.end] == f.text
        assert next_field(text, f.start) == f
        assert previous_field(text, f.end) == f
    assert [f.start for f in fields] == sorted(f.start for f in fields)


@FAST
@given(st.lists(words, min_size=2, max_size=5, unique_by=str.lower), st.data())
def test_saying_any_option_answers_the_pick_list(options, data):
    field = "[" + " / ".join(options) + "]"
    chosen = data.draw(st.sampled_from(options))
    spoken = data.draw(st.sampled_from([chosen, chosen.lower(), chosen.upper() + ".", f" {chosen}, "]))
    assert resolve_choice(field, spoken) == chosen


# -- privacy -----------------------------------------------------------------

identifier = st.text(alphabet=string.ascii_letters + string.digits + " -", min_size=2, max_size=20).filter(
    lambda s: len(s.strip()) >= 2
)


@FAST
@given(identifier, identifier, report_text, report_text)
def test_no_identifier_survives_outside_a_placeholder(name, pid, before, after):
    deid = DeIdentifier({"name": name, "id": pid})
    text = f"{before} {name} {after} {pid}"
    assert deid.validate_clean(deid.deidentify_text(text))
    safe, originals = deid.pseudonymise(text)
    assert deid.validate_clean(safe)
    assert restore(safe, originals) == text


@pytest.fixture
def echo_groq(monkeypatch):
    """A Groq that changes nothing: returns exactly the text it was sent."""
    class _Completions:
        def create(self, **kwargs):
            import json
            sent = kwargs["messages"][-1]["content"]
            body = json.dumps({"corrected_text": sent, "changes": []})
            message = types.SimpleNamespace(content=body)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _Groq:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=_Groq))
    monkeypatch.setattr(llm_cleanup, "is_enabled", lambda: True)
    monkeypatch.setattr(llm_cleanup, "get_groq_key", lambda: "test-key")


@settings(FAST, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(dictation, st.sampled_from(["", "Comparison: 03/04/2024. ", "Accession RX1234567. ", "Jane Doe. "]))
def test_cleanup_that_changes_nothing_gives_back_the_same_report(echo_groq, text, extra):
    """Scrubbing is for the trip out: what comes back must be the radiologist's report."""
    report = extra + text
    cleaned, _ = llm_cleanup.clean_with_llm(report, {"name": "Jane Doe", "id": "RX1234567"})
    assert cleaned == report


# -- text stages ---------------------------------------------------------------

@FAST
@given(report_text)
def test_uk_spelling_is_idempotent(text):
    once = apply_uk_spelling(text, "en-GB")
    assert apply_uk_spelling(once, "en-GB") == once


@FAST
@given(dictation)
def test_scratch_that_only_ever_removes_text(text):
    assert len(apply_scratch_commands(text)) <= len(text)


@FAST
@given(dictation)
def test_pipeline_never_loses_a_number(text):
    """A measurement dropped or invented is a clinical error, not a style one.

    ("scratch that" is excluded: deleting the last sentence is its job.)
    """
    import re
    text = text.replace("scratch that", "")
    out = postprocess_transcript(text, cleanup_level="soft")
    digits = lambda s: sorted(re.findall(r"\d+", s))  # noqa: E731
    assert digits(out) == digits(text)
