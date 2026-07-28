"""The highlight-a-word lookup: what looks like it, and what goes with it.

Covers the two tiers separately (spelling from the SymSpell index, related from
the mined stems *and* the curated file), the rule that a term never appears in
both, and every way a selection can be junk — because this runs on a plain text
selection, which is the least trustworthy input in the app.
"""

from __future__ import annotations

import pytest

import src.medical.medical_dict as md
import src.medical.term_lookup as tl
from src.features.file_manager import radiology_lexicon_path

pytestmark = pytest.mark.skipif(
    not radiology_lexicon_path().exists(),
    reason="curated radiology lexicon not present",
)


def _symspell_over(terms):
    """A real SymSpell index over just the terms given (fast enough per test)."""
    symspellpy = pytest.importorskip("symspellpy")
    sym = symspellpy.SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for term in terms:
        sym.create_dictionary_entry(term, 1)
    return sym


@pytest.fixture(autouse=True)
def lexicon(monkeypatch: pytest.MonkeyPatch):
    """Real curated lexicon + a lexicon-sized spelling index.

    The shared conftest pins the dictionary caches empty, and the full 98k
    membership index takes seconds to build; this loads the one file the
    service is allowed to suggest from and indexes exactly that.
    """
    terms = md._load_lexicon_from_disk(radiology_lexicon_path())
    monkeypatch.setattr(md, "_CORRECTION_TARGETS", terms)
    monkeypatch.setattr(tl, "get_symspell", lambda: _symspell_over(terms))
    tl.reset()
    yield terms
    tl.reset()


def _terms(suggestions):
    return [item.term for item in suggestions]


# --- spelling tier ---------------------------------------------------------

def test_typo_finds_the_real_term():
    result = tl.lookup("pnemothorax")
    assert "pneumothorax" in _terms(result.similar_spelling)


def test_a_correctly_spelled_term_still_gets_neighbours():
    result = tl.lookup("pneumothorax")
    assert "pneumothorax" not in _terms(result.similar_spelling)
    assert result.related, "a term spelled correctly must still show its neighbourhood"
    assert "haemothorax" in _terms(result.related)


def test_only_lexicon_terms_are_offered(monkeypatch: pytest.MonkeyPatch):
    """A generic-wordlist word one edit away must never be suggested.

    The shared index spans ~98k generic medical terms; the same rule the fuzzy
    corrector follows applies here — only the curated lexicon is offered.
    """
    lexicon = md.get_correction_targets()
    monkeypatch.setattr(
        tl, "get_symspell", lambda: _symspell_over(list(lexicon) + ["effusio"]),
    )
    tl.reset()
    offered = _terms(tl.lookup("effusion").similar_spelling)
    assert "effusio" not in offered
    assert "perfusion" in offered


# --- related tier ----------------------------------------------------------

def test_stems_are_mined_from_the_lexicon():
    notes = {item.term: item.note for item in tl.lookup("atelectasis").related}
    assert "bronchiectasis" in notes
    assert notes["bronchiectasis"] == "-ectasis"


def test_curated_relations_reach_past_the_lexicon():
    """"chest drain" is in no wordlist — only a human could write that pair down."""
    suggestions = {item.term: item.note for item in tl.lookup("pneumothorax").related}
    assert "chest drain" in suggestions
    assert suggestions["chest drain"] == "pleural space"


def test_both_layers_appear_together():
    notes = {item.note for item in tl.lookup("pneumothorax").related}
    assert "pleural space" in notes          # curated
    assert any(note.endswith("-") or note.startswith("-") for note in notes)  # mined


def test_an_inflection_is_not_a_neighbour():
    """bronchus/bronchi share a stem and tell the radiologist nothing."""
    assert "bronchi" not in _terms(tl.lookup("bronchus").related)


# --- selections that are not one clean word --------------------------------

def test_multi_word_selection_uses_the_head_term():
    result = tl.lookup("small right pneumothorax")
    assert result.key == "pneumothorax"
    assert "haemothorax" in _terms(result.related)


def test_a_curated_phrase_is_looked_up_whole():
    """"chest drain" is a term in its own right; neither of its words is."""
    assert "pneumothorax" in _terms(tl.lookup("chest drain").related)


def test_case_and_punctuation_are_ignored():
    assert tl.lookup("  Pneumothorax.  ").key == "pneumothorax"


@pytest.mark.parametrize("junk", ["", "   ", "!!!", "1234", "zzqqxx", "x" * 500])
def test_junk_selections_answer_empty(junk):
    result = tl.lookup(junk)
    assert result.similar_spelling == []
    assert result.related == []


def test_the_query_is_never_echoed_back_unbounded():
    """An over-long selection must not become an over-long payload either."""
    limit = tl._index().tuning["max_query_chars"]
    assert len(tl.lookup("x" * 500).query) <= limit


# --- the two tiers never repeat each other ---------------------------------

@pytest.mark.parametrize(
    "query",
    ["pneumothorax", "effusion", "fracture", "meniscus", "osteoporosis", "pnemothorax"],
)
def test_no_term_appears_twice(query):
    result = tl.lookup(query)
    spelling = _terms(result.similar_spelling)
    related = _terms(result.related)
    assert len(set(spelling)) == len(spelling)
    assert len(set(related)) == len(related)
    assert not set(spelling) & set(related)
    assert query.lower() not in spelling + related


def test_a_term_in_two_curated_groups_is_offered_once():
    """"effusion" sits in several groups; a duplicate would be shown twice."""
    related = _terms(tl.lookup("effusion").related)
    assert len(related) == len(set(related))
    assert "pneumothorax" in related


# --- it must never break the editor ----------------------------------------

def test_an_internal_failure_returns_empty(monkeypatch: pytest.MonkeyPatch):
    def boom():
        raise RuntimeError("index is gone")

    monkeypatch.setattr(tl, "_index", boom)
    result = tl.lookup("pneumothorax")
    assert result.similar_spelling == [] and result.related == []


def test_a_missing_relations_file_still_answers(monkeypatch: pytest.MonkeyPatch):
    """Lose the curated file and the mined stems must carry on alone."""
    monkeypatch.setattr(
        tl, "related_terms_path", lambda: radiology_lexicon_path().with_name("nope.json"),
    )
    tl.reset()
    assert "bronchiectasis" in _terms(tl.lookup("atelectasis").related)


def test_results_are_capped():
    limits = tl._index().tuning
    result = tl.lookup("pneumothorax")
    assert len(result.similar_spelling) <= limits["max_similar"]
    assert len(result.related) <= limits["max_related"]
