"""Tests for post-dictation edit tracking (the 'was dictation wrong?' signal).

Covers the word-level diff, the JSONL round-trip, the consent gate, and that
the noise filters (unchanged text, oversized rewrites) keep the log clean.
"""

from __future__ import annotations

import pytest

from src.features import edit_tracking as et


# ---------------------------------------------------------------------------
# diff_edits — the pure diff, no I/O
# ---------------------------------------------------------------------------

def test_single_word_replacement_is_captured() -> None:
    edits = et.diff_edits("the plural effusion", "the pleural effusion")
    assert len(edits) == 1
    e = edits[0]
    assert e["op"] == "replace"
    assert e["before"] == "plural"
    assert e["after"] == "pleural"
    assert "effusion" in e["after_context"]  # context retained


def test_multi_word_replacement_is_captured() -> None:
    edits = et.diff_edits("mild root tier cuff tear", "mild rotator cuff tear")
    assert any(e["before"] == "root tier" and e["after"] == "rotator" for e in edits)


def test_insertion_and_deletion() -> None:
    ins = et.diff_edits("no fracture", "no acute fracture")
    assert any(e["op"] == "insert" and e["after"] == "acute" for e in ins)
    dele = et.diff_edits("no acute fracture", "no fracture")
    assert any(e["op"] == "delete" and e["before"] == "acute" for e in dele)


def test_identical_text_yields_nothing() -> None:
    assert et.diff_edits("same text here", "same text here") == []


def test_oversized_rewrite_is_skipped() -> None:
    # A whole-template paste is not a dictation fix — must not pollute the log.
    before = "short"
    after = "an entirely different and very long replacement block " * 3
    assert et.diff_edits(before, after) == []


# ---------------------------------------------------------------------------
# record_session_edits / load_edits — the JSONL round-trip and gating
# ---------------------------------------------------------------------------

@pytest.fixture()
def _temp_log(tmp_path, monkeypatch) -> None:
    """Point the edit log at a temp file and force learning_enabled on."""
    log = tmp_path / "dictation_edits.jsonl"
    monkeypatch.setattr(et, "_enabled", lambda: True)
    import src.features.file_manager as fm
    monkeypatch.setattr(fm, "dictation_edits_path", lambda: log)
    return log


def test_record_and_load_round_trip(_temp_log) -> None:
    n = et.record_session_edits("the plural effusion", "the pleural effusion")
    assert n == 1
    rows = et.load_edits()
    assert rows[0]["before"] == "plural" and rows[0]["after"] == "pleural"
    assert "ts" in rows[0]


def test_no_change_records_nothing(_temp_log) -> None:
    assert et.record_session_edits("identical", "identical") == 0
    assert et.load_edits() == []


def test_disabled_records_nothing(tmp_path, monkeypatch) -> None:
    log = tmp_path / "dictation_edits.jsonl"
    monkeypatch.setattr(et, "_enabled", lambda: False)
    import src.features.file_manager as fm
    monkeypatch.setattr(fm, "dictation_edits_path", lambda: log)
    assert et.record_session_edits("plural", "pleural") == 0
    assert not log.exists()


def test_stats_and_reset(_temp_log) -> None:
    et.record_session_edits("no fracture seen", "no acute fracture noted")
    stats = et.edit_stats()
    assert stats["total"] >= 1
    et.reset_edits()
    assert et.load_edits() == []
