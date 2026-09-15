"""Stop hands the report back before the polish: and never takes it away again.

The desktop used to sit on the finished report until the confidence-targeted
polish returned (15+ seconds on a 30-second dictation, measured in
``data/runs.jsonl``). It now applies the live text immediately and lets the
polish upgrade it afterwards, which creates the one hazard worth a test: the
radiologist can be editing a clinical report while the polish is still running,
and the polish must never overwrite what they wrote.
"""

from types import SimpleNamespace

import pytest

from src.ui import recording_session as rs


class _Editor:
    def __init__(self, text=""):
        self._text = text

    def toPlainText(self):
        return self._text


def _window(text, monkeypatch):
    """A window stub carrying only what on_processed_text actually touches."""
    applied = []
    monkeypatch.setattr(
        rs, "_replace_dictation_region", lambda w, processed: applied.append(processed)
    )
    monkeypatch.setattr(rs, "_complete_finish", lambda w: finished.append(True))
    finished = []
    window = SimpleNamespace(
        editor=_Editor(text),
        _applied_seq=0,
        _partial_seq=0,
        _final_seq=None,
        _handback_seq=None,
        _handed_over_text=None,
        _corrections_pending=[],
        _corrections_seen=set(),
        _status=SimpleNamespace(last_progress=None),
        btn_record=SimpleNamespace(enabled=False, setEnabled=lambda v: None),
        _show_status=lambda *a, **k: statuses.append(a[0]),
        applied=applied,
        finished=finished,
    )
    statuses = []
    window.statuses = statuses
    enabled = []
    window.btn_record = SimpleNamespace(setEnabled=enabled.append)
    window.record_enabled = enabled
    return window


def test_handback_returns_the_report_before_the_polish(monkeypatch):
    """The handback pass puts text on screen and gives Record back."""
    window = _window("", monkeypatch)
    window._handback_seq = 3

    rs.on_processed_text(window, "The lungs are clear.", [], 3)

    assert window.applied == ["The lungs are clear."]
    assert window.record_enabled == [True]      # can start the next dictation
    assert window.finished == []                # but the session is not closed
    assert window._handed_over_text == ""       # what the editor stub reports


def test_polish_upgrades_text_the_radiologist_left_alone(monkeypatch):
    """Untouched report: the more accurate decode replaces it."""
    window = _window("The lungs are clear.", monkeypatch)
    window._final_seq = 5
    window._handed_over_text = "The lungs are clear."

    rs.on_processed_text(window, "The lungs are clear bilaterally.", [], 5)

    assert window.applied == ["The lungs are clear bilaterally."]
    assert window.finished == [True]


def test_polish_never_overwrites_an_edited_report(monkeypatch):
    """Edited report: the radiologist's words win and the polish is dropped."""
    window = _window("The lungs are clear. No effusion.", monkeypatch)
    window._final_seq = 5
    window._handed_over_text = "The lungs are clear."

    rs.on_processed_text(window, "The lungs are clear bilaterally.", [], 5)

    assert window.applied == []                 # nothing was written over
    assert window.finished == [True]            # the session still closes
    assert "Kept your edits" in window.statuses[0]


def test_a_late_pass_can_never_rewind_the_report(monkeypatch):
    """The oldest hazard, still guarded: results only move forward."""
    window = _window("", monkeypatch)
    window._applied_seq = 7

    rs.on_processed_text(window, "stale text", [], 6)

    assert window.applied == []


def test_the_polish_own_update_cannot_slip_past_the_guard(monkeypatch):
    """The polish emits a partial before its final pass: guard that one too.

    The nearest case the request does not name, and the one that would have
    reached the editor first if only the final pass were checked.
    """
    window = _window("The lungs are clear. No effusion.", monkeypatch)
    window._final_seq = 9
    window._handed_over_text = "The lungs are clear."

    rs.on_processed_text(window, "polished text", [], 8)   # the polish's own update

    assert window.applied == []
    assert window.finished == []               # not the final pass: stay open
    assert window._handed_over_text is not None  # still guarded for what follows

    rs.on_processed_text(window, "polished text", [], 9)   # then the final pass
    assert window.applied == []
    assert window.finished == [True]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
