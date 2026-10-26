"""Stop hands the report back before the polish: and never takes it away again.

The desktop used to sit on the finished report until the confidence-targeted
polish returned (15+ seconds on a 30-second dictation). It now applies the
live text at once and lets the polish upgrade it, which creates the hazard
worth testing: the radiologist can be editing a clinical report while the
polish runs, and the polish must never overwrite what they wrote.

The rule (recording_session.session_may_write): from Stop on, the session
writes only while the editor holds exactly what the session last wrote.
"""

from types import SimpleNamespace

import pytest

from src.ui import recording_session as rs


class _Editor:
    def __init__(self, text=""):
        self._text = text

    def toPlainText(self):
        return self._text


def _window(text, monkeypatch, last_written=None):
    """A window stub after Stop, carrying only what on_processed_text touches."""
    applied, finished, statuses, enabled = [], [], [], []

    def write(w, processed):
        applied.append(processed)
        w.editor._text = processed
        w._session_last_written = processed

    monkeypatch.setattr(rs, "_replace_dictation_region", write)
    monkeypatch.setattr(rs, "_complete_finish", lambda w: finished.append(True))
    return SimpleNamespace(
        editor=_Editor(text),
        recorder=SimpleNamespace(is_recording=False),
        _applied_seq=0, _partial_seq=0, _final_seq=None, _handback_seq=None,
        _session_last_written=last_written, _session_yielded=False,
        _corrections_pending=[], _corrections_seen=set(),
        _active_accent="neutral", _active_cleanup_level="medium", _last_raw_transcript="",
        _status=SimpleNamespace(last_progress=None),
        _show_status=lambda *a, **k: statuses.append(a[0]),
        btn_record=SimpleNamespace(setEnabled=enabled.append),
        applied=applied, finished=finished, statuses=statuses, record_enabled=enabled,
    )


def test_handback_returns_the_report_before_the_polish(monkeypatch):
    window = _window("The lungs", monkeypatch, last_written="The lungs")
    window._handback_seq = 3
    rs.on_processed_text(window, "The lungs are clear.", [], 3)
    assert window.applied == ["The lungs are clear."]
    assert window.record_enabled == [True]      # can start the next dictation
    assert window.finished == []                # but the session is not closed


def test_polish_upgrades_text_the_radiologist_left_alone(monkeypatch):
    window = _window("The lungs are clear.", monkeypatch, last_written="The lungs are clear.")
    window._final_seq = 5
    rs.on_processed_text(window, "The lungs are clear bilaterally.", [], 5)
    assert window.applied == ["The lungs are clear bilaterally."]
    assert window.finished == [True]


def test_polish_never_overwrites_an_edited_report(monkeypatch):
    window = _window("The lungs are clear. No effusion.", monkeypatch,
                     last_written="The lungs are clear.")
    window._final_seq = 5
    rs.on_processed_text(window, "The lungs are clear bilaterally.", [], 5)
    assert window.applied == []                 # nothing was written over
    assert window.finished == [True]            # the session still closes
    assert "Kept your edits" in window.statuses[0]


def test_once_edited_the_session_never_writes_again_even_if_the_edit_is_undone(monkeypatch):
    window = _window("The lungs are clear. No effusion.", monkeypatch,
                     last_written="The lungs are clear.")
    window._final_seq = 9
    rs.on_processed_text(window, "polished text", [], 8)   # the polish's own update
    window.editor._text = "The lungs are clear."           # edit undone
    rs.on_processed_text(window, "polished text", [], 9)   # the final pass
    assert window.applied == []
    assert window.finished == [True]


def test_a_released_report_is_not_rewritten(monkeypatch):
    window = _window("The lungs are clear.", monkeypatch, last_written="The lungs are clear.")
    window.live_worker, window.pp_worker = object(), object()
    rs.yield_document(window)                               # copied / saved / exported
    window._final_seq = 5
    rs.on_processed_text(window, "The lungs are clear bilaterally.", [], 5)
    assert window.applied == []


def test_a_handback_refused_because_of_an_edit_still_gives_record_back(monkeypatch):
    window = _window("The lungs are. Edited", monkeypatch, last_written="The lungs are.")
    window._handback_seq = 4
    rs.on_processed_text(window, "The lungs are clear.", [], 4)
    assert window.applied == [] and window.record_enabled == [True]


def test_a_late_pass_can_never_rewind_the_report(monkeypatch):
    window = _window("", monkeypatch)
    window._applied_seq = 7
    rs.on_processed_text(window, "stale text", [], 6)
    assert window.applied == []


def test_a_decode_landing_after_stop_becomes_the_handback(monkeypatch):
    submitted = []
    window = _window("", monkeypatch)
    window.pp_worker = SimpleNamespace(submit=lambda *a: submitted.append(a))
    window._handback_seq = 1
    window._partial_seq = 1
    rs.on_partial_text(window, "The lungs are clear and", committed_len=12)
    (text, committed_len, *_rest, seq) = submitted[0]
    assert (committed_len, seq, window._handback_seq) == (0, 2, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
