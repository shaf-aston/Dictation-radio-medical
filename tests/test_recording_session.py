"""The dictation region's boundary, and the order of the post-stop shutdown.

Two things are being protected here.

*Where the dictated region starts.* It used to be an integer frozen when
recording began, so anything inserted above it — a template, a typed patient
field — left it pointing into the middle of somebody else's text, and the next
live cycle overwrote from there. It is now a cursor Qt keeps in place; this
covers the part that is not Qt's job, the clamp.

*What runs after Stop.* The final whole-document pass now runs on the
post-process thread instead of freezing the UI, which means everything that
depends on the finished report — the optional AI cleanup, the edit-tracking
snapshot, thread teardown — has to wait for it to come back rather than
running straight after Stop.
"""

from __future__ import annotations

from runtime_stubs import install_test_runtime_stubs, qt_widgets_stub

install_test_runtime_stubs()

with qt_widgets_stub():
    import src.ui.recording_session as rs  # noqa: E402

from src.ui.status import READY, StatusTrack  # noqa: E402


class _FakeCursor:
    def __init__(self, position: int) -> None:
        self._position = position

    def position(self) -> int:
        return self._position


class _FakeButton:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, value: bool) -> None:
        self.enabled = value


class _FakeEditor:
    def __init__(self, text: str) -> None:
        self.text = text

    def toPlainText(self) -> str:
        return self.text


class _FakePostprocessWorker:
    def __init__(self) -> None:
        self.jobs: list = []

    def submit(self, text, committed_len, accent, cleanup_level, seq) -> None:
        self.jobs.append((text, committed_len, accent, cleanup_level, seq))


class _FakeWindow:
    def __init__(self, raw: str = "raw dictated words") -> None:
        self.btn_record = _FakeButton()
        self.btn_record.enabled = False  # state left by on_stop_recording
        self.btn_stop = _FakeButton()
        self.editor = _FakeEditor("Findings: raw dictated words")
        self.pp_worker = _FakePostprocessWorker()
        self.pp_thread = None
        self._partial_seq = 3
        self._applied_seq = 3
        self._final_seq = None
        self._last_raw_transcript = raw
        self._active_accent = "neutral"
        self._active_cleanup_level = "medium"
        self._corrections_pending: list = []
        self._corrections_seen: set = set()
        self._post_dictation_snapshot = None
        self._dictation_start = None
        self.current_wav_path = None
        self.statuses: list = []
        self._status = StatusTrack()

    def _show_status(self, message: str, timeout: int = 0, state: str = "idle") -> None:
        self.statuses.append(message)


def _finish_window(monkeypatch, raw: str = "raw dictated words"):
    """A window at end-of-recording, with the steps that touch Qt recorded."""
    order: list = []
    monkeypatch.setattr(
        rs, "_replace_dictation_region",
        lambda window, processed: order.append(f"apply:{processed}"),
    )
    monkeypatch.setattr(
        rs, "_apply_finished_ai_cleanup", lambda window: order.append("ai_cleanup")
    )
    monkeypatch.setattr(rs, "_prepare_training_audio", lambda: order.append("deid"))
    return _FakeWindow(raw), order


class TestDictationStartOffset:
    def test_reads_the_live_position_of_the_anchor(self) -> None:
        window = _FakeWindow()
        window._dictation_start = _FakeCursor(12)

        assert rs.dictation_start_offset(window, 40) == 12

    def test_clamps_to_a_document_that_shrank_under_it(self) -> None:
        # Open / New / Clear replace the document wholesale; an offset past the
        # end is not a position anything can be inserted at.
        window = _FakeWindow()
        window._dictation_start = _FakeCursor(500)

        assert rs.dictation_start_offset(window, 40) == 40

    def test_no_anchor_means_the_whole_document_is_dictation(self) -> None:
        assert rs.dictation_start_offset(_FakeWindow(), 40) == 0


class TestFinishSequence:
    def test_the_final_pass_is_submitted_as_one_whole_document_job(
        self, monkeypatch
    ) -> None:
        window, _ = _finish_window(monkeypatch)

        rs.on_transcription_finished(window)

        (text, committed_len, accent, level, seq), = window.pp_worker.jobs
        assert text == "raw dictated words"
        assert committed_len == 0  # no prefix treated as already processed
        assert (accent, level) == ("neutral", "medium")
        assert seq > 3  # newer than every live pass, so it is applied last
        assert window._final_seq == seq
        assert "Finalising report..." in window.statuses

    def test_nothing_is_torn_down_until_the_final_pass_lands(
        self, monkeypatch
    ) -> None:
        # This is the freeze fix: Stop returns immediately, so the steps that
        # need the finished report must not have run yet.
        window, order = _finish_window(monkeypatch)

        rs.on_transcription_finished(window)

        assert order == []
        assert window.pp_worker is not None
        assert window._post_dictation_snapshot is None
        assert window.btn_record.enabled is False
        assert "Ready" not in window.statuses

    def test_the_final_result_is_applied_before_the_shutdown_steps(
        self, monkeypatch
    ) -> None:
        window, order = _finish_window(monkeypatch)
        rs.on_transcription_finished(window)
        window.editor.text = "Findings: finished report"

        rs.on_processed_text(window, "finished report", [], window._final_seq)

        assert order == ["apply:finished report", "ai_cleanup", "deid"]
        assert window._post_dictation_snapshot == "Findings: finished report"
        assert window.pp_worker is None
        assert window._dictation_start is None
        assert window.btn_record.enabled is True
        assert window.statuses[-1] == "Ready"

    def test_a_late_live_result_neither_applies_nor_finishes(
        self, monkeypatch
    ) -> None:
        # The live path is latest-only: a pass that finishes after the final one
        # would otherwise rewind the report to a half-processed version of it.
        window, order = _finish_window(monkeypatch)
        rs.on_transcription_finished(window)
        final_seq = window._final_seq
        rs.on_processed_text(window, "finished report", [], final_seq)
        order.clear()

        rs.on_processed_text(window, "stale live text", [], final_seq - 1)

        assert order == []
        assert window._post_dictation_snapshot == "Findings: raw dictated words"

    def test_an_empty_recording_finishes_without_a_final_pass(
        self, monkeypatch
    ) -> None:
        window, order = _finish_window(monkeypatch, raw="   ")

        rs.on_transcription_finished(window)

        assert window.pp_worker is None or window.pp_worker.jobs == []
        assert order == ["ai_cleanup", "deid"]
        assert window.btn_record.enabled is True


class TestStatusOrdering:
    """A timed status message must never overwrite a newer one.

    The corrections banner and the mic-clipping warning both schedule a revert
    seconds later; before this, one landing mid-finalisation put "Ready" on
    screen while the report was still being processed.
    """

    def test_a_stale_revert_is_dropped(self) -> None:
        track = StatusTrack()
        first = track.show()
        track.show()  # a newer message arrived meanwhile

        assert track.revert(first, dictating=False) is None

    def test_the_newest_message_reverts_to_ready_when_idle(self) -> None:
        track = StatusTrack()
        generation = track.show()

        assert track.revert(generation, dictating=False) == READY

    def test_it_reverts_to_the_work_still_running(self) -> None:
        track = StatusTrack()
        track.last_progress = ("Polishing chunk 2/5...", "busy")
        generation = track.show()

        assert track.revert(generation, dictating=True) == ("Polishing chunk 2/5...", "busy")

    def test_two_messages_in_a_row_leave_only_the_last_revertable(self) -> None:
        """The near case the others miss: identical back-to-back messages."""
        track = StatusTrack()
        first = track.show()
        second = track.show()

        assert track.revert(first, dictating=False) is None
        assert track.revert(second, dictating=False) == READY
