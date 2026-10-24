"""Recording session control: transcription worker lifecycle and UI updates."""

from __future__ import annotations

import os
import logging
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QDialog, QMessageBox

from src.core import perf
from src.dictation.asr.models import resolve_model
from src.dictation.worker import LiveTranscribeWorker
from src.core.paths import create_temp_wav
from src.features import run_log
from src.ui.postprocess_worker import PostprocessWorker, build_changes
from src.features.accent_corrections import ACCENT_LABELS, suggest_accent
from src.features.report_release import (
    add_communication_statement,
    communication_statement,
    record_release,
)
from src.features.template_fields import find_fields, resolve_choice
from src.ui.dialogs import CommunicationDialog, confirm_report_checks, offer_ai_cleanup
from src.ui.styles import set_level_state

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cloud training capture helpers (all no-ops unless the user has opted in)
# ---------------------------------------------------------------------------

def _active_model_path():
    """Return the active fine-tuned model directory, or None for the base model."""
    try:
        from src.cloud.framework.registry import ModelRegistry
        from src.training.schemas import TASK_WHISPER_VOICE
        return ModelRegistry().get_active_model_path(TASK_WHISPER_VOICE)
    except Exception as exc:
        logger.debug("Model registry lookup failed: %s", exc)
        return None


def _start_training_capture(window: MainWindow, wav_path: str, model_size: str) -> None:
    """Start a collector session and route voice corrections into it."""
    try:
        from src.training.collector import get_correction_collector
        from src.dictation.postprocess.voice_commands import set_correction_hook

        collector = get_correction_collector()
        # Persist any previous session still open for review before reusing it.
        collector.finalize_session()
        accent = getattr(window, "_active_accent", "neutral")
        version = window.settings.get("active_model_version") or model_size
        collector.start_session(
            session_id=os.path.basename(wav_path),
            wav_path=wav_path,
            patient_info=window._get_patient_info(),
            model_version=version,
            accent_profile=accent,
        )
        # Explicit "X correct word Y" voice edits feed the collector too.
        set_correction_hook(collector.record_text_correction)
    except Exception as exc:
        logger.debug("Training capture not started: %s", exc)


def _on_segments(abs_segments: list) -> None:
    """Forward worker segments to the training collector for timing/PHI scrub."""
    try:
        from src.training.collector import get_correction_collector
        get_correction_collector().update_segments(abs_segments)
    except Exception as exc:
        logger.debug("Segment forwarding to collector failed: %s", exc)


def _prepare_training_audio() -> None:
    """De-identify the session audio while the temp WAV still exists.

    Runs when transcription ends but *before* the WAV is deleted. The session
    stays open so spelling fixes made during review are still captured; the
    voice-command hook is detached since voice edits only occur while recording.
    """
    try:
        from src.training.collector import get_correction_collector
        from src.dictation.postprocess.voice_commands import set_correction_hook
        get_correction_collector().prepare_audio()
        set_correction_hook(None)
    except Exception as exc:
        logger.debug("Training audio prepare failed: %s", exc)


def finalize_training_capture() -> None:
    """Persist captured corrections and close the open collector session.

    Called at the next natural boundary, a new recording (see
    ``_start_training_capture``) or window close, so review-time corrections
    are included.
    """
    try:
        from src.training.collector import get_correction_collector
        get_correction_collector().finalize_session()
    except Exception as exc:
        logger.debug("Training capture finalize failed: %s", exc)


def _abandon_unfinished_session(window: MainWindow) -> None:
    """Retire the previous session, if any part of it still exists.

    Record comes back before the polish finishes, so this is reachable: Stop,
    then Record (or F5) a few seconds later. The old worker must not be able to
    write into the new recording's document, so its signals are disconnected
    *before* anything of the new session exists. The test is "does a worker
    still exist", not a sequence number: the numbers are cleared while the
    polish is still running, which is how a polish used to survive into the
    next recording. Its threads are owned by the window, so dropping the
    references here never destroys a running thread; each deletes itself when
    it finishes. The parts of the shutdown that still matter for the abandoned
    run (its audio, its log record) are done here, since ``_complete_finish``
    will never run for it.
    """
    if window.live_worker is None and window.pp_worker is None:
        return

    worker = window.live_worker
    if worker is not None:
        try:
            for signal, slot in (
                (worker.partial, window._on_partial_text),
                (worker.progress, window._on_worker_progress),
                (worker.finished, window._on_transcription_finished),
                (worker.segments, _on_segments),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass  # that one was already disconnected
            # Its thread still quits itself: finished -> quit stays connected.
            worker.cancel()
        except RuntimeError as exc:
            logger.debug("Previous worker already gone: %s", exc)

    pp_worker = window.pp_worker
    if pp_worker is not None:
        try:
            pp_worker.processed.disconnect(window._on_processed_text)
        except (RuntimeError, TypeError):
            pass
    thread = window.pp_thread
    if thread is not None:
        thread.quit()
    # Still running for a moment (a decode, a pipeline pass). The window keeps
    # them until they end, so closing it can wait for them: a thread destroyed
    # with its window while running aborts the process.
    window._retiring_threads.extend(t for t in (window.pp_thread, window.live_thread) if t is not None)
    window.pp_worker = None
    window.pp_thread = None

    window._handback_seq = None
    window._final_seq = None
    window._post_dictation_snapshot = window.editor.toPlainText()
    _prepare_training_audio()
    _record_the_run(window)   # reads live_worker for its chunk count: before the clear
    cleanup_temp_audio(window)
    window.live_worker = None
    window.live_thread = None
    _close_dictation_region(window)


def session_may_write(window: MainWindow) -> bool:
    """May the dictation session still write into the document?

    While recording, yes: it owns its region. From Stop on, only while the
    editor still holds exactly what the session last wrote. The radiologist
    has the report from the moment it is handed back, so any change they make
    (typing, a template, a release) ends the session's right to rewrite it,
    for good: overwriting a report someone has corrected is the worst outcome
    this feature could have. Comparing with the session's own last write,
    rather than a snapshot taken at hand-back, means a pass that lands late or
    a hand-back that never lands cannot switch the guard off or fake an edit.
    """
    if window.recorder.is_recording:
        return True
    if window._session_yielded:
        return False
    last = window._session_last_written
    if last is not None and window.editor.toPlainText() != last:
        window._session_yielded = True
        return False
    return True


def yield_document(window: MainWindow) -> None:
    """The radiologist has taken the report (released it): no more rewrites."""
    if window.live_worker is not None or window.pp_worker is not None:
        window._session_yielded = True


def insert_before_dictation(window: MainWindow, text: str) -> None:
    """Put *text* (a template, a quick phrase) just before the dictated region.

    Anything inside the region is rewritten by the next live cycle. The region
    start stays put when text is written AT it (see _open_dictation_region),
    so after inserting there the start is moved past the new text by hand.
    """
    start = window._dictation_start
    if start is None:
        window.editor.textCursor().insertText(text)
        return
    position = start.position()
    cursor = window.editor.textCursor()
    cursor.setPosition(position)
    cursor.insertText(text)
    start.setPosition(position + len(text))


def _open_dictation_region(window: MainWindow) -> None:
    """Mark where this recording's text goes: at the caret, or over a selection.

    The region has two ends, both ``QTextCursor``s, because Qt moves a cursor
    along when text is inserted before it, so a template loaded or a field
    typed elsewhere mid-recording never leaves a boundary cutting into text
    that isn't dictation. The start stays put when text is written AT it:
    without that, the first live update (which starts exactly there) pushed
    the anchor past its own words and every later cycle appended instead of
    replacing ("The lungs areThe lungs are clear..."). The end does move with
    text written at it, which is how the region grows.

    A selected template field is replaced, and remembered so that saying one
    of its options ("left" into ``[Right / Left]``) answers it in the
    template's own spelling (``template_fields.resolve_choice``).
    """
    editor = window.editor
    cursor = editor.textCursor()
    selected = cursor.selectedText() if cursor.hasSelection() else ""
    fields = find_fields(selected)
    window._dictation_field = (
        selected if len(fields) == 1 and fields[0].text == selected else None
    )
    cursor.beginEditBlock()
    if cursor.hasSelection():
        cursor.removeSelectedText()
    text = editor.toPlainText()
    pos = cursor.position()
    # Dictation must not weld onto the words either side of it.
    if pos < len(text) and not text[pos].isspace() and text[pos] not in ".,;:)":
        cursor.insertText(" ")
        cursor.setPosition(pos)
    if pos > 0 and not text[pos - 1].isspace():
        cursor.insertText(" ")
        pos += 1
    cursor.endEditBlock()

    start = editor.textCursor()
    start.setPosition(pos)
    start.setKeepPositionOnInsert(True)
    end = editor.textCursor()
    end.setPosition(pos)
    window._dictation_start = start
    window._dictation_end = end
    editor.setTextCursor(end)


def _close_dictation_region(window: MainWindow) -> None:
    """The dictated region no longer exists; nothing may write into it again."""
    window._dictation_start = None
    window._dictation_end = None
    window._dictation_field = None


def on_start_recording(window: MainWindow) -> None:
    """Start a new recording session."""
    if window.recorder.is_recording:
        return
    # Log any edits to the previous dictation before this one overwrites it.
    window.flush_dictation_edits()
    # Anything still finishing from the last dictation stops owning the
    # document here: Record is available before the polish is done.
    _abandon_unfinished_session(window)
    path = create_temp_wav()
    window.current_wav_path = path
    try:
        window.recorder.start(path)
    except Exception as exc:
        QMessageBox.critical(window, "Audio Error", f"Cannot start recording:\n{exc}")
        window.current_wav_path = None
        return

    window.btn_record.setEnabled(False)
    window.btn_stop.setEnabled(True)
    window._corrections_pending = []
    window._corrections_seen = set()
    window._level_timer.start()
    window._record_started = time.monotonic()
    window._status.last_progress = ("Recording...", "rec")
    window._show_status("Recording...", state="rec")

    _open_dictation_region(window)

    model_size = window.model_combo.currentText()
    language = window.language_input.text().strip() or "en"
    vad_enabled = window.vad_checkbox.isChecked()
    window._active_accent = window.accent_combo.currentData() or "neutral"
    window._active_cleanup_level = window.cleanup_combo.currentData() or "medium"

    # Persist settings (batch to reduce disk writes)
    window.settings.batch_set({
        "model_size": model_size,
        "language": language,
        "vad_filter": vad_enabled,
        "accent": window._active_accent,
        "cleanup_level": window._active_cleanup_level,
    })

    # Use an active fine-tuned model if one has been downloaded and activated.
    active_model_path = _active_model_path()

    # Begin capturing corrections for cloud training (no-op without consent).
    _start_training_capture(window, path, model_size)

    # Post-processing runs on its own thread so the ten-stage pipeline can never
    # block the UI (see postprocess_worker). Fresh per recording: no stale
    # incremental cache can survive into the next report.
    perf.reset()
    # Open the run record beside perf.reset(): both mark "a new dictation starts
    # here", and the record is what survives the next reset (see run_log).
    window._run_record = run_log.start(
        "desktop", model=model_size, engine=str(window.settings.get("asr_engine")),
    )
    window._audio_stopped_at = None
    window._partial_seq = 0
    window._applied_seq = 0
    window._final_seq = None
    window._handback_seq = None
    window._session_last_written = None
    window._session_yielded = False
    # Owned by the window: a thread object dropped while it still runs is a
    # crash ("QThread: Destroyed while thread is still running").
    window.pp_thread = QThread(window)
    window.pp_worker = PostprocessWorker()
    window.pp_worker.moveToThread(window.pp_thread)
    # Bound QObject slots, not lambdas: a signal connected to a plain callable
    # has no receiver thread affinity, so Qt would run the slot in the *emitting*
    # (worker) thread: mutating the editor off the UI thread. See MainWindow.
    window.pp_worker.processed.connect(window._on_processed_text)
    window.pp_thread.finished.connect(window.pp_worker.deleteLater)
    window.pp_thread.finished.connect(window.pp_thread.deleteLater)
    window.pp_thread.start()

    window.live_thread = QThread(window)
    window.live_worker = LiveTranscribeWorker(
        path, model_size, language, vad_enabled,
        settings_get=window.settings.get,
        live_model_size=resolve_model(window.settings.get("live_model_size")),
        model_path=active_model_path,
    )
    window.live_worker.moveToThread(window.live_thread)
    window.live_thread.started.connect(window.live_worker.run)
    window.live_worker.partial.connect(window._on_partial_text)
    window.live_worker.progress.connect(window._on_worker_progress)
    window.live_worker.segments.connect(_on_segments)
    window.live_worker.finished.connect(window._on_transcription_finished)
    # Direct: QThread.quit is thread-safe, and queued it would wait for the UI
    # thread, which on close is the one blocked waiting for this thread.
    window.live_worker.finished.connect(window.live_thread.quit, Qt.ConnectionType.DirectConnection)
    window.live_worker.finished.connect(window.live_worker.deleteLater)
    window.live_thread.finished.connect(window.live_thread.deleteLater)
    window.live_thread.start()


def on_stop_recording(window: MainWindow) -> None:
    """Stop the current recording session."""
    if not window.recorder.is_recording:
        return
    try:
        window.recorder.stop()
    finally:
        # The audio ends here; everything after it is the app catching up.
        # Splitting the two is what makes "how long after Stop?" answerable.
        window._audio_stopped_at = time.monotonic()
        window._level_timer.stop()
        window._level_bar.setValue(0)
        set_level_state(window._level_bar, "healthy")
        window.btn_stop.setEnabled(False)
    if window.live_worker is not None:
        window.live_worker.finalize()
    _hand_back_live_text(window)


def _hand_back_live_text(window: MainWindow) -> None:
    """Give the radiologist the text they have been reading, now.

    The confidence-targeted polish runs for seconds after Stop, and the desktop
    used to sit on the finished report until it returned. But the live text is
    already on screen and already theirs to edit: the polish is an *upgrade*,
    not a gate. So the whole-document pass runs on the live transcript straight
    away, Record comes back as soon as it lands, and the polished text replaces
    it later under a higher sequence number (the same "never rewind" rule every
    other pass obeys). This is the web path's behaviour, brought across.
    """
    raw = getattr(window, "_last_raw_transcript", "")
    if window.pp_worker is None or not raw.strip():
        # Nothing decoded yet: there is no text to hand back, so the ordinary
        # wait for the worker to finish is also the short one.
        window._status.last_progress = ("Processing final pass...", "busy")
        window._show_status("Processing final pass...", state="busy")
        return
    window._partial_seq += 1
    window._handback_seq = window._partial_seq
    window.pp_worker.submit(
        raw,
        0,  # whole document: the handback must be a complete report, not a tail
        getattr(window, "_active_accent", "neutral"),
        getattr(window, "_active_cleanup_level", "medium"),
        window._handback_seq,
    )


def on_partial_text(
    window: MainWindow, full_transcript: str, committed_len: int = 0
) -> None:
    """Queue a live transcription update for off-thread post-processing.

    This runs on the UI thread and must stay cheap: it only hands the
    transcript to the post-process worker. ``committed_len`` marks the frozen
    prefix the worker will never revise, letting the pipeline skip text it has
    already corrected.
    """
    window._last_raw_transcript = full_transcript
    if window.pp_worker is None:
        return
    window._partial_seq += 1
    if not window.recorder.is_recording:
        # A decode that was in flight at Stop lands after the hand-back was
        # queued. It is newer text, so it becomes the hand-back: processed whole
        # and the pass that gives Record back. Queued as an ordinary partial it
        # replaced the hand-back in the latest-only mailbox, and the hand-back
        # never landed.
        window._handback_seq = window._partial_seq
        committed_len = 0
    window.pp_worker.submit(
        full_transcript,
        committed_len,
        getattr(window, "_active_accent", "neutral"),
        getattr(window, "_active_cleanup_level", "medium"),
        window._partial_seq,
    )


def on_processed_text(
    window: MainWindow, processed: str, changes: list, seq: int
) -> None:
    """Apply a processed transcript to the dictated region (UI thread).

    Only the region between ``_dictation_start`` and ``_dictation_end`` is
    replaced; text after it is never touched.
    """
    # Results can only move forward. A pass that finishes after a newer one has
    # already been applied would otherwise rewind the editor to older text.
    if seq <= window._applied_seq:
        return
    window._applied_seq = seq
    is_handback = seq == window._handback_seq
    is_final = seq == window._final_seq

    if session_may_write(window):
        if changes:
            # Accumulate unique corrections for the post-recording banner.
            seen = getattr(window, "_corrections_seen", None)
            if seen is None:
                seen = window._corrections_seen = set(window._corrections_pending)
            window._corrections_pending.extend(build_changes(seen, changes))
        if processed:
            with perf.stage("ui.apply_partial"):
                _replace_dictation_region(window, processed)
            if not is_final:
                window._show_status("Receiving...", 800, state="rec")
    elif is_final:
        window._show_status(
            "Kept your edits - the improved version was discarded", 4000, state="ok"
        )

    # The hand-back has been processed: the report is the radiologist's, so
    # Record comes back now rather than after the polish. Whether or not it was
    # written (they may already be editing), the session is NOT closed here: the
    # polish is still running and still owns the teardown.
    if is_handback:
        window._handback_seq = None
        window.btn_record.setEnabled(True)
        if not is_final:
            window._status.last_progress = ("Improving accuracy...", "busy")
            window._show_status("Improving accuracy...", state="busy")

    # The authoritative full-document pass has landed (or been refused): the
    # rest of the shutdown can run.
    if is_final:
        _complete_finish(window)


def dictation_start_offset(window: MainWindow, doc_len: int) -> int:
    """Where the dictated region begins, clamped inside the document.

    The clamp is the last line of defence: a document replaced wholesale
    (Open, New, Clear) leaves an anchor Qt has already reset, and writing
    after the end of the text is not a valid edit.
    """
    anchor = getattr(window, "_dictation_start", None)
    position = anchor.position() if anchor is not None else 0
    return max(0, min(position, doc_len))


def dictation_end_offset(window: MainWindow, doc_len: int) -> int:
    """Where the dictated region ends; the end of the document if unbounded."""
    anchor = getattr(window, "_dictation_end", None)
    position = anchor.position() if anchor is not None else doc_len
    return max(dictation_start_offset(window, doc_len), min(position, doc_len))


def _replace_dictation_region(window: MainWindow, processed: str) -> None:
    """Rewrite only the changed tail of the dictated region.

    ``setPlainText`` rebuilt the entire document on every cycle, which gets
    expensive as the report grows and forces a full re-layout. Nearly all of
    the text is identical between cycles, so only the part of the region that
    differs is replaced, via a cursor selection. Text after the region (the
    rest of the template, when dictating into the middle of it) is never
    touched.
    """
    field = getattr(window, "_dictation_field", None)
    if field:
        processed = resolve_choice(field, processed) or processed
    current = window.editor.toPlainText()
    start = dictation_start_offset(window, len(current))
    end = dictation_end_offset(window, len(current))
    region = current[start:end]
    if region == processed:
        return

    # First index where old and new differ; everything before it is untouched.
    same = 0
    limit = min(len(region), len(processed))
    while same < limit and region[same] == processed[same]:
        same += 1

    cursor = QTextCursor(window.editor.document())
    window.editor.blockSignals(True)
    cursor.beginEditBlock()
    cursor.setPosition(start + same)
    cursor.setPosition(end, cursor.MoveMode.KeepAnchor)
    cursor.insertText(processed[same:])
    cursor.endEditBlock()
    if window.recorder.is_recording:
        # The caret follows the dictation while it is spoken. After Stop it is
        # the radiologist's: a polish landing must not move it, or the next
        # dictation goes wherever the polish left it.
        window.editor.setTextCursor(cursor)
    window.editor.blockSignals(False)
    window._session_last_written = window.editor.toPlainText()
    window._on_text_changed(by_dictation=True)


def _apply_finished_ai_cleanup(window: MainWindow) -> None:
    """The 'hard' cleanup level: offer the AI polish once, on the finished report.

    Never per live chunk (CLAUDE.md invariant), and never applied unseen: it
    goes through the same review dialog as Tools → AI Cleanup.
    """
    if getattr(window, "_active_cleanup_level", "medium") == "hard":
        offer_ai_cleanup(window, automatic=True)


def on_transcription_finished(window: MainWindow) -> None:
    """Recording ended: queue the one authoritative full-document pass.

    The live path is incremental and latest-only, so at this moment the editor
    may hold the output of a superseded cycle, or a tail-only pass. The final
    high-beam transcript is re-processed as a whole document, once, so what
    the radiologist reviews is never a partially-processed artefact.

    That pass used to run right here, on the UI thread, freezing the window for
    seconds after Stop. It now goes through the same post-process thread the
    live path uses, with the highest sequence number of the session, so it is
    guaranteed to be the last text applied. Everything that must happen *after*
    the finished report exists, the optional AI cleanup, the edit-tracking
    snapshot, thread teardown, the corrections banner, waits in
    :func:`_complete_finish` until that result comes back.
    """
    # Record stays disabled until the finished report is on screen: starting a
    # second session on top of an unfinished one would abandon its thread and
    # let its final pass write into the new recording's document.
    window.btn_stop.setEnabled(False)

    raw = getattr(window, "_last_raw_transcript", "")
    if window.pp_worker is None or not raw.strip():
        _complete_finish(window)
        return

    window._status.last_progress = ("Finalising report...", "busy")
    window._show_status("Finalising report...", state="busy")
    window._partial_seq += 1
    window._final_seq = window._partial_seq
    window.pp_worker.submit(
        raw,
        0,  # whole document: no prefix is treated as already-processed
        getattr(window, "_active_accent", "neutral"),
        getattr(window, "_active_cleanup_level", "medium"),
        window._final_seq,
    )
    window._last_raw_transcript = ""


def _complete_finish(window: MainWindow) -> None:
    """Close the session down, once the finished report is on screen."""
    thread = getattr(window, "pp_thread", None)
    if thread is not None:
        thread.quit()
        thread.wait(5000)
        window.pp_thread = None
    window.pp_worker = None
    # Both deleted themselves on finishing; a reference kept past that point
    # is a dead object the next close or recording would call into.
    window.live_worker = None
    window.live_thread = None
    window._final_seq = None
    window._handback_seq = None
    window._last_raw_transcript = ""
    # The overlays paused while the session wrote; the report they missed is
    # the finished one (a final pass identical to the hand-back writes nothing).
    window.finding_gutter.schedule_refresh()
    window.term_marks.schedule_rescan()

    # The 'hard' cleanup level's AI polish runs here, once, on the finished
    # report: kept out of the per-chunk live path (CLAUDE.md invariant).
    # Record is re-enabled only after it: that call can sit on the network for
    # seconds, and a recording started underneath it would be overwritten.
    _apply_finished_ai_cleanup(window)
    window.btn_record.setEnabled(True)
    window._show_status("Ready", state="idle")
    # Snapshot the dictation output so any later manual edit can be diffed
    # against it (the "was dictation itself wrong?" signal). Flushed to the
    # edit log when the report is committed (export / clear / close).
    window._post_dictation_snapshot = window.editor.toPlainText()
    # The dictated region no longer exists: a template loaded from here on
    # replaces the document again, as it does outside a recording.
    _close_dictation_region(window)
    # De-identify the session audio while the WAV still exists; the collector
    # session stays open so spelling fixes made during review are captured too.
    _prepare_training_audio()
    _record_the_run(window)
    cleanup_temp_audio(window)
    check_accent_suggestion(window)
    show_corrections_banner(window)


def _record_the_run(window: MainWindow) -> None:
    """Close the run record now the finished report is on screen.

    Here rather than in ``on_transcription_finished`` because this is the first
    moment the *final* text exists: and it must be before the next recording's
    ``perf.reset()`` wipes the stage timings the record is made of.
    """
    record = getattr(window, "_run_record", None)
    if record is None:
        return
    window._run_record = None
    started = getattr(window, "_record_started", None)
    stopped = getattr(window, "_audio_stopped_at", None)
    now = time.monotonic()
    if started is not None:
        record.duration_sec = round(now - started, 3)
        if stopped is not None:
            record.audio_sec = round(stopped - started, 3)
            record.finalise_sec = round(now - stopped, 3)
    # Read defensively: the worker is torn down around this point, and a
    # missing chunk count must not cost the whole record.
    worker = getattr(window, "live_worker", None)
    record.chunk_count = getattr(worker, "committed_chunks", 0) or 0
    run_log.finish(record, window.editor.toPlainText(), window.settings)


def cleanup_temp_audio(window: MainWindow) -> None:
    """Remove temporary WAV file."""
    try:
        if window.current_wav_path and os.path.exists(window.current_wav_path):
            os.remove(window.current_wav_path)
    except Exception as exc:
        logger.debug("Failed to remove temp audio %s: %s", window.current_wav_path, exc)
    window.current_wav_path = None


def check_accent_suggestion(window: MainWindow) -> None:
    """Analyze transcribed text and suggest accent profile if patterns detected."""
    current_accent = getattr(window, "_active_accent", "neutral")
    if current_accent != "neutral":
        return  # Already using an accent profile

    text = window.editor.toPlainText()
    if len(text) < 100:  # Need enough text to analyze
        return

    suggested = suggest_accent(text)
    if suggested and suggested != current_accent:
        label = ACCENT_LABELS.get(suggested, suggested)
        window._show_status(f"Tip: Consider '{label}' accent profile", 5000, state="idle")
        logger.info("Accent suggestion: %s based on text patterns", suggested)


def show_corrections_banner(window: MainWindow) -> None:
    """Display auto-corrections applied during transcription."""
    if not window._corrections_pending:
        return

    n = len(window._corrections_pending)
    examples = ", ".join(window._corrections_pending[:3])
    suffix = f" (and {n - 3} more)" if n > 3 else ""

    window._show_status(f"Applied {n} correction(s): {examples}{suffix}", 8000, state="ok")
    window._corrections_pending = []
    window._corrections_seen = set()


def confirm_release(window: MainWindow) -> bool:
    """Run both release rules and return True if the report may leave.

    The desktop shape of the shared gate in ``features/report_release.py``: the
    rules and the audit trail live there, alongside the web app's 409. Called
    from every exit (copy / save / export) immediately before the text leaves,
    so a field or finding typed in after dictation ended is still caught.

    Unfilled fields are asked about first, because that is the only answer that
    can cancel: cancelling must never leave a findings acknowledgement in the
    audit log for a report that then did not leave. The findings answer itself
    never withholds the report, both answers proceed and only the audit entry
    differs, so False here always means "the radiologist cancelled".

    The scan runs through ``window.findings`` so the answer given here is the
    same state the gutter marks and the count pill show: acknowledging settles
    the pill, and a finding typed in afterwards puts it back to outstanding.
    Only outstanding findings are asked about (``OutstandingFindings.pending``):
    one already communicated needs no second call, and a second answer used to
    append a second "communicated" sentence that was never audited.
    """
    if window.recorder.is_recording:
        # The live loop is still rewriting the end of the report.
        window._show_status("Stop recording before the report leaves", 3000, state="warn")
        return False
    if not confirm_report_checks(window):
        return False

    window.findings.update(window.editor.toPlainText())
    # Captured before the dialog: msg.exec() runs a nested event loop, and the
    # gutter's debounced re-scan can fire inside it and replace window.findings
    # .check. Answering and auditing from this local means the radiologist's
    # answer always applies to exactly what they were shown.
    check = window.findings.pending
    if not check.needs_acknowledgement:
        window.finding_gutter.show_state()
        yield_document(window)
        return True

    dialog = CommunicationDialog(window, check, str(window.settings.get("radiologist_name") or ""))
    acknowledged = dialog.exec() == QDialog.DialogCode.Accepted
    comm = dialog.communication() if acknowledged else None
    if comm is not None:
        if comm.radiologist:
            window.settings.set("radiologist_name", comm.radiologist)
        if dialog.add_to_report.isChecked():
            # Before the caller reads the text, so the sentence leaves with it.
            # One undoable edit at the end of the report.
            text = window.editor.toPlainText()
            updated = add_communication_statement(
                text, communication_statement(check.findings, comm)
            )
            if updated != text:
                cursor = window.editor.textCursor()
                cursor.beginEditBlock()
                cursor.movePosition(cursor.MoveOperation.End)
                cursor.movePosition(cursor.MoveOperation.Start, cursor.MoveMode.KeepAnchor)
                cursor.insertText(updated)
                cursor.endEditBlock()
    record_release(
        check.findings, window._get_patient_info().get("id", ""), acknowledged, comm
    )
    if acknowledged:
        window.findings.acknowledge(check)
    window.finding_gutter.show_state()
    # What leaves is what the radiologist answered for: a polish landing now
    # would make the screen and the released copy different reports.
    yield_document(window)
    return True


def setup_level_timer(window: MainWindow) -> None:
    """Create the microphone level update timer."""
    window._level_timer = QTimer(window)
    window._level_timer.setInterval(100)   # 10 Hz update
    window._level_timer.timeout.connect(lambda: update_level_display(window))


def update_level_display(window: MainWindow) -> None:
    """Update microphone level meter color based on input."""
    level = window.recorder.current_level
    clipping = window.recorder.is_clipping
    window._level_bar.setValue(int(level * 100))
    # How long this dictation has run. Free: this timer already ticks at 10 Hz
    # while recording, and stops with it.
    seconds = int(time.monotonic() - getattr(window, "_record_started", time.monotonic()))
    window._elapsed_label.setText(f"{seconds // 60}:{seconds % 60:02d}")
    if clipping:
        set_level_state(window._level_bar, "clipping")
        window._show_status("Microphone clipping: reduce input gain", 1500, state="warn")
    elif level < 0.03:
        set_level_state(window._level_bar, "low")
    else:
        set_level_state(window._level_bar, "healthy")
