"""Recording session control — transcription worker lifecycle and UI updates."""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QMessageBox

from src.core import perf
from src.dictation.stream.segmenter import ChunkPolicy
from src.dictation.worker import LiveTranscribeWorker
from src.features.file_manager import create_temp_wav
from src.ui.postprocess_worker import PostprocessWorker, build_changes
from src.features.accent_corrections import ACCENT_LABELS, suggest_accent
from src.features.report_release import check_release, record_release
from src.ui.dialogs import confirm_unfilled_fields
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

    Called at the next natural boundary — a new recording (see
    ``_start_training_capture``) or window close — so review-time corrections
    are included.
    """
    try:
        from src.training.collector import get_correction_collector
        get_correction_collector().finalize_session()
    except Exception as exc:
        logger.debug("Training capture finalize failed: %s", exc)


def on_start_recording(window: MainWindow) -> None:
    """Start a new recording session."""
    if window.recorder.is_recording:
        return
    # Log any edits to the previous dictation before this one overwrites it.
    window.flush_dictation_edits()
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
    window._show_status("Recording...")

    # Remember where dictation text starts so we can replace it each cycle
    current_text = window.editor.toPlainText()
    if current_text and not current_text.endswith(("\n", " ")):
        window.editor.insertPlainText(" ")
    window._dictation_start_pos = len(window.editor.toPlainText())

    model_size = window.model_combo.currentText()
    language = window.language_input.text().strip() or "en"
    vad_enabled = window.vad_checkbox.isChecked()
    pause_threshold = float(window.settings.get("pause_threshold", 2.5))
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
    window._partial_seq = 0
    window._applied_seq = 0
    window.pp_thread = QThread()
    window.pp_worker = PostprocessWorker()
    window.pp_worker.moveToThread(window.pp_thread)
    # Bound QObject slots, not lambdas: a signal connected to a plain callable
    # has no receiver thread affinity, so Qt would run the slot in the *emitting*
    # (worker) thread — mutating the editor off the UI thread. See MainWindow.
    window.pp_worker.processed.connect(window._on_processed_text)
    window.pp_thread.finished.connect(window.pp_worker.deleteLater)
    window.pp_thread.start()

    window.live_thread = QThread()
    window.live_worker = LiveTranscribeWorker(
        path, model_size, language, vad_enabled, pause_threshold,
        model_path=active_model_path,
        chunk_policy=ChunkPolicy(
            min_sec=float(window.settings.get("chunk_min_sec")),
            soft_max_sec=float(window.settings.get("chunk_soft_max_sec")),
            force_cut_sec=float(window.settings.get("chunk_force_cut_sec")),
        ),
        silence_rms_floor=float(window.settings.get("silence_rms_floor", 0.002)),
        live_beam_size=int(window.settings.get("live_beam_size")),
        final_beam_size=int(window.settings.get("final_beam_size")),
        polish_confidence_ceiling=float(window.settings.get("polish_confidence_ceiling")),
    )
    window.live_worker.moveToThread(window.live_thread)
    window.live_thread.started.connect(window.live_worker.run)
    window.live_worker.partial.connect(window._on_partial_text)
    window.live_worker.progress.connect(window._on_worker_progress)
    window.live_worker.segments.connect(_on_segments)
    window.live_worker.finished.connect(window._on_transcription_finished)
    window.live_worker.finished.connect(window.live_thread.quit)
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
        window._level_timer.stop()
        window._level_bar.setValue(0)
        set_level_state(window._level_bar, "healthy")
        window.btn_record.setEnabled(True)
        window.btn_stop.setEnabled(False)
        window._show_status("Processing final pass...")
    if window.live_worker is not None:
        window.live_worker.finalize()


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
    """Apply a processed transcript to the editor (UI thread).

    Replaces the dictated region with the latest transcription: Whisper
    re-segments non-deterministically, so everything from ``_dictation_start_pos``
    onwards is rewritten rather than appended.
    """
    # Results can only move forward. A pass that finishes after a newer one has
    # already been applied would otherwise rewind the editor to older text.
    if seq <= window._applied_seq:
        return
    window._applied_seq = seq

    if changes:
        # Accumulate unique corrections for the post-recording banner.
        seen = getattr(window, "_corrections_seen", None)
        if seen is None:
            seen = window._corrections_seen = set(window._corrections_pending)
        window._corrections_pending.extend(build_changes(seen, changes))

    if not processed:
        return

    with perf.stage("ui.apply_partial"):
        _replace_dictation_region(window, processed)
    window._show_status("Receiving...", 800)


def _replace_dictation_region(window: MainWindow, processed: str) -> None:
    """Rewrite only the changed suffix of the dictated region.

    ``setPlainText`` rebuilt the entire document on every cycle, which gets
    expensive as the report grows and forces a full re-layout. Nearly all of the
    text is identical between cycles, so only the differing tail is replaced,
    via a cursor selection.
    """
    current = window.editor.toPlainText()
    start = window._dictation_start_pos
    new_text = current[:start] + processed
    if new_text == current:
        return

    # First index where old and new differ; everything before it is untouched.
    common = start
    limit = min(len(current), len(new_text))
    while common < limit and current[common] == new_text[common]:
        common += 1

    cursor = window.editor.textCursor()
    window.editor.blockSignals(True)
    cursor.beginEditBlock()
    cursor.setPosition(common)
    cursor.setPosition(len(current), cursor.MoveMode.KeepAnchor)
    cursor.insertText(new_text[common:])
    cursor.endEditBlock()
    window.editor.setTextCursor(cursor)
    window.editor.blockSignals(False)
    window._on_text_changed()


def _apply_finished_ai_cleanup(window: MainWindow) -> None:
    """Run the optional 'hard' AI cleanup once on the finished report.

    This is the *only* place the network-backed AI polish runs (never per live
    chunk). No-op unless the user selected the 'hard' cleanup level. The
    underlying ``clean_with_llm`` de-identifies first, is consent-gated, and
    returns the text unchanged on any error or when disabled — so when the
    feature is off (the default) this neither blocks nor alters the report.
    """
    if getattr(window, "_active_cleanup_level", "medium") != "hard":
        return
    text = window.editor.toPlainText()
    if not text.strip():
        return
    try:
        from src.dictation.postprocess.llm_cleanup import clean_with_llm
        window._show_status("AI cleanup...")
        cleaned, _ = clean_with_llm(text)
    except Exception as exc:
        logger.debug("Finished AI cleanup skipped: %s", exc)
        return
    if cleaned and cleaned != text:
        window.editor.blockSignals(True)
        window.editor.setPlainText(cleaned)
        cursor = window.editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        window.editor.setTextCursor(cursor)
        window.editor.blockSignals(False)
        window._on_text_changed()


def _finalize_postprocess(window: MainWindow) -> None:
    """Shut down the post-process thread and run one authoritative full pass.

    The live path is incremental and latest-only, so at this moment the editor
    may hold the output of a superseded cycle, or a tail-only pass. The final
    high-beam transcript is re-processed here as a whole document — once — so
    what the radiologist reviews is never a partially-processed artefact.
    """
    thread = getattr(window, "pp_thread", None)
    if thread is not None:
        thread.quit()
        thread.wait(5000)
        window.pp_thread = None
    window.pp_worker = None

    raw = getattr(window, "_last_raw_transcript", "")
    if not raw.strip():
        return
    from src.dictation.postprocess import postprocess_transcript_with_changes

    processed, changes = postprocess_transcript_with_changes(
        raw,
        accent=getattr(window, "_active_accent", "neutral"),
        cleanup_level=getattr(window, "_active_cleanup_level", "medium"),
        live=True,  # the 'hard' AI polish runs separately, below
    )
    if changes:
        seen = getattr(window, "_corrections_seen", None) or set()
        window._corrections_seen = seen
        window._corrections_pending.extend(build_changes(seen, changes))
    if processed:
        _replace_dictation_region(window, processed)
    window._last_raw_transcript = ""


def on_transcription_finished(window: MainWindow) -> None:
    """Handle end of transcription session."""
    window.btn_record.setEnabled(True)
    window.btn_stop.setEnabled(False)
    _finalize_postprocess(window)
    # The 'hard' cleanup level's AI polish runs here, once, on the finished
    # report — kept out of the per-chunk live path (CLAUDE.md invariant).
    _apply_finished_ai_cleanup(window)
    window._show_status("Ready")
    # Snapshot the dictation output so any later manual edit can be diffed
    # against it (the "was dictation itself wrong?" signal). Flushed to the
    # edit log when the report is committed (export / clear / close).
    window._post_dictation_snapshot = window.editor.toPlainText()
    # De-identify the session audio while the WAV still exists; the collector
    # session stays open so spelling fixes made during review are captured too.
    _prepare_training_audio()
    cleanup_temp_audio(window)
    check_accent_suggestion(window)
    show_corrections_banner(window)


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
        window._show_status(f"Tip: Consider '{label}' accent profile", 5000)
        logger.info("Accent suggestion: %s based on text patterns", suggested)


def show_corrections_banner(window: MainWindow) -> None:
    """Display auto-corrections applied during transcription."""
    if not window._corrections_pending:
        return

    n = len(window._corrections_pending)
    examples = ", ".join(window._corrections_pending[:3])
    suffix = f" (and {n - 3} more)" if n > 3 else ""

    window._show_status(f"Applied {n} correction(s): {examples}{suffix}", 8000)
    window._corrections_pending = []
    window._corrections_seen = set()


def confirm_release(window: MainWindow) -> bool:
    """Run both release rules and return True if the report may leave.

    The desktop shape of the shared gate in ``features/report_release.py`` — the
    rules and the audit trail live there, alongside the web app's 409. Called
    from every exit (copy / save / export) immediately before the text leaves,
    so a field or finding typed in after dictation ended is still caught.

    Unfilled fields are asked about first, because that is the only answer that
    can cancel: cancelling must never leave a findings acknowledgement in the
    audit log for a report that then did not leave. The findings answer itself
    never withholds the report — both answers proceed and only the audit entry
    differs — so False here always means "the radiologist cancelled".
    """
    if not confirm_unfilled_fields(window):
        return False

    check = check_release(window.editor.toPlainText())
    if not check.needs_acknowledgement:
        return True

    msg = QMessageBox(window)
    msg.setWindowTitle("Critical / Urgent Finding Detected")
    msg.setIcon(
        QMessageBox.Icon.Critical if check.worst_level == 1 else QMessageBox.Icon.Warning
    )
    msg.setText(
        "The following critical or urgent finding(s) were detected in this report.\n\n"
        "Please confirm verbal communication with the referring clinician before saving."
    )
    msg.setDetailedText(check.summary)
    btn_ack = msg.addButton("I have communicated this finding", QMessageBox.ButtonRole.AcceptRole)
    msg.addButton("Proceed without acknowledging", QMessageBox.ButtonRole.RejectRole)
    msg.setDefaultButton(btn_ack)
    msg.exec()

    record_release(
        check,
        window._get_patient_info().get("id", ""),
        msg.clickedButton() == btn_ack,
    )
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
    if clipping:
        set_level_state(window._level_bar, "clipping")
        window._show_status("Microphone clipping — reduce input gain", 1500)
    elif level < 0.03:
        set_level_state(window._level_bar, "low")
    else:
        set_level_state(window._level_bar, "healthy")
