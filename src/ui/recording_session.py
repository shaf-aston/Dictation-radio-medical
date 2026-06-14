"""Recording session control — transcription worker lifecycle and UI updates."""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QMessageBox

from src.dictation.worker import LiveTranscribeWorker
from src.features.file_manager import create_temp_wav
from src.dictation.postprocess import postprocess_transcript_with_changes
from src.features.accent_corrections import ACCENT_LABELS, suggest_accent
from src.features import audit_log
from src.medical.critical_findings import scan_for_critical_findings, format_findings_for_dialog
from src.ui.styles import COLOR_HEALTHY, COLOR_CLIPPING, COLOR_LOW, LEVEL_BAR_STYLESHEET

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
    except Exception:
        pass


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

    # Persist settings (batch to reduce disk writes)
    window.settings.batch_set({
        "model_size": model_size,
        "language": language,
        "vad_filter": vad_enabled,
        "accent": window._active_accent,
    })

    # Use an active fine-tuned model if one has been downloaded and activated.
    active_model_path = _active_model_path()

    # Begin capturing corrections for cloud training (no-op without consent).
    _start_training_capture(window, path, model_size)

    window.live_thread = QThread()
    window.live_worker = LiveTranscribeWorker(
        path, model_size, language, vad_enabled, pause_threshold,
        model_path=active_model_path,
    )
    window.live_worker.moveToThread(window.live_thread)
    window.live_thread.started.connect(window.live_worker.run)
    window.live_worker.partial.connect(lambda transcript: on_partial_text(window, transcript))
    window.live_worker.progress.connect(lambda msg: window._show_status(msg))
    window.live_worker.segments.connect(_on_segments)
    window.live_worker.finished.connect(lambda: on_transcription_finished(window))
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
        window._level_bar.setStyleSheet(LEVEL_BAR_STYLESHEET.format(color=COLOR_HEALTHY))
        window.btn_record.setEnabled(True)
        window.btn_stop.setEnabled(False)
        window._show_status("Processing final pass...")
    if window.live_worker is not None:
        window.live_worker.finalize()


def on_partial_text(window: MainWindow, full_transcript: str) -> None:
    """Handle live transcription update from worker.

    Replace the dictated region with the latest full transcription.
    Whisper's non-deterministic re-segmentation is handled by replacing
    everything from _dictation_start_pos onwards.
    """
    processed, changes = postprocess_transcript_with_changes(
        full_transcript,
        accent=getattr(window, "_active_accent", "neutral"),
    )
    if changes:
        # Accumulate unique corrections for display after recording stops
        for c in changes:
            if c not in window._corrections_pending:
                window._corrections_pending.append(c)
    if not processed:
        return

    # Build: (text before dictation) + (latest transcription)
    current = window.editor.toPlainText()
    prefix = current[:window._dictation_start_pos]
    new_text = prefix + processed

    # Only update if content actually changed to avoid cursor flicker
    if new_text != current:
        window.editor.blockSignals(True)
        window.editor.setPlainText(new_text)
        # Move cursor to end
        cursor = window.editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        window.editor.setTextCursor(cursor)
        window.editor.blockSignals(False)
        window._on_text_changed()

    window._show_status("Receiving...", 800)


def on_transcription_finished(window: MainWindow) -> None:
    """Handle end of transcription session."""
    window.btn_record.setEnabled(True)
    window.btn_stop.setEnabled(False)
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
    check_critical_findings(window)


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


def check_critical_findings(window: MainWindow) -> None:
    """Scan the report for critical/urgent findings and alert the radiologist."""
    text = window.editor.toPlainText()
    if not text.strip():
        return
    try:
        findings = scan_for_critical_findings(text)
    except Exception as exc:
        logger.warning("Critical findings scan failed: %s", exc)
        return
    if not findings:
        return

    pid = window._get_patient_info().get("id", "")
    summary = format_findings_for_dialog(findings)
    level1 = [f for f in findings if f.level == 1]

    msg = QMessageBox(window)
    msg.setWindowTitle("Critical / Urgent Finding Detected")
    msg.setIcon(QMessageBox.Icon.Critical if level1 else QMessageBox.Icon.Warning)
    msg.setText(
        "The following critical or urgent finding(s) were detected in this report.\n\n"
        "Please confirm verbal communication with the referring clinician before saving."
    )
    msg.setDetailedText(summary)
    btn_ack = msg.addButton("I have communicated this finding", QMessageBox.ButtonRole.AcceptRole)
    msg.addButton("Proceed without acknowledging", QMessageBox.ButtonRole.RejectRole)
    msg.setDefaultButton(btn_ack)
    msg.exec()

    if msg.clickedButton() == btn_ack:
        for f in findings:
            audit_log.log_critical_finding_acknowledged(f.term, pid, f.level)
    else:
        terms_str = "; ".join(f.term for f in findings)
        audit_log.log_critical_finding_overridden(terms_str, pid)


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
        window._level_bar.setStyleSheet(LEVEL_BAR_STYLESHEET.format(color=COLOR_CLIPPING))
        window._show_status("Microphone clipping — reduce input gain", 1500)
    elif level < 0.03:
        window._level_bar.setStyleSheet(LEVEL_BAR_STYLESHEET.format(color=COLOR_LOW))
    else:
        window._level_bar.setStyleSheet(LEVEL_BAR_STYLESHEET.format(color=COLOR_HEALTHY))
