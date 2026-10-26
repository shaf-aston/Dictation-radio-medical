"""Cloud-training capture around a dictation: every function a no-op without consent.

What the recording session tells the training collector, and when: start a
capture with the recording, pass it the decoded segments, de-identify the
audio before the temp WAV goes, and persist the corrections at the next
boundary. Kept apart from the session itself, which is about the report.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def active_model_path():
    """Return the active fine-tuned model directory, or None for the base model."""
    try:
        from src.cloud.framework.registry import ModelRegistry
        from src.training.schemas import TASK_WHISPER_VOICE
        return ModelRegistry().get_active_model_path(TASK_WHISPER_VOICE)
    except Exception as exc:
        logger.debug("Model registry lookup failed: %s", exc)
        return None


def start_training_capture(window: MainWindow, wav_path: str, model_size: str) -> None:
    """Start a collector session and route voice corrections into it."""
    try:
        from src.training.collector import get_correction_collector
        from src.dictation.postprocess.voice_commands import set_correction_hook

        collector = get_correction_collector()
        # Persist any previous session still open for review before reusing it.
        collector.finalize_session()
        accent = window._active_accent
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


def on_segments(abs_segments: list) -> None:
    """Forward worker segments to the training collector for timing/PHI scrub."""
    try:
        from src.training.collector import get_correction_collector
        get_correction_collector().update_segments(abs_segments)
    except Exception as exc:
        logger.debug("Segment forwarding to collector failed: %s", exc)


def prepare_training_audio() -> None:
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
    ``start_training_capture``) or window close, so review-time corrections
    are included.
    """
    try:
        from src.training.collector import get_correction_collector
        get_correction_collector().finalize_session()
    except Exception as exc:
        logger.debug("Training capture finalize failed: %s", exc)
