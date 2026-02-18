"""
Background worker for live transcription during recording.

Runs on a QThread, polling the growing WAV file and emitting new text
via Qt signals as segments arrive.
"""

import os
import time
import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal

from transcriber import Transcriber

logger = logging.getLogger(__name__)


class LiveTranscribeWorker(QObject):
    """Polls a WAV file being written by the recorder and emits transcribed text."""

    partial = Signal(str)
    finished = Signal()
    progress = Signal(str)

    def __init__(
        self,
        audio_path: str,
        model_size: str,
        language: str,
        vad_enabled: bool,
        pause_threshold: float = 2.5,
    ) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.language = language
        self.vad_enabled = vad_enabled
        self.pause_threshold = pause_threshold
        self._keep_running = True
        self._last_end: Optional[float] = None
        self._final_requested = False

    def stop(self) -> None:
        self._keep_running = False

    def finalize(self) -> None:
        """Request one final transcription pass after recording stops."""
        self._final_requested = True

    def run(self) -> None:
        start_time = time.time()
        segment_count = 0
        try:
            logger.info("Loading model: %s", self.model_size)
            self.progress.emit("Loading model...")
            transcriber = Transcriber(model_size=self.model_size, device="auto")
            logger.info("Model loaded in %.2fs", time.time() - start_time)
            self.progress.emit("Live transcribing...")

            prev_file_size = 0
            while self._keep_running or self._final_requested:
                try:
                    file_size = os.stat(self.audio_path).st_size
                except Exception:
                    file_size = 0

                if file_size <= prev_file_size and not self._final_requested:
                    time.sleep(0.6)
                    continue
                prev_file_size = file_size

                try:
                    _, segments = transcriber.transcribe(
                        self.audio_path,
                        language=self.language,
                        vad_filter=self.vad_enabled,
                        beam_size=1,
                    )
                except Exception as exc:
                    logger.warning("Transcription cycle failed: %s", exc)
                    time.sleep(0.4)
                    continue

                new_text = self._extract_new_text(segments)
                if new_text:
                    segment_count += 1
                    self.partial.emit(new_text)

                time.sleep(0.6)

                if self._final_requested:
                    self._keep_running = False
                    self._final_requested = False

        except Exception as exc:
            logger.error("Worker error: %s", exc, exc_info=True)
        finally:
            logger.info(
                "Worker finished in %.2fs, %d segments",
                time.time() - start_time,
                segment_count,
            )
            self.finished.emit()

    def _extract_new_text(self, segments: list[dict]) -> str:
        """Build text from segments that appear after the last processed timestamp."""
        new_parts: list[str] = []
        prev_end = self._last_end

        for seg in segments:
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or 0.0)
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if self._last_end is not None and end <= self._last_end + 1e-3:
                continue
            if prev_end is not None:
                gap = start - prev_end
                if gap >= self.pause_threshold:
                    new_parts.append("\n")
                elif new_parts and not new_parts[-1].endswith("\n"):
                    new_parts.append(" ")
            new_parts.append(text)
            prev_end = end

        if new_parts and prev_end is not None:
            self._last_end = prev_end

        return "".join(new_parts).lstrip()
