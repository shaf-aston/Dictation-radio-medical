"""Background worker for live transcription during recording.

Runs on a QThread, polling the growing WAV file and emitting the full
transcription via Qt signals each cycle.

Design notes
------------
1.  **Numpy audio** — the WAV is decoded once into a float32 array and
    passed straight to faster-whisper, skipping the ffmpeg path.
2.  **Sliding window** — once audio exceeds :data:`_WINDOW_SEC` only the
    most recent slice is sent to Whisper.  Earlier text is *committed*
    (frozen) and prepended without re-processing.
3.  **Adaptive sleep** — the pause between cycles scales with how long
    the last transcription took, avoiding wasted CPU.
4.  **Minimum-growth gate** — skips a cycle if the file hasn't grown by
    at least :data:`_MIN_GROWTH_SEC`.
5.  **Boundary de-duplication** — :func:`trim_committed_tail` finds the
    overlap between the committed tail and the new chunk so words are
    never repeated.

Live-mode quality knobs
-----------------------
* ``_LIVE_BEAM_SIZE = 2`` — beam=1 (greedy) caused noticeable repetition
  on long dictation; beam=2 is still fast and much steadier.
* Committed text is *not* appended to the initial prompt.  Doing so
  caused Whisper to echo prior words back into the new window with the
  small live beam.  Continuity is preserved by the overlap dedup logic.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, Signal

from src.dictation.transcriber import Transcriber, _RADIOLOGY_INITIAL_PROMPT
from src.dictation.text_diff import trim_committed_tail
from src.features.adaptive_learning import get_custom_prompt_suffix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_WINDOW_SEC = 25.0       # max audio duration to transcribe per cycle
_OVERLAP_SEC = 3.0       # context overlap when sliding the window
_MIN_AUDIO_SEC = 0.8     # ignore audio shorter than this
_MIN_GROWTH_SEC = 0.5    # min new audio before re-transcribing (was 0.3 — thrashed)
_LIVE_BEAM_SIZE = 2      # beam=1 caused repetition; beam=2 still real-time
_FINAL_BEAM_SIZE = 5     # higher quality for the final pass after stop


class LiveTranscribeWorker(QObject):
    """Polls a WAV file written by the recorder and emits transcribed text.

    Each ``partial`` emission carries the *complete* transcription
    (committed prefix + current window).  The UI must replace the
    dictated region — not append — when it receives one.
    """

    partial = Signal(str)
    finished = Signal()
    progress = Signal(str)
    # Absolute-timed segments for the current window, used by the cloud training
    # collector to locate corrections in audio. Each item: {start, end, text}
    # with timestamps offset to the full recording.
    segments = Signal(list)

    def __init__(
        self,
        audio_path: str,
        model_size: str,
        language: str,
        vad_enabled: bool,
        pause_threshold: float = 2.5,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.language = language
        self.vad_enabled = vad_enabled
        self.pause_threshold = pause_threshold
        # Optional fine-tuned CT2 model directory (overrides model_size).
        self.model_path = model_path
        self._keep_running = True
        self._final_requested = False

        # Sliding-window state.
        self._committed_text: str = ""
        self._committed_samples: int = 0
        self._last_emitted: str = ""
        self._prev_total_samples: int = 0

        # Bootstrap-commit state.
        self._prev_segments: List[dict] = []
        self._prev_chunk_start_sec: float = 0.0

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._keep_running = False

    def finalize(self) -> None:
        """Request one final transcription pass after recording stops."""
        self._final_requested = True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        wall_start = time.time()
        cycle_count = 0
        try:
            self.progress.emit("Loading model...")
            transcriber = Transcriber(
                model_size=self.model_size, device="auto", model_path=self.model_path
            )
            logger.info("Model loaded in %.2fs", time.time() - wall_start)
            self.progress.emit("Live transcribing...")

            while self._keep_running or self._final_requested:
                audio, sr = self._read_audio()
                if audio is None:
                    time.sleep(0.5)
                    continue

                total_samples = len(audio)
                total_sec = total_samples / sr

                if total_sec < _MIN_AUDIO_SEC:
                    time.sleep(0.5)
                    continue

                growth_sec = (total_samples - self._prev_total_samples) / sr
                if growth_sec < _MIN_GROWTH_SEC and not self._final_requested:
                    time.sleep(0.4)
                    continue
                self._prev_total_samples = total_samples

                chunk, chunk_start_sec = self._window(audio, sr, total_sec)
                windowed = chunk_start_sec > 0.0
                chunk_sec = len(chunk) / sr

                t0 = time.time()
                is_final = self._final_requested
                beam = _FINAL_BEAM_SIZE if is_final else _LIVE_BEAM_SIZE
                prompt = self._build_context_prompt()
                try:
                    chunk_text, segments = transcriber.transcribe(
                        chunk,
                        language=self.language,
                        vad_filter=self.vad_enabled,
                        beam_size=beam,
                        pause_threshold=self.pause_threshold,
                        condition_on_previous_text=False,
                        initial_prompt=prompt,
                    )
                except Exception as exc:
                    logger.warning("Transcription cycle failed: %s", exc)
                    time.sleep(0.5)
                    continue
                elapsed = time.time() - t0
                chunk_text = chunk_text.strip()

                # Commit segments that just fell out of the sliding window
                # so the committed prefix keeps advancing even when VAD
                # silenced the early audio in the new chunk.
                if (chunk_start_sec > self._prev_chunk_start_sec + 0.5
                        and self._prev_segments):
                    self._bootstrap_commit(
                        self._prev_segments,
                        self._prev_chunk_start_sec,
                        chunk_start_sec,
                        sr,
                    )
                self._prev_segments = segments
                self._prev_chunk_start_sec = chunk_start_sec

                # Emit absolute-timed segments for the training collector.
                if segments:
                    abs_segments = [
                        {
                            "start": chunk_start_sec + float(s.get("start", 0)),
                            "end": chunk_start_sec + float(s.get("end", 0)),
                            "text": s.get("text", ""),
                        }
                        for s in segments
                    ]
                    self.segments.emit(abs_segments)

                output = self._build_output(chunk_text, chunk_start_sec)

                emitted = False
                if output and output != self._last_emitted:
                    self._last_emitted = output
                    cycle_count += 1
                    emitted = True
                    self.partial.emit(output)

                if total_sec > _WINDOW_SEC and segments:
                    self._advance_commit(segments, chunk_start_sec, total_sec, sr)

                sleep_time = max(0.4, min(elapsed * 0.5, 2.0))
                logger.info(
                    "cycle=%d  total=%.1fs  chunk=%.1fs  window=%s  "
                    "committed=%.1fs  transcribe=%.2fs  sleep=%.2fs  emit=%s",
                    cycle_count, total_sec, chunk_sec,
                    "YES" if windowed else "no",
                    self._committed_samples / sr, elapsed, sleep_time,
                    "YES" if emitted else "skip",
                )
                time.sleep(sleep_time)

                if self._final_requested:
                    self._keep_running = False
                    self._final_requested = False

        except Exception as exc:
            logger.error("Worker error: %s", exc, exc_info=True)
        finally:
            logger.info(
                "Worker finished in %.2fs, %d cycles emitted",
                time.time() - wall_start, cycle_count,
            )
            self.finished.emit()

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    def _read_audio(self) -> Tuple[Optional[np.ndarray], int]:
        try:
            audio, sr = sf.read(self.audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            return audio, sr
        except FileNotFoundError:
            return None, 0
        except Exception as exc:
            logger.warning("Unexpected error reading audio %s: %s", self.audio_path, exc)
            return None, 0

    # ------------------------------------------------------------------
    # Sliding window
    # ------------------------------------------------------------------

    def _window(
        self, audio: np.ndarray, sr: int, total_sec: float
    ) -> Tuple[np.ndarray, float]:
        """Return ``(audio_chunk, chunk_start_seconds)``.

        The chunk is capped at :data:`_WINDOW_SEC` seconds.  When prior
        text has been committed the window starts just before the commit
        frontier (with :data:`_OVERLAP_SEC` of context); otherwise it
        anchors to the most recent ``_WINDOW_SEC`` of audio.
        """
        if total_sec <= _WINDOW_SEC:
            return audio, 0.0

        if self._committed_samples > 0:
            overlap_samples = int(_OVERLAP_SEC * sr)
            anchor = max(0, self._committed_samples - overlap_samples)
            earliest = max(0, len(audio) - int(_WINDOW_SEC * sr))
            start_sample = max(anchor, earliest)
        else:
            start_sample = max(0, len(audio) - int(_WINDOW_SEC * sr))

        return audio[start_sample:], start_sample / sr

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _build_output(self, chunk_text: str, chunk_start_sec: float) -> str:
        """Combine the committed prefix with the new portion of this chunk."""
        if not self._committed_text or chunk_start_sec == 0.0:
            return chunk_text
        return trim_committed_tail(self._committed_text, chunk_text)

    def _advance_commit(
        self,
        segments: List[dict],
        chunk_start_sec: float,
        total_sec: float,
        sr: int,
    ) -> None:
        """Freeze segments that are safely behind the transcription frontier."""
        safe_abs = total_sec - _WINDOW_SEC
        current_end = self._committed_samples / sr
        if safe_abs <= current_end:
            return

        new_parts: List[str] = []
        new_committed_end = current_end
        for seg in segments:
            abs_end = chunk_start_sec + float(seg.get("end", 0))
            if abs_end > current_end and abs_end <= safe_abs:
                if text := (seg.get("text") or "").strip():
                    new_parts.append(text)
                new_committed_end = abs_end

        if not new_parts:
            return

        addition = " ".join(new_parts)
        self._committed_text = (
            f"{self._committed_text} {addition}".strip()
            if self._committed_text
            else addition
        )
        self._committed_samples = int(new_committed_end * sr)
        logger.info(
            "commit  frontier=%.1fs  committed_now=%.1fs  added=%d chars",
            total_sec, new_committed_end, len(addition),
        )

    def _bootstrap_commit(
        self,
        prev_segments: List[dict],
        prev_chunk_start_sec: float,
        new_chunk_start_sec: float,
        sr: int,
    ) -> None:
        """Seed the commit frontier from the previous window when it slides.

        Called when :meth:`_advance_commit` hasn't fired yet but the
        window has moved forward.  Commits prior-cycle segments whose
        absolute end timestamp is now before the new window's start.
        """
        parts: List[str] = []
        new_end = self._committed_samples / sr
        for seg in prev_segments:
            abs_end = prev_chunk_start_sec + float(seg.get("end", 0))
            if abs_end <= new_chunk_start_sec and abs_end > new_end:
                if text := (seg.get("text") or "").strip():
                    parts.append(text)
                new_end = abs_end

        if not parts:
            return

        addition = " ".join(parts)
        self._committed_text = (
            f"{self._committed_text} {addition}".strip()
            if self._committed_text
            else addition
        )
        self._committed_samples = int(new_end * sr)
        logger.info(
            "bootstrap_commit  new_frontier=%.1fs  added=%d chars",
            new_end, len(addition),
        )

    # ------------------------------------------------------------------
    # Context prompt for Whisper
    # ------------------------------------------------------------------

    def _build_context_prompt(self) -> str:
        """Return the initial prompt: base radiology vocab + learned terms.

        Committed text is intentionally NOT appended — doing so caused
        Whisper to echo prior words back into the current window.  Cross-
        window continuity is preserved by the overlap dedup in
        :func:`trim_committed_tail` instead.
        """
        if custom_terms := get_custom_prompt_suffix():
            return f"{_RADIOLOGY_INITIAL_PROMPT} {custom_terms}"
        else:
            return _RADIOLOGY_INITIAL_PROMPT
