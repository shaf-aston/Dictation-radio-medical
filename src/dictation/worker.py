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
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, Signal

from src.core import perf
from src.dictation.asr import AsrEngine, TranscribeContext, create_engine
from src.dictation.transcriber import _RADIOLOGY_INITIAL_PROMPT
from src.dictation.window_state import WindowState
from src.features.adaptive_learning import get_custom_prompt_suffix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_WINDOW_SEC = 25.0       # hard ceiling on audio sent to Whisper per cycle (safety)
_MIN_AUDIO_SEC = 0.8     # ignore audio shorter than this
_MIN_GROWTH_SEC = 0.5    # min new audio before re-transcribing (was 0.3 — thrashed)
_COMMIT_LAG_SEC = 8.0    # trailing audio kept un-committed (still revisable by
                         # Whisper). Older text is frozen, so the live window
                         # shrinks to ~_COMMIT_LAG_SEC + overlap instead of a full
                         # _WINDOW_SEC every cycle — the main live-speed lever.
                         # WindowState clamps it above the window overlap so the
                         # window always re-covers the committed tail (lossless).
_LIVE_BEAM_SIZE = 2      # beam=1 caused repetition; beam=2 still real-time
_FINAL_BEAM_SIZE = 5     # higher quality for the final pass after stop


class LiveTranscribeWorker(QObject):
    """Polls a WAV file written by the recorder and emits transcribed text.

    Each ``partial`` emission carries the *complete* transcription
    (committed prefix + current window).  The UI must replace the
    dictated region — not append — when it receives one.

    ``partial`` also carries the length of the committed prefix within that
    text: the number of leading characters this worker will never revise.
    Consumers use it to skip re-processing frozen text
    (:class:`~src.dictation.postprocess.incremental.IncrementalPostprocessor`).
    A value of ``0`` means "treat the whole text as revisable".
    """

    partial = Signal(str, int)
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
        window_sec: float = _WINDOW_SEC,
        commit_lag_sec: float = _COMMIT_LAG_SEC,
        silence_rms_floor: float = 0.002,
    ) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.language = language
        self.vad_enabled = vad_enabled
        self.pause_threshold = pause_threshold
        self.silence_rms_floor = max(0.0, float(silence_rms_floor))
        # Optional fine-tuned CT2 model directory (overrides model_size).
        self.model_path = model_path
        # Pure sliding-window / commit machine (config-tunable knobs are clamped
        # to safe bounds inside WindowState). This owns all window geometry and
        # commit-frontier arithmetic; the worker only feeds it audio segments.
        self._win = WindowState(window_sec, commit_lag_sec,
                                pause_threshold=self.pause_threshold)
        self._keep_running = True
        self._final_requested = False

        # Loop-local gating (not part of the window machine).
        self._last_emitted: str = ""
        self._prev_total_samples: int = 0

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
            engine = create_engine(
                model_size=self.model_size, device="auto", model_path=self.model_path
            )
            logger.info("Model loaded in %.2fs", time.time() - wall_start)
            self.progress.emit("Live transcribing...")

            final_grace = 0
            while self._keep_running or self._final_requested:
                # Frame count from the header — no decode. The audio itself is
                # read below, and only for the window we actually transcribe.
                total_samples, sr = self._audio_length()
                if not sr or total_samples / sr < _MIN_AUDIO_SEC:
                    # The WAV stops growing once recording ends, so a finalize()
                    # on a too-short or unreadable file would spin this loop
                    # forever. Give the recorder a short grace to flush, then
                    # finish without a final pass.
                    if self._final_requested:
                        final_grace += 1
                        if final_grace > 6:
                            logger.warning(
                                "Recording too short or unreadable (%s frames); "
                                "skipping final pass", total_samples,
                            )
                            break
                    time.sleep(0.5)
                    continue

                total_sec = total_samples / sr

                growth_sec = (total_samples - self._prev_total_samples) / sr
                if growth_sec < _MIN_GROWTH_SEC and not self._final_requested:
                    time.sleep(0.4)
                    continue

                if self._final_requested:
                    # Final pass: re-transcribe the ENTIRE recording at high beam
                    # with cross-segment context, replacing the frozen low-beam
                    # committed prefix. This is the main accuracy win — mumbled
                    # speech committed from the fast live pass gets a second look.
                    audio, sr = self._read_audio()
                    if audio is not None:
                        self._run_final_pass(engine, audio, sr)
                    self._keep_running = False
                    self._final_requested = False
                    break

                start_sample = self._win.window_start(total_samples, sr)
                chunk, sr = self._read_audio(start_sample)
                if chunk is None or not len(chunk):
                    time.sleep(0.4)
                    continue
                chunk_start_sec = start_sample / sr
                windowed = start_sample > 0
                chunk_sec = len(chunk) / sr

                # Silence gate: decoding near-silence is the canonical source of
                # ",,..," / phrase hallucinations. Skip the cycle (but consume the
                # audio so we don't re-evaluate the same silence next loop).
                if self._rms(chunk) < self.silence_rms_floor:
                    self._prev_total_samples = total_samples
                    time.sleep(0.4)
                    continue

                t0 = time.time()
                beam = _LIVE_BEAM_SIZE
                prompt = self._build_context_prompt()
                try:
                    with perf.stage("worker.transcribe_live"):
                        result = engine.transcribe(
                            chunk,
                            TranscribeContext(
                                language=self.language,
                                vad_filter=self.vad_enabled,
                                beam_size=beam,
                                pause_threshold=self.pause_threshold,
                                condition_on_previous_text=False,
                                initial_prompt=prompt,
                                # One greedy decode per live cycle — the
                                # temperature-fallback ladder can re-decode the
                                # window up to 5x and higher temperatures
                                # hallucinate; the final full pass keeps it.
                                temperature=0.0,
                            ),
                        )
                        chunk_text, segments = result.text, result.segments_as_dicts()
                except Exception as exc:
                    logger.warning("Transcription cycle failed: %s", exc)
                    time.sleep(0.5)
                    continue
                # Only advance the growth baseline after a successful transcribe,
                # so a failed cycle's audio is retried rather than skipped.
                self._prev_total_samples = total_samples
                elapsed = time.time() - t0
                chunk_text = chunk_text.strip()

                # Commit segments that just fell out of the sliding window
                # so the committed prefix keeps advancing even when VAD
                # silenced the early audio in the new chunk.
                self._win.maybe_bootstrap(chunk_start_sec, sr)
                self._win.record_segments(segments, chunk_start_sec)

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

                output = self._win.build_output(chunk_text, chunk_start_sec)

                emitted = False
                if output and output != self._last_emitted:
                    self._last_emitted = output
                    cycle_count += 1
                    emitted = True
                    # The committed prefix is a literal prefix of ``output``
                    # (see WindowState.build_output), so its length locates the
                    # frontier.
                    self.partial.emit(output, self._win.committed_prefix_len(output))

                if total_sec > self._win.commit_lag_sec and segments:
                    self._win.advance_commit(segments, chunk_start_sec, total_sec, sr)

                sleep_time = max(0.4, min(elapsed * 0.5, 2.0))
                logger.debug(
                    "cycle=%d  total=%.1fs  chunk=%.1fs  window=%s  "
                    "committed=%.1fs  transcribe=%.2fs  sleep=%.2fs  emit=%s",
                    cycle_count, total_sec, chunk_sec,
                    "YES" if windowed else "no",
                    self._win.committed_samples / sr, elapsed, sleep_time,
                    "YES" if emitted else "skip",
                )
                time.sleep(sleep_time)

        except Exception as exc:
            logger.error("Worker error: %s", exc, exc_info=True)
        finally:
            logger.info(
                "Worker finished in %.2fs, %d cycles emitted",
                time.time() - wall_start, cycle_count,
            )
            perf.log_summary("dictation perf")
            self.finished.emit()

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))

    def _run_final_pass(
        self, engine: AsrEngine, audio: np.ndarray, sr: int
    ) -> None:
        """Re-transcribe the whole recording at high beam and emit as authoritative.

        Bypasses the sliding window so text committed from the fast live pass
        (beam=2) is fully re-decoded with beam=5 + cross-segment context.
        """
        t0 = time.time()
        try:
            result = engine.transcribe(
                audio,
                TranscribeContext(
                    language=self.language,
                    vad_filter=self.vad_enabled,
                    beam_size=_FINAL_BEAM_SIZE,
                    pause_threshold=self.pause_threshold,
                    condition_on_previous_text=True,
                    initial_prompt=self._build_context_prompt(),
                ),
            )
            full_text, segments = result.text, result.segments_as_dicts()
        except Exception as exc:
            logger.warning("Final transcription pass failed: %s", exc)
            return
        full_text = full_text.strip()
        logger.info("final pass  dur=%.1fs  transcribe=%.2fs  chars=%d",
                    len(audio) / sr, time.time() - t0, len(full_text))
        if segments:
            self.segments.emit(
                [{"start": float(s.get("start", 0)),
                  "end": float(s.get("end", 0)),
                  "text": s.get("text", "")} for s in segments]
            )
        if full_text and full_text != self._last_emitted:
            self._last_emitted = full_text
            # committed_len=0: the final pass re-decodes everything, so no part
            # of this text is a carry-over of the frozen live prefix.
            self.partial.emit(full_text, 0)

    def _audio_length(self) -> Tuple[int, int]:
        """Return ``(total_frames, samplerate)`` from the header — no decode.

        ``(0, 0)`` while the file is missing or unreadable (the recorder may not
        have created it yet), which the run loop treats as "wait and retry".
        """
        try:
            info = sf.info(self.audio_path)
            return int(info.frames), int(info.samplerate)
        except (FileNotFoundError, RuntimeError):
            return 0, 0
        except Exception as exc:
            logger.warning("Could not stat audio %s: %s", self.audio_path, exc)
            return 0, 0

    def _read_audio(self, from_sample: int = 0) -> Tuple[Optional[np.ndarray], int]:
        """Read the growing WAV from *from_sample* onwards as mono float32.

        Reading the whole file every cycle re-decodes the entire recording each
        time — the cost grows with the session and dominates late in a long
        dictation. The live loop only needs the sliding window, so it seeks.

        Falls back to a full read if the seek fails: the header of a WAV still
        being written is not guaranteed to describe every frame on disk.
        """
        with perf.stage("worker.read_audio"):
            try:
                if from_sample > 0:
                    try:
                        with sf.SoundFile(self.audio_path) as handle:
                            handle.seek(from_sample)
                            audio = handle.read(dtype="float32")
                            sr = handle.samplerate
                        return self._to_mono(audio), sr
                    except (RuntimeError, ValueError) as exc:
                        logger.debug("Seek read failed (%s); full read", exc)

                audio, sr = sf.read(self.audio_path, dtype="float32")
                mono = self._to_mono(audio)
                # The caller labels the returned chunk as starting at
                # *from_sample* — slice the full read so segment timestamps
                # stay correct when the seek path failed.
                if from_sample > 0:
                    mono = mono[from_sample:]
                return mono, sr
            except FileNotFoundError:
                return None, 0
            except Exception as exc:
                logger.warning(
                    "Unexpected error reading audio %s: %s", self.audio_path, exc
                )
                return None, 0

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        return audio[:, 0] if audio.ndim > 1 else audio

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
