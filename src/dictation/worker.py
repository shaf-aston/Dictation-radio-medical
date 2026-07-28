"""Background worker for live transcription during recording.

Runs on a QThread, polling the growing WAV file and emitting the transcribed
text via Qt signals each cycle.

Chunk-once design
------------------
Superseded design: every cycle re-decoded a sliding window of the last few
seconds, so a long dictation cost several times its own duration in Whisper
decode time (the old ``window_state.py`` + ``text_diff.py``, both deleted —
their whole purpose was deduplicating overlapping re-decodes, which cannot
happen once chunks never overlap). This worker instead:

1. **VAD** (:mod:`src.dictation.stream.vad`) finds silence boundaries in the
   still-open tail only — never the whole growing recording, which would
   itself become O(n^2) over a long dictation.
2. **The segmenter** (:mod:`src.dictation.stream.segmenter`) turns those
   boundaries into chunk cuts: never shorter than the configured minimum,
   cut at the latest usable pause, force-cut only as a last resort.
3. **The ledger** (:mod:`src.dictation.stream.ledger`) decodes each closed
   chunk exactly once and freezes its text permanently.
4. Only the still-open tail — bounded by ``chunk_policy.force_cut_sec`` — is
   ever re-decoded, and only for a stable live preview via LocalAgreement-2
   (:mod:`src.dictation.stream.tail`); that preview is never committed.
5. **That preview is dropped when the machine cannot afford it** — see
   :func:`should_skip_preview`. On a CPU that decodes slower than speech, the
   preview re-decode of a 20-second open tail costs more than the closed chunks
   waiting behind it, so words arrived in late bursts. Skipping it spends every
   remaining second on the text that is actually kept. Because the preview is
   never committed, this changes only what is shown *early*; the committed and
   final text are byte-for-byte unaffected. It also lowers
   ``stream.decode_ratio`` (the preview is the bulk of the re-decode above 1.0x).

After recording stops there is no full re-transcribe. A confidence-targeted
polish (:meth:`LiveTranscribeWorker._run_confidence_targeted_polish`)
re-decodes only the committed chunks whose mean word confidence fell below
``polish_confidence_ceiling``, plus whatever audio was still open, at higher
beam width.

Beam widths and the confidence ceiling arrive as constructor arguments (the
caller reads them from settings — see :mod:`src.ui.recording_session`); this
worker never reads settings itself.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, Signal

from src.core import perf
from src.dictation.asr import (
    RADIOLOGY_PROMPT,
    AsrEngine,
    AsrResult,
    TranscribeContext,
    create_engine,
)
from src.dictation.stream.ledger import ChunkLedger
from src.dictation.stream.segmenter import Chunk, ChunkPolicy
from src.dictation.stream.tail import LocalAgreement2
from src.dictation.stream.vad import detect_speech
from src.features.adaptive_learning import get_custom_prompt_suffix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_MIN_AUDIO_SEC = 0.8     # ignore audio shorter than this
_MIN_GROWTH_SEC = 0.5    # min new audio before re-checking for a chunk cut

# Progress states the UI shows while the live loop runs.
_STATE_LIVE = "Live transcribing..."
_STATE_CATCHING_UP = "Catching up..."


def should_skip_preview(
    open_tail_sec: float, decode_cost: float, max_lag_sec: float
) -> bool:
    """Whether to drop this cycle's live preview decode.

    Two conditions, both required:

    * ``decode_cost`` — measured wall seconds spent decoding per second of
      audio decoded — is above 1.0, i.e. this machine decodes slower than
      speech arrives. A machine that keeps up has slack to spend on a preview
      and always keeps it.
    * the still-open tail is longer than *max_lag_sec*, so the preview
      re-decode of it is expensive and the words it shows are already stale.

    Requiring both is why a fast machine never loses the preview and a slow one
    only loses it once it is genuinely behind. ``max_lag_sec <= 0`` turns the
    skip off entirely; ``decode_cost`` of 0.0 means "not measured yet", which
    keeps the preview for the first cycles of a recording.
    """
    if max_lag_sec <= 0:
        return False
    return decode_cost > 1.0 and open_tail_sec > max_lag_sec


class LiveTranscribeWorker(QObject):
    """Decodes each closed chunk exactly once; the open tail is a stable preview only.

    ``partial`` carries the full display text (frozen chunk text + the
    stable LocalAgreement-2 prefix of the open tail) and the length of the
    frozen prefix within it. Consumers use the prefix length to skip
    re-processing text that this worker will never revise
    (:class:`~src.dictation.postprocess.incremental.IncrementalPostprocessor`).
    A value of ``0`` means "treat the whole text as revisable".
    """

    partial = Signal(str, int)
    finished = Signal()
    progress = Signal(str)
    # Absolute-timed segments for the current chunk, used by the cloud
    # training collector to locate corrections in audio. Each item:
    # {start, end, text} with timestamps offset to the full recording.
    segments = Signal(list)

    def __init__(
        self,
        audio_path: str,
        model_size: str,
        language: str,
        vad_enabled: bool,
        pause_threshold: float = 2.5,
        model_path: Optional[Union[str, Path]] = None,
        chunk_policy: Optional[ChunkPolicy] = None,
        silence_rms_floor: float = 0.002,
        live_beam_size: int = 2,
        final_beam_size: int = 5,
        polish_confidence_ceiling: float = 0.75,
        preview_max_lag_sec: float = 3.0,
    ) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.language = language
        # VAD is now load-bearing for chunk cutting (not just an engine-side
        # filter), so it stays on regardless of this flag; vad_enabled is
        # kept only to still gate faster-whisper's own internal VAD filter
        # inside each chunk decode.
        self.vad_enabled = vad_enabled
        self.pause_threshold = pause_threshold
        self.silence_rms_floor = max(0.0, float(silence_rms_floor))
        self.live_beam_size = max(1, int(live_beam_size))
        self.final_beam_size = max(1, int(final_beam_size))
        self.polish_confidence_ceiling = float(polish_confidence_ceiling)
        self.preview_max_lag_sec = float(preview_max_lag_sec)
        self.model_path = model_path
        self._ledger = ChunkLedger(
            chunk_policy or ChunkPolicy(), pause_threshold=pause_threshold
        )
        self._agreement = LocalAgreement2()
        self._keep_running = True
        self._final_requested = False
        self._last_emitted: str = ""
        self._prev_total_samples: int = 0
        # Every second of audio actually sent to engine.transcribe(), across
        # closed-chunk decodes, open-tail preview decodes, and the polish
        # pass. Divided by the recording's true length at the end to get
        # stream.decode_ratio — the number M2's exit criterion is judged on
        # (target: <= 1.4x, versus the old sliding window's ~8x).
        self._decode_sec_total: float = 0.0
        self._audio_sec_total: float = 0.0
        # Wall time actually spent inside engine.transcribe(). Against
        # _decode_sec_total this gives the measured cost of a second of audio on
        # this machine, which is what decides whether a preview is affordable.
        self._decode_wall_total: float = 0.0
        # The last stable preview shown. Re-shown while previews are being
        # skipped, so the display stalls instead of losing words it already
        # showed. Cleared whenever a chunk closes, since the committed text
        # then covers that audio.
        self._last_stable: str = ""
        self._progress_state: str = ""

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._keep_running = False

    def finalize(self) -> None:
        """Request one final confidence-targeted polish after recording stops."""
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
            self._emit_state(_STATE_LIVE)

            final_grace = 0
            while self._keep_running or self._final_requested:
                total_samples, sr = self._audio_length()
                if not sr or total_samples / sr < _MIN_AUDIO_SEC:
                    # The WAV stops growing once recording ends, so a finalize()
                    # on a too-short or unreadable file would spin this loop
                    # forever. Give the recorder a short grace to flush, then
                    # finish without a polish pass.
                    if self._final_requested:
                        final_grace += 1
                        if final_grace > 6:
                            logger.warning(
                                "Recording too short or unreadable (%s frames); "
                                "skipping final polish", total_samples,
                            )
                            break
                    time.sleep(0.5)
                    continue

                growth_sec = (total_samples - self._prev_total_samples) / sr
                if growth_sec < _MIN_GROWTH_SEC and not self._final_requested:
                    time.sleep(0.4)
                    continue

                if self._final_requested:
                    audio, sr = self._read_audio()
                    if audio is not None:
                        self._audio_sec_total = len(audio) / sr
                        self._run_confidence_targeted_polish(engine, audio, sr)
                    self._keep_running = False
                    self._final_requested = False
                    break

                self._audio_sec_total = total_samples / sr
                emitted, decode_sec = self._run_cycle(engine, total_samples, sr)
                self._prev_total_samples = total_samples
                if emitted:
                    cycle_count += 1

                sleep_time = max(0.4, min(decode_sec * 0.5, 2.0))
                time.sleep(sleep_time)

        except Exception as exc:
            logger.error("Worker error: %s", exc, exc_info=True)
        finally:
            if self._audio_sec_total > 0:
                perf.set_gauge(
                    "stream.decode_ratio", self._decode_sec_total / self._audio_sec_total
                )
            logger.info(
                "Worker finished in %.2fs, %d cycles emitted",
                time.time() - wall_start, cycle_count,
            )
            perf.log_summary("dictation perf")
            self.finished.emit()

    def _run_cycle(self, engine: AsrEngine, total_samples: int, sr: int) -> Tuple[bool, float]:
        """One poll cycle: cut+commit any newly-closed chunks, refresh the
        open-tail preview. Returns ``(emitted, decode_seconds)``."""
        t0 = time.time()
        tail_start = self._ledger.open_start_sample
        tail_audio, sr = self._read_audio(tail_start)
        if tail_audio is None or not len(tail_audio):
            return False, 0.0

        marks = detect_speech(tail_audio)
        chunks = self._ledger.pending_cuts(total_samples, marks)

        committed_any = False
        for chunk in chunks:
            if not chunk.closed:
                continue
            local = slice(chunk.start_sample - tail_start, chunk.end_sample - tail_start)
            chunk_audio = tail_audio[local]
            if self._rms(chunk_audio) < self.silence_rms_floor:
                # Genuinely silent (e.g. a long unspoken pause force-cut by
                # the segmenter) — nothing to decode, nothing to hallucinate.
                self._ledger.commit(chunk, "", None)
                committed_any = True
                continue
            decode_started = time.time()
            try:
                with perf.stage("worker.transcribe_chunk"):
                    result = engine.transcribe(
                        chunk_audio,
                        TranscribeContext(
                            language=self.language,
                            vad_filter=False,  # already cut at a VAD boundary
                            beam_size=self.live_beam_size,
                            pause_threshold=self.pause_threshold,
                            condition_on_previous_text=False,
                            initial_prompt=self._build_context_prompt(),
                            want_word_confidence=True,
                            temperature=0.0,
                        ),
                    )
            except Exception as exc:
                logger.warning("Chunk transcription failed: %s", exc)
                break  # retry this (and any later) chunk next cycle
            self._decode_wall_total += time.time() - decode_started
            self._decode_sec_total += len(chunk_audio) / sr
            self._emit_absolute_segments(result, chunk.start_sample, sr)
            self._ledger.commit(chunk, result.text, _mean_confidence(result))
            committed_any = True

        if committed_any:
            self._agreement.reset()
            self._last_stable = ""

        open_tail_sec = (total_samples - self._ledger.open_start_sample) / sr
        if should_skip_preview(
            open_tail_sec, self._decode_cost(), self.preview_max_lag_sec
        ):
            # Cosmetic only: the last stable preview stays on screen and every
            # remaining second goes to the chunks that are actually kept.
            self._emit_state(_STATE_CATCHING_UP)
            stable_tail = self._last_stable
        else:
            self._emit_state(_STATE_LIVE)
            stable_tail = self._decode_open_tail(
                engine, chunks, tail_audio, tail_start, sr
            )
            self._last_stable = stable_tail

        committed_text = self._ledger.committed_text
        output = committed_text
        if stable_tail:
            output = f"{output} {stable_tail}" if output else stable_tail

        emitted = False
        if output and output != self._last_emitted:
            self._last_emitted = output
            emitted = True
            self.partial.emit(output, len(committed_text))

        return emitted, time.time() - t0

    def _decode_cost(self) -> float:
        """Measured wall seconds of decoding per second of audio decoded.

        ``0.0`` until the first decode has been timed.
        """
        if self._decode_sec_total <= 0:
            return 0.0
        return self._decode_wall_total / self._decode_sec_total

    def _emit_state(self, state: str) -> None:
        """Emit a live-loop status only when it changes."""
        if state != self._progress_state:
            self._progress_state = state
            self.progress.emit(state)

    def _decode_open_tail(
        self, engine: AsrEngine, chunks: List[Chunk], tail_audio: np.ndarray,
        tail_start: int, sr: int,
    ) -> str:
        """Re-decode the still-open portion for a stable live preview only.

        Never committed to the ledger — LocalAgreement-2 (stream/tail.py)
        only shows the word-prefix that agreed between this decode and the
        last one of the same open region, so the preview is stable even
        though the underlying decode is provisional.
        """
        open_chunk = chunks[-1] if chunks and not chunks[-1].closed else None
        if open_chunk is None or open_chunk.length_samples <= 0:
            return self._agreement.update("")

        local = slice(open_chunk.start_sample - tail_start, open_chunk.end_sample - tail_start)
        open_audio = tail_audio[local]
        if self._rms(open_audio) < self.silence_rms_floor:
            return self._agreement.update("")

        decode_started = time.time()
        try:
            with perf.stage("worker.transcribe_live"):
                result = engine.transcribe(
                    open_audio,
                    TranscribeContext(
                        language=self.language,
                        vad_filter=False,
                        beam_size=self.live_beam_size,
                        pause_threshold=self.pause_threshold,
                        condition_on_previous_text=False,
                        initial_prompt=self._build_context_prompt(),
                        temperature=0.0,
                    ),
                )
        except Exception as exc:
            logger.warning("Live preview transcription failed: %s", exc)
            return self._agreement.update("")
        self._decode_wall_total += time.time() - decode_started
        self._decode_sec_total += len(open_audio) / sr
        return self._agreement.update(result.text.strip())

    # ------------------------------------------------------------------
    # Confidence-targeted polish (replaces the old full-WAV final pass)
    # ------------------------------------------------------------------

    def _run_confidence_targeted_polish(
        self, engine: AsrEngine, audio: np.ndarray, sr: int
    ) -> None:
        """Re-decode only what's worth re-decoding, at higher beam width.

        Bounded cost, spent only where it buys something: committed chunks
        whose mean word confidence fell below the ceiling, plus whatever
        audio never made it into a closed chunk before recording stopped.
        """
        t0 = time.time()
        prompt = self._build_context_prompt()

        # This pass can take several seconds per chunk, and it runs after the
        # radiologist has already pressed Stop — say what it is doing rather
        # than leaving the window looking hung.
        targets = list(self._ledger.low_confidence_indices(self.polish_confidence_ceiling))
        for done, i in enumerate(targets, start=1):
            self.progress.emit(f"Polishing chunk {done}/{len(targets)}...")
            c = self._ledger.committed[i]
            clip = audio[c.start_sample:c.end_sample]
            if not len(clip):
                continue
            try:
                result = engine.transcribe(
                    clip,
                    TranscribeContext(
                        language=self.language,
                        vad_filter=self.vad_enabled,
                        beam_size=self.final_beam_size,
                        pause_threshold=self.pause_threshold,
                        condition_on_previous_text=True,
                        initial_prompt=prompt,
                        want_word_confidence=True,
                    ),
                )
            except Exception as exc:
                logger.warning("Confidence-targeted polish failed for chunk %d: %s", i, exc)
                continue
            self._decode_sec_total += len(clip) / sr
            self._ledger.replace(i, result.text, _mean_confidence(result))

        # Whatever never closed before the recording stopped gets its only
        # decode here, at final quality.
        tail = audio[self._ledger.open_start_sample:]
        if len(tail) and self._rms(tail) >= self.silence_rms_floor:
            self.progress.emit("Polishing final section...")
            try:
                result = engine.transcribe(
                    tail,
                    TranscribeContext(
                        language=self.language,
                        vad_filter=self.vad_enabled,
                        beam_size=self.final_beam_size,
                        pause_threshold=self.pause_threshold,
                        condition_on_previous_text=True,
                        initial_prompt=prompt,
                        want_word_confidence=True,
                    ),
                )
            except Exception as exc:
                logger.warning("Final open-tail transcription failed: %s", exc)
                result = None
            if result is not None:
                self._decode_sec_total += len(tail) / sr
                text = result.text.strip()
                if text:
                    self._emit_absolute_segments(result, self._ledger.open_start_sample, sr)
                    closing = Chunk(self._ledger.open_start_sample, len(audio), closed=True)
                    self._ledger.commit(closing, text, _mean_confidence(result))

        full_text = self._ledger.committed_text
        logger.info(
            "confidence-targeted polish  dur=%.1fs  elapsed=%.2fs  chars=%d",
            len(audio) / sr, time.time() - t0, len(full_text),
        )
        if full_text and full_text != self._last_emitted:
            self._last_emitted = full_text
            # committed_len=0: everything here just got a final, authoritative
            # decode, so nothing is a stale carry-over of a live-pass guess.
            self.partial.emit(full_text, 0)

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))

    def _emit_absolute_segments(self, result: AsrResult, chunk_start_sample: int, sr: int) -> None:
        if not result.segments:
            return
        chunk_start_sec = chunk_start_sample / sr
        self.segments.emit([
            {
                "start": chunk_start_sec + seg.start,
                "end": chunk_start_sec + seg.end,
                "text": seg.text,
            }
            for seg in result.segments
        ])

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

        Reading the whole file every cycle re-decodes the entire recording
        each time; this worker only ever needs the still-open tail, so it
        seeks.

        Falls back to a full read if the seek fails: the header of a WAV
        still being written is not guaranteed to describe every frame on disk.
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
        Whisper to echo prior words back into the current chunk under the
        small live beam. Chunks never overlap in this design, so there is no
        boundary-dedup step to lean on instead; the prompt just stays fixed.

        **Order is priority.** Whisper's prompt slot holds 223 tokens and it keeps
        the LAST 223, discarding the front without a word. The curated radiology
        vocabulary is sized to fit that slot on its own, so it goes last and
        always survives. The learned terms go in front, where they fill whatever
        room is left and are the ones dropped when there is none. Putting them
        last instead — as this did — let a full custom vocabulary (capped at 80
        terms, about 216 tokens) push almost the entire shipped dictionary out of
        the decoder, which is invisible from the outside.
        """
        if custom_terms := get_custom_prompt_suffix():
            return f"{custom_terms} {RADIOLOGY_PROMPT}"
        else:
            return RADIOLOGY_PROMPT


def _mean_confidence(result: AsrResult) -> Optional[float]:
    """Mean word confidence across every segment that reported one.

    ``None`` when the engine gave no word timestamps for this call — distinct
    from 0.0 so the ledger's confidence gate never mistakes "no signal" for
    "the model was certain this is wrong".
    """
    vals = [c for seg in result.segments if (c := seg.confidence) is not None]
    return sum(vals) / len(vals) if vals else None
