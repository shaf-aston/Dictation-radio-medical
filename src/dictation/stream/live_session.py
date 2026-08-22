"""Live dictation over a push-fed audio buffer — the front-end-agnostic loop.

The desktop's :class:`src.dictation.worker.LiveTranscribeWorker` runs the same
chunk rules, but it is a ``QObject`` polling a growing WAV file. A browser has
neither Qt nor a file, so the loop lives here owning only a buffer, and both
front-ends keep one set of rules.

Owns no thread, socket, signal, or clock — the caller decides when to call
:meth:`cycle`. That is what makes it drivable from an async handler and
testable with a numpy array and no I/O.

Two engines: a fast one for words that appear while you are still speaking, an
accurate one that re-decodes after Stop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from src.core import perf
from src.dictation.asr import AsrEngine, TranscribeContext
from src.dictation.asr.types import AsrResult
from src.dictation.postprocess.incremental import IncrementalPostprocessor
from src.dictation.stream.ledger import ChunkLedger
from src.dictation.stream.rules import (
    AdaptiveFloor,
    mean_confidence,
    rms,
    should_skip_preview,
)
from src.dictation.stream.segmenter import Chunk, ChunkPolicy
from src.dictation.stream.tail import LocalAgreement2
from src.dictation.stream.vad import SAMPLE_RATE, detect_speech

logger = logging.getLogger(__name__)

# What the caller may show the person dictating. Not free text: the two
# front-ends must not invent their own wording for the same state.
STATE_LIVE = "live"
STATE_CATCHING_UP = "catching_up"


@dataclass(frozen=True)
class LiveUpdate:
    """One cycle's view of the transcript.

    Two fields, and the difference between them is the whole point: *committed*
    is decoded-once-and-frozen and post-processed, and *preview* is a guess
    about audio still being spoken. A front-end that renders them identically
    is telling the radiologist that a provisional word is settled — so they are
    handed over separately and the UI shows the preview dimmed.
    """

    committed: str
    preview: str
    state: str
    audio_sec: float


class LiveSession:
    """A single dictation, fed audio in blocks, yielding transcript updates."""

    def __init__(
        self,
        live_engine: AsrEngine,
        final_engine: AsrEngine,
        *,
        language: str = "en",
        accent: str = "neutral",
        cleanup_level: str = "medium",
        policy: ChunkPolicy = ChunkPolicy(),
        pause_threshold: float = 2.5,
        live_beam_size: int = 2,
        final_beam_size: int = 5,
        silence_rms_floor: float = 0.0005,
        silence_rms_margin: float = 2.5,
        preview_max_lag_sec: float = 3.0,
        polish_confidence_ceiling: float = 0.85,
        initial_prompt: str = "",
        sr: int = SAMPLE_RATE,
    ) -> None:
        self.live_engine = live_engine
        self.final_engine = final_engine
        self.language = language
        self.pause_threshold = pause_threshold
        self.live_beam_size = live_beam_size
        self.final_beam_size = final_beam_size
        self._noise_floor = AdaptiveFloor(silence_rms_floor, silence_rms_margin)
        self.preview_max_lag_sec = preview_max_lag_sec
        self.polish_confidence_ceiling = polish_confidence_ceiling
        self.initial_prompt = initial_prompt
        self.sr = sr

        self._ledger = ChunkLedger(policy, pause_threshold=pause_threshold, sr=sr)
        self._agreement = LocalAgreement2()
        self._post = IncrementalPostprocessor(accent=accent, cleanup_level=cleanup_level)

        # Amortised-doubling buffer: appending a block is O(block), not O(total),
        # so a long dictation does not get slower the longer it runs — the same
        # rule the rest of this pipeline is built on.
        self._buf = np.zeros(sr * 60, dtype=np.float32)
        self._len = 0

        self._decode_wall_total = 0.0
        self._decode_sec_total = 0.0
        self._last_stable = ""
        self._last_update: Optional[LiveUpdate] = None
        self._chunks_decoded = 0

    # -- audio in -------------------------------------------------------

    def feed(self, block: np.ndarray) -> None:
        """Append one block of mono float32 audio at :attr:`sr`."""
        if block.dtype != np.float32:
            block = block.astype(np.float32)
        needed = self._len + len(block)
        if needed > len(self._buf):
            grown = np.zeros(max(needed, len(self._buf) * 2), dtype=np.float32)
            grown[: self._len] = self._buf[: self._len]
            self._buf = grown
        self._buf[self._len : needed] = block
        self._len = needed

    @property
    def total_samples(self) -> int:
        return self._len

    @property
    def audio_sec(self) -> float:
        return self._len / self.sr

    @property
    def chunks_decoded(self) -> int:
        return self._chunks_decoded

    def _audio(self, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        return self._buf[start : self._len if end is None else end]

    # -- the loop -------------------------------------------------------

    def cycle(self) -> Optional[LiveUpdate]:
        """Decode whatever is newly decodable. ``None`` if nothing changed.

        Takes one snapshot of the buffer and its length up front, and works from
        that alone. The caller runs this on a worker thread while more audio is
        still arriving, so re-reading ``self._len`` part-way through would cut
        chunks that run past the audio actually sliced — which numpy clamps
        silently, making two chunks cover the same seconds and the transcript
        repeat itself. :meth:`feed` only ever appends beyond the snapshot, so a
        snapshot stays valid for the whole cycle even if the buffer is replaced.
        """
        buf, total = self._buf, self._len
        tail_start = self._ledger.open_start_sample
        tail_audio = buf[tail_start:total]
        if not len(tail_audio):
            return None

        marks = detect_speech(tail_audio)
        chunks = self._ledger.pending_cuts(total, marks)

        committed_any = self._commit_closed(chunks, tail_audio, tail_start)
        if committed_any:
            self._agreement.reset()
            self._last_stable = ""

        open_tail_sec = (total - self._ledger.open_start_sample) / self.sr
        if should_skip_preview(open_tail_sec, self._decode_cost(), self.preview_max_lag_sec):
            # Cosmetic only: the last stable preview stays on screen and every
            # remaining second goes to the chunks that are actually kept.
            state = STATE_CATCHING_UP
            preview = self._last_stable
        else:
            state = STATE_LIVE
            preview = self._decode_open_tail(chunks, tail_audio, tail_start)
            self._last_stable = preview

        committed_raw = self._ledger.committed_text
        # Only the frozen half is post-processed. The preview is a guess about
        # words still being spoken; running the correction pipeline over it
        # would make finished words visibly change their minds.
        committed = self._post.process(committed_raw, len(committed_raw))[0] if committed_raw else ""

        update = LiveUpdate(committed, preview, state, self.audio_sec)
        if self._last_update is not None and (
            update.committed == self._last_update.committed
            and update.preview == self._last_update.preview
            and update.state == self._last_update.state
        ):
            return None
        self._last_update = update
        return update

    def _commit_closed(
        self, chunks: List[Chunk], tail_audio: np.ndarray, tail_start: int
    ) -> bool:
        committed_any = False
        for chunk in chunks:
            if not chunk.closed:
                continue
            stop = chunk.end_sample - tail_start
            if stop > len(tail_audio):
                # Would decode audio this cycle never sliced. numpy would just
                # clamp and hand back a short clip, which reads as a plausible
                # transcript of the wrong seconds — so stop instead and let the
                # next cycle cut it properly.
                logger.warning("Chunk runs past the sliced tail; deferring")
                break
            clip = tail_audio[chunk.start_sample - tail_start : stop]
            if self._noise_floor.is_silence(rms(clip)):
                # Genuinely silent (a long pause the segmenter force-cut) —
                # nothing to decode, nothing to hallucinate.
                self._ledger.commit(chunk, "", None)
                committed_any = True
                continue
            result = self._decode(
                self.live_engine, clip, self.live_beam_size,
                want_confidence=True, stage="live.chunk",
            )
            if result is None:
                break  # retry this (and any later) chunk next cycle
            self._ledger.commit(chunk, result.text, mean_confidence(result))
            self._chunks_decoded += 1
            committed_any = True
        return committed_any

    def _decode_open_tail(
        self, chunks: List[Chunk], tail_audio: np.ndarray, tail_start: int
    ) -> str:
        """Re-decode the still-open portion for a stable preview only.

        Never committed to the ledger — LocalAgreement-2 only shows the word
        prefix that agreed between this decode and the last one of the same
        open region, so the preview is stable even though the decode is not.
        """
        open_chunk = chunks[-1] if chunks and not chunks[-1].closed else None
        if open_chunk is None or open_chunk.length_samples <= 0:
            return self._agreement.update("")

        clip = tail_audio[
            open_chunk.start_sample - tail_start : open_chunk.end_sample - tail_start
        ]
        if self._noise_floor.is_silence(rms(clip)):
            return self._agreement.update("")

        result = self._decode(
            self.live_engine, clip, self.live_beam_size,
            want_confidence=False, stage="live.preview",
        )
        return self._agreement.update(result.text.strip() if result else "")

    # -- after Stop -----------------------------------------------------

    def committed_text(self) -> str:
        """The post-processed text decoded so far — usable the moment Stop is
        pressed, before :meth:`finalize` improves it."""
        raw = self._ledger.committed_text
        return self._post.process(raw, len(raw))[0] if raw else ""


    def finalize(self, on_progress: Optional[Callable[[str], None]] = None) -> str:
        """Re-decode what is worth re-decoding, with the accurate engine.

        Bounded cost, spent only where it buys something: committed chunks the
        live model was unsure about, plus whatever audio never closed into a
        chunk before Stop. Everything the fast model got confidently right is
        kept as-is, which is why this is seconds and not a full re-transcribe.
        """
        t0 = time.time()
        say = on_progress or (lambda _msg: None)

        targets = self._ledger.low_confidence_indices(self.polish_confidence_ceiling)
        for done, i in enumerate(targets, start=1):
            say(f"Improving section {done} of {len(targets)}")
            c = self._ledger.committed[i]
            clip = self._audio(c.start_sample, c.end_sample)
            if not len(clip):
                continue
            result = self._decode(
                self.final_engine, clip, self.final_beam_size,
                want_confidence=True, stage="final.polish", condition=True,
            )
            if result is not None:
                self._ledger.replace(i, result.text, mean_confidence(result))

        end = self._len
        tail = self._buf[self._ledger.open_start_sample:end]
        if len(tail) and not self._noise_floor.is_silence(rms(tail)):
            say("Improving the last section")
            result = self._decode(
                self.final_engine, tail, self.final_beam_size,
                want_confidence=True, stage="final.tail", condition=True,
            )
            if result is not None and (text := result.text.strip()):
                closing = Chunk(self._ledger.open_start_sample, end, closed=True)
                self._ledger.commit(closing, text, mean_confidence(result))
                self._chunks_decoded += 1

        raw = self._ledger.committed_text
        # committed_len=0: everything here just got an authoritative decode, so
        # no cached prefix from the live pass may survive into the final report.
        self._post.reset()
        final = self._post.process(raw, 0)[0] if raw else ""
        logger.info(
            "final polish  audio=%.1fs  elapsed=%.2fs  polished=%d  chars=%d",
            self.audio_sec, time.time() - t0, len(targets), len(final),
        )
        return final

    # -- internals ------------------------------------------------------

    def _decode(
        self,
        engine: AsrEngine,
        clip: np.ndarray,
        beam_size: int,
        *,
        want_confidence: bool,
        stage: str,
        condition: bool = False,
    ) -> Optional[AsrResult]:
        """One transcribe call. ``None`` on failure — never raises at the caller.

        A failed decode must not end the dictation: the audio is still in the
        buffer and the chunk is still open, so the next cycle simply tries
        again. Losing a recording because one call threw is not a trade worth
        making.
        """
        started = time.time()
        try:
            with perf.stage(f"stream.{stage}"):
                result = engine.transcribe(
                    clip,
                    TranscribeContext(
                        language=self.language,
                        vad_filter=False,  # already cut at a VAD boundary
                        beam_size=beam_size,
                        pause_threshold=self.pause_threshold,
                        condition_on_previous_text=condition,
                        initial_prompt=self.initial_prompt,
                        want_word_confidence=want_confidence,
                        temperature=0.0,
                    ),
                )
        except Exception as exc:
            logger.warning("Decode failed (%s): %s", stage, exc)
            return None
        self._decode_wall_total += time.time() - started
        self._decode_sec_total += len(clip) / self.sr
        return result

    def _decode_cost(self) -> float:
        """Measured wall seconds of decoding per second of audio decoded."""
        if self._decode_sec_total <= 0:
            return 0.0
        return self._decode_wall_total / self._decode_sec_total
