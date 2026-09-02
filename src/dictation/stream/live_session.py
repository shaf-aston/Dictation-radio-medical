"""Live dictation over a push-fed audio buffer: the front-end-agnostic loop.

The desktop's :class:`src.dictation.worker.LiveTranscribeWorker` runs the same
chunk rules, but it is a ``QObject`` polling a growing WAV file. A browser has
neither Qt nor a file, so the loop lives here owning only a buffer, and both
front-ends keep one set of rules.

Owns no thread, socket, signal, or clock: the caller decides when to call
:meth:`cycle`. That is what makes it drivable from an async handler and
testable with a numpy array and no I/O.

Two engines: a fast one for words that appear while you are still speaking, an
accurate one that re-decodes after Stop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.core import event_log
from src.dictation.asr import AsrEngine, TranscribeContext
from src.dictation.asr.types import AsrResult
from src.dictation.postprocess.incremental import IncrementalPostprocessor
from src.dictation.stream.ledger import ChunkLedger, close_sentence
from src.dictation.stream.rules import (
    AdaptiveFloor,
    low_confidence_words,
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
    is telling the radiologist that a provisional word is settled: so they are
    handed over separately and the UI shows the preview dimmed.
    """

    committed: str
    preview: str
    state: str
    audio_sec: float
    #: Words the decoder itself was unsure of, lower-cased. The front-end
    #: underlines them faintly in the committed text -- a report the machine
    #: half-guessed at should say so, rather than reading as settled.
    uncertain: Tuple[str, ...] = ()


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
        preview_min_tail_sec: float = 1.0,
        polish_confidence_ceiling: float = 0.85,
        uncertain_word_confidence: float = 0.6,
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
        self.preview_min_tail_sec = preview_min_tail_sec
        self.polish_confidence_ceiling = polish_confidence_ceiling
        self.uncertain_word_confidence = uncertain_word_confidence
        self.initial_prompt = initial_prompt
        self.sr = sr

        self._ledger = ChunkLedger(policy, pause_threshold=pause_threshold, sr=sr)
        self._agreement = LocalAgreement2()
        self._post = IncrementalPostprocessor(accent=accent, cleanup_level=cleanup_level)

        # Amortised-doubling buffer: appending a block is O(block), not O(total),
        # so a long dictation does not get slower the longer it runs: the same
        # rule the rest of this pipeline is built on.
        self._buf = np.zeros(sr * 60, dtype=np.float32)
        self._len = 0

        # What one preview decode actually costs on this machine, in wall
        # seconds. 0.0 means "not measured yet", which keeps the preview on for
        # the first cycles rather than guessing it is too slow.
        self._preview_cost = 0.0
        # Earliest wall-clock time the next preview may start. A preview costs
        # about the same however much audio it is given, so running one every
        # cycle spends far more than a second of machine per second of speech
        # and the loop falls behind the microphone. Holding off for as long as
        # the last one took caps previews at half the wall clock and leaves the
        # other half for the committed chunks, which are the text that is kept.
        self._preview_earliest = 0.0
        self._last_stable = ""
        self._last_update: Optional[LiveUpdate] = None
        self._chunks_decoded = 0
        # Keyed by committed-chunk index, so the polish pass can replace one
        # chunk's doubts along with its text instead of leaving marks behind
        # on words the accurate model has since settled.
        self._uncertain: dict = {}
        # The chunk closed by close_open_tail_fast(), if any. It was decoded by
        # the FAST model purely to get the report complete in time for Stop, so
        # finalize() must re-decode it whatever its confidence says.
        self._forced_polish_index: Optional[int] = None

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
    def committed_sec(self) -> float:
        """Audio time of the last permanent sample: the committed frontier.

        ``audio_sec`` minus this is how far the kept text trails the
        microphone, which is the number a radiologist is describing when they
        say dictation feels slow. Read by scripts/eval/replay.py.
        """
        return self._ledger.open_start_sample / self.sr

    @property
    def chunks_decoded(self) -> int:
        return self._chunks_decoded

    @property
    def uncertain_words(self) -> Tuple[str, ...]:
        """Every word still flagged low-confidence, across all committed chunks."""
        out: set = set()
        for words in self._uncertain.values():
            out |= words
        return tuple(sorted(out))

    def _note_uncertain(self, result: AsrResult, index: Optional[int] = None) -> None:
        """Record (or replace) one chunk's doubtful words."""
        at = len(self._ledger.committed) - 1 if index is None else index
        self._uncertain[at] = low_confidence_words(result, self.uncertain_word_confidence)

    def _audio(self, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        return self._buf[start : self._len if end is None else end]

    # -- the loop -------------------------------------------------------

    def cycle(self) -> Optional[LiveUpdate]:
        """Decode whatever is newly decodable. ``None`` if nothing changed.

        Takes one snapshot of the buffer and its length up front, and works from
        that alone. The caller runs this on a worker thread while more audio is
        still arriving, so re-reading ``self._len`` part-way through would cut
        chunks that run past the audio actually sliced: which numpy clamps
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

        # Two different reasons to skip the preview, and only one of them is
        # bad news. `behind` means a decode measured slower than the lag
        # ceiling: the machine genuinely cannot keep up, and the radiologist
        # should be told. `pacing` is the healthy half of the design, which
        # never starts a preview until as long has passed as the last one
        # took, so previews can never eat more than half the wall clock. That
        # fires on roughly half of all cycles, and reporting it as "catching
        # up" told the radiologist the app was behind at exactly the moments
        # it was working as intended.
        behind = should_skip_preview(self._preview_cost, self.preview_max_lag_sec)
        pacing = time.time() < self._preview_earliest
        if behind or pacing:
            # Cosmetic only: the last stable preview stays on screen and every
            # remaining second goes to the chunks that are actually kept. The
            # recorded cost decays while skipping so the preview comes back on
            # its own once the machine is free again -- a cost that is only
            # ever written when a preview runs would latch the preview off
            # permanently after one slow decode.
            self._preview_cost *= 0.9
            state = STATE_CATCHING_UP if behind else STATE_LIVE
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

        update = LiveUpdate(committed, preview, state, self.audio_sec, self.uncertain_words)
        if self._last_update is not None and (
            update.committed == self._last_update.committed
            and update.preview == self._last_update.preview
            and update.state == self._last_update.state
            and update.uncertain == self._last_update.uncertain
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
                # transcript of the wrong seconds: so stop instead and let the
                # next cycle cut it properly.
                logger.warning("Chunk runs past the sliced tail; deferring")
                break
            clip = tail_audio[chunk.start_sample - tail_start : stop]
            if self._noise_floor.is_silence(rms(clip)):
                # Genuinely silent (a long pause the segmenter force-cut):
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
            self._note_uncertain(result)
            self._chunks_decoded += 1
            committed_any = True
        return committed_any

    def _decode_open_tail(
        self, chunks: List[Chunk], tail_audio: np.ndarray, tail_start: int
    ) -> str:
        """Re-decode the still-open portion for a stable preview only.

        Never committed to the ledger: LocalAgreement-2 only shows the word
        prefix that agreed between this decode and the last one of the same
        open region, so the preview is stable even though the decode is not.
        """
        open_chunk = chunks[-1] if chunks and not chunks[-1].closed else None
        if open_chunk is None or open_chunk.length_samples <= 0:
            return self._agreement.update("")

        clip = tail_audio[
            open_chunk.start_sample - tail_start : open_chunk.end_sample - tail_start
        ]
        # Too little audio to be worth a decode yet. A decode costs the same
        # whatever it is given, so spending one on a fraction of a second buys
        # almost no words and pushes the first real preview a decode further
        # out. Keep whatever is already shown rather than clearing it.
        if len(clip) < self.preview_min_tail_sec * self.sr:
            return self._agreement.stable()
        if self._noise_floor.is_silence(rms(clip)):
            return self._agreement.update("")

        started = time.time()
        result = self._decode(
            self.live_engine, clip, self.live_beam_size,
            want_confidence=False, stage="live.preview",
        )
        # Smoothed, so one unlucky decode does not switch the preview off and
        # one lucky one does not switch it back on.
        cost = time.time() - started
        self._preview_cost = cost if self._preview_cost <= 0 else 0.6 * self._preview_cost + 0.4 * cost
        self._preview_earliest = time.time() + cost
        return self._agreement.update(result.text.strip() if result else "")

    # -- after Stop -----------------------------------------------------

    def committed_text(self) -> str:
        """The post-processed text decoded so far: usable the moment Stop is
        pressed, before :meth:`finalize` improves it."""
        raw = self._ledger.committed_text
        return self._post.process(raw, len(raw))[0] if raw else ""


    def close_open_tail_fast(self) -> None:
        """Decode whatever never closed into a chunk, using the FAST model.

        Called the moment Stop is pressed, before the report is handed back.
        Without it the hand-back is everything *except* the last chunk -- and
        the last chunk is where the impression lives. Measured on the synthetic
        set: 20 of 48 words handed back, the missing 28 arriving up to 25
        seconds later behind a status that already said "ready to edit". A
        radiologist could copy or export half a report and nothing would say so.

        One fast decode is about a second and a half on this machine, so this
        buys a complete report for a fraction of what the accurate pass costs.
        The chunk it commits is deliberately re-decoded by :meth:`finalize`
        regardless of confidence -- speed was the reason it was decoded by the
        fast model, so it has not earned the benefit of the doubt.
        """
        end = self._len
        tail = self._buf[self._ledger.open_start_sample:end]
        if not len(tail) or self._noise_floor.is_silence(rms(tail)):
            return
        result = self._decode(
            self.live_engine, tail, self.live_beam_size,
            want_confidence=True, stage="stop.tail", condition=True,
        )
        if result is None or not (text := result.text.strip()):
            return
        closing = Chunk(self._ledger.open_start_sample, end, closed=True)
        self._ledger.commit(closing, text, mean_confidence(result))
        self._note_uncertain(result)
        self._chunks_decoded += 1
        self._forced_polish_index = len(self._ledger.committed) - 1

    def finalize(self, on_progress: Optional[Callable[[str], None]] = None) -> str:
        """Re-decode what is worth re-decoding, with the accurate engine.

        Bounded cost, spent only where it buys something: committed chunks the
        live model was unsure about, plus whatever audio never closed into a
        chunk before Stop. Everything the fast model got confidently right is
        kept as-is, which is why this is seconds and not a full re-transcribe.
        """
        t0 = time.time()
        say = on_progress or (lambda _msg: None)

        targets = list(self._ledger.low_confidence_indices(self.polish_confidence_ceiling))
        # The tail closed at Stop was decoded fast on purpose; a confident fast
        # decode is still a fast decode, so it is re-done here either way.
        if self._forced_polish_index is not None and self._forced_polish_index not in targets:
            targets.append(self._forced_polish_index)
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
                self._note_uncertain(result, i)

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
                self._note_uncertain(result)
                self._chunks_decoded += 1

        raw = self._ledger.committed_text
        # committed_len=0: everything here just got an authoritative decode, so
        # no cached prefix from the live pass may survive into the final report.
        self._post.reset()
        final = close_sentence(self._post.process(raw, 0)[0]) if raw else ""
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
        """One transcribe call. ``None`` on failure: never raises at the caller.

        A failed decode must not end the dictation: the audio is still in the
        buffer and the chunk is still open, so the next cycle simply tries
        again. Losing a recording because one call threw is not a trade worth
        making.
        """
        try:
            # perf keeps the rolling average of this stage; event_log keeps the
            # individual call, which is what shows *which* decode blew out.
            with event_log.timed(
                "asr", f"{stage} decode", stage=f"stream.{stage}",
                clip_sec=round(len(clip) / self.sr, 1), beam=beam_size,
            ) as note:
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
                note["words"] = len(result.text.split())
        except Exception as exc:
            logger.warning("Decode failed (%s): %s", stage, exc)
            return None
        return result
