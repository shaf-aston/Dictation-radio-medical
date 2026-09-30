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
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.core import event_log
from src.dictation.asr import AsrEngine, TranscribeContext
from src.dictation.asr.port import AsrStream, StreamingAsrEngine, engine_identity
from src.dictation.asr.types import AsrResult, StreamError, StreamFinal, StreamInterim
from src.dictation.postprocess.incremental import IncrementalPostprocessor
from src.dictation.stream.ledger import ChunkLedger, close_sentence
from src.dictation.stream.polish import decode_run, polish, weak_runs
from src.dictation.stream.rules import (
    low_confidence_words,
    mean_confidence,
    has_speech,
    level_db,
    should_skip_preview,
)
from src.dictation.stream.segmenter import Chunk, ChunkPolicy
from src.dictation.stream.tail import LocalAgreement2
from src.dictation.stream.vad import SAMPLE_RATE, SpeechMark, detect_speech

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
        preview_max_lag_sec: float = 3.0,
        preview_min_tail_sec: float = 1.0,
        polish_confidence_ceiling: float = 0.85,
        uncertain_word_confidence: float = 0.6,
        initial_prompt: str = "",
        sr: int = SAMPLE_RATE,
        postprocess: bool = True,
        polish_vad_filter: bool = False,
        on_decoded: Optional[Callable[[AsrResult, int], None]] = None,
        streaming: bool = True,
        background_polish: bool = True,
    ) -> None:
        self.live_engine = live_engine
        self.final_engine = final_engine
        self.language = language
        self.pause_threshold = pause_threshold
        self.live_beam_size = live_beam_size
        self.final_beam_size = final_beam_size
        self.preview_max_lag_sec = preview_max_lag_sec
        self.preview_min_tail_sec = preview_min_tail_sec
        self.polish_confidence_ceiling = polish_confidence_ceiling
        self.uncertain_word_confidence = uncertain_word_confidence
        self.initial_prompt = initial_prompt
        self.sr = sr
        # False: every text this session hands out is the ledger's raw text.
        # The desktop runs the correction pipeline on its own worker thread
        # (ui/postprocess_worker.py) and must not get it twice.
        self.postprocess = postprocess
        # The polish's own VAD trims the pause each chunk begins with, which is
        # real work there (dictation-speed-review.md); the desktop keeps it on.
        self.polish_vad_filter = polish_vad_filter
        # Called with every decode that becomes committed text, and the absolute
        # sample it starts at: the training collector's segment timings.
        self._on_decoded = on_decoded
        # Seconds of audio handed to an engine, all stages: over the recording's
        # length this is stream.decode_ratio.
        self.decode_audio_sec = 0.0

        # Streaming mode: when the live engine can take audio as it arrives
        # (EngineCaps.streaming, Deepgram's live socket), the engine itself
        # settles text at each endpoint and sends interim guesses, so there is
        # no VAD, no chunk cutting and no re-decoded preview on this side:
        # each settled result is committed straight into the same ledger.
        # "untried" until the first cycle opens it (never in the constructor,
        # which the web app calls on its event loop); "off" for good once it
        # fails, and the chunked loop carries on from the ledger's frontier.
        self._stream_state = "untried" if streaming else "off"
        self._stream: Optional[AsrStream] = None
        self._stream_origin = 0   # absolute sample of the stream's time zero
        self._stream_sent = 0     # absolute samples pushed to the stream so far
        # feed() runs on the caller's thread (the web app's event loop) and the
        # stream opens on the cycle thread: both append-and-push under this.
        self._push_lock = threading.Lock()

        # The background polish: the accurate engine re-decodes weak chunks
        # WHILE the radiologist is still dictating, a chunk or more behind the
        # frontier, so Stop has little or nothing left to improve. It runs on
        # its own thread and only decodes; cycle() applies what it returns, so
        # the ledger keeps one writer. One run at a time, and never the newest
        # committed chunk (its neighbour may still join its run).
        self.background_polish = background_polish
        self._bg_lock = threading.Lock()
        self._bg_thread: Optional[threading.Thread] = None
        self._bg_results: List[Tuple[int, AsrResult]] = []
        self._bg_claimed: set = set()   # handed to the thread (maybe not back yet)
        self._bg_applied: set = set()   # re-decoded by the accurate engine, frozen

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
        self._chunks_polished = 0
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
        with self._push_lock:
            needed = self._len + len(block)
            if needed > len(self._buf):
                grown = np.zeros(max(needed, len(self._buf) * 2), dtype=np.float32)
                grown[: self._len] = self._buf[: self._len]
                self._buf = grown
            self._buf[self._len : needed] = block
            self._len = needed
            if self._stream is not None:
                self._push_unsent()

    def _push_unsent(self) -> None:
        """Send the stream every sample it has not had yet. Hold _push_lock."""
        stream = self._stream
        pcm = self._buf[self._stream_sent : self._len]
        if stream is not None and len(pcm):
            self._stream_sent = self._len
            stream.push((np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())

    @property
    def streaming(self) -> bool:
        """True while the live engine's own socket is doing the live work."""
        return self._stream is not None

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
    def chunks_polished(self) -> int:
        """How many committed chunks the last :meth:`finalize` re-decoded."""
        return self._chunks_polished

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
        if self._stream_state == "untried":
            self._open_stream()
        if self._stream is not None:
            return self._cycle_streaming()

        buf, total = self._buf, self._len
        tail_start = self._ledger.open_start_sample
        tail_audio = buf[tail_start:total]
        if not len(tail_audio):
            return None

        marks = detect_speech(tail_audio)
        chunks = self._ledger.pending_cuts(total, marks)

        committed_any = self._commit_closed(chunks, marks, tail_audio, tail_start)
        if committed_any:
            self._agreement.reset()
            self._last_stable = ""
        self._apply_background()
        self._start_background()

        # Three reasons to skip the preview, and only one of them is bad
        # news. `committed_any` means a chunk just froze, so a preview now
        # would decode audio the committed text already covers. `behind`
        # means a decode measured slower than the lag ceiling: the machine
        # genuinely cannot keep up, and the radiologist should be told.
        # `pacing` is the healthy half of the design, which never starts a
        # preview until as long has passed as the last one took, so previews
        # can never eat more than half the wall clock. That fires on roughly
        # half of all cycles, and reporting it as "catching up" told the
        # radiologist the app was behind at exactly the moments it was
        # working as intended.
        behind = should_skip_preview(self._preview_cost, self.preview_max_lag_sec)
        pacing = time.time() < self._preview_earliest
        if committed_any or behind or pacing:
            # Cosmetic only: the last stable preview stays on screen and every
            # remaining second goes to the chunks that are actually kept. The
            # recorded cost decays on the load-related skips so the preview comes
            # back on its own once the machine is free again -- a cost that is
            # only ever written when a preview runs would latch the preview off
            # permanently after one slow decode. A commit-cycle skip says nothing
            # about load, so it leaves the cost alone: decaying it there would
            # make `behind` slow to admit the machine is struggling.
            if behind or pacing:
                self._preview_cost *= 0.9
            state = STATE_CATCHING_UP if behind else STATE_LIVE
            preview = self._last_stable
        else:
            state = STATE_LIVE
            preview = self._decode_open_tail(chunks, marks, tail_audio, tail_start)
            self._last_stable = preview

        return self._update(preview, state)

    def _update(self, preview: str, state: str) -> Optional[LiveUpdate]:
        """The update to show, or ``None`` when it is what was shown last."""
        # Only the frozen half is post-processed. The preview is a guess about
        # words still being spoken; running the correction pipeline over it
        # would make finished words visibly change their minds.
        committed = self.committed_text()

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

    # -- streaming mode -------------------------------------------------

    def _open_stream(self) -> None:
        """Try the live engine's socket once; on any failure, chunks it is."""
        self._stream_state = "off"
        try:
            engine = self.live_engine
            if not engine.capabilities().streaming or not isinstance(engine, StreamingAsrEngine):
                return
            with event_log.timed("asr", "live stream open", stage="stream.open"):
                stream = engine.open_stream(self._context(
                    self.live_beam_size, want_confidence=True, condition=False,
                ))
        except Exception as exc:
            logger.warning("Live stream unavailable (%s); decoding chunks instead", exc)
            event_log.emit("live", "stream unavailable, decoding chunks", error=str(exc))
            return
        with self._push_lock:
            self._stream = stream
            self._stream_origin = self._stream_sent = self._ledger.open_start_sample
            self._push_unsent()
        self._stream_state = "open"
        event_log.emit("live", "streaming from the engine's live socket")

    def _cycle_streaming(self) -> Optional[LiveUpdate]:
        assert self._stream is not None
        preview = self._last_stable
        for event in self._stream.poll():
            if isinstance(event, StreamFinal):
                self._commit_final(event)
                preview = self._last_stable = ""
            elif isinstance(event, StreamInterim):
                # Interims are revised as more audio arrives; the same
                # never-take-a-word-back rule as the chunked preview applies.
                preview = self._last_stable = self._agreement.update(event.text)
            elif isinstance(event, StreamError):
                self._drop_stream(event.message)
                break
        return self._update(preview, STATE_LIVE)

    def _commit_final(self, final: StreamFinal) -> None:
        start = self._ledger.open_start_sample
        end = self._stream_origin + int(round(final.end * self.sr))
        # Finals tile the stream, so `end` only ever moves forward; clamped so
        # a server's rounding can never commit audio this session never had.
        end = min(max(end, start), self._len)
        result = final.as_result()
        self._ledger.commit(Chunk(start, end, closed=True), final.text, mean_confidence(result))
        self._note_uncertain(result)
        self.decode_audio_sec += (end - start) / self.sr
        self._agreement.reset()
        if final.text:
            self._chunks_decoded += 1
            self._decoded(result, start)

    def _drop_stream(self, why: str) -> None:
        """The socket is gone: carry on decoding chunks from the frontier."""
        logger.warning("Live stream dropped (%s); decoding chunks from %.1fs",
                       why, self.committed_sec)
        event_log.emit("live", "stream dropped, decoding chunks", error=why)
        self.close()
        self._agreement.reset()
        self._last_stable = ""

    def close(self) -> None:
        """Close the live socket, if one is open. Safe to call more than once."""
        with self._push_lock:
            stream, self._stream = self._stream, None
            self._stream_state = "off"
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                logger.debug("Closing the live stream failed: %s", exc)

    # -- background polish --------------------------------------------------

    def _polish_worthwhile(self) -> bool:
        """False when the accurate engine would only repeat the live one."""
        same = engine_identity(self.final_engine) == engine_identity(self.live_engine)
        return not (same and _ignores_beam(self.final_engine))

    def _start_background(self) -> None:
        if not self.background_polish or self._stream is not None:
            return
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        newest = len(self._ledger.committed) - 1
        runs = [
            run for run in weak_runs(self._ledger, self.polish_confidence_ceiling, self._bg_claimed)
            if run[-1] < newest
        ]
        if not runs or not self._polish_worthwhile():
            return
        run = runs[0]
        # Claimed now, so the next cycle cannot start the same run again.
        self._bg_claimed.update(run)
        # A view, not a copy, and safe to read from another thread: feed() only
        # ever writes past the current length, or into a new, larger array
        # (this view keeps the old one alive), never over committed samples.
        audio = self._buf[: self._ledger.committed[run[-1]].end_sample]
        ledger = self._ledger

        def work() -> None:
            results = decode_run(
                ledger, audio,
                lambda clip, stage: self._decode(
                    self.final_engine, clip, self.final_beam_size,
                    want_confidence=True, stage=stage, condition=True,
                    vad_filter=self.polish_vad_filter,
                ),
                run,
            )
            with self._bg_lock:
                self._bg_results.extend(results)

        self._bg_thread = threading.Thread(target=work, name="background-polish", daemon=True)
        self._bg_thread.start()

    def _apply_background(self) -> int:
        """Freeze whatever the background polish has finished. Cycle thread only."""
        with self._bg_lock:
            results, self._bg_results = self._bg_results, []
        for index, result in results:
            self._ledger.replace(index, result.text, mean_confidence(result))
            self._note_uncertain(result, index)
            self._bg_applied.add(index)
        if results:
            event_log.emit("live", "background polish applied", chunks=len(results))
        return len(results)

    def _finish_background(self, cancelled: Callable[[], bool]) -> None:
        """At Stop: let a run already being decoded land, rather than redo it.

        Afterwards only chunks actually APPLIED count as done: a run whose
        decode failed is still covered by the polish after Stop.
        """
        thread = self._bg_thread
        while thread is not None and thread.is_alive() and not cancelled():
            thread.join(timeout=0.1)
        self._apply_background()

    def _commit_closed(
        self, chunks: List[Chunk], marks: List[SpeechMark],
        tail_audio: np.ndarray, tail_start: int,
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
            if not has_speech(marks, chunk.start_sample - tail_start, stop):
                # The VAD heard nothing (a long pause the segmenter force-cut):
                # nothing to decode, nothing to hallucinate. Said out loud with
                # the loudness, so a dictation that came back empty can be told
                # apart from one that was too quiet for the VAD.
                event_log.emit(
                    "live", "chunk skipped, no speech heard",
                    clip_sec=round(len(clip) / self.sr, 1), level_db=level_db(clip),
                )
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
            self._decoded(result, chunk.start_sample)
            self._chunks_decoded += 1
            committed_any = True
        return committed_any

    def _decode_open_tail(
        self, chunks: List[Chunk], marks: List[SpeechMark],
        tail_audio: np.ndarray, tail_start: int,
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
        if not has_speech(marks, open_chunk.start_sample - tail_start, open_chunk.end_sample - tail_start):
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
        pressed, before :meth:`finalize` improves it. Raw when this session
        was built with ``postprocess=False``."""
        raw = self._ledger.committed_text
        if not self.postprocess:
            return raw
        return self._post.process(raw, len(raw))[0] if raw else ""

    @property
    def committed_raw(self) -> str:
        """The ledger's text exactly as decoded, before any correction."""
        return self._ledger.committed_text

    def _decoded(self, result: AsrResult, start_sample: int) -> None:
        if self._on_decoded is not None:
            try:
                self._on_decoded(result, start_sample)
            except Exception as exc:  # a listener must never end the dictation
                logger.debug("on_decoded listener failed: %s", exc)


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
        if self._stream is not None:
            # Ask the socket to settle everything it has been sent. Measured in
            # a few hundred milliseconds when it is healthy; if it is not, the
            # audio is still here and the chunk decode below covers it.
            stream = self._stream
            with event_log.timed("asr", "live stream flush", stage="stream.flush"):
                stream.finalize(timeout=2.0)
            for event in stream.poll():
                if isinstance(event, StreamFinal):
                    self._commit_final(event)
            self.close()
            self._last_stable = ""
        end = self._len
        tail = self._buf[self._ledger.open_start_sample:end]
        if not len(tail) or not detect_speech(tail):
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
        self._decoded(result, closing.start_sample)
        self._chunks_decoded += 1
        self._forced_polish_index = len(self._ledger.committed) - 1

    def finalize(
        self,
        on_progress: Optional[Callable[[str], None]] = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> str:
        """Re-decode what is worth re-decoding, with the accurate engine.

        Bounded cost, spent only where it buys something: committed chunks the
        live model was unsure about, plus whatever audio never closed into a
        chunk before Stop. Everything the fast model got confidently right is
        kept as-is, which is why this is seconds and not a full re-transcribe.
        """
        t0 = time.time()
        self._finish_background(cancelled)

        def decoded(result: AsrResult, index: Optional[int], start: int, _end: int) -> None:
            self._note_uncertain(result, index)
            if index is None:
                self._chunks_decoded += 1
                self._decoded(result, start)

        # The polish exists to hand weak chunks to a BETTER decode. When the
        # accurate engine is the same cloud model as the live one (Deepgram
        # answering both, the default), it would send the same audio to the
        # same model and get the same words back, for a billed call and a
        # slower "final". Then only audio never decoded at all is worth a call.
        # A local model is still re-run at the wider final beam, which is a
        # better decode even on the same weights, so it is not skipped.
        same_model = (
            engine_identity(self.final_engine) == engine_identity(self.live_engine)
            and _ignores_beam(self.final_engine)
        )
        if same_model:
            event_log.emit("live", "polish skipped, final engine is the live engine")
        # The tail closed at Stop was decoded fast on purpose; a confident fast
        # decode is still a fast decode, so it is re-done here either way.
        forced = () if self._forced_polish_index is None else (self._forced_polish_index,)
        polished = self._chunks_polished = polish(
            self._ledger, self._audio(),
            lambda clip, stage: self._decode(
                self.final_engine, clip, self.final_beam_size,
                want_confidence=True, stage=stage, condition=True,
                vad_filter=self.polish_vad_filter,
            ),
            ceiling=float("-inf") if same_model else self.polish_confidence_ceiling,
            force=() if same_model else forced,
            skip=self._bg_applied,
            on_progress=on_progress or (lambda _msg: None),
            on_decoded=decoded,
            cancelled=cancelled,
        )

        raw = self._ledger.committed_text
        # committed_len=0: everything here just got an authoritative decode, so
        # no cached prefix from the live pass may survive into the final report.
        self._post.reset()
        if not self.postprocess:
            final = raw
        else:
            final = close_sentence(self._post.process(raw, 0)[0]) if raw else ""
        logger.info(
            "final polish  audio=%.1fs  elapsed=%.2fs  polished=%d  chars=%d",
            self.audio_sec, time.time() - t0, polished, len(final),
        )
        return final

    # -- internals ------------------------------------------------------

    def _context(
        self, beam_size: int, *, want_confidence: bool, condition: bool,
        vad_filter: bool = False,
    ) -> TranscribeContext:
        return TranscribeContext(
            language=self.language,
            # Live chunks are already cut at a VAD boundary.
            vad_filter=vad_filter,
            beam_size=beam_size,
            pause_threshold=self.pause_threshold,
            condition_on_previous_text=condition,
            initial_prompt=self.initial_prompt,
            want_word_confidence=want_confidence,
            temperature=0.0,
        )

    def _decode(
        self,
        engine: AsrEngine,
        clip: np.ndarray,
        beam_size: int,
        *,
        want_confidence: bool,
        stage: str,
        condition: bool = False,
        vad_filter: bool = False,
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
                clip_sec=round(len(clip) / self.sr, 1), level_db=level_db(clip), beam=beam_size,
            ) as note:
                result = engine.transcribe(
                    clip,
                    self._context(
                        beam_size, want_confidence=want_confidence,
                        condition=condition, vad_filter=vad_filter,
                    ),
                )
                note["words"] = len(result.text.split())
                if result.engine:
                    note["engine"] = result.engine
            self.decode_audio_sec += len(clip) / self.sr
        except Exception as exc:
            logger.warning("Decode failed (%s): %s", stage, exc)
            return None
        return result


def _ignores_beam(engine: AsrEngine) -> bool:
    """A cloud engine decodes the same way whatever beam it is asked for."""
    try:
        return bool(engine.capabilities().network)
    except Exception:
        return False
