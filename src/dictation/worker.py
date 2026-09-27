"""Background worker for live transcription during recording (desktop).

A thin Qt adapter over :class:`~src.dictation.stream.live_session.LiveSession`,
the same loop the web app runs. It used to be a second copy of that loop, and
the two drifted: this one never learned the commit-cycle preview skip, the
minimum preview tail, or the engine-priced chunk plan, so every speed fix had
to land twice or did not land here at all. Now there is one loop, two skins:

* the web app feeds :class:`LiveSession` PCM frames off a WebSocket;
* this worker feeds it the new samples of the growing WAV the recorder writes
  (only what arrived since the last read: it seeks, it never re-reads the
  file), calls :meth:`LiveSession.cycle`, and re-emits the result as Qt
  signals.

Everything about chunking, previews, confidence and the polish after Stop is
documented in :mod:`src.dictation.stream.live_session` and its neighbours.

The session is built with ``postprocess=False``: ``partial`` carries RAW text,
because the desktop runs the correction pipeline on its own worker thread
(:mod:`src.ui.postprocess_worker`), with the committed length telling it which
prefix will never change.

Beam widths and the confidence ceiling arrive as constructor arguments (the
caller reads them from settings: see :mod:`src.ui.recording_session`); this
worker never reads settings itself.
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
from src.dictation.asr import AsrResult, create_engine
from src.dictation.stream import live_session
from src.dictation.stream.live_session import LiveSession
from src.dictation.stream.policy import plan_for
from src.dictation.stream.rules import build_context_prompt
from src.dictation.stream.segmenter import ChunkPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_MIN_AUDIO_SEC = 0.8     # ignore audio shorter than this
_POLL_SEC = 0.25         # how often to look for new audio when nothing changed

# Progress states the UI shows while the live loop runs. Public: the desktop
# window maps them onto the colour of its state pill (ui/main_window.py).
STATE_LOADING = "Loading model..."
STATE_LIVE = "Live transcribing..."
STATE_CATCHING_UP = "Catching up..."

_SESSION_STATES = {
    live_session.STATE_LIVE: STATE_LIVE,
    live_session.STATE_CATCHING_UP: STATE_CATCHING_UP,
}


class LiveTranscribeWorker(QObject):
    """Feeds a :class:`LiveSession` from the recorder's growing WAV.

    ``partial`` carries the full display text (frozen chunk text + the stable
    preview of the open tail) and the length of the frozen prefix within it.
    Consumers use the prefix length to skip re-processing text that will never
    be revised
    (:class:`~src.dictation.postprocess.incremental.IncrementalPostprocessor`).
    A value of ``0`` means "treat the whole text as revisable".
    """

    partial = Signal(str, int)
    finished = Signal()
    progress = Signal(str)
    # Absolute-timed segments for each committed decode, used by the cloud
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
        live_model_size: Optional[str] = None,
        model_path: Optional[Union[str, Path]] = None,
        chunk_policy: Optional[ChunkPolicy] = None,
        live_beam_size: int = 2,
        final_beam_size: int = 5,
        polish_confidence_ceiling: float = 0.75,
        preview_max_lag_sec: float = 3.0,
        trailing_silence_sec: float = 0.6,
    ) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        # The fast model that writes what you see while speaking; the accurate
        # model_size engine only runs the post-stop polish.
        self.live_model_size = live_model_size or model_size
        self.language = language
        # VAD is load-bearing for chunk cutting, so it is always on; this flag
        # only gates the engine's own VAD inside the polish decodes.
        self.vad_enabled = vad_enabled
        self.pause_threshold = pause_threshold
        self.live_beam_size = max(1, int(live_beam_size))
        self.final_beam_size = max(1, int(final_beam_size))
        self.polish_confidence_ceiling = float(polish_confidence_ceiling)
        self.preview_max_lag_sec = float(preview_max_lag_sec)
        self.model_path = model_path
        # None means "size chunks from the live engine's cost", which is only
        # known once run() has built the engine.
        self._chunk_policy = chunk_policy
        self._trailing_silence_sec = float(trailing_silence_sec)
        self._session: Optional[LiveSession] = None
        self._keep_running = True
        self._final_requested = False
        # Set when the session is abandoned (the radiologist started a new
        # recording before the polish finished). The polish checks it between
        # chunks so an abandoned pass stops at the next boundary instead of
        # holding the model for the recording that replaced it.
        self._cancelled = False
        self._fed = 0              # samples of the WAV already handed over
        self._sr = 0
        self._last_emitted = ""
        self._progress_state = ""

    @property
    def committed_chunks(self) -> int:
        """How many chunks the stream froze: the "decode once" count.

        Exposed for the run log, so it does not have to reach into the ledger.
        """
        return self._session.chunks_decoded if self._session is not None else 0

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._keep_running = False

    def finalize(self) -> None:
        """Request one final confidence-targeted polish after recording stops."""
        self._final_requested = True

    def cancel(self) -> None:
        """Abandon this session: stop the loop and any polish still running.

        Its text is no longer wanted: a newer recording owns the document now.
        """
        self._cancelled = True
        self._keep_running = False
        self._final_requested = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        wall_start = time.time()
        cycle_count = 0
        try:
            self.progress.emit(STATE_LOADING)
            self._session = self._build_session()
            logger.info("Engines ready in %.2fs", time.time() - wall_start)
            self._emit_state(STATE_LIVE)

            final_grace = 0
            while self._keep_running or self._final_requested:
                grew = self._feed_new_audio()
                if self._session.audio_sec < _MIN_AUDIO_SEC:
                    # The WAV stops growing once recording ends, so a finalize()
                    # on a too-short or unreadable file would spin this loop
                    # forever. Give the recorder a short grace to flush, then
                    # finish without a polish pass.
                    if self._final_requested:
                        final_grace += 1
                        if final_grace > 6:
                            logger.warning(
                                "Recording too short or unreadable (%d samples); "
                                "skipping final polish", self._fed,
                            )
                            break
                    time.sleep(0.5)
                    continue

                if self._final_requested:
                    self._run_final_polish()
                    self._keep_running = False
                    self._final_requested = False
                    break

                if not grew:
                    time.sleep(_POLL_SEC)
                    continue
                update = self._session.cycle()
                if update is None:
                    time.sleep(_POLL_SEC)
                    continue
                self._emit_state(_SESSION_STATES.get(update.state, STATE_LIVE))
                if self._emit_partial(update.committed, update.preview):
                    cycle_count += 1

        except Exception as exc:
            logger.error("Worker error: %s", exc, exc_info=True)
        finally:
            if self._session is not None and self._session.audio_sec > 0:
                perf.set_gauge(
                    "stream.decode_ratio",
                    self._session.decode_audio_sec / self._session.audio_sec,
                )
            logger.info(
                "Worker finished in %.2fs, %d cycles emitted",
                time.time() - wall_start, cycle_count,
            )
            perf.log_summary("dictation perf")
            self.finished.emit()

    def _build_session(self) -> LiveSession:
        # Engines load their model lazily, so creating both here is free. A
        # fine-tuned voice model (model_path) only ever replaces the accurate
        # engine: live text comes from the stock fast model, as on the web.
        live_engine = create_engine(model_size=self.live_model_size, device="auto")
        if self.live_model_size == self.model_size and not self.model_path:
            final_engine = live_engine
        else:
            final_engine = create_engine(
                model_size=self.model_size, device="auto", model_path=self.model_path
            )
        plan = plan_for(live_engine.capabilities(), self._trailing_silence_sec)
        policy = self._chunk_policy or plan.policy
        logger.info("Chunk plan %s: %s", "manual" if self._chunk_policy else plan.row, policy)
        return LiveSession(
            live_engine,
            final_engine,
            language=self.language,
            policy=policy,
            pause_threshold=self.pause_threshold,
            live_beam_size=self.live_beam_size,
            final_beam_size=self.final_beam_size,
            preview_max_lag_sec=self.preview_max_lag_sec,
            preview_min_tail_sec=plan.preview_min_tail_sec,
            polish_confidence_ceiling=self.polish_confidence_ceiling,
            initial_prompt=build_context_prompt(),
            postprocess=False,
            polish_vad_filter=self.vad_enabled,
            on_decoded=self._emit_absolute_segments,
        )

    def _run_final_polish(self) -> None:
        """After Stop: re-decode only what is worth it (see stream/polish.py).

        The desktop hands the live text back the moment Stop is pressed
        (recording_session._hand_back_live_text), so this is an upgrade that
        lands later, never a wait.
        """
        assert self._session is not None
        self._feed_new_audio()
        t0 = time.time()
        # This pass can take seconds, after the radiologist has pressed Stop:
        # say what it is doing rather than leaving the window looking hung.
        final = self._session.finalize(
            on_progress=self.progress.emit, cancelled=lambda: self._cancelled,
        )
        if self._cancelled:
            return
        logger.info(
            "confidence-targeted polish  dur=%.1fs  elapsed=%.2fs  chars=%d",
            self._session.audio_sec, time.time() - t0, len(final),
        )
        if final and final != self._last_emitted:
            self._last_emitted = final
            # committed_len=0: everything here just got a final, authoritative
            # decode, so nothing is a stale carry-over of a live-pass guess.
            self.partial.emit(final, 0)

    def _emit_partial(self, committed: str, preview: str) -> bool:
        output = f"{committed} {preview}" if committed and preview else (committed or preview)
        if not output or output == self._last_emitted:
            return False
        self._last_emitted = output
        self.partial.emit(output, len(committed))
        return True

    def _emit_state(self, state: str) -> None:
        """Emit a live-loop status only when it changes."""
        if state != self._progress_state:
            self._progress_state = state
            self.progress.emit(state)

    def _emit_absolute_segments(self, result: AsrResult, start_sample: int) -> None:
        if not result.segments or not self._sr:
            return
        offset = start_sample / self._sr
        self.segments.emit([
            {"start": offset + seg.start, "end": offset + seg.end, "text": seg.text}
            for seg in result.segments
        ])

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    def _feed_new_audio(self) -> bool:
        """Hand the session whatever the recorder wrote since the last call."""
        assert self._session is not None
        total, sr = self._audio_length()
        if not sr or total <= self._fed:
            return False
        if sr != self._session.sr:
            # The recorder captures at 16 kHz (audio.py), the rate the VAD is
            # fixed to; anything else is a caller bug, not audio to guess at.
            raise RuntimeError(f"Recording is {sr} Hz; live dictation needs {self._session.sr} Hz")
        audio, _ = self._read_audio(self._fed)
        if audio is None or not len(audio):
            return False
        self._sr = sr
        self._session.feed(audio)
        self._fed += len(audio)
        return True

    def _audio_length(self) -> Tuple[int, int]:
        """Return ``(total_frames, samplerate)`` from the header: no decode.

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

        Seeks rather than re-reading the whole recording. Falls back to a full
        read if the seek fails: the header of a WAV still being written is not
        guaranteed to describe every frame on disk.
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
