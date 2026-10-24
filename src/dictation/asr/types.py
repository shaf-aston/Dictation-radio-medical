"""Engine-agnostic transcription result types: the AsrEngine contract.

Every recognition engine (faster-whisper today, ONNX/Parakeet at M3) returns
these shapes regardless of its own native API. Confidence is part of the
contract, not an optional extra: the whole point of this seam is that
downstream post-processing (M4's token contract) can gate a correction on how
sure the decoder was, and that only works if every engine reports it: an
engine that cannot supply real per-word probabilities must say so via
:class:`EngineCaps` rather than silently returning 1.0 everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class Word:
    """One decoded word with its span and the decoder's own confidence."""

    text: str
    start: float
    end: float
    #: A probability in [0, 1] (faster-whisper's exp(Word.probability)), or
    #: None when the engine cannot measure one. Checked here, so no engine can
    #: hand on a score that is not a probability: the hotword decoder's scores
    #: carry the hotword bonus (up to 2.7), which made every boosted word look
    #: the most certain of all and switched the confidence polish off.
    confidence: Optional[float]

    def __post_init__(self) -> None:
        c = self.confidence
        if c is not None and not 0.0 <= c <= 1.0 + 1e-6:
            raise ValueError(f"word confidence {c!r} for {self.text!r} is not a probability")


@dataclass(frozen=True)
class AsrSegment:
    """One decoded segment (Whisper's unit of output), with its words if known."""

    text: str
    start: float
    end: float
    words: Tuple[Word, ...] = ()

    @property
    def confidence(self) -> Optional[float]:
        """Mean word confidence, or ``None`` when the engine gave no words.

        ``None`` (not 0.0 or 1.0) so a caller can tell "the model was unsure"
        apart from "this engine doesn't expose confidence at all": collapsing
        the two would make the M4 correction gate either over- or under-fire
        for every engine that lacks word timestamps.
        """
        known = [w.confidence for w in self.words if w.confidence is not None]
        return sum(known) / len(known) if known else None


@dataclass(frozen=True)
class AsrResult:
    """Full output of one ``AsrEngine.transcribe()`` call."""

    text: str
    segments: Tuple[AsrSegment, ...] = ()
    #: Which provider produced it, set by the chain that picked one. Empty
    #: from a bare engine, which is only ever itself.
    engine: str = ""


@dataclass(frozen=True)
class CostModel:
    """What one ``transcribe()`` call costs, in wall seconds, on a warm engine.

    ``fixed_sec + per_audio_sec * clip_sec``. The fixed part is what decides
    how the live loop should be shaped: Whisper pads every call to a 30s window
    so it pays ~1-4s however little it is handed, and the stream layer must
    cut few, long chunks to amortise that. A cloud engine pays ~0.2s, so the
    same long chunks only make the kept text trail the microphone for nothing.
    Approximate by nature; measured where the engine says so.
    """

    fixed_sec: float
    per_audio_sec: float = 0.0

    def estimate(self, clip_sec: float) -> float:
        return self.fixed_sec + self.per_audio_sec * max(0.0, clip_sec)


#: A local Whisper-class engine: the price every constant in stream/ was
#: originally sized for, so it is the safe default for an engine that does not
#: say otherwise.
LOCAL_DECODE_COST = CostModel(fixed_sec=1.0, per_audio_sec=0.02)


@dataclass(frozen=True)
class EngineCaps:
    """What an engine can actually provide: never assume, always check.

    ``word_confidence``: real per-word probabilities (not a constant stand-in).
    ``hotwords``: decoder-level vocabulary biasing (M5).
    ``cost``: what a call costs (see :class:`CostModel`); the stream layer
    sizes chunks and preview pacing from it (``stream/policy.py``).
    ``streaming``: also implements ``open_stream()`` (a live socket that
    returns interim and final words as audio arrives).
    ``network``: audio leaves the device when this engine answers.
    ``beam_sensitive``: a wider beam (``TranscribeContext.beam_size``) can
    give a different, better decode; the polish re-runs the same model only
    when it can.
    """

    word_confidence: bool
    hotwords: bool
    cost: CostModel = LOCAL_DECODE_COST
    streaming: bool = False
    network: bool = False
    beam_sensitive: bool = False


@dataclass(frozen=True)
class TranscribeContext:
    """Everything one ``transcribe()`` call needs, gathered in one place.

    Replaces the growing keyword-argument list on the old ``Transcriber``
    wrapper: callers build one of these instead of remembering which knobs
    the live cycle needs versus the final pass.
    """

    language: str = "en"
    vad_filter: bool = True
    beam_size: int = 5
    initial_prompt: Optional[str] = None
    pause_threshold: float = 2.5
    condition_on_previous_text: bool = True
    temperature: Optional[Union[float, List[float]]] = None
    # Word-level confidence costs a little extra decode time; callers that
    # don't need it (nothing downstream reads .words yet) can skip it.
    want_word_confidence: bool = False
    # Reserved for M5 (lexicon-biased decoding); unused engines ignore it.
    hotwords: Optional[Sequence[str]] = None


# -- streaming ---------------------------------------------------------------
#
# A streaming engine is handed audio as it arrives and answers as it hears it,
# instead of being handed a finished clip. Three things come back, and the
# stream layer (stream/live_session.py) maps each one onto what it already has:

@dataclass(frozen=True)
class StreamInterim:
    """A guess at the words since the last final: the live preview."""

    text: str


@dataclass(frozen=True)
class StreamFinal:
    """Settled text for ``[start, end)`` seconds of the stream: one ledger chunk.

    Times are in seconds of audio from the first sample pushed to the stream.
    Consecutive finals tile the stream: each starts where the last ended, so
    they commit in order exactly the way closed chunks do.
    """

    text: str
    start: float
    end: float
    words: Tuple[Word, ...] = ()

    def as_result(self) -> AsrResult:
        return AsrResult(
            text=self.text,
            segments=(AsrSegment(self.text, 0.0, self.end - self.start, self.words),)
            if self.text else (),
        )


@dataclass(frozen=True)
class StreamError:
    """The stream is gone (dropped socket, rejected key). Nothing more will come."""

    message: str


StreamEvent = Union[StreamInterim, StreamFinal, StreamError]


class ProviderUnavailable(RuntimeError):
    """This engine cannot work until something outside the app changes.

    A missing or rejected key, not a dropped connection: retrying it on the
    next decode would only add a wasted round trip in front of every decode.
    A chain skips such a provider until the app restarts.
    """
