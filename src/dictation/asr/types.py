"""Engine-agnostic transcription result types — the AsrEngine contract.

Every recognition engine (faster-whisper today, ONNX/Parakeet at M3) returns
these shapes regardless of its own native API. Confidence is part of the
contract, not an optional extra: the whole point of this seam is that
downstream post-processing (M4's token contract) can gate a correction on how
sure the decoder was, and that only works if every engine reports it — an
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
    confidence: float  # 0..1, e.g. faster-whisper's exp(Word.probability)

    def as_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end,
                "confidence": self.confidence}


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
        apart from "this engine doesn't expose confidence at all" — collapsing
        the two would make the M4 correction gate either over- or under-fire
        for every engine that lacks word timestamps.
        """
        if not self.words:
            return None
        return sum(w.confidence for w in self.words) / len(self.words)

    def as_dict(self) -> dict:
        """Legacy shape (``start``/``end``/``text``) used by the pre-port
        window/commit code — a bridge, not the long-term contract."""
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True)
class AsrResult:
    """Full output of one ``AsrEngine.transcribe()`` call."""

    text: str
    segments: Tuple[AsrSegment, ...] = ()

    def segments_as_dicts(self) -> List[dict]:
        return [s.as_dict() for s in self.segments]


@dataclass(frozen=True)
class EngineCaps:
    """What an engine can actually provide — never assume, always check.

    ``word_confidence``: real per-word probabilities (not a constant stand-in).
    ``hotwords``: decoder-level vocabulary biasing (M5).
    ``streaming``: can decode an open-ended growing buffer efficiently, rather
    than only fixed clips (relevant once M3 adds a CTC-style engine).
    """

    word_confidence: bool
    hotwords: bool
    streaming: bool = True


@dataclass(frozen=True)
class TranscribeContext:
    """Everything one ``transcribe()`` call needs, gathered in one place.

    Replaces the growing keyword-argument list on the old ``Transcriber``
    wrapper — callers build one of these instead of remembering which knobs
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
