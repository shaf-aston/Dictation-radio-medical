"""AsrEngine adapter over the existing faster-whisper wrapper.

Deliberately thin: all the real work (model loading/caching, the compute-type
fallback chain, hallucination filtering) stays in
:class:`src.dictation.transcriber.Transcriber`, which is not being rewritten.
This class only translates the port's :class:`TranscribeContext` in and
:class:`AsrResult` out, so a pure-refactor milestone can prove it changed
nothing by diffing eval numbers against the pre-port baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from src.dictation.asr.types import AsrResult, AsrSegment, EngineCaps, TranscribeContext, Word
from src.dictation.transcriber import Transcriber


class FasterWhisperEngine:
    """Wraps :class:`Transcriber` to satisfy the :class:`AsrEngine` port."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: Optional[str] = None,
        use_msk_prompt: bool = True,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._transcriber = Transcriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            use_msk_prompt=use_msk_prompt,
            model_path=model_path,
        )

    def preload(self) -> None:
        self._transcriber.preload()

    @property
    def compute_type(self) -> Optional[str]:
        """The ctranslate2 quantisation actually selected (None until loaded)."""
        return self._transcriber.compute_type

    def capabilities(self) -> EngineCaps:
        return EngineCaps(word_confidence=True, hotwords=True)

    def transcribe(self, audio: Any, ctx: TranscribeContext) -> AsrResult:
        text, seg_list = self._transcriber.transcribe(
            audio,
            language=ctx.language,
            vad_filter=ctx.vad_filter,
            beam_size=ctx.beam_size,
            initial_prompt=ctx.initial_prompt,
            pause_threshold=ctx.pause_threshold,
            condition_on_previous_text=ctx.condition_on_previous_text,
            temperature=ctx.temperature,
            word_timestamps=ctx.want_word_confidence,
            hotwords=list(ctx.hotwords) if ctx.hotwords else None,
        )
        return AsrResult(text=text, segments=tuple(_to_segment(s) for s in seg_list))


def _to_segment(seg: dict) -> AsrSegment:
    words = tuple(
        Word(text=w["text"], start=w["start"], end=w["end"], confidence=w["probability"])
        for w in seg.get("words") or ()
    )
    return AsrSegment(text=seg["text"], start=seg["start"], end=seg["end"], words=words)
