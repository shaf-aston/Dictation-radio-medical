"""The AsrEngine swap-seam: everything downstream depends on this, not on any
concrete recognition engine.

Adding a second engine (M3: an ONNX/Parakeet CTC model) means writing one new
class satisfying this Protocol: nothing in ``worker.py`` or ``web_app.py``
changes. A ``Protocol`` rather than an ABC on purpose: it lets the eval
harness's ``WhisperRunner`` and any future test double satisfy the interface
structurally, with no inheritance and no import of this module required.
"""

from __future__ import annotations

from typing import Any, Hashable, Protocol, runtime_checkable

from src.dictation.asr.types import AsrResult, EngineCaps, TranscribeContext


@runtime_checkable
class AsrEngine(Protocol):
    """A speech recognition engine: audio in, an :class:`AsrResult` out."""

    def transcribe(self, audio: Any, ctx: TranscribeContext) -> AsrResult:
        """Transcribe *audio* (path, file-like, or float32 ndarray)."""
        ...

    def capabilities(self) -> EngineCaps:
        """What this engine can actually provide: checked, never assumed."""
        ...

    def preload(self) -> None:
        """Load the model now instead of on the first ``transcribe()`` call."""
        ...


def engine_identity(engine: Any) -> Hashable:
    """Which model would answer a call to *engine* right now.

    Two engines with the same identity return the same words for the same
    audio, so re-decoding one's output with the other buys nothing and costs a
    call (for Deepgram, a billed one). An engine says who it is through
    ``identity()``; a provider chain answers with whichever provider it would
    try first. An engine that does not say is only ever equal to itself.
    """
    identify = getattr(engine, "identity", None)
    return identify() if callable(identify) else ("object", id(engine))
