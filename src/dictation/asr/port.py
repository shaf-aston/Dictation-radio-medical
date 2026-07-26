"""The AsrEngine swap-seam: everything downstream depends on this, not on any
concrete recognition engine.

Adding a second engine (M3: an ONNX/Parakeet CTC model) means writing one new
class satisfying this Protocol — nothing in ``worker.py`` or ``web_app.py``
changes. A ``Protocol`` rather than an ABC on purpose: it lets the eval
harness's ``WhisperRunner`` and any future test double satisfy the interface
structurally, with no inheritance and no import of this module required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.dictation.asr.types import AsrResult, EngineCaps, TranscribeContext


@runtime_checkable
class AsrEngine(Protocol):
    """A speech recognition engine: audio in, an :class:`AsrResult` out."""

    def transcribe(self, audio: Any, ctx: TranscribeContext) -> AsrResult:
        """Transcribe *audio* (path, file-like, or float32 ndarray)."""
        ...

    def capabilities(self) -> EngineCaps:
        """What this engine can actually provide — checked, never assumed."""
        ...

    def preload(self) -> None:
        """Load the model now instead of on the first ``transcribe()`` call."""
        ...
