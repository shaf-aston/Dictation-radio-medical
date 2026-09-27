"""The AsrEngine swap-seam: everything downstream depends on this, not on any
concrete recognition engine.

Adding a second engine (M3: an ONNX/Parakeet CTC model) means writing one new
class satisfying this Protocol: nothing in ``worker.py`` or ``web_app.py``
changes. A ``Protocol`` rather than an ABC on purpose: it lets the eval
harness's ``WhisperRunner`` and any future test double satisfy the interface
structurally, with no inheritance and no import of this module required.
"""

from __future__ import annotations

from typing import Any, Hashable, List, Protocol, runtime_checkable

from src.dictation.asr.types import AsrResult, EngineCaps, StreamEvent, TranscribeContext


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


class AsrStream(Protocol):
    """One open streaming session. Thread-safe: audio is pushed from one
    thread while events are polled from another."""

    def push(self, pcm16: bytes) -> None:
        """Send 16-bit little-endian mono PCM at 16 kHz. Never blocks on the network."""
        ...

    def poll(self) -> List[StreamEvent]:
        """Every event received since the last call, oldest first. Never blocks."""
        ...

    def finalize(self, timeout: float) -> bool:
        """Ask for everything pushed so far to be settled, and wait up to
        *timeout* seconds for it. ``True`` when it all arrived."""
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class StreamingAsrEngine(Protocol):
    """An engine that can also transcribe audio as it arrives
    (``EngineCaps.streaming``). Opening may raise; the caller falls back to
    decoding chunks with :meth:`AsrEngine.transcribe`."""

    def open_stream(self, ctx: TranscribeContext) -> AsrStream:
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
