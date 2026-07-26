"""The one place that maps an engine name to a concrete AsrEngine.

Everything else — worker.py, web_app.py, the eval harness — asks for an
engine by name and depends only on the port. Adding engine #2 at M3 means one
new entry in :data:`_ENGINES`; nothing else in this file, or anywhere
downstream, changes.
"""

from __future__ import annotations

from typing import Any

from src.dictation.asr.engines.faster_whisper_engine import FasterWhisperEngine
from src.dictation.asr.port import AsrEngine

_ENGINES = {
    "faster-whisper": FasterWhisperEngine,
}

DEFAULT_ENGINE = "faster-whisper"


def create_engine(name: str = DEFAULT_ENGINE, **kwargs: Any) -> AsrEngine:
    """Build the named engine. Raises ``ValueError`` on an unknown name."""
    cls = _ENGINES.get(name)
    if cls is None:
        raise ValueError(f"Unknown ASR engine {name!r}. Available: {sorted(_ENGINES)}")
    return cls(**kwargs)
