"""The one place that maps an engine name to a concrete AsrEngine.

Everything else — worker.py, web_app.py, the eval harness — asks for an
engine by name and depends only on the port. Adding engine #2 at M3 means one
new entry in :data:`_ENGINES`; nothing else in this file, or anywhere
downstream, changes.
"""

from __future__ import annotations

from typing import Any

from src.dictation.asr.engines.faster_whisper_engine import FasterWhisperEngine
from src.dictation.asr.engines.parakeet_engine import ParakeetEngine
from src.dictation.asr.port import AsrEngine

_ENGINES = {
    "faster-whisper": FasterWhisperEngine,
    "parakeet": ParakeetEngine,
}

#: The constructor keyword each engine uses for "which model". Callers that
#: offer one generic model option (the eval harness's ``--model``) ask here
#: instead of learning engine-specific argument names — which is the whole
#: point of this file being the only name-to-engine mapping.
_MODEL_KWARG = {
    "faster-whisper": "model_size",
    "parakeet": "model_name",
}

DEFAULT_ENGINE = "faster-whisper"

#: Every engine name a caller may ask for — the eval harness builds its
#: ``--engine`` choices from this so a new engine needs no CLI edit.
ENGINE_NAMES = tuple(sorted(_ENGINES))


def create_engine(name: str = DEFAULT_ENGINE, **kwargs: Any) -> AsrEngine:
    """Build the named engine. Raises ``ValueError`` on an unknown name."""
    cls = _ENGINES.get(name)
    if cls is None:
        raise ValueError(f"Unknown ASR engine {name!r}. Available: {sorted(_ENGINES)}")
    return cls(**kwargs)


def model_kwargs(name: str, model: str) -> dict:
    """The kwargs that select *model* on the named engine (empty when blank)."""
    if not model:
        return {}
    keyword = _MODEL_KWARG.get(name)
    if keyword is None:
        raise ValueError(f"Unknown ASR engine {name!r}. Available: {sorted(_ENGINES)}")
    return {keyword: model}
