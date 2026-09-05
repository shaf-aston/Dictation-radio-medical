"""The one place that maps an engine name to a concrete AsrEngine.

Everything else, worker.py, web_app.py, the eval harness, asks for an
engine by name and depends only on the port. Adding engine #2 at M3 means one
new entry in :data:`_ENGINES`; nothing else in this file, or anywhere
downstream, changes.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, List, Tuple

from src.dictation.asr.engines.deepgram_engine import DeepgramEngine
from src.dictation.asr.engines.fallback_engine import ChainEngine
from src.dictation.asr.engines.faster_whisper_engine import FasterWhisperEngine
from src.dictation.asr.engines.parakeet_engine import ParakeetEngine
from src.dictation.asr.port import AsrEngine

logger = logging.getLogger(__name__)


def _make_deepgram_chain(**kwargs: Any) -> AsrEngine:
    """The default provider chain: Deepgram (cloud, medical model) first,
    Parakeet (local, if the optional ``onnx-asr`` package is installed) next,
    faster-whisper (local, always available) last.

    Each tier only runs when the one before it raised, never on a middling
    result, ``ChainEngine`` has no ground truth to grade a decode against.
    Adding a fourth provider later is one more entry in this list, nothing
    downstream (worker.py, web_app.py, the eval harness) changes.

    *kwargs* are whatever the caller already passes for the Whisper tier
    (``model_size``, ``device``, ``model_path``, ...): every existing call
    site built these before Deepgram existed, so they stay Whisper-shaped
    and only need forwarding, not translating.
    """
    providers: List[Tuple[str, Any]] = [("deepgram", DeepgramEngine())]
    if importlib.util.find_spec("onnx_asr") is not None:
        providers.append(("parakeet", ParakeetEngine()))
    else:
        logger.info("onnx-asr not installed; ASR chain skips the Parakeet tier")
    providers.append(("faster-whisper", FasterWhisperEngine(**kwargs)))
    return ChainEngine(providers)


_ENGINES = {
    "faster-whisper": FasterWhisperEngine,
    "parakeet": ParakeetEngine,
    "deepgram": _make_deepgram_chain,
}

#: The constructor keyword each engine uses for "which model". Callers that
#: offer one generic model option (the eval harness's ``--model``) ask here
#: instead of learning engine-specific argument names: which is the whole
#: point of this file being the only name-to-engine mapping.
_MODEL_KWARG = {
    "faster-whisper": "model_size",
    "parakeet": "model_name",
    # Deepgram's own model is fixed to the medical tier (see deepgram_engine.py);
    # "model" here still picks the size of its Whisper fallback.
    "deepgram": "model_size",
}

#: Deepgram is the default: cloud, medical-vocabulary-tuned, with a local
#: Parakeet/Whisper chain behind it (see ``_make_deepgram_chain``). Every call
#: site that does not pass name= (worker.py, web_app.py, warmup.py) picks
#: this up automatically.
DEFAULT_ENGINE = "deepgram"

#: Every engine name a caller may ask for: the eval harness builds its
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
