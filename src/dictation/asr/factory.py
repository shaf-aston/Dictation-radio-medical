"""The one place that maps an engine name to a concrete AsrEngine.

Everything else (the desktop worker, ``ui/web/state.get_engine``, the eval
harness) asks for an engine by name and depends only on the port. Adding an
engine means one new entry in :data:`_ENGINES`; nothing downstream changes.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.dictation.asr.engines.deepgram_engine import DeepgramEngine
from src.dictation.asr.engines.fallback_engine import ChainEngine
from src.dictation.asr.engines.faster_whisper_engine import FasterWhisperEngine
from src.dictation.asr.engines.parakeet_engine import STANDARD, ParakeetEngine, model_dir_for
from src.dictation.asr.engines.parakeet_hotword_engine import ParakeetHotwordEngine
from src.dictation.asr.port import AsrEngine
from src.core.settings import Settings, get_default

logger = logging.getLogger(__name__)


#: parakeet_precision setting -> onnx-asr's quantization argument.
_PARAKEET_QUANTIZATION = {"fp32": None, "int8": "int8"}


def _tuned_dir(settings: Settings) -> Optional[Path]:
    """The fine-tune folder the ``parakeet_model`` setting names; None for the
    standard model. A name no longer installed (folder deleted, data/ wiped) is
    logged and dictation carries on with the standard model, and
    :func:`describe_engine` says so: a missing folder must not stop dictation."""
    name = settings.get("parakeet_model")
    if name == STANDARD:
        return None
    try:
        return model_dir_for(name)
    except ValueError:
        logger.error("Voice model %r is not installed; using the standard model", name)
        return None


def _make_parakeet(settings: Optional[Settings] = None, **kwargs: Any) -> ParakeetEngine:
    """Parakeet with the weights ``parakeet_model`` names. The standard model
    runs at the precision ``parakeet_precision`` names; a fine-tune at the one
    it was exported in, since it ships only that one. Raises ``ValueError`` on
    an unknown precision rather than guessing one."""
    settings = settings or Settings()
    if "model_dir" not in kwargs:
        kwargs["model_dir"] = _tuned_dir(settings)
    if kwargs["model_dir"] is not None:
        full = (kwargs["model_dir"] / "encoder-model.onnx").is_file()
        kwargs.setdefault("quantization", None if full else "int8")
        return ParakeetEngine(**kwargs)
    precision = settings.get("parakeet_precision")
    if precision not in _PARAKEET_QUANTIZATION:
        raise ValueError(
            f"Unknown parakeet_precision {precision!r}. Available: {sorted(_PARAKEET_QUANTIZATION)}"
        )
    kwargs.setdefault("quantization", _PARAKEET_QUANTIZATION[precision])
    return ParakeetEngine(**kwargs)


def _local_tiers(**kwargs: Any) -> List[Tuple[str, Any]]:
    """Parakeet (if the optional ``onnx-asr`` package is installed), then
    faster-whisper, which is always available.

    Each tier only runs when the one before it raised, never on a middling
    result, ``ChainEngine`` has no ground truth to grade a decode against.

    *kwargs* are whatever the caller already passes for the Whisper tier
    (``model_size``, ``device``, ``model_path``, ...): every call site is
    Whisper-shaped, so they are forwarded, not translated. A ``model_path`` is
    a fine-tuned Whisper voice the radiologist switched on, so it skips
    Parakeet: ahead of it, the fine-tune would never be used.
    """
    tiers: List[Tuple[str, Any]] = []
    if kwargs.get("model_path"):
        logger.info("Fine-tuned voice model active; ASR chain skips the Parakeet tier")
    else:
        parakeet = parakeet_tier(Settings())
        if parakeet is not None:
            tiers.append(parakeet)
        else:
            logger.info("Neither sherpa-onnx nor onnx-asr installed; ASR chain skips Parakeet")
    tiers.append(("faster-whisper", FasterWhisperEngine(**kwargs)))
    return tiers


def parakeet_tier(settings: Settings) -> Any:
    """The best Parakeet this install can run, as a (name, engine) tier, or None.

    1. A fine-tune picked as the voice model (``parakeet_model``): onnx-asr
       layout, so the onnx-asr engine, without hotwords (its training already
       targets them).
    2. sherpa-onnx with radiology hotwords (``asr_hotwords``, default on):
       measured 35% fewer medical-term errors than greedy at the same speed.
       It runs its own int8 export, so ``parakeet_precision`` does not apply.
    3. onnx-asr, greedy: the stock model.
    """
    have_onnx_asr = importlib.util.find_spec("onnx_asr") is not None
    tuned = _tuned_dir(settings)
    if tuned is not None and have_onnx_asr:
        return ("parakeet", _make_parakeet(settings, model_dir=tuned))
    if settings.get("asr_hotwords") and importlib.util.find_spec("sherpa_onnx") is not None:
        return ("parakeet-hotwords", ParakeetHotwordEngine(float(settings.get("hotword_score"))))
    if have_onnx_asr:
        return ("parakeet", _make_parakeet(settings))
    return None


def describe_engine(settings: Settings) -> str:
    """What the radiologist should be told is transcribing, in a few words.

    Answers without loading a model: the same choice ``_local_tiers`` makes.
    The Whisper model setting alone is misleading here, because on the
    default engine it only names the fallback.
    """
    if settings.get("asr_engine") == "deepgram":
        return "Deepgram (cloud)"
    tier = parakeet_tier(settings)
    if tier is None:
        return f"Whisper {settings.get('model_size')}"
    name, engine = tier
    if getattr(engine, "model_dir", None):
        return f"Parakeet, tuned: {engine.model_dir.name}"
    label = "Parakeet + radiology terms" if name == "parakeet-hotwords" else "Parakeet"
    chosen = settings.get("parakeet_model")
    if chosen != STANDARD:
        # Installed but unused means the runtime a fine-tune needs is missing.
        why = "needs onnx-asr" if _tuned_dir(settings) is not None else "not installed"
        label += f" ({chosen} {why})"
    return label


def _make_local_chain(**kwargs: Any) -> AsrEngine:
    """Everything on this machine: nothing leaves the device."""
    return ChainEngine(_local_tiers(**kwargs))


def _make_deepgram_chain(**kwargs: Any) -> AsrEngine:
    """Deepgram (cloud, medical model) first, the local chain behind it."""
    return ChainEngine([("deepgram", DeepgramEngine()), *_local_tiers(**kwargs)])


_ENGINES = {
    "faster-whisper": FasterWhisperEngine,
    "parakeet": _make_parakeet,
    "parakeet-hotwords": ParakeetHotwordEngine,
    "deepgram": _make_deepgram_chain,
    "local": _make_local_chain,
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
    "local": "model_size",
}

#: The shipped choice; the ``asr_engine`` setting is what a caller that does
#: not pass name= (worker.py, web_app.py, warmup.py) actually gets.
DEFAULT_ENGINE = get_default("asr_engine")

#: Every engine name a caller may ask for: the eval harness builds its
#: ``--engine`` choices from this so a new engine needs no CLI edit.
ENGINE_NAMES = tuple(sorted(_ENGINES))


def create_engine(name: str = "", **kwargs: Any) -> AsrEngine:
    """Build the named engine, or the ``asr_engine`` setting's when unnamed.
    Raises ``ValueError`` on an unknown name."""
    name = name or Settings().get("asr_engine")
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
    if name == "parakeet" and Path(model).is_dir():
        # A fine-tune's folder, as the kit's gate passes it (--model <dir>).
        return {"model_dir": Path(model)}
    return {keyword: model}
