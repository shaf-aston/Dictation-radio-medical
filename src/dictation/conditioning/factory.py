"""build_chain: the only setting-name -> step mapping."""

from __future__ import annotations

from typing import Callable, Dict

from src.core.settings import get_default
from src.dictation.conditioning.chain import Chain
from src.dictation.conditioning.port import Step

# Fixed run order, not a setting. Gating before gain, so the volume step never
# turns up the gaps between words.
ORDER = ("highpass", "denoise", "expander", "agc")

# step name -> builder(section, sr). A later slice adds one entry per step.
_BUILDERS: Dict[str, Callable[[dict, int], Step]] = {}


def build_chain(cfg: dict, sr: int) -> Chain:
    """cfg is settings["audio_conditioning"]; each section is merged over the
    default's, so an old settings file still gets every knob."""
    base = get_default("audio_conditioning") or {}
    merged = {**base, **(cfg or {})}
    window = merged.pop("window_samples")
    if not isinstance(window, int) or window <= 0 or window % 2:
        raise ValueError(f"window_samples must be a positive even integer, got {window!r}")
    for name in merged:
        if name not in _BUILDERS:
            raise ValueError(f"unknown audio_conditioning section {name!r}")
    steps = []
    for name in ORDER:
        if name not in merged:
            continue
        section = {**base.get(name, {}), **merged[name]}
        if section.get("on", True):
            steps.append(_BUILDERS[name](section, sr))
    return Chain(steps, window)
