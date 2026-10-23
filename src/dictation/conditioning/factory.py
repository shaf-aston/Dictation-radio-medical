"""build_chain: the only setting-name -> step mapping."""

from __future__ import annotations

from typing import Callable, Dict

from src.core.settings import get_default
from src.dictation.conditioning.chain import Chain
from src.dictation.conditioning.port import Step
from src.dictation.conditioning.stages.agc import Agc
from src.dictation.conditioning.stages.denoise import Denoise
from src.dictation.conditioning.stages.expander import Expander
from src.dictation.conditioning.stages.highpass import Highpass

# Fixed run order, not a setting. Gating before gain, so the volume step never
# turns up the gaps between words.
ORDER = ("highpass", "denoise", "expander", "agc")



def _positive(section: dict, *names: str) -> None:
    for name in names:
        if not section[name] > 0:
            raise ValueError(f"audio_conditioning {name} must be positive, got {section[name]!r}")


def _highpass(s: dict, sr: int, window: int) -> Step:
    _positive(s, "cutoff_hz")
    return Highpass(s["cutoff_hz"], sr, window)


def _denoise(s: dict, sr: int, window: int) -> Step:
    _positive(s, "strength", "noise_rise_db_per_sec")
    if not 0.0 <= s["smoothing"] < 1.0:
        raise ValueError(f"audio_conditioning smoothing must be in [0, 1), got {s['smoothing']!r}")
    return Denoise(s["strength"], s["min_gain_db"], s["noise_rise_db_per_sec"],
                   s["smoothing"], _frames_per_sec(sr, window))


def _expander(s: dict, sr: int, window: int) -> Step:
    _positive(s, "below_voice_db", "max_cut_db", "voice_release_db_per_sec")
    return Expander(s["below_voice_db"], s["max_cut_db"], s["voice_release_db_per_sec"],
                    window, _frames_per_sec(sr, window))


def _agc(s: dict, sr: int, window: int) -> Step:
    _positive(s, "max_gain_db", "slew_db_per_sec")
    return Agc(s["target_dbfs"], s["max_gain_db"], s["slew_db_per_sec"],
               window, _frames_per_sec(sr, window))


def _frames_per_sec(sr: int, window: int) -> float:
    return sr / (window // 2)


# step name -> builder(section, sr, window).
_BUILDERS: Dict[str, Callable[[dict, int, int], Step]] = {
    "highpass": _highpass, "denoise": _denoise, "expander": _expander, "agc": _agc,
}


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
            steps.append(_BUILDERS[name](section, sr, window))
    return Chain(steps, window)
