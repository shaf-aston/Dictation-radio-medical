"""build_chain: the only setting-name -> step mapping."""

from __future__ import annotations

import math
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

# Where the agc may aim: above it clips the voice, below it turns it to nothing.
TARGET_DBFS_RANGE = (-60.0, 0.0)


def _bad(name: str, value: object, need: str) -> ValueError:
    return ValueError(f"audio_conditioning {name} must be {need}, got {value!r}")


def _check_section(name: str, section: object, default: dict) -> dict:
    """Merged over the default. Every knob a known one, `on` true or false,
    every other knob a finite number."""
    if not isinstance(section, dict):
        raise _bad(name, section, "a section of knobs")
    merged = {**default, **section}
    for key, value in merged.items():
        if key not in default:
            raise _bad(f"{name}.{key}", value, "a known knob")
        if key == "on":
            if not isinstance(value, bool):
                raise _bad(f"{name}.on", value, "true or false")
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(value)):
            raise _bad(f"{name}.{key}", value, "a finite number")
    return merged


def _positive(section: dict, *names: str) -> None:
    for name in names:
        if not section[name] > 0:
            raise _bad(name, section[name], "positive")


def _highpass(s: dict, sr: int, window: int) -> Step:
    _positive(s, "cutoff_hz")
    if not s["cutoff_hz"] < sr / 2:
        raise _bad("cutoff_hz", s["cutoff_hz"], f"below {sr / 2:g} Hz, or every sound is cut")
    return Highpass(s["cutoff_hz"], sr, window)


def _denoise(s: dict, sr: int, window: int) -> Step:
    _positive(s, "strength", "noise_rise_db_per_sec")
    if not 0.0 <= s["smoothing"] < 1.0:
        raise _bad("smoothing", s["smoothing"], "in [0, 1)")
    if not s["min_gain_db"] < 0:
        raise _bad("min_gain_db", s["min_gain_db"], "negative (how far noise may be turned down)")
    return Denoise(s["strength"], s["min_gain_db"], s["noise_rise_db_per_sec"],
                   s["smoothing"], _frames_per_sec(sr, window))


def _expander(s: dict, sr: int, window: int) -> Step:
    _positive(s, "below_voice_db", "max_cut_db", "voice_release_db_per_sec")
    return Expander(s["below_voice_db"], s["max_cut_db"], s["voice_release_db_per_sec"],
                    window, _frames_per_sec(sr, window))


def _agc(s: dict, sr: int, window: int) -> Step:
    _positive(s, "max_gain_db", "slew_db_per_sec", "voice_release_db_per_sec", "adapt_within_db")
    low, high = TARGET_DBFS_RANGE
    if not low <= s["target_dbfs"] < high:
        raise _bad("target_dbfs", s["target_dbfs"], f"in [{low:g}, {high:g})")
    return Agc(s["target_dbfs"], s["max_gain_db"], s["slew_db_per_sec"],
               s["voice_release_db_per_sec"], s["adapt_within_db"],
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
    if isinstance(sr, bool) or not isinstance(sr, int) or sr <= 0:
        raise ValueError(f"sample rate must be a positive integer, got {sr!r}")
    base = get_default("audio_conditioning") or {}
    merged = {**base, **(cfg or {})}
    window = merged.pop("window_samples")
    for name in merged:
        if name not in _BUILDERS:
            raise ValueError(f"unknown audio_conditioning section {name!r}")
    steps = []
    for name in ORDER:
        if name not in merged:
            continue
        section = _check_section(name, merged[name], base.get(name, {}))
        if section["on"]:
            steps.append(_BUILDERS[name](section, sr, window))
    # Chain checks the window: it is the one that needs it right.
    return Chain(steps, window)
