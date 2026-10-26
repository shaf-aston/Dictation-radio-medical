"""Which Parakeet voice the radiologist dictates with: the choices, their
labels, and switching between them.

Both front-ends call this module, so checking a choice, saving it and loading
it cannot drift apart: written once per front-end, the web checked the name
and the desktop did not.
"""

from __future__ import annotations

from typing import Dict, List

from src.core.settings import Settings
from src.dictation.asr.engines.parakeet_engine import STANDARD, available_models
from src.dictation.asr.factory import describe_engine, parakeet_tier


def label(name: str) -> str:
    """What the picker shows for a ``parakeet_model`` value."""
    return "Standard" if name == STANDARD else f"Tuned: {name}"


def choices() -> List[Dict[str, str]]:
    """Every installed voice, standard first, as picker entries."""
    return [{"value": name, "label": label(name)} for name in available_models()]


def switch(settings: Settings, name: str) -> str:
    """Save *name* as the voice and load it now, so the next dictation's first
    words do not wait on a 2.4 GB load. Blocks while it loads: call it off the
    UI thread. Returns what is now transcribing (:func:`describe_engine`).

    Raises ``ValueError`` for a name that is not installed. If the load fails,
    the previous voice is saved back and the error raised: the radiologist must
    not be left on a voice that cannot run while the picker says it is chosen.
    """
    if name not in available_models():
        raise ValueError(f"Voice model {name!r} is not installed. Available: {available_models()}")
    previous = settings.get("parakeet_model")
    settings.set("parakeet_model", name)
    try:
        tier = parakeet_tier(settings)
        if tier is not None:
            # The tier itself, not the fallback chain: the chain logs a failed
            # tier and carries on, which would hide exactly this failure.
            tier[1].preload()
    except Exception:
        settings.set("parakeet_model", previous)
        raise
    return describe_engine(settings)
