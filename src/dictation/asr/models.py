"""Which Whisper model names this build knows, and the fallback when a setting names another.

Neutral to which engine runs: both front-ends read these to fill the model
picker and to validate ``model_size`` / ``live_model_size`` in settings.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Supported model sizes in order of speed (fastest first).
#
# The `.en` models are English-only. For English dictation they are both faster and
# more accurate than the same-sized multilingual model, because none of the capacity
# is spent on the other 98 languages: so they are the right default here and they
# have to be selectable. Leaving them out silently downgraded anyone whose settings
# named one: the picker ignored the unknown value and fell back to its first entry.
SUPPORTED_MODELS = [
    "tiny.en", "tiny",
    "base.en", "base",
    "small.en", "small",
    "medium.en", "medium",
    "large-v2", "large-v3",
]

# What to fall back to when a setting names a model this build does not know.
DEFAULT_MODEL = "base.en"


def resolve_model(name: Optional[str]) -> str:
    """Return *name* if this build supports it, else :data:`DEFAULT_MODEL`, loudly.

    Both front-ends used to drop an unrecognised model on the floor without a word:
    the web app reassigned it, and the desktop combo box ignored ``setCurrentText``
    for a value it had no item for and stayed on its first entry. Either way the
    radiologist got a different model from the one their settings named, with no
    hint that it had happened. Falling back is fine; doing it in silence is not.
    """
    if name in SUPPORTED_MODELS:
        return name  # type: ignore[return-value]
    logger.warning(
        "Model %r is not one of %s: using %s instead. Check 'model_size' in "
        "dictation_settings.json.", name, SUPPORTED_MODELS, DEFAULT_MODEL,
    )
    return DEFAULT_MODEL
