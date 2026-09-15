"""Single source of truth for logging configuration.

Call ``setup_logging()`` exactly once from each entry point (the desktop
``app.py`` and the FastAPI ``web_app.py``).  Library modules use the
standard ``logger = logging.getLogger(__name__)`` pattern and inherit
from the root handler installed here.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_CONFIGURED = False


def setup_logging(level: int = logging.INFO, *, fmt: Optional[str] = None) -> None:
    """Initialise the root logger.  Idempotent: safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(level=level, format=fmt or _FORMAT)

    # Third-party noise we know about. httpx logs every request URL at INFO,
    # and the Deepgram engine's URL carries a hundred boosted keywords: one
    # such line fills the developer console. The engine's own timed decode
    # event already says the call happened and what it cost.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

    _CONFIGURED = True
