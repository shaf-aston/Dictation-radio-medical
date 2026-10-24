"""
MSK Radiology Quick-Phrase Macro Library.

Loads from data/macros.json for user editability.
Each entry is [button_label, report_text].
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from src.core.json_store import read_json
from src.core.paths import macros_file

logger = logging.getLogger(__name__)

# Type alias for clarity
MacroEntry = Tuple[str, str]
MacroLibrary = Dict[str, List[MacroEntry]]

def _load_macros() -> MacroLibrary:
    """Load macros from JSON file."""
    data = read_json(macros_file(), {})
    try:
        # Convert JSON lists [label, text] to tuples (label, text)
        return {
            region: [tuple(item) for item in phrases]
            for region, phrases in data.items()
        }
    except (AttributeError, TypeError) as exc:
        logger.error("Malformed macros file: %s", exc)
        return {}

#: The phrases used when there is no usable ``macros.json``.
_BUILT_IN: MacroLibrary = {
    "General": [
        ("Normal study", "No acute abnormality identified."),
        ("No fracture", "No acute fracture or dislocation."),
    ],
    "Knee": [
        ("ACL intact", "Anterior cruciate ligament intact."),
    ],
}


def _library() -> MacroLibrary:
    """The file's phrases, or the built-in ones: one rule for start-up and reload."""
    if loaded := _load_macros():
        return loaded
    logger.warning("No usable %s; using the built-in phrases", macros_file())
    return dict(_BUILT_IN)


def reload_macros() -> None:
    """Re-read the phrases (a new library object, which voice commands notice)."""
    global MACROS, REGION_ORDER
    MACROS = _library()
    REGION_ORDER = list(MACROS.keys())
    logger.info("Reloaded %d macro regions", len(MACROS))


MACROS: MacroLibrary = _library()

# Region display order
REGION_ORDER: List[str] = list(MACROS.keys())
