"""
MSK Radiology Quick-Phrase Macro Library.

Loads from data/macros.json for user editability.
Each entry is [button_label, report_text].
"""

import logging
from typing import Dict, List, Tuple

from src.core.json_store import read_json
from src.features.file_manager import macros_file

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

def reload_macros() -> None:
    """Hot-reload macros from JSON. Updates global MACROS and REGION_ORDER."""
    global MACROS, REGION_ORDER

    if new_macros := _load_macros():
        MACROS = new_macros
        REGION_ORDER = list(MACROS.keys())
        logger.info(f"Reloaded {len(MACROS)} macro regions")
    else:
        raise ValueError("Failed to load macros from JSON")

MACROS: MacroLibrary = _load_macros()

# Fallback if load fails
if not MACROS:
    logger.warning("Using fallback macros")
    MACROS = {
        "General": [
            ("Normal study", "No acute abnormality identified."),
            ("No fracture", "No acute fracture or dislocation."),
        ],
        "Knee": [
            ("ACL intact", "Anterior cruciate ligament intact."),
        ],
    }

# Region display order
REGION_ORDER: List[str] = list(MACROS.keys())
