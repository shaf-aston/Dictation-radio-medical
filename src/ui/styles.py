"""Qt stylesheets for the desktop UI.

Sheets live as `.qss` files in `src/ui/styles/` and are read on import; the
`DARK` and `LIGHT` names are kept as module-level constants for compatibility
with existing callers (`from src.ui.styles import DARK, LIGHT`).
"""

from pathlib import Path

_STYLES_DIR = Path(__file__).resolve().parent / "styles"

DARK = (_STYLES_DIR / "dark.qss").read_text(encoding="utf-8")
LIGHT = (_STYLES_DIR / "light.qss").read_text(encoding="utf-8")

# Microphone level meter colors and styling
COLOR_HEALTHY = "#4CAF50"
COLOR_CLIPPING = "#F44336"
COLOR_LOW = "#FF9800"
LEVEL_BAR_STYLESHEET = (
    "QProgressBar {{ border: 1px solid #555; border-radius: 3px; background: #222; }}"
    "QProgressBar::chunk {{ background: {color}; border-radius: 2px; }}"
)
