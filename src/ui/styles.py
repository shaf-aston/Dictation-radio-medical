"""Qt stylesheets for the desktop UI.

Sheets live as `.qss` files in `src/ui/styles/` and are read on import; the
`DARK` and `LIGHT` names are kept as module-level constants for compatibility
with existing callers (`from src.ui.styles import DARK, LIGHT`).
"""

from pathlib import Path

_STYLES_DIR = Path(__file__).resolve().parent / "styles"

DARK = (_STYLES_DIR / "dark.qss").read_text(encoding="utf-8")
LIGHT = (_STYLES_DIR / "light.qss").read_text(encoding="utf-8")
