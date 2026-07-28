"""Qt stylesheets for the desktop UI.

There is one sheet — `styles/app.qss` — rendered once per theme from the colours
in `tokens.json`, the same file the web front-end reads (see `src/ui/theme.py`).
`DARK` and `LIGHT` keep their names so existing callers are unchanged.
"""

from __future__ import annotations

from src.ui.theme import render_qss, tokens

DARK = render_qss("dark")
LIGHT = render_qss("light")

# The microphone level meter. Its three states are colours with a meaning the
# rest of the app already uses: healthy, too quiet to trust, clipping.
_DARK = tokens("dark")
COLOR_HEALTHY = _DARK["ok"]
COLOR_LOW = _DARK["warn"]
COLOR_CLIPPING = _DARK["rec"]

LEVEL_BAR_STYLESHEET = (
    "QProgressBar {{ border: 1px solid %(edge)s; border-radius: 3px; background: %(room)s; }}"
    "QProgressBar::chunk {{ background: {color}; border-radius: 2px; }}"
) % _DARK
