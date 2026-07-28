"""Qt stylesheets for the desktop UI.

There is one sheet — `styles/app.qss` — rendered once per theme from the colours
in `tokens.json`, the same file the web front-end reads (see `src/ui/theme.py`).
`DARK` and `LIGHT` keep their names so existing callers are unchanged.
"""

from __future__ import annotations

from src.ui.theme import render_qss

DARK = render_qss("dark")
LIGHT = render_qss("light")

LEVEL_STATES = ("healthy", "low", "clipping")


def set_level_state(bar, state: str) -> None:
    """Show the microphone meter as healthy, too quiet to trust, or clipping.

    The three colours live in `app.qss` like every other colour, so the meter
    follows the active theme instead of staying dark-themed on a light window.
    This only sets the property the stylesheet selects on.

    Qt does not re-evaluate a property selector on its own, hence the repolish —
    and only doing it when the state actually changes keeps it off the hot path
    of the level timer, which fires many times a second.
    """
    if state not in LEVEL_STATES:
        raise ValueError(f"Unknown meter state {state!r}; expected one of {LEVEL_STATES}")
    if bar.property("level") == state:
        return
    bar.setProperty("level", state)
    bar.style().unpolish(bar)
    bar.style().polish(bar)
