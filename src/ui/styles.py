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
STATUS_STATES = ("idle", "busy", "rec", "warn", "ok")


def _set_state(widget, name: str, state: str, allowed: tuple[str, ...]) -> None:
    """Put a widget into one of its stylesheet states.

    The colours live in `app.qss` like every other colour, so the widget follows
    the active theme instead of staying dark-themed on a light window. This only
    sets the property the stylesheet selects on.

    Qt does not re-evaluate a property selector on its own, hence the repolish —
    and only doing it when the state actually changes keeps it off the hot path
    of the level timer, which fires many times a second.
    """
    if state not in allowed:
        raise ValueError(f"Unknown {name} state {state!r}; expected one of {allowed}")
    if widget.property(name) == state:
        return
    widget.setProperty(name, state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def set_level_state(bar, state: str) -> None:
    """Show the microphone meter as healthy, too quiet to trust, or clipping."""
    _set_state(bar, "level", state, LEVEL_STATES)


def set_status_state(label, state: str) -> None:
    """Show what the dictation is doing: idle, busy, recording, behind, done."""
    _set_state(label, "state", state, STATUS_STATES)
