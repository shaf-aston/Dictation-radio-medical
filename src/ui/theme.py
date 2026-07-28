"""The only module that reads `tokens.json`.

Both front-ends get their colours from here, so a colour is written down once and
the desktop app and the web app cannot drift apart:

    desktop  styles/app.qss  --render_qss()-->  Qt stylesheet
    web      app.html <head> --css_variables()-->  :root custom properties

`app.css` and `app.qss` therefore contain no hex values of their own. Adding a
colour means adding a token here, which is the point.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_TOKENS_FILE = Path(__file__).resolve().parent / "tokens.json"
_QSS_FILE = Path(__file__).resolve().parent / "styles" / "app.qss"

THEMES = ("dark", "light")

# `room2`/`textDim` read naturally in Python and JSON; CSS wants `--room-2`.
CSS_NAMES = {"room2": "room-2", "textDim": "text-dim"}


def css_name(key: str) -> str:
    """The custom-property name a token is published under."""
    return CSS_NAMES.get(key, key)

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _load() -> dict[str, dict[str, str]]:
    raw = json.loads(_TOKENS_FILE.read_text(encoding="utf-8"))
    themes = {name: raw[name] for name in THEMES}

    # Fail loud: a missing or malformed token would otherwise reach the screen as
    # a silent black-on-black widget, which is far harder to spot than a crash.
    reference = sorted(themes["dark"])
    for name, values in themes.items():
        if sorted(values) != reference:
            missing = set(reference) ^ set(values)
            raise ValueError(f"tokens.json: '{name}' theme differs by {sorted(missing)}")
        for key, value in values.items():
            if not _HEX.match(value):
                raise ValueError(f"tokens.json: {name}.{key} is not a #rrggbb colour: {value!r}")
    return themes


_THEMES = _load()


def tokens(theme: str) -> dict[str, str]:
    """The colour set for one theme. Raises on an unknown theme name."""
    try:
        return dict(_THEMES[theme])
    except KeyError:
        raise ValueError(f"Unknown theme {theme!r}; expected one of {THEMES}") from None


def _declarations(theme: str) -> str:
    return "".join(
        f"    --{css_name(key)}: {value};\n"
        for key, value in tokens(theme).items()
    )


def css_variables() -> str:
    """The `<style>` body the web page needs: OS preference plus an explicit override.

    Emitted into the page rather than into `app.css` so there is still exactly one
    source of truth on disk.
    """
    return (
        f":root {{\n{_declarations('dark')}}}\n"
        f"@media (prefers-color-scheme: light) {{\n"
        f":root {{\n{_declarations('light')}}}\n}}\n"
        f':root[data-theme="dark"] {{\n{_declarations("dark")}}}\n'
        f':root[data-theme="light"] {{\n{_declarations("light")}}}\n'
    )


def render_qss(theme: str) -> str:
    """`styles/app.qss` with every `{{token}}` replaced. Qt has no variables of its own."""
    values = tokens(theme)
    sheet = _QSS_FILE.read_text(encoding="utf-8")

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise ValueError(f"app.qss uses {{{{{key}}}}}, which is not a token in tokens.json")
        return values[key]

    return _PLACEHOLDER.sub(substitute, sheet)
