"""What a template field is, declared once for every caller.

A field is the part of a template the radiologist still has to fill in or
delete: ``[Right / Left]``, ``[X]``, ``[With gadolinium contrast if indicated.]``.
Three things read this definition and must never disagree:

* the release gate, which warns before a report leaves with a field in it;
* field navigation (Tab / Shift+Tab, or a dictation-mic button), which jumps
  from one field to the next;
* dictation into a selected pick-list, where saying "left" answers
  ``[Right / Left]`` with the option's own spelling.

Any square-bracketed span on one line is a field. Dictation never produces a
square bracket ("open bracket" gives a parenthesis), and the shipped templates
spell their fields twelve different ways, so a narrower rule misses most of
them: the old SHOUTED-names-only rule saw 1 of the 148. A ``{{name}}``
placeholder is a field too. A bare number with a unit (``[5 mm]``) is ordinary
text.

``FIELD_PATTERN`` is written to be valid in both Python and JavaScript; the web
page receives it in its bootstrap payload instead of keeping its own copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

#: One field: a bracketed span on one line, or a ``{{name}}`` placeholder.
FIELD_PATTERN = r"\[[^\[\]\n]{1,300}\]|\{\{[^{}\n]{1,60}\}\}"

#: Bracketed text that is a measurement, not a field: ``[5 mm]``, ``[2.5cm]``.
NOT_A_FIELD_PATTERN = r"^\[\s*\d+(?:\.\d+)?\s*(?:mm|cm|m|ml|cc|%)?\s*\]$"

_FIELD_RE = re.compile(FIELD_PATTERN)
_NOT_A_FIELD_RE = re.compile(NOT_A_FIELD_PATTERN, re.IGNORECASE)


@dataclass(frozen=True)
class Field:
    """One field in the report text."""

    start: int
    end: int
    text: str  # with its brackets

    @property
    def inner(self) -> str:
        """The field's wording without brackets or braces."""
        return self.text.strip("[]{}").strip()

    @property
    def prefix(self) -> str:
        """Fixed wording before the choice: ``"IV contrast:"`` in ``[IV contrast: yes / no]``."""
        inner = self.inner
        colon = inner.find(":")
        if colon >= 0 and "/" in inner[colon:]:
            return inner[: colon + 1]
        return ""

    @property
    def options(self) -> Tuple[str, ...]:
        """The choices of a pick-list (``[Right / Left]``); empty for a free field."""
        rest = self.inner[len(self.prefix):]
        if "/" not in rest:
            return ()
        return tuple(o.strip() for o in rest.split("/") if o.strip())


def find_fields(text: str) -> Tuple[Field, ...]:
    """Every field in *text*, in reading order."""
    return tuple(
        Field(m.start(), m.end(), m.group(0))
        for m in _FIELD_RE.finditer(text)
        if not _NOT_A_FIELD_RE.match(m.group(0))
    )


def unfilled_fields(text: str) -> Tuple[str, ...]:
    """The wording of each field still in *text*, once each, in order."""
    return tuple(dict.fromkeys(f.inner for f in find_fields(text)))


def next_field(text: str, after: int) -> Optional[Field]:
    """The first field starting at or after *after*, or None. Does not wrap."""
    return next((f for f in find_fields(text) if f.start >= after), None)


def previous_field(text: str, before: int) -> Optional[Field]:
    """The last field ending at or before *before*, or None. Does not wrap."""
    found = [f for f in find_fields(text) if f.end <= before]
    return found[-1] if found else None


def _normalise(words: str) -> str:
    return " ".join(re.sub(r"[^\w\s-]", " ", words.lower()).split())


def resolve_choice(field_text: str, dictated: str) -> Optional[str]:
    """What a pick-list becomes when the dictation into it is one of its options.

    Returns the option in the template's own spelling (with any fixed prefix),
    or None when *dictated* is not exactly one option, in which case the
    dictation stands as spoken. Case, spacing and punctuation are ignored:
    the recogniser may write "Left." for a spoken "left".
    """
    field = Field(0, len(field_text), field_text)
    spoken = _normalise(dictated)
    if not spoken:
        return None
    for option in field.options:
        if _normalise(option) == spoken:
            return f"{field.prefix} {option}" if field.prefix else option
    return None
