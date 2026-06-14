"""Stages 1–2 — voice-edit commands and spoken punctuation.

* ``apply_correction_commands`` handles ``"<word> correct word <new>"``
  and the bare ``"correct word <new>"`` forms.
* ``apply_spoken_commands`` substitutes spoken punctuation tokens
  (``"full stop"`` → ``.``) while guarding against medical context
  (``ascending colon`` stays a noun, ``menstrual period`` stays a noun).
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Pattern, Tuple, Union

# ---------------------------------------------------------------------------
# Correction hook
# ---------------------------------------------------------------------------
# Optional callback fired whenever an explicit "X correct word Y" command
# resolves. The recording session sets this (via set_correction_hook) so the
# cloud training collector can capture the (wrong → correct) pair with audio.
# Left None in tests / offline use so this module has no outward dependency.
_correction_hook: Optional[Callable[[str, str], None]] = None


def set_correction_hook(fn: Optional[Callable[[str, str], None]]) -> None:
    """Register (or clear) the callback invoked on each resolved correction."""
    global _correction_hook
    _correction_hook = fn


def _fire_correction_hook(wrong: str, correct: str) -> None:
    if _correction_hook is not None and wrong and correct and wrong.lower() != correct.lower():
        try:
            _correction_hook(wrong, correct)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. Voice correction commands
# ---------------------------------------------------------------------------

_CORRECTION_PATTERN1 = re.compile(
    r"([A-Za-z0-9'-]+)(\s*)(?:[,.!?;:]*)\s+(?:correct(?:\s+the)?\s+word)(?:\s+to)?\s+([A-Za-z0-9'-]+)\b",
    re.IGNORECASE,
)
_CORRECTION_PATTERN2 = re.compile(
    r"(?:^|[\s\n])(?:correct(?:\s+the)?\s+word)(?:\s+to)?\s+([A-Za-z0-9'-]+)\b",
    re.IGNORECASE,
)
_PREV_WORD_PATTERN = re.compile(r"\b([A-Za-z0-9'-]+)\b(?!.*\b[A-Za-z0-9'-]+\b)")


def _correction1_repl(m: re.Match) -> str:
    _fire_correction_hook(m[1], m[3])
    return f"{m[3]}{m[2]}"


def apply_correction_commands(text: str) -> str:
    """Handle ``"X correct word Y"`` and ``"correct word Y"`` voice edits."""
    while True:
        text, n = _CORRECTION_PATTERN1.subn(_correction1_repl, text, count=1)
        if n == 0:
            break
    while True:
        m = _CORRECTION_PATTERN2.search(text)
        if not m:
            break
        new_word = m.group(1)
        cmd_start, cmd_end = m.start(), m.end()
        before = text[:cmd_start]
        m_prev = _PREV_WORD_PATTERN.search(before)
        if not m_prev:
            text = before.rstrip() + text[cmd_end:]
            continue
        ps, pe = m_prev.span(1)
        _fire_correction_hook(m_prev.group(1), new_word)
        text = text[:ps] + new_word + text[pe:cmd_start] + text[cmd_end:]
    return text


# ---------------------------------------------------------------------------
# 2. Spoken punctuation → symbols
# ---------------------------------------------------------------------------

# Anatomical/medical adjectives that turn "colon" and "period" into nouns.
_COLON_ANATOMY_BEFORE = re.compile(
    r"\b(?:ascending|descending|transverse|sigmoid|hepatic|splenic|"
    r"the|entire|proximal|distal|redundant|tortuous|dilated|normal)\s+$",
    re.IGNORECASE,
)
_PERIOD_MEDICAL_BEFORE = re.compile(
    r"\b(?:menstrual|postoperative|post-operative|recovery|washout|"
    r"follow-up|refractory|latent|incubation)\s+$",
    re.IGNORECASE,
)


def _replace_colon(match: re.Match) -> str:
    if _COLON_ANATOMY_BEFORE.search(match.string[:match.start()]):
        return match[0]
    return ":"


def _replace_period(match: re.Match) -> str:
    if _PERIOD_MEDICAL_BEFORE.search(match.string[:match.start()]):
        return match[0]
    return "."


_SPOKEN_PATTERNS: List[Tuple[Pattern, Union[str, Callable[[re.Match], str]]]] = [
    (re.compile(r"\bfull[- ]?stop\b", re.IGNORECASE), "."),
    (re.compile(r"\bperiod\b", re.IGNORECASE), _replace_period),
    (re.compile(r"\bcomma\b", re.IGNORECASE), ","),
    (re.compile(r"\bnew\s+line\b", re.IGNORECASE), "\n"),
    (re.compile(r"\bnewline\b", re.IGNORECASE), "\n"),
    (re.compile(r"\bnew\s+paragraph\b", re.IGNORECASE), "\n\n"),
    (re.compile(r"\bopen\s+bracket\b", re.IGNORECASE), "("),
    (re.compile(r"\bclose\s+bracket\b", re.IGNORECASE), ")"),
    (re.compile(r"\bquestion\s+mark\b", re.IGNORECASE), "?"),
    (re.compile(r"\bexclamation\s+mark\b", re.IGNORECASE), "!"),
    (re.compile(r"\bcolon\b", re.IGNORECASE), _replace_colon),
    (re.compile(r"\bsemi[- ]?colon\b", re.IGNORECASE), ";"),
    (re.compile(r"\bhyphen\b", re.IGNORECASE), "-"),
    (re.compile(r"\bdash\b", re.IGNORECASE), " – "),
    (re.compile(r"\b(?:new|next)\s+section\b", re.IGNORECASE), "\n\n"),
]


def apply_spoken_commands(text: str) -> str:
    """Replace spoken punctuation tokens with their symbol equivalents."""
    for pattern, repl in _SPOKEN_PATTERNS:
        text = pattern.sub(repl, text)  # type: ignore[call-overload]
    return text
