"""Stages 1–2: voice-edit commands and spoken punctuation.

* ``apply_correction_commands`` handles ``"<word> correct word <new>"``
  and the bare ``"correct word <new>"`` forms.
* ``apply_spoken_commands`` substitutes spoken punctuation tokens
  (``"full stop"`` → ``.``) while guarding against medical context
  (``ascending colon`` stays a noun, ``menstrual period`` stays a noun).
* ``apply_scratch_commands``: "scratch that" removes the last thing said.
* ``apply_phrase_commands``: "insert <quick phrase name>" writes the phrase.

Commands that only edit text live here, as pipeline stages, so both front-ends
get them. Commands that move the caret ("next field") belong to the front-end.
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


# ---------------------------------------------------------------------------
# 3. "Scratch that": remove the last thing said
# ---------------------------------------------------------------------------

#: The command, with whatever punctuation the recogniser put around it.
SCRATCH_PATTERN = re.compile(r"[,;:]?\s*\bscratch\s+that\b[.,!?;:]*", re.IGNORECASE)
_BOUNDARY = ".!?\n"


def _start_of_last_sentence(text: str) -> int:
    """Index where the last sentence of *text* starts (after a boundary)."""
    core = text.rstrip()
    if core and core[-1] in ".!?":
        core = core[:-1]  # the sentence just finished is the one to remove
    return max(core.rfind(c) for c in _BOUNDARY) + 1


def apply_scratch_commands(text: str) -> str:
    """Remove the sentence before each "scratch that", and the command itself.

    "The last thing said" is the sentence being spoken, or, if the command
    starts a new sentence, the one just finished: a long pause already ends a
    sentence (``ledger.close_sentence``), so a radiologist who stops, thinks,
    and says "scratch that" removes what they said before stopping. Text before
    a new line is never reached across: a heading stays.
    """
    while (m := SCRATCH_PATTERN.search(text)) is not None:
        before, rest = text[: m.start()], text[m.end():].lstrip(" ")
        keep = before[: _start_of_last_sentence(before)].rstrip(" ")
        joiner = " " if keep and rest and not keep.endswith("\n") else ""
        text = keep + joiner + rest
    return text


# ---------------------------------------------------------------------------
# 4. "Insert <phrase>": a quick phrase by voice
# ---------------------------------------------------------------------------

_phrase_cache: Tuple[int, Optional[Pattern], dict] = (0, None, {})


def _phrase_pattern() -> Tuple[Optional[Pattern], dict]:
    """The "insert <name>" pattern for the current quick phrases, cached.

    Rebuilt when the phrase library object changes (``macros.reload_macros``
    replaces it), so a phrase added and reloaded works without a restart.
    """
    global _phrase_cache
    from src.medical import macros

    library = macros.MACROS
    if _phrase_cache[0] != id(library):
        by_name = {}
        for entries in library.values():
            for label, phrase in entries:
                by_name.setdefault(" ".join(label.lower().split()), phrase)
        names = sorted(by_name, key=len, reverse=True)
        pattern = re.compile(
            r"\binsert\s+(" + "|".join(
                r"\s+".join(re.escape(w) for w in name.split()) for name in names
            ) + r")\b[.,]?",
            re.IGNORECASE,
        ) if names else None
        _phrase_cache = (id(library), pattern, by_name)
    return _phrase_cache[1], _phrase_cache[2]


def apply_phrase_commands(text: str) -> str:
    """Replace "insert <quick phrase name>" with the phrase it names.

    Only an exact name after the word "insert" fires, so "insert a drain"
    stays as dictated. Names are matched without case.
    """
    pattern, by_name = _phrase_pattern()
    if pattern is None:
        return text
    return pattern.sub(lambda m: by_name[" ".join(m.group(1).lower().split())], text)
