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

import logging
import re
from typing import Callable, List, Optional, Pattern, Tuple, Union

# ---------------------------------------------------------------------------
# Correction hook
# ---------------------------------------------------------------------------
# Optional callback fired whenever an explicit "X correct word Y" command
# resolves. The recording session sets this (via set_correction_hook) so the
# cloud training collector can capture the (wrong → correct) pair with audio.
# Left None in tests / offline use so this module has no outward dependency.
logger = logging.getLogger(__name__)

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
            logger.warning("Correction hook failed", exc_info=True)


# ---------------------------------------------------------------------------
# 1. Voice correction commands
# ---------------------------------------------------------------------------

#: One spoken word: letters, digits, apostrophes and hyphens, and a decimal
#: number whole ("2.5"). Starting only where a word starts keeps the match
#: linear on long input, and keeps "2.5" from being split at its point.
_WORD = r"(?<![\w'.-])[A-Za-z0-9'-]+(?:\.\d+)?"
_CORRECTION_PATTERN1 = re.compile(
    rf"({_WORD})(\s*)(?:[,.!?;:]*)\s+(?:correct(?:\s+the)?\s+word)(?:\s+to)?\s+({_WORD})\b",
    re.IGNORECASE,
)
_CORRECTION_PATTERN2 = re.compile(
    rf"(?:^|[\s\n])(?:correct(?:\s+the)?\s+word)(?:\s+to)?\s+({_WORD})\b",
    re.IGNORECASE,
)
_PREV_WORD_PATTERN = re.compile(rf"({_WORD})(?!.*[A-Za-z0-9])", re.DOTALL)


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

# A sentence cannot end on these, so a punctuation word right after one is an
# ordinary word: "over a period of", "the rectum and colon", "a comma shaped
# lesion", "a new line has been placed". One rule for every command, not a
# list of nouns per command.
_FUNCTION_WORDS = frozenset(
    "a an the this that these those its their his her our your any each every some "
    "of in on at to for with without by from into onto via per and or nor but no".split()
)
_LAST_WORD = re.compile(r"([A-Za-z]+)[^\S\n]*$")

# Anatomical/medical adjectives that turn "colon" and "period" into nouns.
_COLON_ANATOMY_BEFORE = re.compile(
    r"\b(?:ascending|descending|transverse|sigmoid|hepatic|splenic|left|right|"
    r"entire|proximal|distal|redundant|tortuous|dilated|normal)\s+$",
    re.IGNORECASE,
)
_PERIOD_MEDICAL_BEFORE = re.compile(
    r"\b(?:menstrual|postoperative|post-operative|recovery|washout|"
    r"follow-up|refractory|latent|incubation)\s+$",
    re.IGNORECASE,
)
#: Where a heading's label starts: "Clinical history colon" -> "Clinical history:".
_LABEL_START = re.compile(r"[.\n:;!?]")
_MAX_LABEL_WORDS = 4


def _after_function_word(before: str) -> bool:
    last = _LAST_WORD.search(before)
    return bool(last) and last.group(1).lower() in _FUNCTION_WORDS


def _replace_colon(match: re.Match) -> str:
    """A colon ends a short label (a heading), never a sentence about the colon."""
    before = match.string[:match.start()]
    label = _LABEL_START.split(before)[-1].split()
    if not 1 <= len(label) <= _MAX_LABEL_WORDS or _COLON_ANATOMY_BEFORE.search(before):
        return match[0]
    return ":"


def _replace_period(match: re.Match) -> str:
    if _PERIOD_MEDICAL_BEFORE.search(match.string[:match.start()]):
        return match[0]
    return "."


def _command(replacement: Union[str, Callable[[re.Match], str]]) -> Callable[[re.Match], str]:
    def apply(match: re.Match) -> str:
        if _after_function_word(match.string[:match.start()]):
            return match[0]
        return replacement(match) if callable(replacement) else replacement
    return apply


_SPOKEN_PATTERNS: List[Tuple[Pattern, Callable[[re.Match], str]]] = [
    (re.compile(pattern, re.IGNORECASE), _command(replacement))
    for pattern, replacement in (
        (r"\bfull[- ]?stop\b", "."),
        (r"\bperiod\b", _replace_period),
        (r"\bcomma\b", ","),
        (r"\bnew\s+line\b", "\n"),
        (r"\bnewline\b", "\n"),
        (r"\bnew\s+paragraph\b", "\n\n"),
        (r"\bopen\s+(?:round\s+)?(?:bracket|brackets|parenthesis|parentheses|paren)\b", "("),
        (r"\bclosed?\s+(?:round\s+)?(?:bracket|brackets|parenthesis|parentheses|paren)\b", ")"),
        (r"\bopen\s+square\s+brackets?\b", "["),
        (r"\bclosed?\s+square\s+brackets?\b", "]"),
        # Before a word it marks a query and stays on that word: "?PE".
        (r"\bquestion\s+mark[^\S\n]+(?=\w)", " ?"),
        (r"\bquestion\s+mark\b", "?"),
        (r"\bexclamation\s+mark\b", "!"),
        # Before "colon", which would otherwise turn "semi-colon" into "semi-:".
        (r"\bsemi[- ]?colon\b", ";"),
        (r"\bcolon\b", _replace_colon),
        # Joins the words either side: "ultrasound hyphen guided" is one word.
        (r"\s*\bhyphen\b\s*", "-"),
        (r"\bdash\b", " – "),
        (r"\b(?:new|next)\s+section\b", "\n\n"),
    )
]


#: Every spoken punctuation command, as one pattern. What a command writes
#: (a break, a stop) joins with the text on either side, so the live
#: post-processor never splits its text beside one (incremental.py).
SPOKEN_COMMANDS = "|".join(f"(?:{pattern.pattern})" for pattern, _ in _SPOKEN_PATTERNS)


#: "Slash" joins the words either side ("right slash left" -> "right/left").
#: It skips the function-word guard: "and slash or" is the whole point, and a
#: report never means the word itself. Commas the recogniser hears in the pause
#: around it ("right, slash, left") go too.
_SLASH = re.compile(r"[^\S\n]*[,;]?[^\S\n]*\b(?:forward\s+)?slash\b[,;]?[^\S\n]*", re.IGNORECASE)
#: Stray commas the recogniser puts in the pause before a bracket command.
_COMMA_BEFORE_CLOSE = re.compile(r"[,;]\s*([)\]])")
_COMMA_AFTER_OPEN = re.compile(r"([(\[])\s*[,;]\s*")

SPOKEN_COMMANDS = SPOKEN_COMMANDS + "|(?:" + _SLASH.pattern + ")"


def apply_spoken_commands(text: str) -> str:
    """Replace spoken punctuation tokens with their symbol equivalents."""
    text = _SLASH.sub("/", text)
    for pattern, repl in _SPOKEN_PATTERNS:
        text = pattern.sub(repl, text)
    text = _COMMA_BEFORE_CLOSE.sub(r"\1", text)
    return _COMMA_AFTER_OPEN.sub(r"\1", text)


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
