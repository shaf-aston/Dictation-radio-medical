"""Stages 3 and 9: whitespace normalisation and sentence capitalisation."""

from __future__ import annotations

import re
from typing import List

_RE_HSPACE = re.compile(r"[^\S\n]+")
#: A space before punctuation closes up, except before a "?" that leads a
#: word (a query: "Indication: ?PE") and a "." that leads a digit (".5 cm").
_RE_PREPUNCT = re.compile(r"[ \t]+([,;:!)]|\.(?!\d)|\?(?!\w))")
#: A sentence ends at terminal punctuation, or at a line break: what follows
#: "new line" / "new paragraph" (or a heading's colon) starts a sentence.
_RE_SENT = re.compile(r"([.!?]+\s+|\n+)")
#: Spaces the spoken-punctuation stage leaves inside a line break or a bracket.
_RE_SPACE_AT_BREAK = re.compile(r"[ \t]*\n[ \t]*")
_RE_SPACE_IN_BRACKET = re.compile(r"\([ \t]+")
# Whisper decoding near-silence emits stray punctuation runs (", , . . ,").
# After _RE_PREPUNCT strips the inner spaces these collapse to ",,..,": so fold
# any run of two-or-more punctuation marks (optionally space-separated) down to
# its first mark. Single spaced marks are left alone: those are the legitimate
# output of the spoken-punctuation stage (" : ", " . ") and get joined by
# _RE_PREPUNCT below.
#: "?" is not folded: after another mark it is a query marker
#: ("Clinical history: ?scaphoid fracture"), and folding it turned a question
#: into a statement.
_RE_PUNCT_RUN = re.compile(r"([.,;:!?])[.,;:!\s]*[.,;:!]")


def normalize_spaces(text: str) -> str:
    """Collapse horizontal whitespace, strip hallucinated punctuation runs, close gaps."""
    text = _RE_HSPACE.sub(" ", text)
    text = _RE_PUNCT_RUN.sub(r"\1", text)
    text = _RE_PREPUNCT.sub(r"\1", text)
    text = _RE_SPACE_AT_BREAK.sub("\n", text)
    text = _RE_SPACE_IN_BRACKET.sub("(", text)
    return _RE_HSPACE.sub(" ", text).strip()


# Tokens whose canonical case is already uppercase: leave them alone.
_PRESERVE_CAPS = re.compile(
    r"^(CT|MRI|US|CXR|STIR|PDFS|FLAIR|DWI|ADC|ACL|PCL|MCL|LCL|UCL|TFCC|"
    r"SLAP|HAGL|ARCO|SPARCC|DEXA|PET|BMD|FAI|SI|APL|EPB|AVN|BMO)\b"
)


def smart_capitalize(text: str) -> str:
    """Capitalise the first letter of every sentence; preserve known acronyms."""

    def _cap(s: str) -> str:
        s = s.strip()
        if not s:
            return s
        return s if _PRESERVE_CAPS.match(s) else s[0].upper() + s[1:]

    parts = _RE_SENT.split(text)
    out: List[str] = []
    for i in range(0, len(parts), 2):
        out.append(_cap(parts[i]))
        if i + 1 < len(parts):
            out.append(parts[i + 1])
    return "".join(out)
