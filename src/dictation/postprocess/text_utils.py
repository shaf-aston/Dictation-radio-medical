"""Stages 3 and 9 — whitespace normalisation and sentence capitalisation."""

from __future__ import annotations

import re
from typing import List


def normalize_spaces(text: str) -> str:
    """Collapse horizontal whitespace and remove spaces before punctuation."""
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?)])", r"\1", text)
    return text.strip()


# Tokens whose canonical case is already uppercase — leave them alone.
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

    parts = re.split(r"([.!?]+\s+)", text)
    out: List[str] = []
    for i in range(0, len(parts), 2):
        out.append(_cap(parts[i]))
        if i + 1 < len(parts):
            out.append(parts[i + 1])
    return "".join(out)
