"""Consistency checks on a finished report: side, sex, and a missing impression.

Asked at release, with the unfilled-field question, and never blocking: the
radiologist can always send the report as it is. Three rules:

* **Side.** When the ``EXAMINATION:`` line names exactly one side, a sentence
  naming the other side is flagged. Checked against the exam, not against
  every side word: Minn et al. (J Digit Imaging 2015) found side flags 26%
  precise when any opposite-side word fired, so sentences about the other
  side *on purpose* (comparison, contralateral, prior) are skipped.
* **Sex.** When the patient's sex is recorded, an organ only the other sex
  has is flagged (93% precise in the same study). Skipped when sex is blank
  or anything but F/M: the check is never guessed.
* **Impression.** A report with a ``FINDINGS:`` heading and no impression, or
  an impression heading with nothing under it, is flagged. A free-text report
  with no headings is not checked.

Precision and recall are measured on a labelled corpus by
``scripts/eval/report_checks_corpus.py``; read its caveat before quoting them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Issue:
    """One thing worth a second look before the report leaves."""

    kind: str  # "side", "sex" or "impression"
    message: str
    start: int = 0
    end: int = 0


_EXAM_LINE = re.compile(r"^[ \t]*EXAM(?:INATION)?[ \t]*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
_SIDE = re.compile(r"\b(right|left)\b", re.IGNORECASE)
_BOTH_SIDES = re.compile(r"\b(bilateral|both)\b", re.IGNORECASE)
#: A sentence that mentions the other side on purpose.
_ON_PURPOSE = re.compile(
    r"\b(compar\w*|contralateral|opposite|prior|previous|both|bilateral|"
    r"for reference|asymmetr\w*)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")

_FEMALE_ONLY = re.compile(
    r"\b(uter(?:us|ine)|ovar(?:y|ies|ian)|endometri(?:um|al)|cervix|adnexa\w*|"
    r"fallopian|vagina\w*)\b",
    re.IGNORECASE,
)
_MALE_ONLY = re.compile(
    r"\b(prostat(?:e|ic)|testi(?:s|cle|cles|cular)|testes|scrot(?:um|al)|"
    r"seminal vesicles?|epididym\w*|penile|penis)\b",
    re.IGNORECASE,
)

_FINDINGS_HEADING = re.compile(r"^[ \t]*FINDINGS[ \t]*:", re.IGNORECASE | re.MULTILINE)
_CONCLUSION_HEADING = re.compile(
    r"^[ \t]*(IMPRESSION|CONCLUSION|OPINION|SUMMARY)S?[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
#: The next heading: a line of capitals ending in a colon.
_NEXT_HEADING = re.compile(r"^[ \t]*[A-Z][A-Z /&-]{2,}:", re.MULTILINE)


def exam_side(text: str) -> Optional[str]:
    """``"right"`` or ``"left"`` when the exam line names exactly one side."""
    m = _EXAM_LINE.search(text)
    if not m or "[" in m.group(1) or _BOTH_SIDES.search(m.group(1)):
        return None
    sides = {s.lower() for s in _SIDE.findall(m.group(1))}
    return sides.pop() if len(sides) == 1 else None


def _side_issues(text: str) -> List[Issue]:
    side = exam_side(text)
    if side is None:
        return []
    other = "left" if side == "right" else "right"
    body_start = _EXAM_LINE.search(text).end()  # type: ignore[union-attr]
    issues = []
    for sentence in _SENTENCE.finditer(text, body_start):
        words = sentence.group(0)
        if re.search(rf"\b{other}\b", words, re.IGNORECASE) and not _ON_PURPOSE.search(words):
            issues.append(Issue(
                "side",
                f'Exam is {side.upper()}, but this says {other}: "{words.strip()[:80]}"',
                sentence.start(), sentence.end(),
            ))
    return issues


def _sex_issues(text: str, sex: str) -> List[Issue]:
    code = (sex or "").strip()[:1].upper()
    pattern = {"M": _FEMALE_ONLY, "F": _MALE_ONLY}.get(code)
    if pattern is None:
        return []
    label = "male" if code == "M" else "female"
    seen, issues = set(), []
    for m in pattern.finditer(text):
        word = m.group(0).lower()
        if word not in seen:
            seen.add(word)
            issues.append(Issue(
                "sex", f'Patient is {label}, but the report describes "{m.group(0)}"',
                m.start(), m.end(),
            ))
    return issues


def _impression_issues(text: str) -> List[Issue]:
    if not _FINDINGS_HEADING.search(text):
        return []
    heading = _CONCLUSION_HEADING.search(text)
    if heading is None:
        return [Issue("impression", "The report has findings but no impression")]
    nxt = _NEXT_HEADING.search(text, heading.end())
    body = text[heading.end(): nxt.start() if nxt else len(text)]
    if not re.sub(r"\d+\s*[.)]|[\s.\-]", "", body):
        return [Issue("impression", f"The {heading.group(1).lower()} is empty",
                      heading.start(), heading.end())]
    return []


def check_report(text: str, sex: str = "") -> Tuple[Issue, ...]:
    """Every consistency issue in *text*, side first, then sex, then impression."""
    return tuple(_side_issues(text) + _sex_issues(text, sex) + _impression_issues(text))
