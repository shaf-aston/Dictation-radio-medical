"""
Critical and urgent findings detection with NegEx-style negation parsing.

Scans radiology report text for high-risk clinical terms before the
radiologist finalises the report. A lightweight negation parser prevents
false-positive alerts for phrases like "no pneumothorax" while still
flagging uncertainty phrases like "cannot exclude pulmonary embolism".
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Negation / uncertainty phrase sets
# ---------------------------------------------------------------------------

# Pre-finding negation triggers (appear BEFORE the term, up to 60 chars back)
_NEGATION_PRE = [
    r"no\s+(?:acute\s+|new\s+|definite\s+|significant\s+)?(?:evidence\s+of\s+)?",
    r"without\s+(?:evidence\s+of\s+)?",
    r"ruled?\s+out\s+",
    r"excluded\s+",
    r"negative\s+for\s+",
    r"unremarkable\s+for\s+",
    r"absent\s+",
    r"absence\s+of\s+",
    r"free\s+of\s+",
    r"no\s+",
]

# Post-finding negation triggers (appear AFTER the term, up to 60 chars ahead)
_NEGATION_POST = [
    r"\s+(?:is|are|was|were)\s+(?:not|absent|excluded|ruled\s+out)",
    r"\s+not\s+(?:seen|identified|detected|noted|demonstrated|visualised|visualized|present|apparent)",
    r"\s+(?:is|are|was|were)\s+not\s+(?:seen|identified|detected|noted|present)",
]

# Uncertainty / hedging triggers (pre-finding; NOT a clean negative)
_UNCERTAINTY_PRE = [
    r"possible\s+",
    r"possibly\s+",
    r"probable\s+",
    r"probably\s+",
    r"likely\s+",
    r"cannot\s+(?:fully\s+)?exclude\s+",
    r"not\s+excluded\s+",
    r"suspicious\s+(?:for\s+)?",
    r"concerning\s+(?:for\s+)?",
    r"query\s+",
    r"(?:may|might|could)\s+(?:represent|be|indicate)\s+",
    r"clinical\s+concern\s+for\s+",
    r"raise[ds]?\s+(?:the\s+)?(?:question|possibility|concern)\s+(?:of\s+)?",
]

_WINDOW = 70  # character window to scan before/after term

# ---------------------------------------------------------------------------
# Critical term lists
# ---------------------------------------------------------------------------

# Level 1 — life-threatening, immediate verbal communication required
_LEVEL_1_TERMS: List[str] = [
    "tension pneumothorax",
    "pneumothorax",
    "aortic dissection",
    "aortic rupture",
    "acute stroke",
    "acute infarct",
    "cerebral infarction",
    "subdural haematoma",
    "subdural hematoma",
    "subdural haemorrhage",
    "subdural hemorrhage",
    "epidural haematoma",
    "epidural hematoma",
    "epidural haemorrhage",
    "subarachnoid haemorrhage",
    "subarachnoid hemorrhage",
    "intracranial haemorrhage",
    "intracranial hemorrhage",
    "midline shift",
    "saddle embolism",
    "bilateral pulmonary embolism",
    "massive pulmonary embolism",
    "pericardial effusion with tamponade",
    "cardiac tamponade",
    "bowel ischaemia",
    "bowel ischemia",
    "mesenteric ischaemia",
    "mesenteric ischemia",
    "splenic rupture",
    "hepatic rupture",
    "pneumoperitoneum",
    "free perforation",
    "cauda equina syndrome",
    "cauda equina",
    "spinal cord compression",
    "acute myocardial infarction",
    "myocardial infarction",
]

# Level 2 — urgent, same-day communication required
_LEVEL_2_TERMS: List[str] = [
    "pulmonary embolism",
    "filling defect",
    "haemothorax",
    "hemothorax",
    "empyema",
    "abscess",
    "haemorrhage",
    "hemorrhage",
    "cord compression",
    "hydrocephalus",
    "acute appendicitis",
    "appendicitis",
    "bowel obstruction",
    "volvulus",
    "intussusception",
    "ruptured aortic aneurysm",
    "ruptured aneurysm",
    "acute cholecystitis",
    "cholangitis",
    "meningitis",
    "encephalitis",
    "acute pyelonephritis",
    "ureteric calculus",
    "testicular torsion",
    "ovarian torsion",
    "ectopic pregnancy",
    "pericardial effusion",
]

# Sort longest first so multi-word phrases match before their substrings
_ALL_TERMS_SORTED: List[Tuple[str, int]] = sorted(
    [(t, 1) for t in _LEVEL_1_TERMS] + [(t, 2) for t in _LEVEL_2_TERMS],
    key=lambda x: len(x[0]),
    reverse=True,
)

# Pre-compile term patterns
_TERM_PATTERNS = [
    (re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE), level)
    for term, level in _ALL_TERMS_SORTED
]

# Pre-compile negation / uncertainty patterns
_NEG_PRE_RE = re.compile(
    r"(?:" + "|".join(_NEGATION_PRE) + r")$",
    re.IGNORECASE,
)
_NEG_POST_RE = re.compile(
    r"^(?:" + "|".join(_NEGATION_POST) + r")",
    re.IGNORECASE,
)
_UNC_PRE_RE = re.compile(
    r"(?:" + "|".join(_UNCERTAINTY_PRE) + r")$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class CriticalFinding:
    term: str
    level: int        # 1 = life-threatening, 2 = urgent
    negated: bool
    uncertain: bool
    context: str      # surrounding text snippet for display


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _check_negation(text: str, start: int, end: int) -> Tuple[bool, bool]:
    """Return ``(is_negated, is_uncertain)`` for the term at ``text[start:end]``."""
    before = text[max(0, start - _WINDOW):start]
    after  = text[end:end + _WINDOW]
    negated  = bool(_NEG_PRE_RE.search(before)) or bool(_NEG_POST_RE.match(after))
    uncertain = not negated and bool(_UNC_PRE_RE.search(before))
    return negated, uncertain


def scan_for_critical_findings(text: str) -> List[CriticalFinding]:
    """
    Scan *text* for critical and urgent radiology findings.

    Returns a list of :class:`CriticalFinding` objects, sorted by severity
    (Level 1 first).  Cleanly negated Level-2 findings are suppressed;
    uncertain and Level-1 findings are always returned so the radiologist
    can confirm communication or clarify the report.
    """
    findings: List[CriticalFinding] = []
    seen_terms: set = set()

    for pattern, level in _TERM_PATTERNS:
        for m in pattern.finditer(text):
            term_key = m.group(0).lower()
            if term_key in seen_terms:
                continue
            seen_terms.add(term_key)

            negated, uncertain = _check_negation(text, m.start(), m.end())

            # Skip cleanly negated non-life-threatening findings
            if negated and not uncertain and level == 2:
                continue
            # Still report negated Level-1 findings so the radiologist can verify
            if negated and not uncertain and level == 1:
                # Only report if the negation itself is uncertain
                continue

            ctx_start = max(0, m.start() - 45)
            ctx_end   = min(len(text), m.end() + 45)
            context   = "…" + text[ctx_start:ctx_end].strip() + "…"

            findings.append(CriticalFinding(
                term=m.group(0),
                level=level,
                negated=negated,
                uncertain=uncertain,
                context=context,
            ))

    # Sort: Level 1 first, then certain before uncertain
    findings.sort(key=lambda f: (f.level, f.uncertain))
    return findings


def format_findings_for_dialog(findings: List[CriticalFinding]) -> str:
    """Return a human-readable summary for the confirmation dialog."""
    lines = []
    for f in findings:
        qualifier = ""
        if f.uncertain:
            qualifier = " [UNCERTAIN — cannot exclude]"
        elif f.negated:
            qualifier = " [negated — verify context]"
        severity = "⚠ LIFE-THREATENING" if f.level == 1 else "⚠ URGENT"
        lines.extend(
            (
                f"{severity}: {f.term.upper()}{qualifier}",
                f"   Context: {f.context}",
                "",
            )
        )
    return "\n".join(lines).rstrip()
