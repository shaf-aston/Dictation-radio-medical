"""The patient-information schema: declared once.

The same six fields were hand-listed in four places (desktop form reader, web
API model, the de-identifier's PHI field set, and the report header). Adding a
seventh field meant remembering all four: and the one that is easiest to
forget is the de-identifier, where an omission means an identifier is *not*
scrubbed before upload. So the list lives here, and everything derives from it.

``PATIENT_FIELDS`` is ordered: report headers render in this order.
"""

from __future__ import annotations

from typing import Dict, Tuple

# (key, human label): the label is what a report header prints.
PATIENT_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("name", "Patient"),
    ("id", "Patient ID"),
    ("dob", "Date of Birth"),
    ("study_date", "Study Date"),
    ("referring", "Referring"),
    ("accession", "Accession #"),
    # F, M, X or blank. Read by the sex-mismatch check (features/report_checks).
    ("sex", "Sex"),
)

#: Field keys only, in order.
PATIENT_KEYS: Tuple[str, ...] = tuple(key for key, _ in PATIENT_FIELDS)

#: Fields whose value names one person, and so is scrubbed from any text that
#: leaves the device (src/medical/deid.py). Sex is left out on purpose: as a
#: one-letter code it is too short to scrub, and as a word ("female") scrubbing
#: it would delete clinical wording, not an identifier. A new field is an
#: identifier unless it is added to this exclusion with a reason.
IDENTIFYING_KEYS: Tuple[str, ...] = tuple(key for key in PATIENT_KEYS if key != "sex")


def empty_patient_info() -> Dict[str, str]:
    """A patient-info dict with every field present and blank."""
    return {key: "" for key in PATIENT_KEYS}


def normalize_patient_info(raw: Dict[str, object] | None) -> Dict[str, str]:
    """Return a complete, stripped patient-info dict from partial input.

    Unknown keys are dropped and missing ones default to empty, so downstream
    code (de-identification especially) can rely on every field existing.
    """
    raw = raw or {}
    return {key: str(raw.get(key) or "").strip() for key in PATIENT_KEYS}
