"""Stage 4: standardise measurements.

* ``5 millimetres`` → ``5 mm``
* ``5 by 3 millimetres`` → ``5 x 3 mm``
* ``5 by 3 by 2 millimetres`` → ``5 x 3 x 2 mm``
* ``15 degrees`` → ``15°``
* ``1.5 Tesla`` / ``one point five Tesla`` → ``1.5 T``

3-D and 2-D forms are applied before 1-D so longer spans match first.
"""

from __future__ import annotations

import re

# Inline regex tuples instead of a sequential 7-element list: naming each
# pattern matters because the apply order is significant.

_MEASURE_MM_3D = re.compile(
    r"(\d+(?:\.\d+)?)\s+(?:by|x|X)\s+(\d+(?:\.\d+)?)\s+(?:by|x|X)\s+(\d+(?:\.\d+)?)\s+millimet(?:re|er)s?\b",
    re.IGNORECASE,
)
_MEASURE_CM_3D = re.compile(
    r"(\d+(?:\.\d+)?)\s+(?:by|x|X)\s+(\d+(?:\.\d+)?)\s+(?:by|x|X)\s+(\d+(?:\.\d+)?)\s+(?:centimetre|centimeter)s?\b",
    re.IGNORECASE,
)
_MEASURE_MM_2D = re.compile(
    r"(\d+(?:\.\d+)?)\s+(?:by|x|X)\s+(\d+(?:\.\d+)?)\s+millimet(?:re|er)s?\b",
    re.IGNORECASE,
)
_MEASURE_CM_2D = re.compile(
    r"(\d+(?:\.\d+)?)\s+(?:by|x|X)\s+(\d+(?:\.\d+)?)\s+(?:centimetre|centimeter)s?\b",
    re.IGNORECASE,
)
_MEASURE_MM_1D = re.compile(r"(\d+(?:\.\d+)?)\s+millimet(?:re|er)s?\b", re.IGNORECASE)
_MEASURE_CM_1D = re.compile(r"(\d+(?:\.\d+)?)\s+(?:centimetre|centimeter)s?\b", re.IGNORECASE)
_MEASURE_DEG = re.compile(r"(\d+(?:\.\d+)?)\s+degrees?\b", re.IGNORECASE)
#: MRI field strength. Only the strengths scanners are sold at are spelled out,
#: so "three" in "three Tesla" is safe to turn into a number here.
_TESLA_WORDS = {"one point five": "1.5", "three": "3", "three point zero": "3.0",
                "one": "1", "seven": "7", "zero point five": "0.5"}
_MEASURE_TESLA_WORDS = re.compile(
    r"\b((?i:" + "|".join(sorted(_TESLA_WORDS, key=len, reverse=True)) + r"))[\s-]+(?:(?i:teslas?)|T)\b")
_MEASURE_TESLA = re.compile(r"(\d+(?:\.\d+)?)[\s-]*teslas?\b", re.IGNORECASE)


def apply_measurement_standardisation(text: str) -> str:
    """Normalise written-out units to symbols."""
    text = _MEASURE_MM_3D.sub(r"\1 x \2 x \3 mm", text)
    text = _MEASURE_CM_3D.sub(r"\1 x \2 x \3 cm", text)
    text = _MEASURE_MM_2D.sub(r"\1 x \2 mm", text)
    text = _MEASURE_CM_2D.sub(r"\1 x \2 cm", text)
    text = _MEASURE_MM_1D.sub(r"\1 mm", text)
    text = _MEASURE_CM_1D.sub(r"\1 cm", text)
    text = _MEASURE_DEG.sub(r"\1°", text)
    text = _MEASURE_TESLA_WORDS.sub(lambda m: _TESLA_WORDS[" ".join(m[1].lower().split())] + " T", text)
    text = _MEASURE_TESLA.sub(r"\1 T", text)
    return text


#: Field strengths a scanner is sold at, then "tesla" as Parakeet mishears it
#: (seen in the protocol eval set), then the word that says it is a scanner.
#: The trailing scanner word keeps this away from ordinary text.
_FIELD_MISHEARD = re.compile(
    r"\b(0\.5|1\.0|1\.5|3\.0|3|7)[\s-]*"
    r"(?:teslas?|tisla|tesler|tysler|tessler|disla|distel|distiller|distillar)"
    r"(?=\s+(?:magnet|scanner|system|unit|machine|MRI|MR)\b)",
    re.IGNORECASE)
_FIELD_NOSPACE = re.compile(r"\b(0\.5|1\.0|1\.5|3\.0|3|7)T\b")


def apply_field_strength(text: str) -> str:
    """Field strength written as "1.5 T" once later stages have fixed the spelling.

    Runs after the dictionary and context stages: "3-tisla" only becomes
    "3-tesla" there, after apply_measurement_standardisation has run.
    """
    text = _MEASURE_TESLA.sub(r"\1 T", text)
    text = _FIELD_MISHEARD.sub(r"\1 T", text)
    return _FIELD_NOSPACE.sub(r"\1 T", text)
