"""Stage 4 — standardise measurements.

* ``5 millimetres`` → ``5 mm``
* ``5 by 3 millimetres`` → ``5 x 3 mm``
* ``5 by 3 by 2 millimetres`` → ``5 x 3 x 2 mm``
* ``15 degrees`` → ``15°``

3-D and 2-D forms are applied before 1-D so longer spans match first.
"""

from __future__ import annotations

import re

# Inline regex tuples instead of a sequential 7-element list — naming each
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


def apply_measurement_standardisation(text: str) -> str:
    """Normalise written-out units to symbols."""
    text = _MEASURE_MM_3D.sub(r"\1 x \2 x \3 mm", text)
    text = _MEASURE_CM_3D.sub(r"\1 x \2 x \3 cm", text)
    text = _MEASURE_MM_2D.sub(r"\1 x \2 mm", text)
    text = _MEASURE_CM_2D.sub(r"\1 x \2 cm", text)
    text = _MEASURE_MM_1D.sub(r"\1 mm", text)
    text = _MEASURE_CM_1D.sub(r"\1 cm", text)
    text = _MEASURE_DEG.sub(r"\1°", text)
    return text
