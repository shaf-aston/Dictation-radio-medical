"""Radiology transcript post-processing.

Public entry point: :func:`postprocess_transcript`.

Internally a sequence of small, pure-function stages run in fixed order
(see :mod:`src.dictation.postprocess.pipeline`).  Each stage lives in its
own module so the regex tables are easy to grep and extend.
"""

from src.dictation.postprocess.pipeline import (
    postprocess_transcript,
    postprocess_transcript_with_changes,
    CLEANUP_LEVELS,
    CLEANUP_LEVEL_LABELS,
)
from src.dictation.postprocess.hallucinations import filter_hallucinations
from src.dictation.postprocess.voice_commands import (
    apply_correction_commands,
    apply_spoken_commands,
)
from src.dictation.postprocess.text_utils import normalize_spaces, smart_capitalize
from src.dictation.postprocess.measurements import apply_measurement_standardisation
from src.dictation.postprocess.terminology import apply_terminology
from src.dictation.postprocess.medical_dict_match import (
    apply_medical_dictionary_suggestions,
    PROTECTED_TERMS,
)

__all__ = [
    "postprocess_transcript",
    "postprocess_transcript_with_changes",
    "CLEANUP_LEVELS",
    "CLEANUP_LEVEL_LABELS",
    "filter_hallucinations",
    "apply_correction_commands",
    "apply_spoken_commands",
    "normalize_spaces",
    "smart_capitalize",
    "apply_measurement_standardisation",
    "apply_terminology",
    "apply_medical_dictionary_suggestions",
    "PROTECTED_TERMS",
]
