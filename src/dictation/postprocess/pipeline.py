"""Post-processing orchestrator.

The pipeline is a small ordered list of pure functions; reordering is
behaviour-changing so don't reshuffle without updating the tests in
``tests/test_worker_logic.py::TestPostProcessingPipeline``.

Stage order
-----------
0. Hallucination filter        — :func:`filter_hallucinations`
1. Voice correction commands   — :func:`apply_correction_commands`
2. Spoken punctuation          — :func:`apply_spoken_commands`
3. Whitespace                  — :func:`normalize_spaces`
4. Measurements                — :func:`apply_measurement_standardisation`
5. Radiology terminology       — :func:`apply_terminology`
6. Accent-specific Whisper fix — :func:`apply_accent_corrections`
7. Medical dictionary fuzzy    — :func:`apply_medical_dictionary_suggestions`
8. Learned user corrections    — :func:`apply_learned_corrections` (runs last
                                  so user overrides win)
9. Smart capitalisation        — :func:`smart_capitalize`
"""

from __future__ import annotations

import difflib
import logging
import time
from typing import List, Tuple

from src.dictation.postprocess.hallucinations import filter_hallucinations
from src.dictation.postprocess.voice_commands import (
    apply_correction_commands,
    apply_spoken_commands,
)
from src.dictation.postprocess.text_utils import normalize_spaces, smart_capitalize
from src.dictation.postprocess.measurements import apply_measurement_standardisation
from src.dictation.postprocess.terminology import apply_terminology
from src.dictation.postprocess.medical_dict_match import apply_medical_dictionary_suggestions
from src.features.accent_corrections import apply_accent_corrections
from src.features.adaptive_learning import apply_learned_corrections
from src.dictation.postprocess.analysis import PipelineAnalyzer

logger = logging.getLogger(__name__)


def postprocess_transcript(text: str, accent: str = "neutral") -> str:
    """Run the full pipeline on a raw Whisper transcript.

    Args:
        text:   Raw transcript from Whisper (single chunk or full document).
        accent: Accent profile key for stage 6
            (``"neutral"``, ``"south_asian"``, ``"middle_eastern"``,
            ``"east_asian"``, ``"west_african"``).

    Returns:
        Cleaned, terminology-corrected, capitalised text.
    """
    t_start = time.time()
    analyzer = PipelineAnalyzer()
    analyzer.record_input(text)

    stages = [
        ("filter_hallucinations", filter_hallucinations, None),
        ("apply_correction_commands", apply_correction_commands, None),
        ("apply_spoken_commands", apply_spoken_commands, None),
        ("normalize_spaces", normalize_spaces, None),
        ("apply_measurement_standardisation", apply_measurement_standardisation, None),
        ("apply_terminology", apply_terminology, None),
        ("apply_accent_corrections", apply_accent_corrections, accent),
        ("apply_medical_dictionary_suggestions", apply_medical_dictionary_suggestions, None),
        ("apply_learned_corrections", apply_learned_corrections, None),
        ("smart_capitalize", smart_capitalize, None),
    ]

    for stage_name, func, extra_arg in stages:
        stage_start = time.time()
        text = func(text, extra_arg) if extra_arg else func(text)
        stage_duration = time.time() - stage_start
        analyzer.record_stage(stage_name, text, stage_duration)

    elapsed = time.time() - t_start
    analyzer.save_analysis(accent)
    logger.info("Post-processing [%.2fs]", elapsed)
    return text


def postprocess_transcript_with_changes(
    text: str, accent: str = "neutral"
) -> Tuple[str, List[str]]:
    """Run the pipeline and additionally return word-level change examples.

    Returns:
        Tuple of ``(processed_text, changes)`` where ``changes`` lists up to
        eight ``"orig" → "repl"`` substitutions for the corrections banner.
    """
    t_start = time.time()
    original_words = text.split()
    processed = postprocess_transcript(text, accent)
    processed_words = processed.split()

    changes: List[str] = []
    matcher = difflib.SequenceMatcher(
        None, original_words, processed_words, autojunk=False
    )
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "replace":
            orig = " ".join(original_words[i1:i2])
            repl = " ".join(processed_words[j1:j2])
            if orig.lower() != repl.lower():
                changes.append(f'"{orig}" → "{repl}"')
        if len(changes) >= 8:
            break

    elapsed = time.time() - t_start
    logger.info("Post-processing with changes [%.2fs]", elapsed)
    return processed, changes

