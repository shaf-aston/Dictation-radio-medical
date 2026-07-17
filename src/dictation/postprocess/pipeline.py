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

Cleanup levels
--------------
``cleanup_level`` lets the dictating radiologist control how much the pipeline
rewrites their words:

* ``"soft"``   — only structural cleanup (stages 0-4, 9): hallucination
  removal, spoken voice/punctuation commands, whitespace, measurement
  formatting, and capitalisation. Skips terminology, accent, fuzzy-dictionary,
  and learned-correction stages (5-8) — the stages that rewrite a word based
  on similarity to a vocabulary, so the dictated words pass through almost
  verbatim.
* ``"medium"`` (default) — the full ten-stage pipeline above, unchanged.
* ``"hard"``  — the full pipeline, plus an AI cleanup pass
  (:func:`~src.dictation.postprocess.llm_cleanup.clean_with_llm`) if that
  optional, consent-gated feature is enabled; otherwise identical to
  ``"medium"``.
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
from src.core import perf

logger = logging.getLogger(__name__)

CLEANUP_LEVELS = ("soft", "medium", "hard")

CLEANUP_LEVEL_LABELS: dict = {
    "soft": "Soft (minimal changes)",
    "medium": "Medium (standard)",
    "hard": "Hard (standard + AI polish)",
}

# Stages skipped in "soft" mode: terminology, accent corrections, fuzzy
# medical-dictionary matching, and learned corrections all rewrite a word
# based on similarity to a vocabulary -- the highest-risk-of-meaning-change
# stages. Skipping them leaves the dictated words almost verbatim.
_SOFT_SKIP_STAGES = {
    "apply_terminology",
    "apply_accent_corrections",
    "apply_medical_dictionary_suggestions",
    "apply_learned_corrections",
}

# The word-level diff powering the corrections banner is quadratic in document
# length (difflib): measured ~24 ms at 1,000 words but ~1.3 s at 4,000 — ~10x
# the cost of the whole pipeline. A radiology report is typically 150–600
# words, so above this bound the diff is skipped (the banner just shows no
# examples) rather than stalling the live fallback path or the final pass.
_MAX_DIFF_WORDS = 1500


def postprocess_transcript(
    text: str, accent: str = "neutral", cleanup_level: str = "medium",
    live: bool = False,
) -> str:
    """Run the pipeline on a raw Whisper transcript.

    Args:
        text:   Raw transcript from Whisper (single chunk or full document).
        accent: Accent profile key for stage 6
            (``"neutral"``, ``"south_asian"``, ``"middle_eastern"``,
            ``"east_asian"``, ``"west_african"``).
        cleanup_level: ``"soft"``, ``"medium"`` (default), or ``"hard"`` —
            see the module docstring.
        live: ``True`` when called per-chunk during live dictation. Forces the
            ``"hard"`` AI-cleanup (network) stage to be skipped — that pass is
            for the *finished* document only (see ``llm_cleanup`` docstring and
            the CLAUDE.md "AI cleanup is on-demand only" invariant), never on
            every live cycle.

    Returns:
        Cleaned, terminology-corrected, capitalised text.
    """
    _debug = logger.isEnabledFor(logging.DEBUG)
    t_start = time.time()
    analyzer = PipelineAnalyzer() if _debug else None
    if analyzer:
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
        if cleanup_level == "soft" and stage_name in _SOFT_SKIP_STAGES:
            continue
        stage_start = time.time() if _debug else 0.0
        with perf.stage(f"postprocess.{stage_name}"):
            text = func(text, extra_arg) if extra_arg else func(text)
        if analyzer:
            analyzer.record_stage(stage_name, text, time.time() - stage_start)

    if cleanup_level == "hard" and not live:
        from src.dictation.postprocess.llm_cleanup import clean_with_llm
        stage_start = time.time() if _debug else 0.0
        with perf.stage("postprocess.clean_with_llm"):
            text, _ = clean_with_llm(text)
        if analyzer:
            analyzer.record_stage("clean_with_llm", text, time.time() - stage_start)

    elapsed = time.time() - t_start
    if analyzer:
        analyzer.save_analysis(accent)
    logger.debug("Post-processing [%.2fs]", elapsed)
    return text


def postprocess_transcript_with_changes(
    text: str, accent: str = "neutral", cleanup_level: str = "medium",
    live: bool = False,
) -> Tuple[str, List[str]]:
    """Run the pipeline and additionally return word-level change examples.

    ``live=True`` skips the per-cycle AI-cleanup network call (see
    :func:`postprocess_transcript`).

    Returns:
        Tuple of ``(processed_text, changes)`` where ``changes`` lists up to
        eight ``"orig" → "repl"`` substitutions for the corrections banner.
    """
    t_start = time.time()
    original_words = text.split()
    processed = postprocess_transcript(text, accent, cleanup_level, live=live)

    if len(original_words) > _MAX_DIFF_WORDS:
        # Quadratic diff not worth it on very long documents — see
        # _MAX_DIFF_WORDS. The processed text is unaffected; only the banner's
        # example list is dropped.
        logger.debug(
            "Skipping change diff: %d words > %d", len(original_words), _MAX_DIFF_WORDS
        )
        return processed, []

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

