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
6.5 Split-compound rejoin      — :func:`rejoin_split_compounds` (before fuzzy so
                                  "hydro nephrosis" becomes a real term instead
                                  of being mangled per-fragment)
7. Medical dictionary fuzzy    — :func:`apply_medical_dictionary_suggestions`
7.5 Context real-word fix      — :func:`apply_context_correction` (confusable
                                  real words: "spinal chord"->"cord")
8. Learned user corrections    — :func:`apply_learned_corrections` (runs last
                                  so user overrides win)
9. Smart capitalisation        — :func:`smart_capitalize`

Confidence veto
---------------
Stages 6-7.5 are the *guessing* stages: they rewrite a word because it resembles
something in a vocabulary. Given the decoder's per-word confidences (optional —
see :mod:`~src.dictation.postprocess.confidence_gate`), the block's output is
diffed against its input and any rewrite of words the decoder was already sure
about is reverted. With no confidences the veto is inert and the pipeline
behaves exactly as before.

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
from typing import List, Optional, Sequence, Tuple

from src.dictation.postprocess.confidence_gate import (
    carry_confidences,
    veto_confident_rewrites,
)
from src.dictation.postprocess.hallucinations import filter_hallucinations
from src.dictation.postprocess.voice_commands import (
    apply_correction_commands,
    apply_spoken_commands,
)
from src.dictation.postprocess.text_utils import normalize_spaces, smart_capitalize
from src.dictation.postprocess.measurements import apply_measurement_standardisation
from src.dictation.postprocess.terminology import apply_terminology
from src.dictation.postprocess.medical_dict_match import apply_medical_dictionary_suggestions
from src.dictation.postprocess.context_correct import (
    apply_context_correction,
    rejoin_split_compounds,
)
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
    "rejoin_split_compounds",
    "apply_medical_dictionary_suggestions",
    "apply_context_correction",
    "apply_learned_corrections",
}

# The contiguous run of stages that rewrite a word because it *resembles*
# something in a vocabulary -- the guessing stages, and the only ones the
# confidence veto covers (see confidence_gate). Deliberately excludes
# apply_terminology (deliberate standardisation of correctly-heard forms, e.g.
# "x ray" -> "X-ray") and apply_learned_corrections (the user's own overrides,
# which run last precisely so they win over everything, gate included).
_GATED_STAGES = (
    "apply_accent_corrections",
    "rejoin_split_compounds",
    "apply_medical_dictionary_suggestions",
    "apply_context_correction",
)

# The word-level diff powering the corrections banner is quadratic in document
# length (difflib): measured ~24 ms at 1,000 words but ~1.3 s at 4,000 — ~10x
# the cost of the whole pipeline. A radiology report is typically 150–600
# words, so above this bound the diff is skipped (the banner just shows no
# examples) rather than stalling the live fallback path or the final pass.
_MAX_DIFF_WORDS = 1500


def _apply_veto(
    block_input: str, block_output: str,
    confidences: Sequence[Optional[float]], ceiling: Optional[float],
    sink: Optional[List[str]] = None,
) -> str:
    """Run the confidence veto over one guessing block and log what it refused."""
    text, vetoed = veto_confident_rewrites(
        block_input, block_output, confidences, ceiling
    )
    if vetoed:
        logger.info(
            "Confidence veto (>=%.2f) kept %d span(s): %s",
            ceiling, len(vetoed), "; ".join(vetoed),
        )
        if sink is not None:
            sink.extend(vetoed)
    return text


def postprocess_transcript(
    text: str, accent: str = "neutral", cleanup_level: str = "medium",
    live: bool = False,
    confidences: Optional[Sequence[Optional[float]]] = None,
    confidence_ceiling: Optional[float] = None,
    vetoed_out: Optional[List[str]] = None,
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
        confidences: The decoder's per-word confidence for *text*, one entry per
            word, ``None`` where unknown. Omit it (the default) and the
            confidence veto is inert — output is byte-identical to a run without
            it.
        confidence_ceiling: Confidence at or above which the guessing stages may
            not rewrite a word (see :mod:`confidence_gate`). ``None`` disables
            the veto. Both this and *confidences* are required for it to fire.
        vetoed_out: Optional list the vetoed ``'"orig" -> "repl"'`` spans are
            appended to. The veto always logs them; this is for a caller that
            needs to *count* them (the eval harness reports the number per clip).

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
        ("rejoin_split_compounds", rejoin_split_compounds, None),
        ("apply_medical_dictionary_suggestions", apply_medical_dictionary_suggestions, None),
        ("apply_context_correction", apply_context_correction, None),
        ("apply_learned_corrections", apply_learned_corrections, None),
        ("smart_capitalize", smart_capitalize, None),
    ]

    # The veto needs the text as it enters the guessing block, plus the input
    # confidences realigned onto that text (earlier stages have edited it).
    gate_on = bool(confidences) and confidence_ceiling is not None
    input_words = text.split() if gate_on else []
    if gate_on and len(input_words) > _MAX_DIFF_WORDS:
        # Same quadratic word-diff as the corrections banner, same bound. The
        # gate going off is a real change in behaviour, so say so.
        logger.warning(
            "Confidence veto skipped: %d words > %d", len(input_words), _MAX_DIFF_WORDS
        )
        gate_on = False
        input_words = []
    block_input: Optional[str] = None
    block_confidences: List[Optional[float]] = []

    for stage_name, func, extra_arg in stages:
        if cleanup_level == "soft" and stage_name in _SOFT_SKIP_STAGES:
            continue
        if gate_on and stage_name in _GATED_STAGES and block_input is None:
            block_input = text
            block_confidences = carry_confidences(
                input_words, text.split(), list(confidences or [])
            )
        elif block_input is not None and stage_name not in _GATED_STAGES:
            text = _apply_veto(
                block_input, text, block_confidences, confidence_ceiling, vetoed_out
            )
            block_input = None
        stage_start = time.time() if _debug else 0.0
        with perf.stage(f"postprocess.{stage_name}"):
            text = func(text, extra_arg) if extra_arg else func(text)
        if analyzer:
            analyzer.record_stage(stage_name, text, time.time() - stage_start)

    if block_input is not None:  # gated stages ran last (no un-gated stage after)
        text = _apply_veto(
            block_input, text, block_confidences, confidence_ceiling, vetoed_out
        )

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
    confidences: Optional[Sequence[Optional[float]]] = None,
    confidence_ceiling: Optional[float] = None,
    vetoed_out: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """Run the pipeline and additionally return word-level change examples.

    ``live=True`` skips the per-cycle AI-cleanup network call, and
    ``confidences`` / ``confidence_ceiling`` / ``vetoed_out`` drive the
    confidence veto — all four behave exactly as in
    :func:`postprocess_transcript`.

    Returns:
        Tuple of ``(processed_text, changes)`` where ``changes`` lists up to
        eight ``"orig" → "repl"`` substitutions for the corrections banner.
    """
    t_start = time.time()
    original_words = text.split()
    processed = postprocess_transcript(
        text, accent, cleanup_level, live=live,
        confidences=confidences, confidence_ceiling=confidence_ceiling,
        vetoed_out=vetoed_out,
    )

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

