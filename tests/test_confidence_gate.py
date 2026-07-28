"""Regression tests for the confidence veto (no model, no audio).

The gate exists to cut the false-correction rate: the guessing post-processing
stages must not rewrite a word the decoder was already sure about. These tests
pin the two halves of that contract — the pure functions in
``postprocess/confidence_gate.py``, and the block they wrap inside
``postprocess/pipeline.py``.
"""

from __future__ import annotations

from src.dictation.postprocess.confidence_gate import (
    carry_confidences,
    veto_confident_rewrites,
)
from src.dictation.postprocess.pipeline import _GATED_STAGES, postprocess_transcript

CEILING = 0.9


class TestVetoConfidentRewrites:
    """The gate itself: which edits survive and which get put back."""

    def test_confident_word_is_not_rewritten(self):
        text, vetoed = veto_confident_rewrites(
            "the femer is intact", "the femur is intact",
            [0.5, 0.97, 0.5, 0.5], CEILING,
        )
        assert text == "the femer is intact"
        assert vetoed == ['"femer" -> "femur"']

    def test_unsure_word_is_rewritten(self):
        text, vetoed = veto_confident_rewrites(
            "the femer is intact", "the femur is intact",
            [0.5, 0.42, 0.5, 0.5], CEILING,
        )
        assert text == "the femur is intact"
        assert vetoed == []

    def test_unknown_confidence_is_not_confident(self):
        """``None`` means "no signal", which must never protect a word."""
        text, vetoed = veto_confident_rewrites(
            "the femer is intact", "the femur is intact",
            [0.5, None, 0.5, 0.5], CEILING,
        )
        assert text == "the femur is intact"
        assert vetoed == []

    def test_exactly_at_ceiling_is_confident(self):
        text, _ = veto_confident_rewrites(
            "the femer is intact", "the femur is intact",
            [0.5, CEILING, 0.5, 0.5], CEILING,
        )
        assert text == "the femer is intact"

    def test_part_confident_span_is_rewritten(self):
        """A span mixing a sure word with an unsure one is let through.

        A ``replace`` span is restored as a unit — there is no way to keep the
        confident word and still apply the rewrite the unsure word probably
        needs. Vetoing the whole span would therefore block a legitimate fix
        because of a neighbour, so the gate only fires when the whole span is
        known-confident. This also keeps the gate strictly conservative: it can
        only ever fire *less* than a per-word rule, never more.
        """
        text, vetoed = veto_confident_rewrites(
            "hydro nephrosis noted", "hydronephrosis noted",
            [0.99, 0.30, 0.90], CEILING,
        )
        assert text == "hydronephrosis noted"
        assert vetoed == []

    def test_deletion_is_never_reverted(self):
        """Deletions are how "scratch that" and the hallucination filter work."""
        text, vetoed = veto_confident_rewrites(
            "thank you for watching the liver is normal", "the liver is normal",
            [0.99] * 8, CEILING,
        )
        assert text == "the liver is normal"
        assert vetoed == []

    def test_insertion_is_never_reverted(self):
        """Inserted punctuation has no original word to be confident about."""
        text, vetoed = veto_confident_rewrites(
            "the liver is normal", "the liver is normal .",
            [0.99] * 4, CEILING,
        )
        assert text == "the liver is normal ."
        assert vetoed == []

    def test_mismatched_confidence_length_disables_the_gate(self):
        text, vetoed = veto_confident_rewrites(
            "the femer is intact", "the femur is intact", [0.99, 0.99], CEILING,
        )
        assert text == "the femur is intact"
        assert vetoed == []

    def test_no_signal_leaves_the_text_untouched(self):
        for confidences, ceiling in ((None, CEILING), ([], CEILING),
                                     ([0.99] * 4, None)):
            text, vetoed = veto_confident_rewrites(
                "the femer is intact", "the femur is intact", confidences, ceiling,
            )
            assert (text, vetoed) == ("the femur is intact", [])

    def test_surrounding_whitespace_is_preserved(self):
        """Only the vetoed span is spliced back; paragraph breaks survive."""
        text, _ = veto_confident_rewrites(
            "findings\n\nthe femer is intact", "findings\n\nthe femur is intact",
            [0.5, 0.5, 0.99, 0.5, 0.5], CEILING,
        )
        assert text == "findings\n\nthe femer is intact"

    def test_several_spans_are_all_reported(self):
        text, vetoed = veto_confident_rewrites(
            "the femer and the tibea", "the femur and the tibia",
            [0.5, 0.99, 0.5, 0.5, 0.95], CEILING,
        )
        assert text == "the femer and the tibea"
        assert vetoed == ['"femer" -> "femur"', '"tibea" -> "tibia"']


class TestCarryConfidences:
    """Realigning confidences onto text a later stage has already edited."""

    def test_round_trip_over_equal_replace_delete_insert(self):
        before = ["um", "the", "femer", "is", "intact"]
        after = ["the", "femur", "is", "clearly", "intact"]
        carried = carry_confidences(before, after, [0.10, 0.80, 0.99, 0.70, 0.60])

        assert len(carried) == len(after)
        # "the" keeps its own value; "um" (deleted) drops out; "femur" and the
        # inserted "clearly" are words no decoder emitted, so they are unknown.
        assert carried == [0.80, None, 0.70, None, 0.60]

    def test_unchanged_text_carries_everything(self):
        words = ["the", "liver", "is", "normal"]
        assert carry_confidences(words, words, [0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]

    def test_mismatched_length_yields_all_unknown(self):
        assert carry_confidences(["a", "b"], ["a", "b", "c"], [0.9]) == [None, None, None]

    def test_empty_inputs(self):
        assert carry_confidences([], [], []) == []


class TestPipelineIntegration:
    """The gate as wired into the stage list."""

    def test_gate_covers_only_the_guessing_stages(self):
        assert _GATED_STAGES == (
            "apply_accent_corrections",
            "rejoin_split_compounds",
            "apply_medical_dictionary_suggestions",
            "apply_context_correction",
        )
        assert "apply_terminology" not in _GATED_STAGES
        assert "apply_learned_corrections" not in _GATED_STAGES

    def test_confident_word_survives_the_accent_stage(self):
        assert postprocess_transcript(
            "wertebra", accent="south_asian",
            confidences=[0.98], confidence_ceiling=CEILING,
        ) == "Wertebra"

    def test_unsure_word_is_still_corrected_by_the_accent_stage(self):
        assert postprocess_transcript(
            "wertebra", accent="south_asian",
            confidences=[0.40], confidence_ceiling=CEILING,
        ) == "Vertebra"

    def test_confident_word_survives_the_context_stage(self):
        assert postprocess_transcript(
            "spinal chord compression",
            confidences=[0.5, 0.97, 0.5], confidence_ceiling=CEILING,
        ) == "Spinal chord compression"

    def test_unsure_word_is_still_corrected_by_the_context_stage(self):
        assert postprocess_transcript(
            "spinal chord compression",
            confidences=[0.5, 0.55, 0.5], confidence_ceiling=CEILING,
        ) == "Spinal cord compression"

    def test_terminology_is_not_gated(self):
        """Standardising a correctly-heard spoken form is not a guess."""
        assert postprocess_transcript(
            "meniscal tier", confidences=[0.99, 0.99], confidence_ceiling=CEILING,
        ) == "Meniscal tear"

    def test_soft_level_skips_every_gated_stage_without_crashing(self):
        text = "wertebra and spinal chord"
        assert postprocess_transcript(
            text, accent="south_asian", cleanup_level="soft",
            confidences=[0.98, 0.5, 0.5, 0.98], confidence_ceiling=CEILING,
        ) == postprocess_transcript(text, accent="south_asian", cleanup_level="soft")

    def test_defaults_are_inert(self):
        for text in ("wertebra", "spinal chord compression", "meniscal tier"):
            assert postprocess_transcript(text, accent="south_asian") == (
                postprocess_transcript(
                    text, accent="south_asian",
                    confidences=None, confidence_ceiling=CEILING,
                )
            )

    def test_vetoed_spans_are_reported_to_the_caller(self):
        vetoed: list[str] = []
        postprocess_transcript(
            "spinal chord compression",
            confidences=[0.5, 0.97, 0.5], confidence_ceiling=CEILING,
            vetoed_out=vetoed,
        )
        assert vetoed == ['"chord" -> "cord"']
