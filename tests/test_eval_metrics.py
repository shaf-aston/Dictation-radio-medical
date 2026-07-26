"""Tests for the dictation evaluation metrics.

Pure string maths — no audio, no Whisper, no model. These lock down the three
numbers every later milestone is judged against, so a silent change in how
accuracy is scored would invalidate every comparison in the report history.
"""

from __future__ import annotations

import pytest

from scripts.eval.metrics import (
    correction_effect,
    normalize_words,
    term_error_rate,
    word_error_rate,
)

LEXICON = ["hydronephrosis", "pneumothorax", "meniscal", "effusion", "atelectasis"]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_normalize_strips_punctuation_and_case():
    assert normalize_words("The LEFT kidney, mildly enlarged.") == [
        "the", "left", "kidney", "mildly", "enlarged",
    ]


def test_normalize_splits_hyphens_and_keeps_apostrophes():
    # Hyphens split on both sides so a deliberate terminology join
    # ("full thickness" -> "full-thickness") is not scored as two errors.
    assert normalize_words("T2-weighted patient's") == ["t2", "weighted", "patient's"]


def test_normalize_drops_edge_punctuation():
    assert normalize_words("- fluid-filled -") == ["fluid", "filled"]


# ---------------------------------------------------------------------------
# Word error rate
# ---------------------------------------------------------------------------

def test_wer_perfect_match_is_zero():
    r = word_error_rate("no acute intracranial abnormality", "No acute intracranial abnormality.")
    assert r.wer == 0.0
    assert r.hits == 4


def test_wer_counts_substitution_deletion_insertion():
    ref = "the left kidney shows mild hydronephrosis"
    hyp = "the right kidney shows mild hydronephrosis today"
    r = word_error_rate(ref, hyp)
    assert r.substitutions == 1
    assert r.insertions == 1
    assert r.deletions == 0
    assert r.wer == pytest.approx(2 / 6)


def test_wer_empty_hypothesis_is_total_loss():
    assert word_error_rate("small pleural effusion", "").wer == 1.0


def test_wer_empty_reference_does_not_divide_by_zero():
    # Reported as a failure, not a suspiciously perfect score.
    assert word_error_rate("", "spurious words").wer == 1.0
    assert word_error_rate("", "").wer == 0.0


# ---------------------------------------------------------------------------
# Medical-term error rate
# ---------------------------------------------------------------------------

def test_term_error_rate_ignores_non_lexicon_errors():
    # Two ordinary words wrong, but both lexicon terms survived intact.
    ref = "mild hydronephrosis with a small pneumothorax"
    hyp = "mild hydronephrosis without the small pneumothorax"
    assert term_error_rate(ref, hyp, LEXICON).error_rate == 0.0


def test_term_error_rate_catches_a_mangled_term():
    ref = "mild hydronephrosis with a small pneumothorax"
    hyp = "mild hydro nephrosis with a small pneumothorax"
    result = term_error_rate(ref, hyp, LEXICON)
    assert result.term_tokens == 2
    assert result.term_errors == 1
    assert result.error_rate == pytest.approx(0.5)
    assert "hydronephrosis" in result.missed_terms


def test_term_error_rate_is_zero_when_reference_has_no_terms():
    assert term_error_rate("the study is unremarkable", "the study is remarkable",
                           LEXICON).error_rate == 0.0


def test_term_error_rate_ignores_terms_only_in_hypothesis():
    # Inventing lexicon words must not improve the score.
    ref = "the study is unremarkable"
    hyp = "the study is pneumothorax unremarkable"
    assert term_error_rate(ref, hyp, LEXICON).term_tokens == 0


# ---------------------------------------------------------------------------
# Correction effect — the pipeline's honest scoreboard
# ---------------------------------------------------------------------------

def test_correction_effect_credits_a_true_fix():
    ref = "small pleural effusion noted"
    raw = "small pleural effusiun noted"
    post = "small pleural effusion noted"
    eff = correction_effect(ref, raw, post)
    assert eff.true_fixes == 1
    assert eff.false_corrections == 0
    assert eff.net_gain == 1
    assert eff.false_correction_rate == 0.0


def test_correction_effect_penalises_breaking_a_correct_word():
    # The classic failure: a correctly-heard word snapped to a lexicon term.
    ref = "the chord was intact"
    raw = "the chord was intact"
    post = "the cord was intact"
    eff = correction_effect(ref, raw, post)
    assert eff.false_corrections == 1
    assert eff.true_fixes == 0
    assert eff.net_gain == -1
    assert eff.false_correction_rate == 1.0


def test_correction_effect_reports_a_net_liability():
    ref = "the chord shows effusion and the cord is normal"
    raw = "the chord shows effusiun and the chord is normal"
    post = "the cord shows effusion and the cord is normal"
    eff = correction_effect(ref, raw, post)
    # Fixed "effusiun" and "chord"->"cord" in position 2, but broke position 1.
    assert eff.true_fixes >= 1
    assert eff.false_corrections >= 1


def test_correction_effect_ignores_casing_and_punctuation_changes():
    ref = "no acute abnormality"
    raw = "no acute abnormality"
    post = "No acute abnormality."
    eff = correction_effect(ref, raw, post)
    assert eff.changes == 0


def test_correction_effect_classifies_wrong_to_wrong_as_neutral():
    ref = "small pleural effusion"
    raw = "small pleural effusiun"
    post = "small pleural effusionn"
    eff = correction_effect(ref, raw, post)
    assert eff.neutral_changes == 1
    assert eff.true_fixes == 0
    assert eff.false_corrections == 0


def test_correction_effect_counts_a_deletion_as_a_change():
    ref = "small pleural effusion noted"
    raw = "small pleural effusion noted"
    post = "small pleural effusion"
    eff = correction_effect(ref, raw, post)
    assert eff.false_corrections == 1


# ---------------------------------------------------------------------------
# Normalisation of deliberate pipeline behaviour
# ---------------------------------------------------------------------------

def test_spoken_units_match_the_symbols_the_pipeline_writes():
    # measurements.py deliberately rewrites "twelve millimetres" as "12 mm";
    # scoring that as an error would mark the product down for working.
    eff = correction_effect(
        "a nodule measuring fourteen millimetres",
        "a nodule measuring fourteen millimetres",
        "a nodule measuring fourteen mm",
    )
    assert eff.false_corrections == 0


def test_hyphen_join_is_not_two_word_errors():
    # terminology.py joins "full thickness" -> "full-thickness" on purpose.
    r = word_error_rate(
        "a full thickness supraspinatus tear",
        "a full-thickness supraspinatus tear",
    )
    assert r.wer == 0.0


def test_spelling_variants_remain_visible_as_changes():
    # British -> American IS a real change to the radiologist's text, so unlike
    # units and hyphens it must NOT be normalised away.
    eff = correction_effect(
        "normal in calibre", "normal in calibre", "normal in caliber",
    )
    assert eff.false_corrections == 1
