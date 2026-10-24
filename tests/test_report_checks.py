"""Side, sex and impression checks: single sentences, then the labelled corpus.

The corpus floors are what the checks were built to: every injected error
caught, no alarm when the other side is named on purpose, and nothing at all
when the patient's sex is not recorded. The one known false alarm (a side word
in passing, "lies left side down") is pinned too, so a change that fixes or
worsens it shows up here.
"""

import pytest

from scripts.eval.report_checks_corpus import build_corpus, false_alarms, score
from src.features.report_checks import check_report, exam_side


def _kinds(text, sex=""):
    return [i.kind for i in check_report(text, sex)]


@pytest.mark.parametrize(
    "exam, side",
    [
        ("EXAMINATION: MRI Right Knee", "right"),
        ("EXAMINATION: MRI left shoulder", "left"),
        ("EXAMINATION: MRI [Right / Left] Knee", None),  # not chosen yet
        ("EXAMINATION: Radiograph Hands Bilateral", None),
        ("EXAMINATION: CT Chest", None),
    ],
)
def test_exam_side(exam, side):
    assert exam_side(exam + "\nFINDINGS:\nNormal.") == side


@pytest.mark.parametrize(
    "body, flagged",
    [
        ("Marrow oedema in the left medial femoral condyle.", True),
        ("Marrow oedema in the right medial femoral condyle.", False),
        ("Similar to the contralateral left knee.", False),
        ("Compared with the left side, mild asymmetry.", False),
        ("Unchanged from the prior left knee study.", False),
    ],
)
def test_side_is_checked_against_the_exam(body, flagged):
    text = f"EXAMINATION: MRI Right Knee\nFINDINGS:\n{body}\nIMPRESSION:\n1. Normal."
    assert ("side" in _kinds(text)) is flagged


@pytest.mark.parametrize(
    "sex, body, flagged",
    [
        ("M", "The uterus is normal.", True),
        ("F", "The prostate is not enlarged.", True),
        ("F", "The uterus is normal.", False),
        ("", "The uterus is normal.", False),  # not recorded: never guessed
        ("X", "The uterus is normal.", False),  # not F or M: never guessed
        ("M", "Cervical spine alignment is normal.", False),  # cervical is not cervix
    ],
)
def test_sex_specific_organs(sex, body, flagged):
    text = f"EXAMINATION: CT Pelvis\nFINDINGS:\n{body}\nIMPRESSION:\n1. Normal."
    assert ("sex" in _kinds(text, sex)) is flagged


@pytest.mark.parametrize(
    "text, flagged",
    [
        ("FINDINGS:\nNormal.\nIMPRESSION:\n1. Normal study.", False),
        ("FINDINGS:\nNormal.\nCONCLUSION:\nNormal study.", False),
        ("FINDINGS:\nNormal.", True),
        ("FINDINGS:\nNormal.\nIMPRESSION:\n1.\n", True),
        ("FINDINGS:\nNormal.\nIMPRESSION:\n\nRECOMMENDATIONS:\nNone.", True),
        ("Free text with no headings at all.", False),
    ],
)
def test_impression(text, flagged):
    assert ("impression" in _kinds(text)) is flagged


def test_corpus_every_injected_error_is_caught():
    for kind, result in score(build_corpus()).items():
        assert result["recall"] == 1.0, kind


def test_corpus_false_alarms_by_kind_of_clean_report():
    alarms = false_alarms(build_corpus())
    assert alarms["clean"][0] == 0
    assert alarms["tempting"][0] == 0      # the other side named on purpose
    assert alarms["unrecorded"][0] == 0    # sex not given
    # Known limit: a side word in passing ("lies left side down") is flagged.
    assert alarms["passing"][0] == alarms["passing"][1]


def _messages(text, sex=""):
    return [i.message for i in check_report(text, sex)]


def test_side_warning_quotes_the_sentence():
    text = "EXAMINATION: MRI Right Knee\nFINDINGS: The left meniscus is torn.\nIMPRESSION: Tear."
    assert _messages(text) == ['Exam is RIGHT, but this says left: "FINDINGS: The left meniscus is torn."']


def test_an_exam_of_both_sides_is_not_checked_for_side():
    assert exam_side("EXAMINATION: XR Both Knees (right painful)") is None


@pytest.mark.parametrize(
    "sex, body, message",
    [
        ("Male", "The uterus is normal.", 'Patient is male, but the report describes "uterus"'),
        ("M", "Ovarian cyst.", 'Patient is male, but the report describes "Ovarian"'),
        ("Female", "The prostate is enlarged.", 'Patient is female, but the report describes "prostate"'),
        ("female", "Scrotal oedema.", 'Patient is female, but the report describes "Scrotal"'),
    ],
)
def test_sex_warning_names_the_patient_and_the_word(sex, body, message):
    assert _messages(body, sex) == [message]


@pytest.mark.parametrize(
    "text, message",
    [
        ("FINDINGS: Normal.", "The report has findings but no impression"),
        ("FINDINGS: Normal.\nCONCLUSION:\n1.\n", "The conclusion is empty"),
    ],
)
def test_impression_warning_wording(text, message):
    assert _messages(text) == [message]
