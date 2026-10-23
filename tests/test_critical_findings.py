"""What the critical-findings scan must say about real report sentences.

Each row is a sentence a radiologist could dictate and the alert it must (or
must not) raise. The scan is the only thing between a dictated pneumothorax and
a report that leaves without anyone being asked whether it was phoned through,
so a change to the negation rules is judged against this table first.
"""

import pytest

from src.medical.critical_findings import scan_for_critical_findings


def _terms(text: str) -> list:
    return [(f.term.lower(), f.level, f.uncertain) for f in scan_for_critical_findings(text)]


@pytest.mark.parametrize(
    "sentence, expected",
    [
        # A plain positive finding is reported at its level.
        ("Large right pneumothorax.", [("pneumothorax", 1, False)]),
        ("Small subdural haematoma.", [("subdural haematoma", 1, False)]),
        ("Acute appendicitis.", [("acute appendicitis", 2, False)]),
        # A clean denial is not an alert, at either level. Every normal chest
        # report says "no pneumothorax"; alerting on it trains the click-away.
        ("No pneumothorax.", []),
        ("Pneumothorax is not seen.", []),
        ("No evidence of appendicitis.", []),
        ("The aorta is normal. No aortic dissection.", []),
        # A denial carries down an "or" list...
        ("No consolidation, effusion, or pneumothorax.", []),
        # ...but not across a bare comma, which can start a new statement.
        ("No effusion, large pneumothorax.", [("pneumothorax", 1, False)]),
        # Hedged findings are reported, marked uncertain.
        ("Cannot exclude pulmonary embolism.", [("pulmonary embolism", 2, True)]),
        ("Possible appendicitis.", [("appendicitis", 2, True)]),
        # A denial of an old finding must not silence a new one.
        (
            "No pneumothorax on the prior film. Now a large right pneumothorax.",
            [("pneumothorax", 1, False)],
        ),
    ],
)
def test_sentence_raises_the_right_alert(sentence, expected):
    assert _terms(sentence) == expected


def test_life_threatening_findings_come_first():
    found = scan_for_critical_findings("Possible appendicitis. Tension pneumothorax.")
    assert [f.level for f in found] == sorted(f.level for f in found)
    assert found[0].level == 1


def test_positions_point_at_the_words_in_the_text():
    text = "History of trauma. Large right pneumothorax."
    (finding,) = scan_for_critical_findings(text)
    assert text[finding.start:finding.end].lower() == "pneumothorax"
