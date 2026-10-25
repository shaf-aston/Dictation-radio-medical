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
        # Free air is a perforation, however it is worded.
        ("Free air under the diaphragm.", [("free air", 1, False)]),
        ("Free intraperitoneal air.", [("free intraperitoneal air", 1, False)]),
        ("No free air.", []),
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


# -- every term, every cue ----------------------------------------------------
# Written out here, not read from the module: a test that iterated over the
# module's own list would pass with a term deleted or misspelt in it.

LEVEL_1 = [
    "tension pneumothorax", "pneumothorax", "aortic dissection", "aortic rupture",
    "acute stroke", "acute infarct", "cerebral infarction",
    "subdural haematoma", "subdural hematoma", "subdural haemorrhage", "subdural hemorrhage",
    "epidural haematoma", "epidural hematoma", "epidural haemorrhage",
    "subarachnoid haemorrhage", "subarachnoid hemorrhage",
    "intracranial haemorrhage", "intracranial hemorrhage", "midline shift",
    "saddle embolism", "bilateral pulmonary embolism", "massive pulmonary embolism",
    "pericardial effusion with tamponade", "cardiac tamponade",
    "bowel ischaemia", "bowel ischemia", "mesenteric ischaemia", "mesenteric ischemia",
    "splenic rupture", "hepatic rupture", "pneumoperitoneum", "free perforation",
    "cauda equina syndrome", "cauda equina", "spinal cord compression",
    "acute myocardial infarction", "myocardial infarction",
]
LEVEL_2 = [
    "pulmonary embolism", "filling defect", "haemothorax", "hemothorax", "empyema",
    "abscess", "haemorrhage", "hemorrhage", "cord compression", "hydrocephalus",
    "acute appendicitis", "appendicitis", "bowel obstruction", "volvulus",
    "intussusception", "ruptured aortic aneurysm", "ruptured aneurysm",
    "acute cholecystitis", "cholangitis", "meningitis", "encephalitis",
    "acute pyelonephritis", "ureteric calculus", "testicular torsion",
    "ovarian torsion", "ectopic pregnancy", "pericardial effusion",
]


@pytest.mark.parametrize("term, level", [(t, 1) for t in LEVEL_1] + [(t, 2) for t in LEVEL_2])
def test_every_critical_term_is_flagged_at_its_level(term, level):
    assert _terms(f"There is {term} on this study.") == [(term, level, False)]


@pytest.mark.parametrize("cue", [
    "no", "no new", "no definite", "no acute", "no significant", "no evidence of",
    "without", "without evidence of", "ruled out", "rule out", "excluded",
    "negative for", "unremarkable for", "absent", "absence of", "free of",
])
def test_every_denial_before_the_term_clears_it(cue):
    assert _terms(f"Chest: {cue} pneumothorax.") == []


@pytest.mark.parametrize("cue", [
    "is not", "are absent", "was excluded", "were ruled out",
    "not seen", "not identified", "not detected", "not demonstrated", "not visualised",
    "not visualized", "not present", "not apparent", "not noted",
])
def test_every_denial_after_the_term_clears_it(cue):
    assert _terms(f"Pneumothorax {cue}.") == []


@pytest.mark.parametrize("cue", [
    "possible", "possibly", "probable", "probably", "likely", "cannot exclude",
    "cannot fully exclude", "not excluded", "suspicious for", "suspicious",
    "concerning for", "query", "may represent", "might be", "could indicate",
    "clinical concern for", "raises the possibility of", "raised concern of",
])
def test_every_hedge_before_the_term_is_kept_as_uncertain(cue):
    assert _terms(f"Appearances: {cue} pneumothorax.") == [("pneumothorax", 1, True)]


@pytest.mark.parametrize("sentence", [
    # "Not excluded" is a hedge, not a denial: these used to raise nothing.
    "Pneumothorax is not excluded.",
    "Pulmonary embolism was not ruled out.",
    "Aortic dissection cannot be excluded.",
    "Aortic dissection cannot be ruled out.",
])
def test_a_finding_that_cannot_be_excluded_is_kept_as_uncertain(sentence):
    found = scan_for_critical_findings(sentence)
    assert len(found) == 1 and found[0].uncertain and not found[0].negated


def test_a_denial_reaches_seventy_characters_down_an_or_list():
    # "No " + 16 items + "or " is exactly 70 characters; one more item is out of reach.
    assert _terms("No " + "aa, " * 16 + "or pneumothorax.") == []
    assert _terms("No " + "aa, " * 17 + "or pneumothorax.") == [("pneumothorax", 1, False)]


def test_context_shows_the_words_around_the_finding():
    text = "A" * 50 + " large pneumothorax here " + "B" * 50
    (f,) = scan_for_critical_findings(text)
    assert f.context == "…" + text[f.start - 45:f.end + 45].strip() + "…"


def test_each_term_is_reported_once_and_a_later_separate_term_still_counts():
    found = _terms("Pneumothorax. Pneumothorax again. Separately, acute appendicitis.")
    assert found == [("pneumothorax", 1, False), ("acute appendicitis", 2, False)]


def test_the_dialog_summary_names_severity_term_and_hedge():
    from src.medical.critical_findings import format_findings_for_dialog
    summary = format_findings_for_dialog(
        scan_for_critical_findings("Large pneumothorax. Possible appendicitis.")
    )
    lines = summary.split("\n")
    assert lines[0] == "⚠ LIFE-THREATENING: PNEUMOTHORAX"
    assert lines[1].startswith("   Context: …") and lines[2] == ""
    assert lines[3] == "⚠ URGENT: APPENDICITIS [UNCERTAIN, cannot exclude]"
    assert not summary.endswith("\n")


def test_a_separate_mention_after_a_longer_finding_still_counts():
    found = _terms("Tension pneumothorax on the right. Small pneumothorax on the left.")
    assert found == [("tension pneumothorax", 1, False), ("pneumothorax", 1, False)]


def test_context_at_the_very_start_of_the_report():
    (f,) = scan_for_critical_findings("Pneumothorax.")
    assert f.context == "…Pneumothorax.…"
