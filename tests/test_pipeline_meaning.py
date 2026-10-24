"""The correction pipeline must not change what the radiologist meant.

Every row came from the 2026-10-24 bug hunt, each a real input whose meaning
the pipeline changed. The first test is the general rule behind one of them
(a rule matching inside its own output): processing twice changes nothing.
"""

import pytest

from src.core.paths import templates_dir
from src.dictation.postprocess.pipeline import postprocess_transcript
from src.dictation.postprocess.terminology import apply_terminology


def _reports():
    from pathlib import Path
    corpus = Path(__file__).parent.parent / "scripts/eval/resources/report_corpus.txt"
    texts = [p.read_text(encoding="utf-8") for p in sorted(templates_dir().glob("*.txt"))]
    return texts + [line for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("level", ["soft", "medium"])
def test_processing_twice_changes_nothing(level):
    for text in _reports():
        once = postprocess_transcript(text, cleanup_level=level)
        assert postprocess_transcript(once, cleanup_level=level) == once, text[:80]


def test_every_terminology_output_is_left_alone():
    from src.dictation.postprocess import terminology
    for _pattern, replacement in terminology._RULES:
        if isinstance(replacement, str) and "\\" not in replacement:
            assert apply_terminology(replacement) == replacement


@pytest.mark.parametrize("said, written", [
    # A cue in another sentence is about something else (bone vs bowel).
    ("The ilium is intact. The terminal ileum is thickened.",
     "The ilium is intact. The terminal ileum is thickened."),
    ("Undisplaced fracture of the right ilium. Bowel loops are unremarkable.",
     "Undisplaced fracture of the right ilium. Bowel loops are unremarkable."),
    ("Perineal abscess. The hamstring tendons are intact.",
     "Perineal abscess. The hamstring tendons are intact."),
    ("Peroneal tendinopathy. The anal canal is normal.",
     "Peroneal tendinopathy. The anal canal is normal."),
    ("the spinal chord is normal", "The spinal cord is normal"),   # still fixed
    # A misspelling with two equally near terms is left for the radiologist.
    ("cholithiasis is present", "Cholithiasis is present"),
    ("hepatic stetosis", "Hepatic stetosis"),
    ("efusion in the knee", "Effusion in the knee"),               # one answer: fixed
    # Capitals are an abbreviation, never a typo.
    ("Status post TEVAR", "Status post TEVAR"),
    ("CVST is seen", "CVST is seen"),
    # A query stays a query; a decimal keeps its point.
    ("Indication: ?PE", "Indication: ?PE"),
    ("No fracture. ?Small effusion.", "No fracture. ?Small effusion."),
    ("Measures .5 cm", "Measures .5 cm"),
    # The hallucination filter removes a whole line, never the words after it.
    ("Thank you\nno acute fracture", "No acute fracture"),
    ("Thank you.", ""),
    # A rule never writes a word twice.
    ("Bone marrow oedema in the medial femoral condyle.",
     "Bone marrow oedema in the medial femoral condyle."),
    ("marrow oedema in the condyle", "Bone marrow oedema in the condyle"),
    # A rule keeps the part of speech dictated; "an echoic" is not "anechoic".
    ("An echoic structure.", "An echoic structure."),
    ("Hypo echoic nodule.", "Hypoechoic nodule."),
    ("Fat saturation was inhomogeneous.", "Fat saturation was inhomogeneous."),
    ("Comminution of the fragments.", "Comminution of the fragments."),
    ("Signal in the substance of the tendon.", "Signal in the substance of the tendon."),
    # A derived form of a term is spelled as meant.
    ("Enhancing craniocaudally.", "Enhancing craniocaudally."),
    ("Osteonecrotic focus.", "Osteonecrotic focus."),
    ("Isoattenuating mass.", "Isoattenuating mass."),
    ("Multisequence MRI.", "Multisequence MRI."),
    ("Splenomaly is present.", "Splenomegaly is present."),       # a typo: still fixed
    # Capitals survive a fix; an acronym's expansion is written as spelled.
    ("SUBCHONDRAL BONE OEDEMA.", "SUBCHONDRAL BONE OEDEMA."),
    ("BMO in the femur.", "Bone marrow oedema in the femur."),
    # A unit after a number, "size" with a prefix, a dotted abbreviation.
    ("Treated with 50 Gray.", "Treated with 50 Gray."),
    ("Oversized cyst.", "Oversized cyst."),
    ("Small lesions, e.g. the cyst.", "Small lesions, e.g. the cyst."),
])
def test_meaning_is_kept(said, written):
    assert postprocess_transcript(said) == written


def test_correct_the_word_keeps_a_decimal_whole():
    assert postprocess_transcript("measures 2.5 correct word 3.5 cm") == "Measures 3.5 cm"


def test_every_accent_rule_can_fire():
    # The gate in front of the rules is built from them, so none is unreachable.
    from src.features.accent_corrections import ACCENT_PROFILES, apply_accent_corrections
    assert apply_accent_corrections("Severe artritis.", "middle_eastern") == "Severe arthritis."
    for accent, rules in ACCENT_PROFILES.items():
        for pattern, _ in rules:
            sample = pattern.pattern.replace("\\b", "")
            if sample.isalpha():  # a plain word: it must be rewritten
                assert apply_accent_corrections(sample, accent) != sample, (accent, sample)
