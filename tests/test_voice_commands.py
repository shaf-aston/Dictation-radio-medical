"""Voice commands that edit the text: "scratch that" and "insert <phrase>".

Both are pipeline stages, so the desktop and the web app get the same
behaviour. The last test runs every shipped template through the stages to
show ordinary report wording never triggers a command by accident.
"""

import pytest

from src.dictation.postprocess import voice_commands
from src.dictation.postprocess.pipeline import postprocess_transcript
from src.core.paths import templates_dir


@pytest.mark.parametrize(
    "spoken, expected",
    [
        # Mid-sentence: the sentence so far goes.
        ("The ACL is torn scratch that the ACL is intact.", "the ACL is intact."),
        # After a full stop: the sentence just finished goes.
        ("Normal alignment. The ACL is torn. Scratch that. The ACL is intact.",
         "Normal alignment. The ACL is intact."),
        # A new line is a boundary too, and it stays.
        ("FINDINGS:\nACL torn, scratch that, ACL intact.", "FINDINGS:\nACL intact."),
        # Twice in a row removes two sentences.
        ("One. Two. Scratch that. Scratch that. Three.", "Three."),
        # Nothing before it: nothing to remove, the command still disappears.
        ("Scratch that. Normal study.", "Normal study."),
    ],
)
def test_scratch_that_removes_the_last_thing_said(spoken, expected):
    assert voice_commands.apply_scratch_commands(spoken) == expected


def test_scratch_that_through_the_whole_pipeline():
    out = postprocess_transcript("the ACL is torn. scratch that. the ACL is intact.")
    assert out == "The ACL is intact."


def test_scratch_that_reaches_a_sentence_already_frozen_in_live_dictation():
    from src.dictation.postprocess.incremental import IncrementalPostprocessor

    live = IncrementalPostprocessor()
    first = "Normal alignment. The ACL is torn. "
    live.process(first, committed_len=len(first))
    text, _ = live.process(first + "Scratch that. The ACL is intact.", committed_len=len(first))
    assert text == "Normal alignment. The ACL is intact."


@pytest.fixture
def phrases(monkeypatch):
    from src.medical import macros
    monkeypatch.setattr(macros, "MACROS", {
        "Knee": [("ACL intact", "Anterior cruciate ligament intact."),
                 ("Normal knee", "No internal derangement of the knee.")],
    })
    return macros


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("Insert ACL intact.", "Anterior cruciate ligament intact."),
        ("Medial meniscus torn. insert normal knee", "Medial meniscus torn. No internal derangement of the knee."),
        # Not a phrase name: left exactly as dictated.
        ("Insert a drain.", "Insert a drain."),
    ],
)
def test_insert_a_quick_phrase_by_name(phrases, spoken, expected):
    assert voice_commands.apply_phrase_commands(spoken) == expected


def test_no_template_wording_triggers_a_command(phrases):
    for path in sorted(templates_dir().glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        assert voice_commands.apply_scratch_commands(text) == text, path.name
        assert voice_commands.apply_phrase_commands(text) == text, path.name


# -- spoken punctuation, through the whole pipeline ----------------------------

@pytest.mark.parametrize(
    "spoken, written",
    [
        ("no effusion full stop", "No effusion."),
        ("no effusion full-stop", "No effusion."),
        ("no effusion period", "No effusion."),
        ("mild comma moderate", "Mild, moderate"),
        ("first new line second", "First\nSecond"),
        ("first newline second", "First\nSecond"),
        ("findings colon new paragraph the lungs are clear", "Findings:\n\nThe lungs are clear"),
        ("first new section second", "First\n\nSecond"),
        ("first next section second", "First\n\nSecond"),
        ("open bracket see prior close bracket", "(see prior)"),
        ("is it stable question mark", "Is it stable?"),
        ("wow exclamation mark", "Wow!"),
        ("mild semicolon moderate", "Mild; moderate"),
        ("mild semi colon moderate", "Mild; moderate"),
        ("mild semi-colon moderate", "Mild; moderate"),  # was "A semi-: b": colon ran first
        ("ultrasound hyphen guided biopsy", "Ultrasound-guided biopsy"),
        ("mild dash moderate", "Mild – moderate"),
        # A punctuation word after a word no sentence ends on is a word.
        ("the rectum and colon are normal", "The rectum and colon are normal"),
        ("wall thickening of the left colon", "Wall thickening of the left colon"),
        ("over a period of six months", "Over a period of six months"),
        ("a new line has been placed", "A new line has been placed"),
        ("a comma shaped lesion", "A comma shaped lesion"),
        ("clinical history colon question mark scaphoid fracture full stop",
         "Clinical history: ?scaphoid fracture."),
        # "colon" and "period" as nouns are left alone.
        ("the sigmoid colon is normal", "The sigmoid colon is normal"),
        ("the transverse colon is dilated", "The transverse colon is dilated"),
        ("in the postoperative period", "In the postoperative period"),
        ("since the menstrual period", "Since the menstrual period"),
    ],
)
def test_spoken_punctuation(spoken, written):
    assert postprocess_transcript(spoken) == written


# -- "correct the word": fixes the word and teaches the learner ------------------

@pytest.fixture
def taught(monkeypatch):
    pairs = []
    monkeypatch.setattr(voice_commands, "_correction_hook", lambda w, c: pairs.append((w, c)))
    return pairs


@pytest.mark.parametrize(
    "spoken, written, learned",
    [
        ("the menscus correct the word to meniscus", "the meniscus", [("menscus", "meniscus")]),
        ("the menscus correct word meniscus is torn", "the meniscus is torn", [("menscus", "meniscus")]),
        ("the menscus, correct the word meniscus", "the meniscus", [("menscus", "meniscus")]),
        ("a tare correct the word to tear and a menscus correct word meniscus",
         "a tear and a meniscus", [("tare", "tear"), ("menscus", "meniscus")]),
        # Nothing before it: the command just disappears.
        ("correct the word to meniscus", "", []),
        # Same word in another case is not a correction worth learning.
        ("Meniscus correct the word to meniscus", "meniscus", []),
    ],
)
def test_correct_the_word(taught, spoken, written, learned):
    assert voice_commands.apply_correction_commands(spoken) == written
    assert taught == learned


def test_correct_the_word_reaches_back_over_a_line_break(taught):
    assert voice_commands.apply_correction_commands("menscus\ncorrect word meniscus") == "meniscus"
    assert taught == [("menscus", "meniscus")]


def test_correct_the_word_reaches_back_past_a_bracket(taught):
    # The second form: punctuation the first one does not skip sits in between.
    assert voice_commands.apply_correction_commands("(menscus) correct word meniscus") == "(meniscus)"
    assert taught == [("menscus", "meniscus")]


def test_a_failing_learner_never_breaks_dictation(monkeypatch):
    def broken(*_):
        raise RuntimeError("learning store unavailable")
    monkeypatch.setattr(voice_commands, "_correction_hook", broken)
    assert voice_commands.apply_correction_commands("menscus correct word meniscus") == "meniscus"


def test_the_learner_hook_is_set_and_cleared():
    try:
        seen = []
        voice_commands.set_correction_hook(lambda w, c: seen.append((w, c)))
        voice_commands.apply_correction_commands("menscus correct word meniscus")
        assert seen == [("menscus", "meniscus")]
    finally:
        voice_commands.set_correction_hook(None)
    assert voice_commands._correction_hook is None
