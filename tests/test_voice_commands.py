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
