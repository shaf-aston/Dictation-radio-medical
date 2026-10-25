"""What the app may learn from the radiologist's edits, and what it may not.

A learned correction is applied to every later dictation, blind to context.
So it may only fix a mishearing (a word that is not real). One report where
"medial" was changed to "lateral" is a change of mind about that patient;
learned, it rewrote "medial" to "lateral" in every report after it.
"""

import json

import pytest

from src.dictation.postprocess.pipeline import postprocess_transcript
from src.features import adaptive_learning as al


@pytest.fixture
def learning(tmp_path, monkeypatch):
    monkeypatch.setattr(al.AdaptiveLearning, "_instance", None)
    instance = al.AdaptiveLearning(data_dir=tmp_path)
    monkeypatch.setattr(al, "_adaptive_learning", instance)
    monkeypatch.setattr(instance, "_maybe_save", lambda: None)
    yield instance
    monkeypatch.setattr(al.AdaptiveLearning, "_instance", None)


@pytest.mark.parametrize("wrong, right", [
    ("medial", "lateral"), ("hypodense", "hyperdense"), ("hypoechoic", "hyperechoic"),
    ("anterior", "posterior"), ("benign", "malignant"), ("increased", "decreased"),
    ("normal", "abnormal"), ("present", "absent"), ("superior", "inferior"),
    ("ipsilateral", "contralateral"),
])
def test_a_change_between_two_real_words_is_never_learned(learning, wrong, right):
    assert learning.learn_correction(wrong, right) is False
    assert postprocess_transcript(f"the {wrong} aspect") == f"The {wrong} aspect"


def test_a_mishearing_is_learned_and_applied(learning):
    assert learning.learn_correction("menscus", "meniscus") is True
    assert postprocess_transcript("torn menscus", cleanup_level="soft") == "Torn menscus"  # soft skips learning
    assert "meniscus" in postprocess_transcript("torn menscus")


def test_a_session_counts_only_what_it_learned(learning):
    dictated = "The medial compartment is normal. Torn menscus."
    final = "The lateral compartment is normal. Torn meniscus."
    assert al.learn_from_session(dictated, final) == 1
    assert learning.export_corrections() == {"menscus": "meniscus"}


def test_a_bad_rule_learned_before_the_guard_no_longer_fires(tmp_path, monkeypatch):
    (tmp_path / "learned_corrections.json").write_text(
        json.dumps({"word_corrections": {"medial": "lateral", "menscus": "meniscus"}})
    )
    monkeypatch.setattr(al.AdaptiveLearning, "_instance", None)
    instance = al.AdaptiveLearning(data_dir=tmp_path)
    try:
        assert instance.apply_learned_corrections("medial menscus") == "medial meniscus"
    finally:
        monkeypatch.setattr(al.AdaptiveLearning, "_instance", None)


def test_inserting_a_word_is_not_learned_as_replacing_the_next_one(learning):
    al.learn_from_session("the menicus is torn", "the medial menicus is torn")
    assert learning.export_corrections() == {}


def test_a_typed_fix_of_one_mishearing_is_learned(learning):
    al.learn_from_session("the menicus is torn", "the meniscus is torn")
    assert learning.export_corrections() == {"menicus": "meniscus"}
