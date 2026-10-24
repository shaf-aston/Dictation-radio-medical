"""Nothing changes the report's clinical meaning without the radiologist seeing it.

Two paths used to: an accent rule and a terminology rule wrote "tendinopathy"
for a spoken "tendonitis" / "tendinitis" (inflamed vs degenerate: different
diagnoses), and the 'hard' cleanup level swapped the finished report for an
AI rewrite with no review step and without the patient's identifiers.
"""
import os
from types import SimpleNamespace

import pytest

from src.dictation.postprocess import llm_cleanup
from src.dictation.postprocess.pipeline import postprocess_transcript


@pytest.mark.parametrize("said", ["tendonitis", "tendinitis"])
@pytest.mark.parametrize("accent", ["neutral", "south_asian"])
def test_the_diagnosis_said_is_the_diagnosis_written(said, accent):
    out = postprocess_transcript(f"supraspinatus {said} without tear", accent)
    assert said in out and "tendinopathy" not in out


def test_the_pipeline_never_calls_the_ai(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("pipeline reached the network")
    monkeypatch.setattr(llm_cleanup, "clean_with_llm", boom)
    assert postprocess_transcript("no effusion", cleanup_level="hard") == "No effusion"


# -- the review dialog ------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from src.ui import dialogs  # noqa: E402  (after the Qt guard above)

PATIENT = {"name": "Jane Doe", "id": "MRN12345678"}


@pytest.fixture(scope="module")
def qt_app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qt_app, monkeypatch):
    editor = QtWidgets.QTextEdit()
    editor.setPlainText("jane doe has a meniscal tear")
    sent = []

    def fake_clean(text, patient_info=None):
        sent.append(patient_info)
        return "Meniscal tear.", ["grammar"]

    monkeypatch.setattr(llm_cleanup, "is_enabled", lambda: True)
    monkeypatch.setattr(llm_cleanup, "clean_with_llm", fake_clean)
    return SimpleNamespace(editor=editor, sent=sent, _show_status=lambda *a, **k: None,
                           _get_patient_info=lambda: PATIENT,
                           recorder=SimpleNamespace(is_recording=False),
                           replace_report=editor.setPlainText)


def _answer(monkeypatch, button):
    asked = []
    monkeypatch.setattr(dialogs.QMessageBox, "question",
                        lambda *a, **k: asked.append(a) or button)
    return asked


def test_declined_cleanup_leaves_the_report_alone(window, monkeypatch):
    asked = _answer(monkeypatch, dialogs._No)
    dialogs.offer_ai_cleanup(window, automatic=True)
    assert asked and window.editor.toPlainText() == "jane doe has a meniscal tear"


def test_accepted_cleanup_applies_and_scrubs_with_the_patients_details(window, monkeypatch):
    _answer(monkeypatch, dialogs._Yes)
    dialogs.offer_ai_cleanup(window, automatic=True)
    assert window.editor.toPlainText() == "Meniscal tear."
    assert window.sent == [PATIENT]


def test_automatic_cleanup_is_silent_when_the_feature_is_off(window, monkeypatch):
    monkeypatch.setattr(llm_cleanup, "is_enabled", lambda: False)
    asked = _answer(monkeypatch, dialogs._Yes)
    dialogs.offer_ai_cleanup(window, automatic=True)
    assert not asked and not window.sent
