"""The dictated region is rewritten each cycle -- it must not accumulate.

Found by driving the real desktop window end-to-end: every report began with
its own first few words twice ("The lungs areThe lungs are clear..."). Qt moves
a QTextCursor along when text is inserted at its position, and the first live
update inserts exactly at the dictation anchor, so the anchor ended up after
those words and every later cycle appended instead of replacing.
"""
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from src.ui import recording_session as rs  # noqa: E402  (after the Qt guard above)

PARTIALS = ["The lungs are", "The lungs are clear",
            "The lungs are clear without focal consolidation."]


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _window(qt_app, existing: str):
    """A window stub holding a real QTextEdit and a real anchor cursor."""
    editor = QtWidgets.QTextEdit()
    editor.setPlainText(existing)
    anchor = editor.textCursor()
    anchor.movePosition(anchor.MoveOperation.End)
    anchor.setKeepPositionOnInsert(True)     # what on_start_recording sets
    return SimpleNamespace(editor=editor, _dictation_start=anchor,
                           _dictation_end=None, _dictation_field=None,
                           _on_text_changed=lambda **_: None,
                           recorder=SimpleNamespace(is_recording=True))


def test_each_update_replaces_the_dictation_instead_of_appending(qt_app):
    w = _window(qt_app, "PRIOR REPORT ")
    for partial in PARTIALS:
        rs._replace_dictation_region(w, partial)
    assert w.editor.toPlainText() == "PRIOR REPORT " + PARTIALS[-1]


def test_the_anchor_does_not_drift_as_text_is_written_at_it(qt_app):
    w = _window(qt_app, "PRIOR REPORT ")
    start = w._dictation_start.position()
    for partial in PARTIALS:
        rs._replace_dictation_region(w, partial)
        assert w._dictation_start.position() == start


def test_text_typed_before_the_dictation_still_moves_the_anchor(qt_app):
    """The anchor's original job, which the fix must not break."""
    w = _window(qt_app, "PRIOR REPORT ")
    rs._replace_dictation_region(w, PARTIALS[0])
    start = w._dictation_start.position()

    cursor = w.editor.textCursor()          # a template loaded above the dictation
    cursor.setPosition(0)
    cursor.insertText("HEADER\n")
    assert w._dictation_start.position() == start + len("HEADER\n")

    rs._replace_dictation_region(w, PARTIALS[-1])
    assert w.editor.toPlainText() == "HEADER\nPRIOR REPORT " + PARTIALS[-1]


def test_dictation_into_an_empty_report(qt_app):
    w = _window(qt_app, "")
    for partial in PARTIALS:
        rs._replace_dictation_region(w, partial)
    assert w.editor.toPlainText() == PARTIALS[-1]


# -- dictating at the caret (R1) ---------------------------------------------

TEMPLATE = "EXAMINATION: MRI [Right / Left] Knee\nMedial meniscus: \nLateral meniscus: Intact.\n"


def _open_at(qt_app, text: str, start: int, end: int = -1):
    """A window whose editor holds *text* with the caret (or a selection) set."""
    editor = QtWidgets.QTextEdit()
    editor.setPlainText(text)
    cursor = editor.textCursor()
    cursor.setPosition(start)
    if end >= 0:
        cursor.setPosition(end, cursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)
    w = SimpleNamespace(editor=editor, _on_text_changed=lambda **_: None,
                        recorder=SimpleNamespace(is_recording=True))
    rs._open_dictation_region(w)
    return w


def test_dictation_goes_in_at_the_caret_and_leaves_the_rest_alone(qt_app):
    caret = TEMPLATE.index("Medial meniscus: ") + len("Medial meniscus: ")
    w = _open_at(qt_app, TEMPLATE, caret)
    for partial in ["Oblique", "Oblique tear of the posterior horn."]:
        rs._replace_dictation_region(w, partial)
    assert w.editor.toPlainText() == TEMPLATE.replace(
        "Medial meniscus: ", "Medial meniscus: Oblique tear of the posterior horn."
    )


def test_a_shorter_revision_does_not_eat_the_text_after_it(qt_app):
    caret = TEMPLATE.index("Medial meniscus: ") + len("Medial meniscus: ")
    w = _open_at(qt_app, TEMPLATE, caret)
    rs._replace_dictation_region(w, "Oblique tear of the posterior horn.")
    rs._replace_dictation_region(w, "Oblique tear.")
    assert "Medial meniscus: Oblique tear.\nLateral meniscus: Intact." in w.editor.toPlainText()


def test_a_selected_pick_list_is_answered_by_saying_an_option(qt_app):
    start = TEMPLATE.index("[Right / Left]")
    w = _open_at(qt_app, TEMPLATE, start, start + len("[Right / Left]"))
    rs._replace_dictation_region(w, "Left.")
    assert w.editor.toPlainText().startswith("EXAMINATION: MRI Left Knee\n")


def test_a_selected_field_takes_free_dictation_as_spoken(qt_app):
    start = TEMPLATE.index("[Right / Left]")
    w = _open_at(qt_app, TEMPLATE, start, start + len("[Right / Left]"))
    rs._replace_dictation_region(w, "Left sided")
    assert w.editor.toPlainText().startswith("EXAMINATION: MRI Left sided Knee\n")


def test_dictation_mid_word_gets_a_space_on_both_sides(qt_app):
    w = _open_at(qt_app, "alphaomega", 5)
    rs._replace_dictation_region(w, "beta")
    assert w.editor.toPlainText() == "alpha beta omega"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
