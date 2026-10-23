"""Tab and Shift+Tab move between template fields in the desktop editor.

Driven with real key presses on a real QTextEdit, because the thing most
likely to break is Qt's own handling: Shift+Tab arrives as a separate key
(Backtab), and Tab types a tab unless the filter takes it first.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from src.ui.field_navigation import FieldKeys  # noqa: E402

TEXT = "EXAMINATION: MRI [Right / Left] Knee on a [X]T magnet."


@pytest.fixture(scope="module")
def qt_app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _editor(qt_app, text=TEXT, enabled=True):
    editor = QtWidgets.QTextEdit()
    editor.setPlainText(text)
    cursor = editor.textCursor()
    cursor.setPosition(0)
    editor.setTextCursor(cursor)
    editor._keys = FieldKeys(editor, "Tab", "Shift+Tab", enabled=lambda: enabled)
    return editor


def test_tab_selects_each_field_in_turn(qt_app):
    editor = _editor(qt_app)
    QTest.keyClick(editor, Qt.Key.Key_Tab)
    assert editor.textCursor().selectedText() == "[Right / Left]"
    QTest.keyClick(editor, Qt.Key.Key_Tab)
    assert editor.textCursor().selectedText() == "[X]"


def test_shift_tab_goes_back(qt_app):
    editor = _editor(qt_app)
    QTest.keyClick(editor, Qt.Key.Key_Tab)
    QTest.keyClick(editor, Qt.Key.Key_Tab)
    QTest.keyClick(editor, Qt.Key.Key_Backtab, Qt.KeyboardModifier.ShiftModifier)
    assert editor.textCursor().selectedText() == "[Right / Left]"


def test_tab_types_a_tab_when_there_is_no_field(qt_app):
    editor = _editor(qt_app, text="Free text.")
    QTest.keyClick(editor, Qt.Key.Key_Tab)
    assert editor.toPlainText() == "\tFree text."


def test_fields_do_not_move_while_dictation_owns_the_text(qt_app):
    editor = _editor(qt_app, enabled=False)
    QTest.keyClick(editor, Qt.Key.Key_Tab)
    assert editor.textCursor().selectedText() != "[Right / Left]"
