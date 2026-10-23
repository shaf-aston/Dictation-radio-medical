"""Jump between template fields from the keyboard or a dictation-mic button.

The desktop half of field navigation; what a field *is* lives in
:mod:`src.features.template_fields`, which the web page uses too. Bound to
Tab / Shift+Tab by default (``key_next_field`` / ``key_previous_field``).
When there is no field in that direction the key does what it always did, so
Tab still types a tab in a report with no template.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QEvent, QKeyCombination, QObject, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence

from src.features.template_fields import next_field, previous_field

if TYPE_CHECKING:
    from PySide6.QtWidgets import QTextEdit


def key_text(event: QKeyEvent) -> str:
    """The pressed keys in QKeySequence spelling, e.g. ``"Shift+Tab"``."""
    key, mods = event.key(), event.modifiers()
    if key == Qt.Key.Key_Backtab:  # Qt reports Shift+Tab as its own key
        key, mods = Qt.Key.Key_Tab, mods | Qt.KeyboardModifier.ShiftModifier
    return QKeySequence(QKeyCombination(mods, Qt.Key(key))).toString()


def select_field(editor: QTextEdit, forward: bool) -> bool:
    """Select the next (or previous) field. False if there is none that way."""
    cursor = editor.textCursor()
    text = editor.toPlainText()
    field = (
        next_field(text, cursor.selectionEnd())
        if forward
        else previous_field(text, cursor.selectionStart())
    )
    if field is None:
        return False
    cursor.setPosition(field.start)
    cursor.setPosition(field.end, cursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)
    editor.ensureCursorVisible()
    return True


class FieldKeys(QObject):
    """Event filter on the editor: the bound keys move between fields.

    *enabled* is asked on every key press; while dictation owns a region of
    the document the selection must not move under it.
    """

    def __init__(
        self, editor: QTextEdit, next_key: str, previous_key: str,
        enabled: Callable[[], bool],
    ) -> None:
        super().__init__(editor)
        self._editor = editor
        self._next = QKeySequence(next_key).toString()
        self._previous = QKeySequence(previous_key).toString()
        self._enabled = enabled
        editor.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._editor and event.type() == QEvent.Type.KeyPress:
            pressed = key_text(event)  # type: ignore[arg-type]
            if pressed in (self._next, self._previous) and self._enabled():
                return select_field(self._editor, forward=pressed == self._next)
        return super().eventFilter(watched, event)
