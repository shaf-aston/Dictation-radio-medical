"""The report editor: a QTextEdit whose copies go through the release gate.

Copy, Cut, the context menu and dragging all leave the editor through
``createMimeDataFromSelection``, Qt's one virtual for "the selection is about
to leave". Ctrl+C in the editor never reached the window's Copy action (the
editor takes the shortcut itself), so a report with a critical finding could
be selected and copied out with no question asked and nothing audited.

Only a selection the gate has something to say about is stopped: one holding
a critical finding or an unfilled template field. Copying a phrase to paste
elsewhere in the report is not a release.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QMimeData
from PySide6.QtWidgets import QTextEdit

from src.features.report_release import check_release, unfilled_fields


def needs_release_gate(text: str) -> bool:
    """Would the release gate ask about *text* if it left the app?"""
    return bool(unfilled_fields(text)) or check_release(text).needs_acknowledgement


class ReportEditor(QTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._release_gate: Callable[[str], bool] = lambda _text: True

    def set_release_gate(self, gate: Callable[[str], bool]) -> None:
        """*gate* gets the selected text; False keeps it in the editor."""
        self._release_gate = gate

    def createMimeDataFromSelection(self) -> QMimeData:  # noqa: N802 - Qt override
        text = self.textCursor().selection().toPlainText()
        if text and needs_release_gate(text) and not self._release_gate(text):
            return QMimeData()
        return super().createMimeDataFromSelection()
