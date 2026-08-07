"""The desktop half of "highlight a word, see its neighbourhood".

Highlight a term in the report editor and a small panel appears beside it with
two tiers: terms that could be what the word should have been (prominent), and
terms that merely travel with it (quieter, below). Clicking one replaces the
highlighted text. That is the only thing it can do to the report — nothing is
ever applied on its own.

Both lists and their order come from :mod:`src.medical.term_lookup`, the same
service ``src/ui/web_app.py`` calls, so the desktop and the browser cannot
suggest different things for the same word. This module is a renderer: it
decides nothing.

Two behaviours matter as much as the lists:

* **It never takes focus.** A tool window with ``WindowDoesNotAcceptFocus``
  plus ``WA_ShowWithoutActivating`` — the radiologist keeps typing into the
  editor with the panel open.
* **It never opens during dictation.** While a recording session still owns the
  region the text is being rewritten every second, and a panel pinned to moving
  text is noise. Lookups are for reviewing.

The lookup itself runs on the UI thread: it measures ~2 ms warm and its indexes
are pre-built at startup (``src/dictation/warmup.py``).
"""

from __future__ import annotations

import logging
from typing import Callable, List

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.medical.term_lookup import Suggestion, lookup

logger = logging.getLogger(__name__)

#: Wait for the highlight to settle before asking. Same as the web front-end.
LOOKUP_DELAY_MS = 300

#: Gap between the highlighted line and the panel.
POPUP_OFFSET_PX = 6


class TermPopup(QFrame):
    """A frameless panel of term suggestions for the editor's current selection."""

    #: Emitted when a suggestion is actually taken. The window counts these to
    #: decide when the "you can highlight a word" hint has done its teaching.
    applied = Signal()

    def __init__(
        self,
        editor: QTextEdit,
        is_dictating: Callable[[], bool],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Tool
                         | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setObjectName("term_popup")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._editor = editor
        self._is_dictating = is_dictating

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self._key = QLabel()
        self._key.setObjectName("tp_key")
        layout.addWidget(self._key)

        self._spelling = QGridLayout()
        self._spelling.setContentsMargins(0, 0, 0, 0)
        self._spelling.setHorizontalSpacing(8)
        self._spelling.setVerticalSpacing(3)
        layout.addLayout(self._spelling)

        self._divider = QFrame()
        self._divider.setObjectName("tp_divider")
        self._divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(self._divider)

        self._related = QGridLayout()
        self._related.setContentsMargins(0, 0, 0, 0)
        self._related.setHorizontalSpacing(8)
        self._related.setVerticalSpacing(3)
        layout.addLayout(self._related)

        foot = QLabel("Click one to replace the highlighted text.")
        foot.setObjectName("tp_foot")
        layout.addWidget(foot)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(LOOKUP_DELAY_MS)
        self._timer.timeout.connect(self._run_lookup)

        editor.selectionChanged.connect(self._schedule)
        editor.textChanged.connect(self.hide)
        editor.installEventFilter(self)
        if parent is not None:
            parent.installEventFilter(self)
        self.hide()

    # -- what opens and closes it ------------------------------------------

    def _schedule(self) -> None:
        self._timer.start()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Escape closes it; so does moving, resizing or leaving the window."""
        kind = event.type()
        if kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            if self.isVisible():
                self.hide()
                return True
        elif kind in (QEvent.Type.Move, QEvent.Type.Resize,
                      QEvent.Type.WindowDeactivate, QEvent.Type.Hide):
            self.hide()
        return super().eventFilter(watched, event)

    def hide(self) -> None:
        self._timer.stop()
        super().hide()

    # -- the two tiers ------------------------------------------------------

    def _run_lookup(self) -> None:
        if self._is_dictating():
            self.hide()
            return
        selected = self._editor.textCursor().selectedText().strip()
        if not selected:
            self.hide()
            return

        result = lookup(selected)
        if not result.similar_spelling and not result.related:
            self.hide()
            return

        self._key.setText(result.key.upper())
        self._fill(self._spelling, result.similar_spelling, quiet=False)
        self._fill(self._related, result.related, quiet=True)
        self._divider.setVisible(bool(result.similar_spelling and result.related))
        self.adjustSize()
        self._place()
        self.show()

    def _fill(self, grid: QGridLayout, items: List[Suggestion], quiet: bool) -> None:
        while grid.count():
            widget = grid.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        for row, item in enumerate(items):
            button = QPushButton(item.term)
            button.setObjectName("tp_related" if quiet else "tp_match")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _=False, term=item.term: self._apply(term))
            grid.addWidget(button, row, 0)

            note = QLabel(item.note)
            note.setObjectName("tp_why")
            grid.addWidget(note, row, 1)

    def _place(self) -> None:
        """Below the highlighted line, kept on the screen it is on."""
        rect = self._editor.cursorRect()
        point = self._editor.mapToGlobal(rect.bottomLeft())
        x, y = point.x(), point.y() + POPUP_OFFSET_PX
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            x = max(area.left(), min(x, area.right() - self.width()))
            if y + self.height() > area.bottom():
                y = max(area.top(), point.y() - rect.height() - POPUP_OFFSET_PX - self.height())
        self.move(x, y)

    def _apply(self, term: str) -> None:
        """Replace the highlighted text with *term*, then get out of the way."""
        self.hide()
        cursor = self._editor.textCursor()
        if cursor.hasSelection():
            cursor.insertText(term)
            self.applied.emit()
        self._editor.setFocus()
