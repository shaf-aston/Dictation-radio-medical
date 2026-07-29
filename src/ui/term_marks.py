"""The desktop half of "which words are worth a second look".

:mod:`src.ui.term_popup` answers a highlighted word. This points at the words
worth highlighting in the first place — otherwise the lookup only helps a
radiologist who already suspects something, and the words worth suspecting are
exactly the ones that read as plausible.

Every mark is drawn with ``QTextEdit.setExtraSelections``, and that choice is
the whole safety argument: extra selections live in the *view*, not the
document. Nothing is inserted, so the undo stack is untouched and Copy, Save
TXT and Export Word all emit exactly what the radiologist typed. A marking
scheme that edited the text — inserting tags, or even zero-width characters —
could leak a mark into a signed report, and this one cannot.

Two behaviours it shares with the popup, for the same reasons:

* **It never runs during dictation.** While a recording session owns the region
  the text is rewritten every second, and marks that flicker under moving words
  are noise. Marking is for reviewing.
* **It decides nothing.** Which words are suspect is
  :func:`src.medical.term_lookup.suspect_terms`, the same function the browser
  calls, so the two front-ends cannot mark different words.

The colour is read from the theme tokens rather than written down here: extra
selections are painted in code, so they are the one surface a stylesheet cannot
reach. ``glow`` is the token for the machine's own suggestions — never ``rec``,
which this app reserves for recording and clinical severity.
"""

from __future__ import annotations

import logging
from typing import Callable, List

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QTextEdit

from src.medical.term_lookup import Span, suspect_terms
from src.ui.theme import tokens

logger = logging.getLogger(__name__)

#: Wait for typing to settle before re-scanning. Longer than the popup's delay
#: because this reads the whole document, not one selection.
SCAN_DELAY_MS = 400


class TermMarks(QObject):
    """Underlines the words in *editor* that the lookup has something to say about."""

    #: How many words are marked, emitted after every scan. The scan is
    #: debounced, so a listener cannot read the count off ``textChanged``.
    changed = Signal(int)

    def __init__(
        self,
        editor: QTextEdit,
        is_dictating: Callable[[], bool],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._editor = editor
        self._is_dictating = is_dictating
        self._spans: List[Span] = []
        self._theme = "dark"

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(SCAN_DELAY_MS)
        self._timer.timeout.connect(self._rescan)

        editor.textChanged.connect(self._schedule)

    # -- what triggers a rescan ---------------------------------------------

    def _schedule(self) -> None:
        self._timer.start()

    def set_theme(self, theme: str) -> None:
        """Repaint the existing marks in the new theme's ``glow``."""
        self._theme = theme
        self._paint()

    def clear(self) -> None:
        self._timer.stop()
        self._spans = []
        self._paint()

    # -- the scan -----------------------------------------------------------

    def _rescan(self) -> None:
        if self._is_dictating():
            # Committed text is still being rewritten; marking it now would
            # mark a sentence that no longer exists a second later.
            self.clear()
            return
        self._spans = suspect_terms(self._editor.toPlainText())
        self._paint()

    def count(self) -> int:
        """How many words are currently marked — what the hint line reports."""
        return len(self._spans)

    # -- drawing ------------------------------------------------------------

    def _format(self) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.DotLine)
        fmt.setUnderlineColor(QColor(tokens(self._theme)["glow"]))
        return fmt

    def _paint(self) -> None:
        selections: List[QTextEdit.ExtraSelection] = []
        if self._spans:
            fmt = self._format()
            document = self._editor.document()
            for span in self._spans:
                selection = QTextEdit.ExtraSelection()
                cursor = QTextCursor(document)
                cursor.setPosition(span.start)
                cursor.setPosition(span.end, QTextCursor.MoveMode.KeepAnchor)
                selection.cursor = cursor
                selection.format = fmt
                selections.append(selection)
        self._editor.setExtraSelections(selections)
        self.changed.emit(len(self._spans))
