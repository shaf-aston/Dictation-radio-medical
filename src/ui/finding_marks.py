"""A thin strip beside the editor: one mark per critical or urgent finding.

The constraint that shapes this whole file is that **the radiologist's own
characters are never touched**. No underline, no highlight, no background, no
character format of any kind — the report is the thing being judged, and the
software does not write on it. So the marks live in a gutter widget of their
own, painted next to the text rather than on it.

That is a different answer from :mod:`src.ui.term_marks`, which does underline
the words it points at. The difference is deliberate: a spelling suspect is a
suggestion about a word, so it belongs on the word; a critical finding is a
statement about the *report*, and drawing it on the sentence would put the
machine's opinion inside the radiologist's text.

The interaction is one thing only: click a mark, and the phrase is scrolled to
and selected in the editor.

Like the term marks, it never runs during dictation (the region is being
rewritten every second) and it decides nothing — which findings exist is
:class:`src.features.report_release.OutstandingFindings`, the shared service
both front-ends read, so the desktop cannot count differently from the browser.

The colour is read from the theme tokens because a painted widget is one of the
few surfaces a stylesheet cannot reach: ``rec`` for a finding still outstanding
(clinical severity, the one thing red is for) and ``textDim`` once it has been
acknowledged.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Tuple

from PySide6.QtCore import QPoint, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QTextCursor
from PySide6.QtWidgets import QTextEdit, QWidget

from src.features.report_release import OutstandingFindings
from src.medical.critical_findings import CriticalFinding
from src.ui.theme import tokens

logger = logging.getLogger(__name__)

#: Wait for typing to settle before re-scanning — the scan reads the whole
#: document, so it must not run per keystroke.
SCAN_DELAY_MS = 400

#: How wide the strip is, and how tall and thick one mark is drawn.
GUTTER_WIDTH = 12
MARK_HEIGHT = 14
MARK_WIDTH = 4

#: How far from a mark a click still counts as hitting it.
CLICK_SLACK = 8


class FindingGutter(QWidget):
    """The strip of marks, and the click that jumps to a finding."""

    #: How many findings the report has, emitted after every scan. The scan is
    #: debounced, so a listener cannot read the count off ``textChanged``.
    changed = Signal(int)

    def __init__(
        self,
        editor: QTextEdit,
        is_dictating: Callable[[], bool],
        findings: OutstandingFindings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._editor = editor
        self._is_dictating = is_dictating
        self._findings = findings
        self._theme = "dark"
        #: Where each mark was last drawn, so a click can be matched to it.
        self._hit_boxes: List[Tuple[int, CriticalFinding]] = []

        self.setFixedWidth(GUTTER_WIDTH)
        self.setToolTip("Critical and urgent findings — click a mark to jump to it")

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(SCAN_DELAY_MS)
        self._timer.timeout.connect(self.refresh)

        editor.textChanged.connect(self._timer.start)
        # A mark maps a document position to a y, so scrolling moves every mark.
        editor.verticalScrollBar().valueChanged.connect(self.update)

    # -- state ---------------------------------------------------------------

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.update()

    def refresh(self) -> None:
        """Re-scan the report now and show what it found."""
        if self._is_dictating():
            # The dictated region is rewritten every cycle; marks that jump
            # under moving words are noise, and the release gate re-scans at
            # export anyway. Marking is for reviewing.
            self._findings.update("")
        else:
            self._findings.update(self._editor.toPlainText())
        self.show_state()

    def show_state(self) -> None:
        """Repaint and re-announce the count, for a change made elsewhere.

        The release gate scans and acknowledges through the same state object
        (``recording_session.confirm_release``); this is how that answer reaches
        the marks and the count without scanning the report a second time.
        """
        self.update()
        self.changed.emit(self._findings.count)

    # -- drawing -------------------------------------------------------------

    def _mark_y(self, finding: CriticalFinding) -> int | None:
        """Where in this widget the finding's first line sits, or None if scrolled away."""
        document = self._editor.document()
        cursor = QTextCursor(document)
        cursor.setPosition(min(finding.start, max(0, document.characterCount() - 1)))
        rect = self._editor.cursorRect(cursor)
        # Through global coordinates rather than by assuming the gutter and the
        # editor's viewport share a top edge: they are separated by the editor's
        # frame and padding, and that offset is the stylesheet's business.
        viewport = self._editor.viewport()
        y = self.mapFromGlobal(viewport.mapToGlobal(QPoint(0, rect.center().y()))).y()
        return None if y < 0 or y > self.height() else y

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt name)
        self._hit_boxes = []
        findings = self._findings.check.findings
        if not findings:
            return
        outstanding = {id(f) for f in self._findings.outstanding}
        palette = tokens(self._theme)
        painter = QPainter(self)
        x = (self.width() - MARK_WIDTH) // 2
        for finding in findings:
            y = self._mark_y(finding)
            if y is None:
                continue
            colour = palette["rec"] if id(finding) in outstanding else palette["textDim"]
            painter.fillRect(
                x, y - MARK_HEIGHT // 2, MARK_WIDTH, MARK_HEIGHT, QColor(colour)
            )
            self._hit_boxes.append((y, finding))
        painter.end()

    # -- the one interaction -------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt name)
        click_y = int(event.position().y())
        hit = min(
            (box for box in self._hit_boxes if abs(box[0] - click_y) <= CLICK_SLACK),
            key=lambda box: abs(box[0] - click_y),
            default=None,
        )
        if hit is None:
            return
        self._select(hit[1])

    def _select(self, finding: CriticalFinding) -> None:
        """Scroll to the finding and select its words — the whole interaction."""
        document = self._editor.document()
        last = max(0, document.characterCount() - 1)
        cursor = QTextCursor(document)
        cursor.setPosition(min(finding.start, last))
        cursor.setPosition(min(finding.end, last), QTextCursor.MoveMode.KeepAnchor)
        self._editor.setTextCursor(cursor)
        self._editor.ensureCursorVisible()
        self._editor.setFocus()
