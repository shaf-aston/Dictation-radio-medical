"""One folding section — the desktop's version of the web app's `<details>` panel.

Every secondary group in the window (Template, Patient, Settings) is built from
this one class, so they fold, look and behave identically. The header reuses the
existing `fold_header` style, so no new colour is introduced.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget


class Section(QWidget):
    """A click-to-fold header with one content widget underneath."""

    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget, expanded: bool = False) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.header = QToolButton()
        self.header.setObjectName("fold_header")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.clicked.connect(self.set_expanded)

        self.content = content
        layout.addWidget(self.header)
        layout.addWidget(content)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        """Show or hide the content and point the arrow at what happens next."""
        self.content.setVisible(expanded)
        self.header.setChecked(expanded)
        self.header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.toggled.emit(expanded)

    def is_expanded(self) -> bool:
        return self.header.isChecked()
