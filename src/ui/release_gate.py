"""The release gate, desktop shape: the dialogs every exit (Copy/Save/Word) asks.

The rules and the audit trail live in ``features/report_release.py``, beside
the web app's 409; this module owns the two dialogs and the order they are
asked in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QDateTime
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDateTimeEdit, QDialog, QDialogButtonBox, QFormLayout,
    QLabel, QLineEdit, QMessageBox, QVBoxLayout, QWidget,
)

from src.features.report_release import (
    COMMUNICATION_METHODS,
    Communication,
    ReleaseCheck,
    add_communication_statement,
    communication_statement,
    record_release,
    report_warnings,
)
from src.ui.recording_session import yield_document

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow

_Yes = QMessageBox.StandardButton.Yes
_No = QMessageBox.StandardButton.No


def confirm_report_checks(window: MainWindow) -> bool:
    """Ask before releasing a report with unfilled fields or a consistency issue.

    The desktop shape of ``report_release.report_warnings``: the rules live
    there, alongside the web app's 409. This function owns only the message
    box, including how much fits in it. Returns True to proceed; False means
    the radiologist went back to fix it and the report stays put.
    """
    found = report_warnings(
        window.editor.toPlainText(), window._get_patient_info().get("sex", "")
    )
    if not found:
        return True
    reply = QMessageBox.warning(window, "Check the report", found.message(), _Yes | _No, _No)
    return reply == _Yes


class CommunicationDialog(QDialog):
    """Ask whether a critical finding was communicated, and to whom, how, when.

    The desktop shape of the web app's #commModal. Never withholds the report:
    both buttons release it and only the audit entry differs. "Communicated"
    needs a recipient, because the record it writes says someone was told.
    """

    def __init__(self, parent: QWidget, check: ReleaseCheck, radiologist: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(check.title)
        layout = QVBoxLayout(self)
        summary = QLabel(
            check.summary + "\n\nConfirm it was communicated to the referring clinician."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        form = QFormLayout()
        self.recipient = QLineEdit()
        self.recipient.setPlaceholderText("Dr Patel, ED registrar")
        self.method = QComboBox()
        self.method.addItems([m[0].upper() + m[1:] for m in COMMUNICATION_METHODS])
        self.at = QDateTimeEdit(QDateTime.currentDateTime())
        self.at.setDisplayFormat("dd/MM/yyyy HH:mm")
        self.radiologist = QLineEdit(radiologist)
        form.addRow("Told to:", self.recipient)
        form.addRow("How:", self.method)
        form.addRow("When:", self.at)
        form.addRow("Your name:", self.radiologist)
        layout.addLayout(form)
        self.add_to_report = QCheckBox("Add this to the report")
        self.add_to_report.setChecked(True)
        layout.addWidget(self.add_to_report)

        buttons = QDialogButtonBox()
        self.btn_communicated = buttons.addButton(
            "Communicated", QDialogButtonBox.ButtonRole.AcceptRole
        )
        buttons.addButton(
            "Not communicated: release anyway", QDialogButtonBox.ButtonRole.RejectRole
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.btn_communicated.setEnabled(False)
        self.recipient.textChanged.connect(
            lambda text: self.btn_communicated.setEnabled(bool(text.strip()))
        )
        self.recipient.setFocus()

    def communication(self) -> Communication:
        return Communication(
            self.recipient.text().strip(),
            COMMUNICATION_METHODS[self.method.currentIndex()],
            self.at.dateTime().toPython(),  # type: ignore[arg-type]
            self.radiologist.text().strip(),
        )


def confirm_release(window: MainWindow) -> bool:
    """Run both release rules and return True if the report may leave.

    The desktop shape of the shared gate in ``features/report_release.py``: the
    rules and the audit trail live there, alongside the web app's 409. Called
    from every exit (copy / save / export) immediately before the text leaves,
    so a field or finding typed in after dictation ended is still caught.

    Unfilled fields are asked about first, because that is the only answer that
    can cancel: cancelling must never leave a findings acknowledgement in the
    audit log for a report that then did not leave. The findings answer itself
    never withholds the report, both answers proceed and only the audit entry
    differs, so False here always means "the radiologist cancelled".

    The scan runs through ``window.findings`` so the answer given here is the
    same state the gutter marks and the count pill show: acknowledging settles
    the pill, and a finding typed in afterwards puts it back to outstanding.
    Only outstanding findings are asked about (``OutstandingFindings.pending``):
    one already communicated needs no second call, and a second answer used to
    append a second "communicated" sentence that was never audited.
    """
    if window.recorder.is_recording:
        # The live loop is still rewriting the end of the report.
        window._show_status("Stop recording before the report leaves", 3000, state="warn")
        return False
    if not confirm_report_checks(window):
        return False

    window.findings.update(window.editor.toPlainText())
    # Captured before the dialog: msg.exec() runs a nested event loop, and the
    # gutter's debounced re-scan can fire inside it and replace window.findings
    # .check. Answering and auditing from this local means the radiologist's
    # answer always applies to exactly what they were shown.
    check = window.findings.pending
    if not check.needs_acknowledgement:
        window.finding_gutter.show_state()
        yield_document(window)
        return True

    dialog = CommunicationDialog(window, check, window.settings.get("radiologist_name"))
    acknowledged = dialog.exec() == QDialog.DialogCode.Accepted
    comm = dialog.communication() if acknowledged else None
    if comm is not None:
        if comm.radiologist:
            window.settings.set("radiologist_name", comm.radiologist)
        if dialog.add_to_report.isChecked():
            # Before the caller reads the text, so the sentence leaves with it.
            # One undoable edit at the end of the report.
            text = window.editor.toPlainText()
            updated = add_communication_statement(
                text, communication_statement(check.findings, comm)
            )
            if updated != text:
                cursor = window.editor.textCursor()
                cursor.beginEditBlock()
                cursor.movePosition(cursor.MoveOperation.End)
                cursor.movePosition(cursor.MoveOperation.Start, cursor.MoveMode.KeepAnchor)
                cursor.insertText(updated)
                cursor.endEditBlock()
    record_release(
        check.findings, window._get_patient_info().get("id", ""), acknowledged, comm
    )
    if acknowledged:
        window.findings.acknowledge(check)
    window.finding_gutter.show_state()
    # What leaves is what the radiologist answered for: a polish landing now
    # would make the screen and the released copy different reports.
    yield_document(window)
    return True
