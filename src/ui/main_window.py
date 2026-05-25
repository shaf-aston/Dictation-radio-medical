"""Main application window — layout assembly and event dispatch."""

from __future__ import annotations

import os
import sys
import subprocess
import logging
import warnings
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut

from src.dictation.audio import Recorder
from src.core.settings import Settings
from src.ui.styles import DARK, LIGHT
from src.features.report_manager import (
    autosave_report, save_report_txt, export_to_word, DOCX_AVAILABLE
)
from src.medical import macros
from src.medical.macros import reload_macros
from src.features.file_manager import startup_cleanup, autosave_dir, macros_file
from src.features.adaptive_learning import get_adaptive_learning, learn_from_edit
from src.features import audit_log

# View construction
from src.ui.views import (
    build_ui, build_menu, load_templates, rebuild_macro_buttons, rebuild_recent_menu
)

# Recording control
from src.ui.recording_session import (
    on_start_recording, on_stop_recording, on_partial_text, on_transcription_finished,
    setup_level_timer
)

# Dialog handling
from src.ui.dialogs import (
    show_learning_consent_if_needed, show_disclaimer_if_needed, validate_template_fields,
    on_show_learning_stats, on_reset_learning
)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _open_in_system(path: str) -> None:
    """Open a file or folder using the OS default handler."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])


class MainWindow(QMainWindow):
    """MSK Radiology Dictation workstation.

    Keyboard shortcuts:
    F5              Start recording
    F6              Stop recording
    Ctrl+S          Save report as plain text
    Ctrl+Shift+W    Export report to Word (.docx)
    Ctrl+T          Load template
    Ctrl+C          Copy report to clipboard
    Ctrl+L          Clear editor
    Ctrl+]          Increase font size
    Ctrl+[          Decrease font size
    Ctrl+D          Toggle dark / light theme
    Ctrl+P          Toggle patient info panel
    Ctrl+M          Toggle macros panel
    Ctrl+R          Reload macros from JSON
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings()
        self.recorder = Recorder()
        self.current_wav_path: Optional[str] = None
        self.live_thread: Optional[object] = None
        self.live_worker: Optional[object] = None
        self._dictation_start_pos: int = 0
        self._last_editor_text: str = ""
        self._corrections_pending: list = []

        build_ui(self)
        build_menu(self)
        self._setup_shortcuts()
        self._setup_autosave_timer()
        setup_level_timer(self)
        self._apply_theme(self.settings.get("theme", "dark"))

        self.setWindowTitle("MSK Radiology Dictation")
        width = self.settings.get("window_width", 1200)
        height = self.settings.get("window_height", 760)
        self.resize(width, height)

        # Restore splitter sizes
        sizes = self.settings.get("splitter_sizes", [220, 980])
        self.splitter.setSizes(sizes)

        # Restore panel visibility
        self._set_patient_panel_visible(self.settings.get("patient_info_visible", True))
        self._set_macros_panel_visible(self.settings.get("macros_panel_visible", True))

        # First-launch consent and disclaimer (after UI is built)
        show_learning_consent_if_needed(self)
        show_disclaimer_if_needed(self)

        # Load macro region from settings
        region = self.settings.get("last_macro_region", "Knee")
        if region in macros.REGION_ORDER:
            self.macro_region_combo.setCurrentText(region)
        rebuild_macro_buttons(self, region)

        # Load templates
        load_templates(self)
        last_template = self.settings.get("last_template", "")
        if last_template:
            idx = self.template_combo.findText(last_template)
            if idx >= 0:
                self.template_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("F5"), self, self.on_start_recording)
        QShortcut(QKeySequence("F6"), self, self.on_stop_recording)

    # ------------------------------------------------------------------
    # Auto-save timer
    # ------------------------------------------------------------------

    def _setup_autosave_timer(self) -> None:
        interval_seconds = self.settings.get("auto_save_interval", 60)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(interval_seconds * 1000)
        self._autosave_timer.timeout.connect(self._do_autosave)
        self._autosave_timer.start()

    def _do_autosave(self) -> None:
        text = self.editor.toPlainText().strip()
        if not text:
            return
        path = autosave_report(text, self._get_patient_info())
        if path:
            timestamp = datetime.now().strftime("%H:%M")
            self._autosave_label.setText(f"Auto-saved: {timestamp}")
            logger.info("Auto-saved to %s", path)
            audit_log.log_autosave(path, self._get_patient_info().get("id", ""))

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def on_start_recording(self) -> None:
        on_start_recording(self)

    def on_stop_recording(self) -> None:
        on_stop_recording(self)

    def _on_partial_text(self, full_transcript: str) -> None:
        on_partial_text(self, full_transcript)

    def _on_transcription_finished(self) -> None:
        on_transcription_finished(self)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self, theme: str) -> None:
        QApplication.instance().setStyleSheet(DARK if theme == "dark" else LIGHT)
        self.settings.set("theme", theme)

    def on_toggle_theme(self) -> None:
        current = self.settings.get("theme", "dark")
        self._apply_theme("light" if current == "dark" else "dark")

    # ------------------------------------------------------------------
    # Panel toggles
    # ------------------------------------------------------------------

    def _set_patient_panel_visible(self, visible: bool) -> None:
        self.patient_panel.setVisible(visible)
        self.settings.set("patient_info_visible", visible)

    def _set_macros_panel_visible(self, visible: bool) -> None:
        self.macros_panel.setVisible(visible)
        self.settings.set("macros_panel_visible", visible)

    def on_toggle_patient_panel(self) -> None:
        self._set_patient_panel_visible(not self.patient_panel.isVisible())

    def on_toggle_macros_panel(self) -> None:
        self._set_macros_panel_visible(not self.macros_panel.isVisible())

    # ------------------------------------------------------------------
    # Font size
    # ------------------------------------------------------------------

    def on_font_increase(self) -> None:
        self._change_font_size(+1)

    def on_font_decrease(self) -> None:
        self._change_font_size(-1)

    def _change_font_size(self, delta: int) -> None:
        from PySide6.QtGui import QFont
        size = self.settings.get("font_size", 13) + delta
        size = max(8, min(size, 28))
        self.settings.set("font_size", size)
        font = self.editor.font()
        font.setPointSize(size)
        self.editor.setFont(font)

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def on_insert_template(self) -> None:
        from src.features.file_manager import templates_dir
        name = self.template_combo.currentText()
        if not name:
            return
        path = templates_dir() / name
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            self.editor.setPlainText(content)
            self.settings.set("last_template", name)
            self._show_status("Template loaded: " + name, 2000)
            audit_log.log_template_loaded(name)
        except Exception as exc:
            QMessageBox.warning(self, "Template Error", str(exc))

    # ------------------------------------------------------------------
    # Macro management
    # ------------------------------------------------------------------

    def _rebuild_macro_buttons(self, region: str = "") -> None:
        rebuild_macro_buttons(self, region)

    def _insert_macro(self, text: str) -> None:
        current = self.editor.toPlainText()
        if current and not current.endswith(("\n", " ")):
            self.editor.insertPlainText(" ")
        self.editor.insertPlainText(text)
        self.editor.setFocus()

    def on_reload_macros(self) -> None:
        """Hot-reload macros from JSON without restart."""
        try:
            reload_macros()

            # Update combo box
            current_region = self.macro_region_combo.currentText()
            self.macro_region_combo.clear()
            self.macro_region_combo.addItems(macros.REGION_ORDER)

            # Restore selection if still exists
            if current_region in macros.REGION_ORDER:
                self.macro_region_combo.setCurrentText(current_region)
            else:
                self.macro_region_combo.setCurrentIndex(0)

            self._rebuild_macro_buttons(self.macro_region_combo.currentText())
            self._show_status("Macros reloaded", 2000)
        except Exception as exc:
            QMessageBox.warning(self, "Reload Error", f"Failed to reload macros:\n{exc}")

    def on_edit_macros(self) -> None:
        """Open macros.json in system default editor."""
        try:
            _open_in_system(str(macros_file()))
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Cannot open macros file:\n{exc}")

    def on_open_autosave_folder(self) -> None:
        try:
            _open_in_system(str(autosave_dir()))
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Cannot open autosave folder:\n{exc}")

    # ------------------------------------------------------------------
    # Report actions
    # ------------------------------------------------------------------

    def on_new_report(self) -> None:
        if self.editor.toPlainText().strip():
            reply = QMessageBox.question(
                self, "New Report",
                "Clear the current report and start a new one?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self.editor.clear()
        self.patient_name.clear()
        self.patient_id.clear()
        self.patient_dob.clear()
        self.patient_study_date.clear()
        self.patient_referrer.clear()
        self.patient_accession.clear()
        self._show_status("New report started.")

    def on_open_report(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Report", "", "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.editor.setPlainText(fh.read())
            self.settings.add_recent_report(path)
            self._rebuild_recent_menu()
            self._show_status(f"Opened: {os.path.basename(path)}", 2000)
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", str(exc))

    def on_copy(self) -> None:
        QApplication.clipboard().setText(self.editor.toPlainText())
        self._show_status("Copied to clipboard.", 1500)

    def on_save_txt(self) -> None:
        if not validate_template_fields(self):
            return
        default = self._default_filename(".txt")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default, "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            text = self.editor.toPlainText()
            save_report_txt(path, text, self._get_patient_info())
            self.settings.add_recent_report(path)
            self._rebuild_recent_menu()
            self._show_status(f"Saved: {os.path.basename(path)}", 2000)
            pid = self._get_patient_info().get("id", "")
            audit_log.log_report_saved(path, pid, len(text.split()))
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def on_export_word(self) -> None:
        if not DOCX_AVAILABLE:
            QMessageBox.warning(
                self, "Word Export Unavailable",
                "python-docx is not installed.\n\n"
                "Install it by running in the terminal:\n"
                "  pip install python-docx"
            )
            return
        if not validate_template_fields(self):
            return
        default = self._default_filename(".docx")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to Word", default, "Word Documents (*.docx);;All Files (*)"
        )
        if not path:
            return
        try:
            text = self.editor.toPlainText()
            export_to_word(path, text, self._get_patient_info())
            self.settings.add_recent_report(path)
            self._rebuild_recent_menu()
            self._show_status(f"Exported: {os.path.basename(path)}", 2000)
            pid = self._get_patient_info().get("id", "")
            audit_log.log_report_exported(path, pid, "docx")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def on_clear(self) -> None:
        if self.editor.toPlainText().strip():
            audit_log.log_report_cleared(self._get_patient_info().get("id", ""))
        self.editor.clear()

    # ------------------------------------------------------------------
    # Recent reports menu
    # ------------------------------------------------------------------

    def _rebuild_recent_menu(self) -> None:
        rebuild_recent_menu(self)

    def _open_recent(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.editor.setPlainText(fh.read())
            self._show_status(f"Opened: {os.path.basename(path)}", 2000)
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", str(exc))

    # ------------------------------------------------------------------
    # Learning / settings dialogs
    # ------------------------------------------------------------------

    def on_show_learning_stats(self) -> None:
        on_show_learning_stats(self)

    def on_reset_learning(self) -> None:
        on_reset_learning(self)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_patient_info(self) -> dict:
        return {
            "name": self.patient_name.text().strip(),
            "id": self.patient_id.text().strip(),
            "dob": self.patient_dob.text().strip(),
            "study_date": self.patient_study_date.text().strip(),
            "referring": self.patient_referrer.text().strip(),
            "accession": self.patient_accession.text().strip(),
        }

    def _default_filename(self, ext: str) -> str:
        patient_id = self.patient_id.text().strip().replace(" ", "_") or "report"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{patient_id}_{timestamp}{ext}"

    def _show_status(self, message: str, timeout: int = 0) -> None:
        self._status_label.setText(message)
        if timeout:
            QTimer.singleShot(timeout, lambda: self._status_label.setText("Ready"))

    def _on_text_changed(self) -> None:
        text = self.editor.toPlainText()
        word_count = len(text.split()) if text.strip() else 0
        line_count = text.count("\n") + 1 if text else 0
        self._info_words.setText(f"Words: {word_count}  |  Lines: {line_count}")
        self._wordcount_label.setText(f"Words: {word_count}")

        # Passive learning: track user edits
        if (not self.recorder.is_recording
                and self._last_editor_text
                and self.settings.get("learning_enabled", True)):
            if text != self._last_editor_text:
                learn_from_edit(self._last_editor_text, text)
        self._last_editor_text = text

    # ------------------------------------------------------------------
    # Window close – persist settings
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self.settings.set("window_width", self.width())
        self.settings.set("window_height", self.height())
        self.settings.set("splitter_sizes", self.splitter.sizes())
        self.settings.save()

        # Save adaptive learning data on exit
        try:
            get_adaptive_learning().force_save()
        except Exception:
            pass
        if self.recorder.is_recording:
            self.recorder.stop()
        super().closeEvent(event)


def main() -> None:
    """Application entry point."""
    settings = Settings()
    retention_days = settings.get("autosave_retention_days", 30)
    startup_cleanup(retention_days)

    app = QApplication(sys.argv)
    app.setApplicationName("MSK Radiology Dictation")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
