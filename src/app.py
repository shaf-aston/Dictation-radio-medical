"""
MSK Radiology Dictation – Main Application
==========================================
Professional dictation workstation for musculoskeletal radiologists.

Keyboard shortcuts
------------------
F5              Start recording
F6              Stop recording
Ctrl+S          Save report as plain text
Ctrl+Shift+W    Export report to Word (.docx)
Ctrl+T          Load template (cycles focus to template selector)
Ctrl+C          Copy report to clipboard
Ctrl+L          Clear editor
Ctrl+]          Increase font size
Ctrl+[          Decrease font size
Ctrl+D          Toggle dark / light theme
Ctrl+P          Toggle patient info panel
Ctrl+M          Toggle macros panel
"""

import os
import sys
import tempfile
import logging
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QComboBox,
    QLabel, QLineEdit, QCheckBox, QFrame, QScrollArea, QSplitter,
    QSizePolicy, QStatusBar, QMenu,
)
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QAction, QKeySequence, QFont, QShortcut

from audio import Recorder
from transcriber import SUPPORTED_MODELS
from transcribe_worker import LiveTranscribeWorker
from postprocess import postprocess_transcript
from settings import Settings
from styles import DARK, LIGHT
from report_manager import (
    autosave_report, save_report_txt, export_to_word, DOCX_AVAILABLE
)
from macros import MACROS, REGION_ORDER

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings()
        self.recorder = Recorder()
        self.current_wav_path: Optional[str] = None
        self.live_thread: Optional[QThread] = None
        self.live_worker: Optional[LiveTranscribeWorker] = None

        self._build_ui()
        self._build_menu()
        self._setup_shortcuts()
        self._setup_autosave_timer()
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

        # Load macro region from settings
        region = self.settings.get("last_macro_region", "Knee")
        if region in REGION_ORDER:
            self.macro_region_combo.setCurrentText(region)
        self._rebuild_macro_buttons(region)

        # Load templates
        self._load_templates()
        last_template = self.settings.get("last_template", "")
        if last_template:
            idx = self.template_combo.findText(last_template)
            if idx >= 0:
                self.template_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(4)

        # Patient info panel
        self.patient_panel = self._build_patient_panel()
        root_layout.addWidget(self.patient_panel)

        # Horizontal splitter: macros | editor
        self.splitter = QSplitter(Qt.Horizontal)
        self.macros_panel = self._build_macros_panel()
        self.splitter.addWidget(self.macros_panel)
        self.splitter.addWidget(self._build_editor_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        root_layout.addWidget(self.splitter, stretch=1)

        # Recording / action toolbar
        root_layout.addWidget(self._build_recording_bar())

        # Status bar
        self._status_label = QLabel("Ready")
        self._wordcount_label = QLabel("Words: 0")
        self._autosave_label = QLabel("Auto-save: -")
        status_bar = QStatusBar()
        status_bar.addWidget(self._status_label, 1)
        status_bar.addPermanentWidget(self._wordcount_label)
        status_bar.addPermanentWidget(self._autosave_label)
        self.setStatusBar(status_bar)

    def _build_patient_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("patient_panel")
        frame.setFixedHeight(68)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(3)

        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        row1.setSpacing(6)
        row2.setSpacing(6)

        def make_field(placeholder: str, width: int = 160) -> QLineEdit:
            field = QLineEdit()
            field.setPlaceholderText(placeholder)
            field.setFixedWidth(width)
            return field

        self.patient_name = make_field("Patient name", 200)
        self.patient_id = make_field("Patient ID", 120)
        self.patient_dob = make_field("DOB (dd/mm/yyyy)", 110)
        self.patient_study_date = make_field("Study date", 100)
        self.patient_referrer = make_field("Referring clinician", 180)
        self.patient_accession = make_field("Accession #", 130)

        row1.addWidget(QLabel("Name:"))
        row1.addWidget(self.patient_name)
        row1.addWidget(QLabel("ID:"))
        row1.addWidget(self.patient_id)
        row1.addWidget(QLabel("DOB:"))
        row1.addWidget(self.patient_dob)
        row1.addWidget(QLabel("Study:"))
        row1.addWidget(self.patient_study_date)
        row1.addStretch()

        row2.addWidget(QLabel("Referring:"))
        row2.addWidget(self.patient_referrer)
        row2.addWidget(QLabel("Acc #:"))
        row2.addWidget(self.patient_accession)
        row2.addStretch()

        layout.addLayout(row1)
        layout.addLayout(row2)
        return frame

    def _build_macros_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("macros_panel")
        frame.setFixedWidth(228)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("Quick Phrases")
        title.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(title)

        self.macro_region_combo = QComboBox()
        self.macro_region_combo.addItems(REGION_ORDER)
        self.macro_region_combo.currentTextChanged.connect(self._rebuild_macro_buttons)
        layout.addWidget(self.macro_region_combo)

        # Scroll area for dynamic macro buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._macro_container = QWidget()
        self._macro_layout = QVBoxLayout(self._macro_container)
        self._macro_layout.setSpacing(2)
        self._macro_layout.setContentsMargins(0, 0, 0, 0)
        self._macro_layout.addStretch()

        scroll.setWidget(self._macro_container)
        layout.addWidget(scroll, stretch=1)
        return frame

    def _build_editor_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Template bar
        template_bar = QHBoxLayout()
        template_bar.setSpacing(4)

        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(220)
        btn_load = QPushButton("Load")
        btn_load.setToolTip("Insert template into editor (Ctrl+T)")
        btn_load.setFixedWidth(52)
        btn_load.clicked.connect(self.on_insert_template)

        btn_font_up = QPushButton("A+")
        btn_font_up.setFixedWidth(34)
        btn_font_up.setToolTip("Increase font size (Ctrl+])")
        btn_font_up.clicked.connect(self.on_font_increase)

        btn_font_down = QPushButton("A-")
        btn_font_down.setFixedWidth(34)
        btn_font_down.setToolTip("Decrease font size (Ctrl+[)")
        btn_font_down.clicked.connect(self.on_font_decrease)

        template_bar.addWidget(QLabel("Template:"))
        template_bar.addWidget(self.template_combo)
        template_bar.addWidget(btn_load)
        template_bar.addSpacing(8)
        template_bar.addWidget(btn_font_up)
        template_bar.addWidget(btn_font_down)
        template_bar.addStretch()

        layout.addLayout(template_bar)

        # Main editor
        self.editor = QTextEdit()
        self.editor.setPlaceholderText(
            "Dictated text appears here.\n\n"
            "F5 = Start recording   F6 = Stop\n"
            "Load a template, then dictate into each section.\n"
            "Use quick phrases on the left for common findings."
        )
        font = QFont("Consolas", self.settings.get("font_size", 13))
        self.editor.setFont(font)
        self.editor.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.editor, stretch=1)

        # Info bar beneath editor
        info_bar = QHBoxLayout()
        self._info_words = QLabel("Words: 0  |  Lines: 0")
        self._info_words.setStyleSheet("font-size: 11px;")
        self._info_autosave = QLabel("Auto-save: -")
        self._info_autosave.setStyleSheet("font-size: 11px;")
        info_bar.addWidget(self._info_words)
        info_bar.addStretch()
        info_bar.addWidget(self._info_autosave)
        layout.addLayout(info_bar)

        return panel

    def _build_recording_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # Record / stop buttons
        self.btn_record = QPushButton("Record  F5")
        self.btn_record.setObjectName("btn_record")
        self.btn_record.setMinimumWidth(130)
        self.btn_record.clicked.connect(self.on_start_recording)

        self.btn_stop = QPushButton("Stop  F6")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setMinimumWidth(110)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop_recording)

        # Model / VAD controls
        self.model_combo = QComboBox()
        self.model_combo.addItems(SUPPORTED_MODELS)
        self.model_combo.setCurrentText(self.settings.get("model_size", "base"))
        self.model_combo.setToolTip(
            "tiny/base = fastest  |  small/medium = better accuracy  |  large-v2/v3 = best (needs GPU)"
        )
        self.model_combo.setFixedWidth(90)

        self.vad_checkbox = QCheckBox("VAD")
        self.vad_checkbox.setToolTip("Voice activity detection - filters silence (recommended)")
        self.vad_checkbox.setChecked(self.settings.get("vad_filter", True))

        self.language_input = QLineEdit(self.settings.get("language", "en"))
        self.language_input.setFixedWidth(36)
        self.language_input.setToolTip("ISO language code, e.g. 'en'")

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)

        # Action buttons
        btn_copy = QPushButton("Copy")
        btn_copy.setToolTip("Copy report to clipboard (Ctrl+C)")
        btn_copy.clicked.connect(self.on_copy)

        btn_save = QPushButton("Save TXT")
        btn_save.setToolTip("Save report as plain text (Ctrl+S)")
        btn_save.clicked.connect(self.on_save_txt)

        self.btn_export_word = QPushButton("Export Word")
        self.btn_export_word.setObjectName("btn_word")
        self.btn_export_word.setToolTip("Export report to Word document (Ctrl+Shift+W)")
        self.btn_export_word.setEnabled(DOCX_AVAILABLE)
        if not DOCX_AVAILABLE:
            self.btn_export_word.setToolTip("Install python-docx to enable Word export: pip install python-docx")
        self.btn_export_word.clicked.connect(self.on_export_word)

        btn_clear = QPushButton("Clear")
        btn_clear.setToolTip("Clear editor (Ctrl+L)")
        btn_clear.clicked.connect(self.on_clear)

        layout.addWidget(self.btn_record)
        layout.addWidget(self.btn_stop)
        layout.addWidget(separator)
        layout.addWidget(QLabel("Model:"))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel("Lang:"))
        layout.addWidget(self.language_input)
        layout.addWidget(self.vad_checkbox)
        layout.addStretch()
        layout.addWidget(btn_copy)
        layout.addWidget(btn_save)
        layout.addWidget(self.btn_export_word)
        layout.addWidget(btn_clear)
        return bar

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")
        self._add_action(file_menu, "New Report", self.on_new_report, "Ctrl+N")
        self._add_action(file_menu, "Open Report...", self.on_open_report, "Ctrl+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "Save as Text...", self.on_save_txt, "Ctrl+S")
        self._add_action(file_menu, "Export to Word...", self.on_export_word, "Ctrl+Shift+W")
        file_menu.addSeparator()

        self.recent_menu = QMenu("Recent Reports", self)
        file_menu.addMenu(self.recent_menu)
        self._rebuild_recent_menu()

        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", self.close, "Alt+F4")

        # Report
        report_menu = menubar.addMenu("&Report")
        self._add_action(report_menu, "Load Template", self.on_insert_template, "Ctrl+T")
        self._add_action(report_menu, "Copy to Clipboard", self.on_copy, "Ctrl+C")
        report_menu.addSeparator()
        self._add_action(report_menu, "Clear Editor", self.on_clear, "Ctrl+L")

        # View
        view_menu = menubar.addMenu("&View")
        self._add_action(view_menu, "Toggle Dark / Light Theme", self.on_toggle_theme, "Ctrl+D")
        view_menu.addSeparator()
        self._add_action(view_menu, "Toggle Patient Panel", self.on_toggle_patient_panel, "Ctrl+P")
        self._add_action(view_menu, "Toggle Quick Phrases Panel", self.on_toggle_macros_panel, "Ctrl+M")
        view_menu.addSeparator()
        self._add_action(view_menu, "Increase Font Size", self.on_font_increase, "Ctrl+]")
        self._add_action(view_menu, "Decrease Font Size", self.on_font_decrease, "Ctrl+[")

        # Settings
        settings_menu = menubar.addMenu("&Settings")
        self._add_action(settings_menu, "Open Auto-save Folder", self.on_open_autosave_folder)

    @staticmethod
    def _add_action(menu: QMenu, label: str, slot, shortcut: str = "") -> QAction:
        action = QAction(label)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # ------------------------------------------------------------------
    # Keyboard shortcuts (global, independent of menu)
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
            self._info_autosave.setText(f"Auto-saved: {timestamp}")
            self._autosave_label.setText(f"Auto-saved: {timestamp}")
            logger.info("Auto-saved to %s", path)

    # ------------------------------------------------------------------
    # Macro buttons
    # ------------------------------------------------------------------

    def _rebuild_macro_buttons(self, region: str = "") -> None:
        region = region or self.macro_region_combo.currentText()
        phrases = MACROS.get(region, [])

        # Clear existing dynamic buttons (everything except the stretch)
        while self._macro_layout.count() > 1:
            item = self._macro_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for label, text in phrases:
            btn = QPushButton(label)
            btn.setToolTip(text)
            btn.setWordWrap(True)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            btn.setStyleSheet("text-align: left; padding: 4px 6px; font-size: 11px;")
            btn.clicked.connect(lambda checked=False, t=text: self._insert_macro(t))
            self._macro_layout.insertWidget(self._macro_layout.count() - 1, btn)

        self.settings.set("last_macro_region", region)

    def _insert_macro(self, text: str) -> None:
        current = self.editor.toPlainText()
        if current and not current.endswith(("\n", " ")):
            self.editor.insertPlainText(" ")
        self.editor.insertPlainText(text)
        self.editor.setFocus()

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
        size = self.settings.get("font_size", 13) + delta
        size = max(8, min(size, 28))
        self.settings.set("font_size", size)
        font = self.editor.font()
        font.setPointSize(size)
        self.editor.setFont(font)

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def _templates_dir(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    def _load_templates(self) -> None:
        self.template_combo.clear()
        templates_path = self._templates_dir()
        if os.path.isdir(templates_path):
            names = sorted(
                f for f in os.listdir(templates_path) if f.lower().endswith(".txt")
            )
            self.template_combo.addItems(names)

    def on_insert_template(self) -> None:
        name = self.template_combo.currentText()
        if not name:
            return
        path = os.path.join(self._templates_dir(), name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            self.editor.setPlainText(content)
            self.settings.set("last_template", name)
            self._show_status("Template loaded: " + name, 2000)
        except Exception as exc:
            QMessageBox.warning(self, "Template Error", str(exc))

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def on_start_recording(self) -> None:
        if self.recorder.is_recording:
            return
        fd, path = tempfile.mkstemp(prefix="dictation_", suffix=".wav")
        os.close(fd)
        self.current_wav_path = path
        try:
            self.recorder.start(path)
        except Exception as exc:
            QMessageBox.critical(self, "Audio Error", f"Cannot start recording:\n{exc}")
            self.current_wav_path = None
            return

        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._show_status("Recording...")

        model_size = self.model_combo.currentText()
        language = self.language_input.text().strip() or "en"
        vad_enabled = self.vad_checkbox.isChecked()
        pause_threshold = float(self.settings.get("pause_threshold", 2.5))

        # Persist settings
        self.settings.set("model_size", model_size)
        self.settings.set("language", language)
        self.settings.set("vad_filter", vad_enabled)

        self.live_thread = QThread()
        self.live_worker = LiveTranscribeWorker(
            path, model_size, language, vad_enabled, pause_threshold
        )
        self.live_worker.moveToThread(self.live_thread)
        self.live_thread.started.connect(self.live_worker.run)
        self.live_worker.partial.connect(self._on_partial_text)
        self.live_worker.progress.connect(lambda msg: self._show_status(msg))
        self.live_worker.finished.connect(self._on_transcription_finished)
        self.live_worker.finished.connect(self.live_thread.quit)
        self.live_worker.finished.connect(self.live_worker.deleteLater)
        self.live_thread.finished.connect(self.live_thread.deleteLater)
        self.live_thread.start()

    def on_stop_recording(self) -> None:
        if not self.recorder.is_recording:
            return
        try:
            self.recorder.stop()
        finally:
            self.btn_record.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._show_status("Processing final pass...")
        if self.live_worker is not None:
            self.live_worker.finalize()

    def _on_partial_text(self, chunk: str) -> None:
        chunk = postprocess_transcript(chunk)
        if not chunk:
            return
        current = self.editor.toPlainText()
        if current and not chunk.startswith("\n") and not current.endswith(("\n", " ")):
            self.editor.insertPlainText(" ")
        self.editor.insertPlainText(chunk)
        self._show_status("Receiving...", 800)

    def _on_transcription_finished(self) -> None:
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._show_status("Ready")
        self._cleanup_temp_audio()

    def _cleanup_temp_audio(self) -> None:
        try:
            if self.current_wav_path and os.path.exists(self.current_wav_path):
                os.remove(self.current_wav_path)
        except Exception:
            pass
        self.current_wav_path = None

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
        default = self._default_filename(".txt")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default, "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            save_report_txt(path, self.editor.toPlainText(), self._get_patient_info())
            self.settings.add_recent_report(path)
            self._rebuild_recent_menu()
            self._show_status(f"Saved: {os.path.basename(path)}", 2000)
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
        default = self._default_filename(".docx")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to Word", default, "Word Documents (*.docx);;All Files (*)"
        )
        if not path:
            return
        try:
            export_to_word(path, self.editor.toPlainText(), self._get_patient_info())
            self.settings.add_recent_report(path)
            self._rebuild_recent_menu()
            self._show_status(f"Exported: {os.path.basename(path)}", 2000)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def on_clear(self) -> None:
        self.editor.clear()

    def on_open_autosave_folder(self) -> None:
        from report_manager import get_autosave_dir
        folder = get_autosave_dir()
        os.startfile(folder)

    # ------------------------------------------------------------------
    # Recent reports menu
    # ------------------------------------------------------------------

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self.settings.get_recent_reports()
        if not recent:
            self.recent_menu.addAction("(none)").setEnabled(False)
            return
        for path in recent:
            action = QAction(os.path.basename(path), self)
            action.setToolTip(path)
            action.triggered.connect(lambda checked=False, fp=path: self._open_recent(fp))
            self.recent_menu.addAction(action)

    def _open_recent(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self.editor.setPlainText(fh.read())
            self._show_status(f"Opened: {os.path.basename(path)}", 2000)
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", str(exc))

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

    # ------------------------------------------------------------------
    # Window close – persist settings
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self.settings.set("window_width", self.width())
        self.settings.set("window_height", self.height())
        self.settings.set("splitter_sizes", self.splitter.sizes())
        self.settings.save()
        if self.recorder.is_recording:
            self.recorder.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("MSK Radiology Dictation")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
