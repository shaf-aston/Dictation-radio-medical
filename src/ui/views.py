"""UI view construction — panels, buttons, layouts, and menus."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QLineEdit, QCheckBox, QFrame, QScrollArea, QSplitter,
    QSizePolicy, QMenu, QProgressBar,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QAction, QKeySequence

from src.dictation.transcriber import SUPPORTED_MODELS
from src.features.accent_corrections import ACCENT_LABELS
from src.medical import macros
from src.features.file_manager import templates_dir

if TYPE_CHECKING:
    from src.ui.app import MainWindow

_COLOR_HEALTHY = "#4CAF50"
_COLOR_CLIPPING = "#F44336"
_COLOR_LOW = "#FF9800"
_LEVEL_BAR_STYLESHEET = (
    "QProgressBar { border: 1px solid #555; border-radius: 3px; background: #222; }"
    "QProgressBar::chunk { background: {color}; border-radius: 2px; }"
)


def build_ui(window: MainWindow) -> None:
    """Construct the main window layout."""
    central = QWidget()
    window.setCentralWidget(central)
    root_layout = QVBoxLayout(central)
    root_layout.setContentsMargins(6, 6, 6, 6)
    root_layout.setSpacing(4)

    # Patient info panel
    window.patient_panel = build_patient_panel(window)
    root_layout.addWidget(window.patient_panel)

    # Horizontal splitter: macros | editor
    window.splitter = QSplitter(Qt.Horizontal)
    window.macros_panel = build_macros_panel(window)
    window.splitter.addWidget(window.macros_panel)
    window.splitter.addWidget(build_editor_panel(window))
    window.splitter.setStretchFactor(0, 0)
    window.splitter.setStretchFactor(1, 1)
    root_layout.addWidget(window.splitter, stretch=1)

    # Recording / action toolbar
    root_layout.addWidget(build_recording_bar(window))

    # Status bar
    window._status_label = QLabel("Ready")
    window._wordcount_label = QLabel("Words: 0")
    window._autosave_label = QLabel("Auto-save: -")
    status_bar = window.statusBar()
    status_bar.addWidget(window._status_label, 1)
    status_bar.addPermanentWidget(window._wordcount_label)
    status_bar.addPermanentWidget(window._autosave_label)


def build_patient_panel(window: MainWindow) -> QFrame:
    """Build the patient information entry panel."""
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

    window.patient_name = make_field("Patient name", 200)
    window.patient_id = make_field("Patient ID", 120)
    window.patient_dob = make_field("DOB (dd/mm/yyyy)", 110)
    window.patient_study_date = make_field("Study date", 100)
    window.patient_referrer = make_field("Referring clinician", 180)
    window.patient_accession = make_field("Accession #", 130)

    row1.addWidget(QLabel("Name:"))
    row1.addWidget(window.patient_name)
    row1.addWidget(QLabel("ID:"))
    row1.addWidget(window.patient_id)
    row1.addWidget(QLabel("DOB:"))
    row1.addWidget(window.patient_dob)
    row1.addWidget(QLabel("Study:"))
    row1.addWidget(window.patient_study_date)
    row1.addStretch()

    row2.addWidget(QLabel("Referring:"))
    row2.addWidget(window.patient_referrer)
    row2.addWidget(QLabel("Acc #:"))
    row2.addWidget(window.patient_accession)
    row2.addStretch()

    layout.addLayout(row1)
    layout.addLayout(row2)
    return frame


def build_macros_panel(window: MainWindow) -> QFrame:
    """Build the quick phrases panel."""
    frame = QFrame()
    frame.setObjectName("macros_panel")
    frame.setFixedWidth(228)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(4)

    title = QLabel("Quick Phrases")
    title.setStyleSheet("font-weight: bold; font-size: 11px;")
    layout.addWidget(title)

    window.macro_region_combo = QComboBox()
    window.macro_region_combo.addItems(macros.REGION_ORDER)
    window.macro_region_combo.currentTextChanged.connect(window._rebuild_macro_buttons)
    layout.addWidget(window.macro_region_combo)

    # Scroll area for dynamic macro buttons
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    window._macro_container = QWidget()
    window._macro_layout = QVBoxLayout(window._macro_container)
    window._macro_layout.setSpacing(2)
    window._macro_layout.setContentsMargins(0, 0, 0, 0)
    window._macro_layout.addStretch()

    scroll.setWidget(window._macro_container)
    layout.addWidget(scroll, stretch=1)
    return frame


def build_editor_panel(window: MainWindow) -> QWidget:
    """Build the text editor and template panel."""
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    # Template bar
    template_bar = QHBoxLayout()
    template_bar.setSpacing(4)

    window.template_combo = QComboBox()
    window.template_combo.setMinimumWidth(220)
    btn_load = QPushButton("Load")
    btn_load.setToolTip("Insert template into editor (Ctrl+T)")
    btn_load.setFixedWidth(52)
    btn_load.clicked.connect(window.on_insert_template)

    btn_font_up = QPushButton("A+")
    btn_font_up.setFixedWidth(34)
    btn_font_up.setToolTip("Increase font size (Ctrl+])")
    btn_font_up.clicked.connect(window.on_font_increase)

    btn_font_down = QPushButton("A-")
    btn_font_down.setFixedWidth(34)
    btn_font_down.setToolTip("Decrease font size (Ctrl+[)")
    btn_font_down.clicked.connect(window.on_font_decrease)

    template_bar.addWidget(QLabel("Template:"))
    template_bar.addWidget(window.template_combo)
    template_bar.addWidget(btn_load)
    template_bar.addSpacing(8)
    template_bar.addWidget(btn_font_up)
    template_bar.addWidget(btn_font_down)
    template_bar.addStretch()

    layout.addLayout(template_bar)

    # Main editor
    window.editor = QTextEdit()
    window.editor.setPlaceholderText(
        "Dictated text appears here.\n\n"
        "F5 = Start recording   F6 = Stop\n"
        "Load a template, then dictate into each section.\n"
        "Use quick phrases on the left for common findings."
    )
    font = QFont("Consolas", window.settings.get("font_size", 13))
    window.editor.setFont(font)
    window.editor.textChanged.connect(window._on_text_changed)
    layout.addWidget(window.editor, stretch=1)

    # Info bar beneath editor
    info_bar = QHBoxLayout()
    window._info_words = QLabel("Words: 0  |  Lines: 0")
    window._info_words.setStyleSheet("font-size: 11px;")
    info_bar.addWidget(window._info_words)
    info_bar.addStretch()
    layout.addLayout(info_bar)

    return panel


def build_recording_bar(window: MainWindow) -> QFrame:
    """Build the recording controls and action bar."""
    bar = QFrame()
    bar.setFrameShape(QFrame.StyledPanel)
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(6, 4, 6, 4)
    layout.setSpacing(8)

    # Record / stop buttons
    window.btn_record = QPushButton("Record  F5")
    window.btn_record.setObjectName("btn_record")
    window.btn_record.setMinimumWidth(130)
    window.btn_record.clicked.connect(window.on_start_recording)

    window.btn_stop = QPushButton("Stop  F6")
    window.btn_stop.setObjectName("btn_stop")
    window.btn_stop.setMinimumWidth(110)
    window.btn_stop.setEnabled(False)
    window.btn_stop.clicked.connect(window.on_stop_recording)

    # Model / VAD controls
    window.model_combo = QComboBox()
    window.model_combo.addItems(SUPPORTED_MODELS)
    window.model_combo.setCurrentText(window.settings.get("model_size", "base"))
    window.model_combo.setToolTip(
        "tiny/base = fastest  |  small/medium = better accuracy  |  large-v2/v3 = best (needs GPU)"
    )
    window.model_combo.setFixedWidth(90)

    window.vad_checkbox = QCheckBox("VAD")
    window.vad_checkbox.setToolTip("Voice activity detection - filters silence (recommended)")
    window.vad_checkbox.setChecked(window.settings.get("vad_filter", True))

    window.language_input = QLineEdit(window.settings.get("language", "en"))
    window.language_input.setFixedWidth(36)
    window.language_input.setToolTip("ISO language code, e.g. 'en'")

    # Accent correction profile
    window.accent_combo = QComboBox()
    for key, label in ACCENT_LABELS.items():
        window.accent_combo.addItem(label, key)
    saved_accent = window.settings.get("accent", "neutral")
    idx = window.accent_combo.findData(saved_accent)
    if idx >= 0:
        window.accent_combo.setCurrentIndex(idx)
    window.accent_combo.setToolTip("Accent correction profile for Whisper error patterns")
    window.accent_combo.setFixedWidth(110)

    # Separator
    separator = QFrame()
    separator.setFrameShape(QFrame.VLine)

    # Action buttons
    btn_copy = QPushButton("Copy")
    btn_copy.setToolTip("Copy report to clipboard (Ctrl+C)")
    btn_copy.clicked.connect(window.on_copy)

    btn_save = QPushButton("Save TXT")
    btn_save.setToolTip("Save report as plain text (Ctrl+S)")
    btn_save.clicked.connect(window.on_save_txt)

    from src.features.report_manager import DOCX_AVAILABLE
    window.btn_export_word = QPushButton("Export Word")
    window.btn_export_word.setObjectName("btn_word")
    window.btn_export_word.setToolTip("Export report to Word document (Ctrl+Shift+W)")
    window.btn_export_word.setEnabled(DOCX_AVAILABLE)
    if not DOCX_AVAILABLE:
        window.btn_export_word.setToolTip("Install python-docx to enable Word export: pip install python-docx")
    window.btn_export_word.clicked.connect(window.on_export_word)

    btn_clear = QPushButton("Clear")
    btn_clear.setToolTip("Clear editor (Ctrl+L)")
    btn_clear.clicked.connect(window.on_clear)

    # Microphone level meter
    window._level_bar = QProgressBar()
    window._level_bar.setRange(0, 100)
    window._level_bar.setValue(0)
    window._level_bar.setFixedWidth(80)
    window._level_bar.setFixedHeight(14)
    window._level_bar.setTextVisible(False)
    window._level_bar.setToolTip("Microphone input level")
    window._level_bar.setStyleSheet(_LEVEL_BAR_STYLESHEET.format(color=_COLOR_HEALTHY))

    layout.addWidget(window.btn_record)
    layout.addWidget(window.btn_stop)
    layout.addWidget(QLabel("Mic:"))
    layout.addWidget(window._level_bar)
    layout.addWidget(separator)
    layout.addWidget(QLabel("Model:"))
    layout.addWidget(window.model_combo)
    layout.addWidget(QLabel("Lang:"))
    layout.addWidget(window.language_input)
    layout.addWidget(QLabel("Accent:"))
    layout.addWidget(window.accent_combo)
    layout.addWidget(window.vad_checkbox)
    layout.addStretch()
    layout.addWidget(btn_copy)
    layout.addWidget(btn_save)
    layout.addWidget(window.btn_export_word)
    layout.addWidget(btn_clear)
    return bar


def build_menu(window: MainWindow) -> None:
    """Build the menu bar."""
    menubar = window.menuBar()

    # File
    file_menu = menubar.addMenu("&File")
    _add_action(file_menu, "New Report", window.on_new_report, "Ctrl+N")
    _add_action(file_menu, "Open Report...", window.on_open_report, "Ctrl+O")
    file_menu.addSeparator()
    _add_action(file_menu, "Save as Text...", window.on_save_txt, "Ctrl+S")
    _add_action(file_menu, "Export to Word...", window.on_export_word, "Ctrl+Shift+W")
    file_menu.addSeparator()

    window.recent_menu = QMenu("Recent Reports", window)
    file_menu.addMenu(window.recent_menu)
    window._rebuild_recent_menu()

    file_menu.addSeparator()
    _add_action(file_menu, "Exit", window.close, "Alt+F4")

    # Report
    report_menu = menubar.addMenu("&Report")
    _add_action(report_menu, "Load Template", window.on_insert_template, "Ctrl+T")
    _add_action(report_menu, "Copy to Clipboard", window.on_copy, "Ctrl+C")
    report_menu.addSeparator()
    _add_action(report_menu, "Clear Editor", window.on_clear, "Ctrl+L")

    # View
    view_menu = menubar.addMenu("&View")
    _add_action(view_menu, "Toggle Dark / Light Theme", window.on_toggle_theme, "Ctrl+D")
    view_menu.addSeparator()
    _add_action(view_menu, "Toggle Patient Panel", window.on_toggle_patient_panel, "Ctrl+P")
    _add_action(view_menu, "Toggle Quick Phrases Panel", window.on_toggle_macros_panel, "Ctrl+M")
    view_menu.addSeparator()
    _add_action(view_menu, "Increase Font Size", window.on_font_increase, "Ctrl+]")
    _add_action(view_menu, "Decrease Font Size", window.on_font_decrease, "Ctrl+[")

    # Settings
    settings_menu = menubar.addMenu("&Settings")
    _add_action(settings_menu, "Open Auto-save Folder", window.on_open_autosave_folder)
    settings_menu.addSeparator()
    _add_action(settings_menu, "Edit Macros (JSON)", window.on_edit_macros)
    _add_action(settings_menu, "Reload Macros", window.on_reload_macros, "Ctrl+R")
    settings_menu.addSeparator()
    _add_action(settings_menu, "Learning Statistics...", window.on_show_learning_stats)
    _add_action(settings_menu, "Reset Learning Data...", window.on_reset_learning)


@staticmethod
def _add_action(menu: QMenu, label: str, slot, shortcut: str = "") -> QAction:
    """Add an action to a menu."""
    action = QAction(label)
    if shortcut:
        action.setShortcut(QKeySequence(shortcut))
    action.triggered.connect(slot)
    menu.addAction(action)
    return action


def load_templates(window: MainWindow) -> None:
    """Populate the template combo box from disk."""
    window.template_combo.clear()
    tpl_path = templates_dir()
    if tpl_path.is_dir():
        names = sorted(
            f for f in os.listdir(tpl_path) if f.lower().endswith(".txt")
        )
        window.template_combo.addItems(names)


def rebuild_macro_buttons(window: MainWindow, region: str = "") -> None:
    """Rebuild the macro buttons for the selected region."""
    region = region or window.macro_region_combo.currentText()
    phrases = macros.MACROS.get(region, [])

    # Clear existing dynamic buttons (everything except the stretch)
    while window._macro_layout.count() > 1:
        item = window._macro_layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()

    for label, text in phrases:
        btn = QPushButton(label)
        btn.setToolTip(text)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        btn.setStyleSheet("text-align: left; padding: 4px 6px; font-size: 11px; white-space: pre-wrap;")
        btn.clicked.connect(lambda checked=False, t=text: window._insert_macro(t))
        window._macro_layout.insertWidget(window._macro_layout.count() - 1, btn)

    window.settings.set("last_macro_region", region)


def rebuild_recent_menu(window: MainWindow) -> None:
    """Rebuild the Recent Reports menu."""
    window.recent_menu.clear()
    recent = window.settings.get_recent_reports()
    if not recent:
        window.recent_menu.addAction("(none)").setEnabled(False)
        return
    for path in recent:
        action = QAction(os.path.basename(path), window)
        action.setToolTip(path)
        action.triggered.connect(lambda checked=False, fp=path: window._open_recent(fp))
        window.recent_menu.addAction(action)
