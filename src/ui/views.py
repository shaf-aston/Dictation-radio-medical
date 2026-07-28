"""UI view construction — panels, buttons, layouts, and menus."""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QLineEdit, QCheckBox, QFrame, QScrollArea, QSplitter,
    QSizePolicy, QMenu, QProgressBar,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QAction, QKeySequence

from src.dictation.transcriber import SUPPORTED_MODELS, resolve_model
from src.dictation.postprocess import CLEANUP_LEVEL_LABELS
from src.features.accent_corrections import ACCENT_LABELS
from src.medical import macros
from src.features.file_manager import templates_dir
from src.ui.collapsible import Section
from src.ui.styles import set_level_state, set_status_state
from src.features.report_manager import DOCX_AVAILABLE
from src.ui.recording_session import on_start_recording, on_stop_recording
from src.ui.dialogs import (
    on_show_learning_stats, on_reset_learning, show_cloud_training_dialog,
    show_ai_cleanup_settings_dialog, on_run_ai_cleanup, show_scan_assistant_dialog,
    show_correction_rules_dialog,
)

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow


def build_ui(window: MainWindow) -> None:
    """Construct the main window layout.

    One surface, the same shape as the web app (docs/ui-decisions.md): a single
    thin bar of the controls used while dictating, the editor taking the rest of
    the window, and everything else folded into three sections underneath.
    """
    central = QWidget()
    window.setCentralWidget(central)
    root_layout = QVBoxLayout(central)
    root_layout.setContentsMargins(6, 6, 6, 6)
    root_layout.setSpacing(4)

    # 1. The only permanently visible controls.
    root_layout.addWidget(build_top_bar(window))

    # 2. The editor, dominant, with the quick phrases beside it.
    window.splitter = QSplitter(Qt.Orientation.Horizontal)
    window.macros_panel = build_macros_panel(window)
    window.splitter.addWidget(window.macros_panel)
    window.splitter.addWidget(build_editor_panel(window))
    window.splitter.setStretchFactor(0, 0)
    window.splitter.setStretchFactor(1, 1)
    root_layout.addWidget(window.splitter, stretch=1)

    # 3. Template / Patient / Settings, folded away until wanted.
    root_layout.addWidget(build_panels(window))

    # Status bar
    window._status_label = QLabel("Ready")
    window._wordcount_label = QLabel("Words: 0")
    window._autosave_label = QLabel("Auto-save: -")
    status_bar = window.statusBar()
    status_bar.addWidget(window._status_label, 1)
    status_bar.addPermanentWidget(window._wordcount_label)
    status_bar.addPermanentWidget(window._autosave_label)


def build_panels(window: MainWindow) -> QWidget:
    """Stack the three secondary panels, each in the same folding section.

    One mechanism for all three (``ui/collapsible.Section``), so they fold and
    look identically and a fourth panel costs three lines. Template is open by
    default because it is the one reached for at the start of every report; the
    patient section's state is restored from the saved setting by MainWindow,
    and the View menu's Ctrl+P still drives the same state.
    """
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)

    window.template_section = Section("Template and text size", build_template_panel(window), True)
    window.patient_panel = build_patient_panel(window)
    window.patient_section = Section("Patient details", window.patient_panel)
    window.patient_section.header.setToolTip("Show or hide the patient fields (Ctrl+P)")
    window.settings_section = Section("Dictation settings", build_settings_panel(window))

    for section in (window.template_section, window.patient_section, window.settings_section):
        layout.addWidget(section)
    return holder


def build_patient_panel(window: MainWindow) -> QFrame:
    """Build the patient information entry panel."""
    frame = QFrame()
    frame.setObjectName("patient_panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(4)

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
    window.macro_region_combo.currentTextChanged.connect(partial(rebuild_macro_buttons, window))
    layout.addWidget(window.macro_region_combo)

    # Scroll area for dynamic macro buttons
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    window._macro_container = QWidget()
    window._macro_layout = QVBoxLayout(window._macro_container)
    window._macro_layout.setSpacing(2)
    window._macro_layout.setContentsMargins(0, 0, 0, 0)
    window._macro_layout.addStretch()

    scroll.setWidget(window._macro_container)
    layout.addWidget(scroll, stretch=1)
    return frame


def build_template_panel(window: MainWindow) -> QFrame:
    """Build the template picker and the editor's text size controls."""
    frame = QFrame()
    frame.setObjectName("panel_body")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(6)

    window.template_combo = QComboBox()
    window.template_combo.setMinimumWidth(220)

    btn_load = QPushButton("Load")
    btn_load.setToolTip("Insert template into editor (Ctrl+T)")
    btn_load.setFixedWidth(60)
    btn_load.clicked.connect(window.on_insert_template)

    btn_font_up = QPushButton("A+")
    btn_font_up.setFixedWidth(38)
    btn_font_up.setToolTip("Increase font size (Ctrl+])")
    btn_font_up.clicked.connect(window.on_font_increase)

    btn_font_down = QPushButton("A-")
    btn_font_down.setFixedWidth(38)
    btn_font_down.setToolTip("Decrease font size (Ctrl+[)")
    btn_font_down.clicked.connect(window.on_font_decrease)

    layout.addWidget(QLabel("Template:"))
    layout.addWidget(window.template_combo)
    layout.addWidget(btn_load)
    layout.addSpacing(12)
    layout.addWidget(QLabel("Text size:"))
    layout.addWidget(btn_font_up)
    layout.addWidget(btn_font_down)
    layout.addStretch()
    return frame


def build_settings_panel(window: MainWindow) -> QFrame:
    """Build the dictation settings: model, language, accent, cleanup, VAD."""
    frame = QFrame()
    frame.setObjectName("panel_body")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(6)

    window.model_combo = QComboBox()
    window.model_combo.addItems(SUPPORTED_MODELS)
    # resolve_model, not the raw setting: setCurrentText is a no-op on a
    # non-editable combo when no item matches, so an unknown name would leave the
    # picker on its first entry and silently dictate with the wrong model.
    window.model_combo.setCurrentText(resolve_model(window.settings.get("model_size")))
    window.model_combo.setToolTip(
        "'.en' models are English-only — faster and more accurate for English "
        "dictation than the same size multilingual model.\n"
        "tiny/base = fastest  |  small/medium = better accuracy  |  large-v2/v3 = best (needs GPU)"
    )
    window.model_combo.setFixedWidth(104)

    window.language_input = QLineEdit(window.settings.get("language", "en"))
    window.language_input.setFixedWidth(44)
    window.language_input.setToolTip("ISO language code, e.g. 'en'")

    window.accent_combo = QComboBox()
    for key, label in ACCENT_LABELS.items():
        window.accent_combo.addItem(label, key)
    saved_accent = window.settings.get("accent", "neutral")
    idx = window.accent_combo.findData(saved_accent)
    if idx >= 0:
        window.accent_combo.setCurrentIndex(idx)
    window.accent_combo.setToolTip("Accent correction profile for Whisper error patterns")
    window.accent_combo.setFixedWidth(110)

    window.cleanup_combo = QComboBox()
    for key, label in CLEANUP_LEVEL_LABELS.items():
        window.cleanup_combo.addItem(label, key)
    saved_cleanup = window.settings.get("cleanup_level", "medium")
    idx = window.cleanup_combo.findData(saved_cleanup)
    if idx >= 0:
        window.cleanup_combo.setCurrentIndex(idx)
    window.cleanup_combo.setToolTip(
        "Soft = minimal rewriting (your words, almost verbatim)\n"
        "Medium = standard correction pipeline (default)\n"
        "Hard = standard pipeline + AI polish (if enabled)"
    )
    window.cleanup_combo.setFixedWidth(170)

    window.vad_checkbox = QCheckBox("Filter silence (VAD)")
    window.vad_checkbox.setToolTip("Voice activity detection - filters silence (recommended)")
    window.vad_checkbox.setChecked(window.settings.get("vad_filter", True))

    layout.addWidget(QLabel("Model:"))
    layout.addWidget(window.model_combo)
    layout.addWidget(QLabel("Language:"))
    layout.addWidget(window.language_input)
    layout.addWidget(QLabel("Accent:"))
    layout.addWidget(window.accent_combo)
    layout.addWidget(QLabel("Cleanup:"))
    layout.addWidget(window.cleanup_combo)
    layout.addWidget(window.vad_checkbox)
    layout.addStretch()
    return frame


def build_editor_panel(window: MainWindow) -> QWidget:
    """Build the text editor and its word count."""
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

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


def build_top_bar(window: MainWindow) -> QFrame:
    """Build the one always-visible bar: record, what the app is doing, exports.

    Everything here is either used while dictating or is how the report leaves.
    Anything set once and forgotten lives in a folded panel instead
    (:func:`build_panels`).
    """
    bar = QFrame()
    bar.setObjectName("top_bar")
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(8)

    # Record / stop buttons
    window.btn_record = QPushButton("Record  F5")
    window.btn_record.setObjectName("btn_record")
    window.btn_record.setMinimumWidth(130)
    window.btn_record.clicked.connect(partial(on_start_recording, window))

    window.btn_stop = QPushButton("Stop  F6")
    window.btn_stop.setObjectName("btn_stop")
    window.btn_stop.setMinimumWidth(110)
    window.btn_stop.setEnabled(False)
    window.btn_stop.clicked.connect(partial(on_stop_recording, window))

    # What the app is doing, where the eye already is. The text is the same
    # message the status bar carries; the colour is the state.
    window._state_pill = QLabel("Ready")
    window._state_pill.setObjectName("state_pill")
    window._state_pill.setMinimumWidth(150)
    window._state_pill.setMaximumWidth(280)
    window._state_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
    set_status_state(window._state_pill, "idle")

    # How long this dictation has been running (blank when not recording).
    window._elapsed_label = QLabel("")
    window._elapsed_label.setObjectName("elapsed")
    window._elapsed_label.setFixedWidth(44)
    window._elapsed_label.setToolTip("Length of the current recording")

    # Separator
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.VLine)

    # Action buttons
    btn_copy = QPushButton("Copy")
    btn_copy.setToolTip("Copy report to clipboard (Ctrl+C)")
    btn_copy.clicked.connect(window.on_copy)

    btn_save = QPushButton("Save TXT")
    btn_save.setToolTip("Save report as plain text (Ctrl+S)")
    btn_save.clicked.connect(window.on_save_txt)

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
    window._level_bar.setObjectName("level_bar")
    window._level_bar.setRange(0, 100)
    window._level_bar.setValue(0)
    window._level_bar.setFixedWidth(80)
    window._level_bar.setFixedHeight(14)
    window._level_bar.setTextVisible(False)
    window._level_bar.setToolTip("Microphone input level")
    set_level_state(window._level_bar, "healthy")

    layout.addWidget(window.btn_record)
    layout.addWidget(window.btn_stop)
    layout.addWidget(window._state_pill)
    layout.addWidget(QLabel("Mic:"))
    layout.addWidget(window._level_bar)
    layout.addWidget(window._elapsed_label)
    layout.addWidget(separator)
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
    rebuild_recent_menu(window)

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

    # Tools
    tools_menu = menubar.addMenu("&Tools")
    _add_action(tools_menu, "Scan Assistant...", partial(show_scan_assistant_dialog, window))
    _add_action(tools_menu, "AI Cleanup (Groq)", partial(on_run_ai_cleanup, window))

    # Settings
    settings_menu = menubar.addMenu("&Settings")
    _add_action(settings_menu, "Open Auto-save Folder", window.on_open_autosave_folder)
    settings_menu.addSeparator()
    _add_action(settings_menu, "Edit Macros (JSON)", window.on_edit_macros)
    _add_action(settings_menu, "Reload Macros", window.on_reload_macros, "Ctrl+R")
    settings_menu.addSeparator()
    _add_action(settings_menu, "Correction Rules...", partial(show_correction_rules_dialog, window))
    _add_action(settings_menu, "Learning Statistics...", partial(on_show_learning_stats, window))
    _add_action(settings_menu, "Reset Learning Data...", partial(on_reset_learning, window))
    settings_menu.addSeparator()
    _add_action(settings_menu, "Cloud Voice Training...", partial(show_cloud_training_dialog, window))
    _add_action(settings_menu, "AI Cleanup (Groq)...", partial(show_ai_cleanup_settings_dialog, window))

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
        widget = item.widget() if item else None
        if widget:
            widget.deleteLater()

    for label, text in phrases:
        btn = QPushButton(label)
        btn.setToolTip(text)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        btn.setStyleSheet("text-align: left; padding: 4px 6px; font-size: 11px; white-space: pre-wrap;")
        btn.clicked.connect(lambda checked=False, t=text: window._insert_macro(t))
        window._macro_layout.insertWidget(window._macro_layout.count() - 1, btn)

    if window.settings.get("last_macro_region") != region:
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
