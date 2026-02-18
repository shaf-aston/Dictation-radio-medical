"""
Qt stylesheets for dark and light themes.

Catppuccin Mocha (dark) and iOS-inspired (light) colour palettes.
"""

DARK = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QTextEdit {
    background-color: #11111b;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 8px;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 28px;
}
QPushButton:hover  { background-color: #45475a; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { background-color: #1e1e2e; color: #6c7086; border-color: #313244; }
QPushButton#btn_record {
    background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;
}
QPushButton#btn_record:hover { background-color: #94e2d5; }
QPushButton#btn_record:disabled { background-color: #2e4a2e; color: #6c7086; }
QPushButton#btn_stop {
    background-color: #f38ba8; color: #1e1e2e; font-weight: bold;
}
QPushButton#btn_stop:hover { background-color: #eba0ac; }
QPushButton#btn_word {
    background-color: #89b4fa; color: #1e1e2e; font-weight: bold;
}
QPushButton#btn_word:hover { background-color: #74c7ec; }
QComboBox {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px;
    padding: 4px 8px; min-height: 26px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #1e1e2e; color: #cdd6f4;
    border: 1px solid #45475a;
    selection-background-color: #313244;
}
QLineEdit {
    background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px;
    padding: 4px 8px; min-height: 26px;
}
QLabel { color: #bac2de; }
QStatusBar {
    background-color: #11111b; color: #a6adc8;
    border-top: 1px solid #45475a;
}
QStatusBar::item { border: none; }
QMenuBar {
    background-color: #11111b; color: #cdd6f4;
    border-bottom: 1px solid #45475a;
}
QMenuBar::item:selected { background-color: #313244; }
QMenu {
    background-color: #1e1e2e; color: #cdd6f4;
    border: 1px solid #45475a;
}
QMenu::item:selected { background-color: #313244; }
QMenu::separator { background-color: #45475a; height: 1px; }
QFrame#patient_panel {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 4px;
}
QFrame#macros_panel {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 4px;
}
QScrollArea { background-color: transparent; border: none; }
QScrollBar:vertical {
    background-color: #1e1e2e; width: 8px;
}
QScrollBar::handle:vertical { background-color: #45475a; border-radius: 4px; }
QSplitter::handle { background-color: #45475a; width: 2px; }
QCheckBox { color: #cdd6f4; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #45475a; border-radius: 3px;
    background-color: #313244;
}
QCheckBox::indicator:checked { background-color: #89b4fa; border-color: #89b4fa; }
QToolBar {
    background-color: #11111b; border-bottom: 1px solid #45475a; spacing: 4px;
}
"""

LIGHT = """
QMainWindow, QWidget {
    background-color: #f4f4f6;
    color: #1c1c1e;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QTextEdit {
    background-color: #ffffff; color: #1c1c1e;
    border: 1px solid #c7c7cc; border-radius: 4px;
    padding: 8px;
    selection-background-color: #007aff;
    selection-color: #ffffff;
}
QPushButton {
    background-color: #e5e5ea; color: #1c1c1e;
    border: 1px solid #c7c7cc; border-radius: 4px;
    padding: 5px 12px; min-height: 28px;
}
QPushButton:hover  { background-color: #d1d1d6; }
QPushButton:pressed { background-color: #c7c7cc; }
QPushButton:disabled { background-color: #f4f4f6; color: #8e8e93; }
QPushButton#btn_record {
    background-color: #34c759; color: #ffffff; font-weight: bold;
}
QPushButton#btn_record:hover { background-color: #2db34a; }
QPushButton#btn_record:disabled { background-color: #c7e6cf; color: #8e8e93; }
QPushButton#btn_stop {
    background-color: #ff3b30; color: #ffffff; font-weight: bold;
}
QPushButton#btn_stop:hover { background-color: #e03328; }
QPushButton#btn_word {
    background-color: #007aff; color: #ffffff; font-weight: bold;
}
QPushButton#btn_word:hover { background-color: #0062cc; }
QComboBox {
    background-color: #ffffff; color: #1c1c1e;
    border: 1px solid #c7c7cc; border-radius: 4px;
    padding: 4px 8px; min-height: 26px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff; color: #1c1c1e;
    border: 1px solid #c7c7cc;
    selection-background-color: #e5e5ea;
}
QLineEdit {
    background-color: #ffffff; color: #1c1c1e;
    border: 1px solid #c7c7cc; border-radius: 4px;
    padding: 4px 8px; min-height: 26px;
}
QLabel { color: #3c3c43; }
QStatusBar { background-color: #e5e5ea; color: #3c3c43; border-top: 1px solid #c7c7cc; }
QStatusBar::item { border: none; }
QMenuBar { background-color: #e5e5ea; color: #1c1c1e; border-bottom: 1px solid #c7c7cc; }
QMenuBar::item:selected { background-color: #d1d1d6; }
QMenu {
    background-color: #ffffff; color: #1c1c1e; border: 1px solid #c7c7cc;
}
QMenu::item:selected { background-color: #e5e5ea; }
QMenu::separator { background-color: #c7c7cc; height: 1px; }
QFrame#patient_panel {
    background-color: #eff6ff; border: 1px solid #93c5fd; border-radius: 4px;
}
QFrame#macros_panel {
    background-color: #f9f9fb; border: 1px solid #c7c7cc; border-radius: 4px;
}
QScrollArea { background-color: transparent; border: none; }
QScrollBar:vertical { background-color: #e5e5ea; width: 8px; }
QScrollBar::handle:vertical { background-color: #c7c7cc; border-radius: 4px; }
QSplitter::handle { background-color: #c7c7cc; width: 2px; }
QCheckBox { color: #1c1c1e; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #c7c7cc; border-radius: 3px;
    background-color: #ffffff;
}
QCheckBox::indicator:checked { background-color: #007aff; border-color: #007aff; }
QToolBar { background-color: #e5e5ea; border-bottom: 1px solid #c7c7cc; spacing: 4px; }
"""
