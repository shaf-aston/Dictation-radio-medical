"""MSK Radiology Dictation application entry point.

This module is a compatibility shim. The main application has been refactored
into focused modules: main_window.py, views.py, recording_session.py, dialogs.py.

Import from src.ui.main_window for the actual implementation.
"""

from src.ui.main_window import MainWindow, main

__all__ = ["MainWindow", "main"]
