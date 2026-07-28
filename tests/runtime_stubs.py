"""Shared runtime stubs for tests that import optional app dependencies."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager


@contextmanager
def qt_widgets_stub():
    """Shadow ``PySide6.QtWidgets`` for the duration of an import, then undo it.

    A UI module needs ``QMessageBox`` to exist at import time, but leaving a
    stand-in registered would tell every later test that a real Qt is
    installed — ``pytest.importorskip("PySide6.QtWidgets")`` would stop
    skipping and then fail on the first real widget it asked for.
    """
    class _FakeEnum:
        """Any attribute access returns the attribute's own name."""

        def __getattr__(self, name: str) -> str:
            return name

    class _FakeMessageBox:
        StandardButton = _FakeEnum()
        Icon = _FakeEnum()
        ButtonRole = _FakeEnum()

    stub = types.ModuleType("PySide6.QtWidgets")
    stub.QMessageBox = _FakeMessageBox  # type: ignore
    had_real = "PySide6.QtWidgets" in sys.modules
    if not had_real:
        sys.modules["PySide6.QtWidgets"] = stub
    try:
        yield
    finally:
        if not had_real:
            sys.modules.pop("PySide6.QtWidgets", None)


def install_test_runtime_stubs():
    """Install lightweight stand-ins and return patched project modules."""
    pyside6_stub = types.ModuleType("PySide6")
    qtcore_stub = types.ModuleType("PySide6.QtCore")

    class _FakeSignal:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def emit(self, *args) -> None:
            pass

    class _FakeQObject:
        def __init__(self) -> None:
            pass

    def _fake_slot(*args, **kwargs):
        """Stand-in for @Slot(...) — returns the method untouched."""
        def decorate(func):
            return func
        return decorate

    qtcore_stub.QObject = _FakeQObject  # type: ignore
    qtcore_stub.Signal = _FakeSignal  # type: ignore
    qtcore_stub.Slot = _fake_slot  # type: ignore
    qtcore_stub.QThread = _FakeQObject  # type: ignore
    qtcore_stub.QTimer = _FakeQObject  # type: ignore

    sys.modules.setdefault("PySide6", pyside6_stub)
    sys.modules.setdefault("PySide6.QtCore", qtcore_stub)

    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = object  # type: ignore
    sys.modules.setdefault("faster_whisper", faster_whisper_stub)
    sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

    import src.features.file_manager as file_manager
    import src.features.adaptive_learning as adaptive_learning
    import src.medical.medical_dict as medical_dict

    sys.modules.setdefault("file_manager", file_manager)
    return adaptive_learning, medical_dict
