"""Shared runtime stubs for tests that import optional app dependencies."""

from __future__ import annotations

import sys
import types


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

    qtcore_stub.QObject = _FakeQObject
    qtcore_stub.Signal = _FakeSignal

    sys.modules.setdefault("PySide6", pyside6_stub)
    sys.modules.setdefault("PySide6.QtCore", qtcore_stub)

    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = object
    sys.modules.setdefault("faster_whisper", faster_whisper_stub)
    sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

    import src.features.file_manager as file_manager
    import src.features.adaptive_learning as adaptive_learning
    import src.medical.medical_dict as medical_dict

    sys.modules.setdefault("file_manager", file_manager)
    return adaptive_learning, medical_dict
