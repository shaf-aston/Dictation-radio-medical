"""Shared fixtures."""

import os
import time

import pytest


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    """A real desktop MainWindow, offscreen, writing only under *tmp_path*.

    No model loads (warm-up off), no first-launch dialogs, and the recorder and
    live transcription worker are stand-ins the test drives: ``desktop.worker()``
    is the latest worker, whose ``partial`` / ``finished`` signals the test
    emits as the real one would.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt = pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import QObject, Signal

    from src.core import paths

    data = tmp_path / "data"
    data.mkdir()
    paths._data_dir.cache_clear()
    monkeypatch.setattr(paths, "_data_dir", lambda: data)
    monkeypatch.setattr(paths, "settings_file", lambda: tmp_path / "dictation_settings.json")
    from src.core import settings as core_settings
    monkeypatch.setattr(core_settings, "settings_file", lambda: tmp_path / "dictation_settings.json",
                        raising=False)
    app = qt.QApplication.instance() or qt.QApplication([])

    from src.dictation import warmup
    from src.ui import main_window, recording_session

    monkeypatch.setattr(warmup, "warm_up_async", lambda *_a, **_k: None)
    monkeypatch.setattr(main_window, "show_learning_consent_if_needed", lambda *_: None)
    monkeypatch.setattr(main_window, "show_disclaimer_if_needed", lambda *_: None)

    workers = []

    class Worker(QObject):
        partial = Signal(str, int)
        finished = Signal()
        progress = Signal(str)
        segments = Signal(list)

        def __init__(self, *_a, **_k):
            super().__init__()
            self.cancelled = self.finalized = False
            self.committed_chunks = 0
            workers.append(self)

        def run(self):
            pass

        def finalize(self):
            self.finalized = True

        def cancel(self):
            self.cancelled = True

        def stop(self):
            pass

    class Recorder:
        is_recording = False
        current_level = 0.2
        is_clipping = False

        def start(self, path):
            open(path, "wb").close()
            self.is_recording = True

        def stop(self):
            self.is_recording = False

    monkeypatch.setattr(recording_session, "LiveTranscribeWorker", Worker)
    window = main_window.MainWindow()
    window.recorder = Recorder()
    window.show()

    def pump(seconds=0.2):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.005)

    def wait_for(predicate, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            app.processEvents()
            if predicate():
                return True
            time.sleep(0.005)
        return False

    window.worker = lambda: workers[-1]
    window.pump = pump
    window.wait_for = wait_for
    pump()
    yield window
    window.close()
    pump()
