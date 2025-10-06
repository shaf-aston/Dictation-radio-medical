import os
import sys
import tempfile
import traceback
from typing import Dict

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTextEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QCheckBox,
)
from PySide6.QtCore import Qt, QObject, QThread, Signal

from audio import Recorder
from transcriber import Transcriber
from postprocess import postprocess_transcript


class TranscribeWorker(QObject):
    finished = Signal(str, str)  # text, error

    def __init__(self, audio_path: str, model_size: str, language: str, vad: bool):
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.language = language
        self.vad = vad

    def run(self):
        try:
            # Force a safe compute_type to avoid hardware/cache issues
            transcriber = Transcriber(model_size=self.model_size, device="auto", compute_type="int8")
            text, _segments = transcriber.transcribe(self.audio_path, language=self.language, vad_filter=self.vad)
            text = postprocess_transcript(text)
            self.finished.emit(text, "")
        except Exception as e:
            err = f"Transcription failed: {e}\n\n{traceback.format_exc()}"
            self.finished.emit("", err)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiology Dictation")
        self.resize(900, 600)

        self.recorder = Recorder()
        self.current_wav_path = None
        self.thread: QThread | None = None
        self.worker: TranscribeWorker | None = None

        # UI
        self.text = QTextEdit()
        self.text.setPlaceholderText("Dictated text will appear here...")

        self.btn_start = QPushButton("Start Recording")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_copy = QPushButton("Copy")
        self.btn_save = QPushButton("Save As…")
        self.btn_clear = QPushButton("Clear")

        self.templates = QComboBox()
        self.btn_insert_template = QPushButton("Insert Template")

        self.model_label = QLabel("Model:")
        self.model_select = QComboBox()
        self.model_select.addItems(["tiny", "base", "small", "medium"])  # choose based on speed/accuracy
        self.model_select.setCurrentText("base")

        self.lang_label = QLabel("Language:")
        self.lang_edit = QLineEdit("en")
        self.vad_check = QCheckBox("VAD filter")
        self.vad_check.setChecked(True)

        # Layout
        top_controls = QHBoxLayout()
        top_controls.addWidget(self.btn_start)
        top_controls.addWidget(self.btn_stop)
        top_controls.addStretch(1)
        top_controls.addWidget(self.model_label)
        top_controls.addWidget(self.model_select)
        top_controls.addWidget(self.lang_label)
        top_controls.addWidget(self.lang_edit)
        top_controls.addWidget(self.vad_check)

        template_bar = QHBoxLayout()
        template_bar.addWidget(self.templates)
        template_bar.addWidget(self.btn_insert_template)
        template_bar.addStretch(1)

        bottom_controls = QHBoxLayout()
        bottom_controls.addWidget(self.btn_copy)
        bottom_controls.addWidget(self.btn_save)
        bottom_controls.addWidget(self.btn_clear)
        bottom_controls.addStretch(1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_controls)
        main_layout.addLayout(template_bar)
        main_layout.addWidget(self.text)
        main_layout.addLayout(bottom_controls)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Events
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_copy.clicked.connect(self.on_copy)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_clear.clicked.connect(self.text.clear)
        self.btn_insert_template.clicked.connect(self.on_insert_template)

        self.load_templates()

    # Template support
    def templates_dir(self) -> str:
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "templates")

    def load_templates(self):
        self.templates.clear()
        tdir = self.templates_dir()
        items = []
        if os.path.isdir(tdir):
            for name in sorted(os.listdir(tdir)):
                if name.lower().endswith(".txt"):
                    items.append(name)
        if not items:
            items = []
        self.templates.addItems(items)

    def on_insert_template(self):
        name = self.templates.currentText()
        if not name:
            return
        path = os.path.join(self.templates_dir(), name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.text.insertPlainText(f.read() + "\n")
        except Exception as e:
            QMessageBox.warning(self, "Template Error", str(e))

    # Recording
    def on_start(self):
        if self.recorder.is_recording:
            return
        fd, path = tempfile.mkstemp(prefix="dictation_", suffix=".wav")
        os.close(fd)
        self.current_wav_path = path
        try:
            self.recorder.start(path)
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Could not start recording: {e}")
            self.current_wav_path = None
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Recording…")

    def on_stop(self):
        if not self.recorder.is_recording:
            return
        try:
            self.recorder.stop()
        finally:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.statusBar().showMessage("Transcribing…")

        # Start transcription in background
        model_size = self.model_select.currentText()
        language = self.lang_edit.text().strip() or "en"
        vad = self.vad_check.isChecked()

        self.thread = QThread()
        self.worker = TranscribeWorker(self.current_wav_path, model_size, language, vad)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_transcription_done)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_transcription_done(self, text: str, error: str):
        if error:
            QMessageBox.critical(self, "Transcription Error", error)
            self.statusBar().clearMessage()
            return
        if text:
            if self.text.toPlainText():
                self.text.append("\n")
            self.text.insertPlainText(text)
        self.statusBar().showMessage("Ready", 2000)
        # Cleanup temp file
        try:
            if self.current_wav_path and os.path.exists(self.current_wav_path):
                os.remove(self.current_wav_path)
        except Exception:
            pass
        self.current_wav_path = None

    # Utilities
    def on_copy(self):
        QApplication.clipboard().setText(self.text.toPlainText())
        self.statusBar().showMessage("Copied", 1500)

    def on_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Report", "report.txt", "Text Files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text.toPlainText())
            self.statusBar().showMessage(f"Saved to {os.path.basename(path)}", 2000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
