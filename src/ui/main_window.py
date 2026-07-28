"""Main application window — layout assembly and event dispatch."""

from __future__ import annotations

import contextlib
import os
import sys
import subprocess
import logging
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from functools import partial

if TYPE_CHECKING:
    from PySide6.QtCore import QThread
    from src.dictation.worker import LiveTranscribeWorker
    from src.ui.postprocess_worker import PostprocessWorker

from PySide6.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QMessageBox,
    QTextEdit, QPushButton, QComboBox, QLabel, QLineEdit,
    QCheckBox, QFrame, QSplitter, QProgressBar, QMenu, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor

from src.dictation.audio import Recorder
from src.core.settings import Settings
from src.core.patient_schema import normalize_patient_info
from src.features.file_manager import report_filename
from src.ui.collapsible import Section
from src.ui.status import StatusTrack
from src.ui.styles import DARK, LIGHT, set_status_state
from src.dictation.worker import STATE_CATCHING_UP, STATE_LIVE, STATE_LOADING
from src.features.report_manager import (
    autosave_report, save_report_txt, export_to_word, DOCX_AVAILABLE
)
from src.medical import macros
from src.medical.macros import reload_macros
from src.features.file_manager import startup_cleanup, autosave_dir, macros_file, templates_dir
from src.features.adaptive_learning import get_adaptive_learning, learn_from_edit
from src.features import audit_log

# View construction
from src.ui.views import (
    build_ui, build_menu, load_templates, rebuild_macro_buttons, rebuild_recent_menu
)

# Recording control
from src.ui.recording_session import (
    confirm_release, on_partial_text, on_processed_text, on_start_recording,
    on_stop_recording, on_transcription_finished, setup_level_timer,
)

# Dialog handling
from src.ui.dialogs import show_learning_consent_if_needed, show_disclaimer_if_needed

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# The pill colour for each state the transcription worker reports. Anything
# else it emits (the per-chunk polishing counter) is simply "busy".
_WORKER_STATES = {
    STATE_LOADING: "busy",
    STATE_LIVE: "rec",
    STATE_CATCHING_UP: "warn",
}


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

    # Attributes set by build_ui / build_menu / setup_level_timer
    patient_panel: QFrame
    patient_section: Section
    template_section: Section
    settings_section: Section
    patient_name: QLineEdit
    patient_id: QLineEdit
    patient_dob: QLineEdit
    patient_study_date: QLineEdit
    patient_referrer: QLineEdit
    patient_accession: QLineEdit
    splitter: QSplitter
    macros_panel: QFrame
    macro_region_combo: QComboBox
    _macro_container: QWidget
    _macro_layout: QVBoxLayout
    template_combo: QComboBox
    editor: QTextEdit
    _info_words: QLabel
    btn_record: QPushButton
    btn_stop: QPushButton
    model_combo: QComboBox
    vad_checkbox: QCheckBox
    language_input: QLineEdit
    accent_combo: QComboBox
    cleanup_combo: QComboBox
    btn_export_word: QPushButton
    _level_bar: QProgressBar
    _state_pill: QLabel
    _elapsed_label: QLabel
    _status_label: QLabel
    _wordcount_label: QLabel
    _autosave_label: QLabel
    recent_menu: QMenu
    _level_timer: QTimer
    _active_accent: str
    _active_cleanup_level: str

    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings()
        self.recorder = Recorder()
        self.current_wav_path: Optional[str] = None
        self.live_thread: Optional[QThread] = None
        self.live_worker: Optional[LiveTranscribeWorker] = None
        # Post-processing runs off the UI thread (src/ui/postprocess_worker.py);
        # both are created per recording and torn down when it finishes.
        self.pp_thread: Optional[QThread] = None
        self.pp_worker: Optional[PostprocessWorker] = None
        self._partial_seq: int = 0
        self._applied_seq: int = 0
        self._last_raw_transcript: str = ""
        # Sequence number of the one authoritative full-document pass submitted
        # after recording stops; None while no such pass is outstanding.
        self._final_seq: Optional[int] = None
        # Where the dictated region starts. A QTextCursor, not an integer: Qt
        # moves it along when text is inserted before it, so loading a template
        # or typing into a form field mid-recording can no longer leave the
        # boundary pointing into the middle of someone else's text.
        self._dictation_start: Optional[QTextCursor] = None
        self._last_editor_text: str = ""
        self._corrections_pending: list = []
        self._corrections_seen: set = set()
        # Snapshot of the editor right after dictation finishes; diffed against
        # the delivered text at commit time to log what the radiologist changed.
        self._post_dictation_snapshot: Optional[str] = None
        # Debounce adaptive learning: only call learn_from_edit 500 ms after the
        # last keystroke, not on every character (avoids blocking the UI thread).
        self._learn_prev_text: str = ""
        self._learn_curr_text: str = ""
        self._learn_timer = QTimer(self)
        self._learn_timer.setSingleShot(True)
        self._learn_timer.setInterval(500)
        self._learn_timer.timeout.connect(self._flush_learn)

        # Orders status messages so a timed revert cannot overwrite a newer
        # one (src/ui/status.py).
        self._status = StatusTrack()

        build_ui(self)
        self.patient_section.toggled.connect(self._on_patient_section_toggled)
        self.template_section.toggled.connect(self._on_template_section_toggled)
        self.settings_section.toggled.connect(self._on_settings_section_toggled)
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

        # Background services: cloud training monitor + local report analysis.
        self._cloud_monitor = None
        self._cloud_monitor_thread = None
        self._start_background_services()

        # Load macro region from settings
        region = self.settings.get("last_macro_region", "Knee")
        if region in macros.REGION_ORDER:
            self.macro_region_combo.setCurrentText(region)
        rebuild_macro_buttons(self, region)

        # Load templates
        load_templates(self)
        if last_template := self.settings.get("last_template", ""):
            idx = self.template_combo.findText(last_template)
            if idx >= 0:
                self.template_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("F5"), self, partial(on_start_recording, self))
        QShortcut(QKeySequence("F6"), self, partial(on_stop_recording, self))

    # ------------------------------------------------------------------
    # Background services (cloud training + report analysis)
    # ------------------------------------------------------------------

    def _start_background_services(self) -> None:
        """Launch the cloud job monitor and a one-shot report analysis pass.

        Both degrade silently: the monitor is inert unless cloud training is
        enabled, and report analysis is skipped if disabled in settings. Failures
        here must never block app startup.
        """
        # Pre-build the slow spelling index and Whisper model in the background
        # so the radiologist's first spoken chunk is instant, not a ~1.3 s stall.
        try:
            from src.dictation.warmup import (
                postprocess_warmers,
                transcriber_warmer,
                warm_up_async,
            )

            from src.dictation.transcriber import resolve_model

            model_size = resolve_model(self.settings.get("model_size"))
            # Warm the model the recording worker will actually load: an active
            # fine-tuned directory overrides model_size (recording_session
            # passes it to LiveTranscribeWorker), and the process-wide model
            # cache is keyed on that reference — warming the stock model while
            # a fine-tune is active would miss the cache AND pin an unused
            # model in RAM.
            from src.ui.recording_session import _active_model_path

            active_model = _active_model_path()
            warm_up_async(postprocess_warmers() + [
                transcriber_warmer(
                    model_size,
                    model_path=str(active_model) if active_model else None,
                )
            ])
        except Exception as exc:  # warm-up is an optimisation, never fatal
            logger.debug("Could not start dictation warm-up: %s", exc)

        if self.settings.get("cloud_enabled", False):
            try:
                from PySide6.QtCore import QThread
                from src.cloud.framework.job_monitor import CloudJobMonitor
                self._cloud_monitor_thread = QThread()
                self._cloud_monitor = CloudJobMonitor()
                self._cloud_monitor.moveToThread(self._cloud_monitor_thread)
                self._cloud_monitor_thread.started.connect(self._cloud_monitor.run)
                self._cloud_monitor.model_available.connect(self._on_model_available)
                self._cloud_monitor_thread.start()
            except Exception as exc:
                logger.warning("Could not start cloud monitor: %s", exc)

        if self.settings.get("report_analysis_enabled", True):
            try:
                from PySide6.QtCore import QThreadPool, QRunnable

                class _AnalysisTask(QRunnable):
                    def run(self) -> None:
                        try:
                            from src.features.report_analyzer import ReportAnalyzer
                            ReportAnalyzer().analyze_and_save()
                        except Exception as exc:  # background, never fatal
                            logger.debug("Report analysis failed: %s", exc)

                QThreadPool.globalInstance().start(_AnalysisTask())
            except Exception as exc:
                logger.debug("Could not schedule report analysis: %s", exc)

    def _on_model_available(self, version: str) -> None:
        """A fine-tuned model finished training — offer to activate it."""
        try:
            from src.ui.dialogs import show_model_update_notification
            show_model_update_notification(self, version)
        except Exception as exc:
            logger.warning("Model update notification failed: %s", exc)

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
        if path := autosave_report(text, self._get_patient_info()):
            timestamp = datetime.now().strftime("%H:%M")
            self._autosave_label.setText(f"Auto-saved: {timestamp}")
            logger.info("Auto-saved to %s", path)
            audit_log.log_autosave(path, self._get_patient_info().get("id", ""))


    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self, theme: str) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(DARK if theme == "dark" else LIGHT)  # type: ignore[union-attr]
        if self.settings.get("theme") != theme:
            self.settings.set("theme", theme)

    def on_toggle_theme(self) -> None:
        current = self.settings.get("theme", "dark")
        self._apply_theme("light" if current == "dark" else "dark")

    # ------------------------------------------------------------------
    # Panel toggles
    # ------------------------------------------------------------------

    def _set_patient_panel_visible(self, visible: bool) -> None:
        # The fold header, the View menu and the saved setting are three ways
        # into one piece of state. The section owns the widgets and reports what
        # it did through `toggled`, so whichever of the three started the change
        # ends up saving it exactly once.
        self.patient_section.set_expanded(visible)

    def _on_patient_section_toggled(self, visible: bool) -> None:
        if self.settings.get("patient_info_visible") != visible:
            self.settings.set("patient_info_visible", visible)

    def _on_template_section_toggled(self, expanded: bool) -> None:
        if self.settings.get("panel_template_open") != expanded:
            self.settings.set("panel_template_open", expanded)

    def _on_settings_section_toggled(self, expanded: bool) -> None:
        if self.settings.get("panel_settings_open") != expanded:
            self.settings.set("panel_settings_open", expanded)

    def _set_macros_panel_visible(self, visible: bool) -> None:
        self.macros_panel.setVisible(visible)
        if self.settings.get("macros_panel_visible") != visible:
            self.settings.set("macros_panel_visible", visible)

    # Toggles read the saved setting rather than the widget. Qt reports a child
    # as not visible whenever any ancestor is hidden, so asking the widget
    # inverts the wrong value before the window is first shown — and the setting
    # is the single writer for this state anyway.
    def on_toggle_patient_panel(self) -> None:
        self._set_patient_panel_visible(not self.settings.get("patient_info_visible", True))

    def on_toggle_macros_panel(self) -> None:
        self._set_macros_panel_visible(not self.settings.get("macros_panel_visible", True))

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

    def dictation_active(self) -> bool:
        """True while a recording session still owns the dictated region.

        Stays true through the post-stop final pass: ``pp_worker`` is only
        cleared once that pass has landed, and until then the live path is
        still rewriting the region.
        """
        return self.recorder.is_recording or self.pp_worker is not None

    def _dictation_start_position(self) -> int:
        """Character offset where the dictated region begins."""
        return self._dictation_start.position() if self._dictation_start else 0

    def on_insert_template(self) -> None:
        name = self.template_combo.currentText()
        if not name:
            return
        path = templates_dir() / name
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if self.dictation_active():
                # Replacing the document mid-recording would destroy the
                # dictation the radiologist is in the middle of. Put the
                # template above it instead; the cursor anchor shifts with it,
                # so dictation keeps appending underneath.
                cursor = self.editor.textCursor()
                cursor.setPosition(0)
                cursor.insertText(content if content.endswith("\n") else content + "\n")
                self._show_status(f"Template inserted above dictation: {name}", 2000, state="ok")
            else:
                self.editor.setPlainText(content)
                self._show_status(f"Template loaded: {name}", 2000, state="ok")
            self.settings.set("last_template", name)
            audit_log.log_template_loaded(name)
        except Exception as exc:
            QMessageBox.warning(self, "Template Error", str(exc))

    # ------------------------------------------------------------------
    # Macro management
    # ------------------------------------------------------------------

    def _insert_macro(self, text: str) -> None:
        if self.dictation_active():
            # Anything dropped inside the dictated region is overwritten by the
            # next live cycle, so a macro goes in immediately before it.
            cursor = self.editor.textCursor()
            cursor.setPosition(self._dictation_start_position())
            cursor.insertText(f"{text}\n")
            self._show_status("Macro inserted above dictation", 2000, state="ok")
        else:
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

            rebuild_macro_buttons(self, self.macro_region_combo.currentText())
            self._show_status("Macros reloaded", 2000, state="ok")
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
            _Yes = QMessageBox.StandardButton.Yes
            reply = QMessageBox.question(
                self, "New Report",
                "Clear the current report and start a new one?",
                _Yes | QMessageBox.StandardButton.No,
            )
            if reply != _Yes:
                return
        self.flush_dictation_edits()
        self.editor.clear()
        self.patient_name.clear()
        self.patient_id.clear()
        self.patient_dob.clear()
        self.patient_study_date.clear()
        self.patient_referrer.clear()
        self.patient_accession.clear()
        self._show_status("New report started.")

    def _load_report_from_path(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            self.editor.setPlainText(fh.read())
        self._show_status(f"Opened: {os.path.basename(path)}", 2000, state="ok")

    def on_open_report(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Report", "", "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            self._load_report_from_path(path)
            self.settings.add_recent_report(path)
            rebuild_recent_menu(self)
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", str(exc))

    def on_copy(self) -> None:
        # The clipboard leaves the app just as surely as a file does.
        if not confirm_release(self):
            return
        QApplication.clipboard().setText(self.editor.toPlainText())
        self._show_status("Copied to clipboard.", 1500, state="ok")

    def _post_save(self, path: str, verb: str, log_fn) -> None:  # type: ignore[type-arg]
        self.settings.add_recent_report(path)
        rebuild_recent_menu(self)
        self._show_status(f"{verb}: {os.path.basename(path)}", 2000, state="ok")
        log_fn(path, self._get_patient_info().get("id", ""))

    def on_save_txt(self) -> None:
        # Before the file dialog: these are the questions worth full attention,
        # and asking them once someone has already picked a filename catches them
        # in "just save it" mode. Acknowledging and then cancelling the dialog
        # only over-records — the acknowledgement did happen, and the release
        # itself is logged separately by _post_save.
        if not confirm_release(self):
            return
        default = self._default_filename(".txt")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default, "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            text = self.editor.toPlainText()
            self.flush_dictation_edits()
            save_report_txt(path, text, self._get_patient_info())
            self._post_save(path, "Saved",
                            lambda p, pid: audit_log.log_report_saved(p, pid, len(text.split())))
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
        if not confirm_release(self):  # before the dialog — see on_save_txt
            return
        default = self._default_filename(".docx")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to Word", default, "Word Documents (*.docx);;All Files (*)"
        )
        if not path:
            return
        try:
            self.flush_dictation_edits()
            export_to_word(path, self.editor.toPlainText(), self._get_patient_info())
            self._post_save(path, "Exported",
                            lambda p, pid: audit_log.log_report_exported(p, pid, "docx"))
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def on_clear(self) -> None:
        if self.editor.toPlainText().strip():
            audit_log.log_report_cleared(self._get_patient_info().get("id", ""))
        self.editor.clear()

    # ------------------------------------------------------------------
    # Recent reports menu
    # ------------------------------------------------------------------

    def _open_recent(self, path: str) -> None:
        try:
            self._load_report_from_path(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", str(exc))


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Worker slots
    #
    # These MUST be bound methods of this QObject, not lambdas. A signal
    # connected to a plain callable has no receiver thread affinity, so Qt
    # invokes it in the *emitting* thread — which for the transcription and
    # post-process workers means touching QTextEdit from a background thread.
    # Binding them here gives Qt a UI-thread receiver, so it queues the call.
    # ------------------------------------------------------------------

    @Slot(str, int)
    def _on_partial_text(self, transcript: str, committed_len: int) -> None:
        on_partial_text(self, transcript, committed_len)

    @Slot(str, list, int)
    def _on_processed_text(self, text: str, changes: list, seq: int) -> None:
        on_processed_text(self, text, changes, seq)

    @Slot(str)
    def _on_worker_progress(self, message: str) -> None:
        # The worker reports what it is doing as text; the pill colour is this
        # window's reading of it. "Catching up" is a warning, not a failure: the
        # decoder is behind the speech and the preview will lag.
        state = _WORKER_STATES.get(message, "busy")
        self._status.last_progress = (message, state)
        self._show_status(message, state=state)

    @Slot()
    def _on_transcription_finished(self) -> None:
        on_transcription_finished(self)

    def _get_patient_info(self) -> dict:
        """Read the patient form into the shared patient-info dict."""
        fields = {
            "name": self.patient_name,
            "id": self.patient_id,
            "dob": self.patient_dob,
            "study_date": self.patient_study_date,
            "referring": self.patient_referrer,
            "accession": self.patient_accession,
        }
        return normalize_patient_info(
            {key: widget.text() for key, widget in fields.items()}
        )

    def _default_filename(self, ext: str) -> str:
        return report_filename(self._get_patient_info(), ext)

    def _show_status(self, message: str, timeout: int = 0, state: str = "idle") -> None:
        """Say what the app is doing, in the status bar and on the state pill.

        ``timeout`` reverts to Ready — but only if nothing newer has been shown
        since. Without the generation counter a 5-second tip posted before Stop
        would fire in the middle of finalising and claim the app was idle while
        it was still working.
        """
        generation = self._status.show()
        self._status_label.setText(message)
        # The pill is narrow by design; a long message (the corrections banner)
        # is cut with an ellipsis there and read in full in the status bar.
        pill_width = self._state_pill.maximumWidth() - 24
        self._state_pill.setText(
            self._state_pill.fontMetrics().elidedText(
                message, Qt.TextElideMode.ElideRight, pill_width
            )
        )
        self._state_pill.setToolTip(message)
        set_status_state(self._state_pill, state)
        if timeout:
            QTimer.singleShot(timeout, partial(self._clear_status, generation))

    def _clear_status(self, generation: int) -> None:
        """Revert a timed message — to Ready, or back to the work still running.

        A short message shown mid-dictation (a clipping warning, a correction
        count) must not leave the app claiming to be idle while the worker is
        still transcribing, so while a session is live it reverts to whatever
        the worker last reported instead.
        """
        revert = self._status.revert(generation, self.dictation_active())
        if revert is not None:
            self._show_status(revert[0], state=revert[1])

    def _on_text_changed(self) -> None:
        text = self.editor.toPlainText()
        word_count = len(text.split()) if text.strip() else 0
        line_count = text.count("\n") + 1 if text else 0
        self._info_words.setText(f"Words: {word_count}  |  Lines: {line_count}")
        self._wordcount_label.setText(f"Words: {word_count}")

        # Passive learning: track user edits (debounced — fires 500 ms after
        # the last keystroke rather than on every character).
        if (not self.recorder.is_recording
                        and self._last_editor_text
                        and self.settings.get("learning_enabled", True)) and text != self._last_editor_text:
            self._learn_prev_text = self._last_editor_text
            self._learn_curr_text = text
            self._learn_timer.start()
        self._last_editor_text = text

    def _flush_learn(self) -> None:
        learn_from_edit(self._learn_prev_text, self._learn_curr_text)

    def flush_dictation_edits(self) -> None:
        """Log how the last dictation's output differs from the delivered text.

        Called at every commit point (export, save, clear, open, close, and the
        start of the next recording). Compares the snapshot taken when dictation
        finished against the current editor text and records the radiologist's
        edits — the signal for whether dictation itself was mistaken. Cleared
        after flushing so each session is logged once.
        """
        snapshot = self._post_dictation_snapshot
        if snapshot is None:
            return
        self._post_dictation_snapshot = None
        if not self.settings.get("learning_enabled", True):
            return
        with contextlib.suppress(Exception):
            from src.features.edit_tracking import record_session_edits
            record_session_edits(snapshot, self.editor.toPlainText())

    # ------------------------------------------------------------------
    # Window close – persist settings
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self.settings.batch_set({
            "window_width": self.width(),
            "window_height": self.height(),
            "splitter_sizes": self.splitter.sizes(),
        })

        # Make sure any active recording session is finalized before exit.
        if self.recorder.is_recording:
            on_stop_recording(self)
        elif self.live_worker is not None:
            self.live_worker.finalize()
        if self.live_thread is not None:
            self.live_thread.quit()

        # Stop the cloud job monitor thread if running.
        if self._cloud_monitor is not None:
            with contextlib.suppress(Exception):
                self._cloud_monitor.stop()
                if self._cloud_monitor_thread is not None:
                    self._cloud_monitor_thread.quit()
                    self._cloud_monitor_thread.wait(2000)

        # Persist any open training-capture session (review-time corrections).
        with contextlib.suppress(Exception):
            from src.ui.recording_session import finalize_training_capture
            finalize_training_capture()
        # Save adaptive learning data on exit
        with contextlib.suppress(Exception):
            get_adaptive_learning().force_save()
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
