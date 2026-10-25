"""Main application window: layout assembly and event dispatch."""

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
    QPushButton, QComboBox, QLabel, QLineEdit,
    QCheckBox, QFrame, QSplitter, QProgressBar, QMenu, QVBoxLayout, QWidget,
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor

from src.dictation.audio import Recorder
from src.core.logging_setup import setup_logging
from src.core.settings import Settings
from src.core.patient_schema import normalize_patient_info
from src.core.paths import report_filename
from src.ui.collapsible import Section
from src.ui.status import StatusTrack
from src.ui.finding_marks import FindingGutter
from src.ui.report_editor import ReportEditor
from src.ui.term_marks import TermMarks
from src.ui.term_popup import TermPopup
from src.ui.styles import DARK, LIGHT, set_findings_state, set_status_state
from src.features.report_release import OutstandingFindings
from src.dictation.worker import STATE_CATCHING_UP, STATE_LIVE, STATE_LOADING
from src.features.report_manager import (
    autosave_report, save_report_txt, export_to_word, DOCX_AVAILABLE
)
from src.medical import macros
from src.medical.macros import reload_macros
from src.medical.term_lookup import lookup_hint_active, note_lookup_used
from src.core.paths import startup_cleanup, autosave_dir, macros_file, templates_dir
from src.features.adaptive_learning import get_adaptive_learning, learn_from_session
from src.features import audit_log, run_log

# View construction
from src.ui.views import (
    build_ui, build_menu, load_templates, rebuild_macro_buttons, rebuild_recent_menu
)

# Recording control
from src.ui.field_navigation import FieldKeys
from src.ui.recording_session import (
    on_handback, _abandon_unfinished_session, insert_before_dictation, on_partial_text,
    on_processed_text, on_start_recording, on_stop_recording, on_transcription_finished,
    setup_level_timer,
)

# Dialog handling
from src.ui.dialogs import show_learning_consent_if_needed, show_disclaimer_if_needed
from src.ui.release_gate import confirm_release

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

setup_logging()
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
    patient_sex: QComboBox
    patient_inputs: dict
    splitter: QSplitter
    macros_panel: QFrame
    macro_region_combo: QComboBox
    _macro_container: QWidget
    _macro_layout: QVBoxLayout
    template_combo: QComboBox
    editor: ReportEditor
    finding_gutter: FindingGutter
    _findings_pill: QLabel
    _info_words: QLabel
    btn_record: QPushButton
    btn_stop: QPushButton
    model_combo: QComboBox
    parakeet_combo: QComboBox
    engine_label: QLabel
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
    _info_marks: QLabel
    _active_accent: str
    _active_cleanup_level: str
    # Set per recording by recording_session.
    _record_started: float
    _run_record: Optional[run_log.RunRecord]
    _run_chunks: int
    _audio_stopped_at: Optional[float]

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
        # Sequence number of the pass that hands the live text straight back at
        # Stop, ahead of the polish. Applying it is what re-enables Record.
        self._handback_seq: Optional[int] = None
        # The editor's text right after the session's last write, and whether
        # the radiologist has taken the report back (recording_session.
        # session_may_write): from Stop on, the session writes only into a
        # report nobody has touched since its own last write.
        self._session_last_written: Optional[str] = None
        self._session_yielded: bool = False
        # Threads of sessions retired before they finished (see
        # recording_session._abandon_unfinished_session); waited for on close.
        self._retiring_threads: list = []
        # Where the dictated region starts. A QTextCursor, not an integer: Qt
        # moves it along when text is inserted before it, so loading a template
        # or typing into a form field mid-recording can no longer leave the
        # boundary pointing into the middle of someone else's text.
        self._dictation_start: Optional[QTextCursor] = None
        # Where it ends: dictation goes in at the caret, so text after the
        # region (the rest of a template) must be left alone. Moves with text
        # written at it, unlike the start (recording_session._open_dictation_region).
        self._dictation_end: Optional[QTextCursor] = None
        # The template field the dictation replaced, if it replaced one: saying
        # one of a pick-list's options answers it in the template's spelling.
        self._dictation_field: Optional[str] = None
        self._last_editor_text: str = ""
        self._corrections_pending: list = []
        self._corrections_seen: set = set()
        # Snapshot of the editor right after dictation finishes; diffed against
        # the delivered text at commit time to log what the radiologist changed.
        self._post_dictation_snapshot: Optional[str] = None

        # Orders status messages so a timed revert cannot overwrite a newer
        # one (src/ui/status.py).
        self._status = StatusTrack()

        # Which critical findings this report has and which have been answered
        # for. Built before the UI because the gutter and the count pill are
        # two views of this one object, and the release gate updates it: the
        # rule itself lives in the shared service, not in this front-end.
        self.findings = OutstandingFindings()

        build_ui(self)
        self.patient_section.toggled.connect(self._on_patient_section_toggled)
        self.template_section.toggled.connect(self._on_template_section_toggled)
        self.settings_section.toggled.connect(self._on_settings_section_toggled)
        # Highlight a term in the editor and its neighbourhood appears beside
        # it. Owned here rather than in build_ui because it needs
        # dictation_active(): it stays shut while the text is being rewritten.
        self.term_popup = TermPopup(self.editor, self.dictation_active, self)
        # And the marks that say which word to highlight: without them the
        # popup only helps someone who already suspects the word.
        self.term_marks = TermMarks(self.editor, self.dictation_active, self)
        self.term_marks.changed.connect(self._on_marks_changed)
        self.finding_gutter.changed.connect(self._on_findings_changed)
        self.term_popup.applied.connect(self._on_lookup_used)
        build_menu(self)
        self._setup_shortcuts()
        self.editor.set_release_gate(self._release_selection)
        self._setup_autosave_timer()
        setup_level_timer(self)
        self._apply_theme(self.settings.get("theme"))

        self.setWindowTitle("MSK Radiology Dictation")
        width = self.settings.get("window_width")
        height = self.settings.get("window_height")
        self.resize(width, height)

        # Restore splitter sizes
        sizes = self.settings.get("splitter_sizes")
        self.splitter.setSizes(sizes)

        # Restore panel visibility
        self._set_patient_panel_visible(self.settings.get("patient_info_visible"))
        self._set_macros_panel_visible(self.settings.get("macros_panel_visible"))

        # First-launch consent and disclaimer (after UI is built)
        show_learning_consent_if_needed(self)
        show_disclaimer_if_needed(self)

        # Background services: cloud training monitor + local report analysis.
        self._cloud_monitor = None
        self._cloud_monitor_thread = None
        self._start_background_services()

        # Load macro region from settings
        region = self.settings.get("last_macro_region")
        if region in macros.REGION_ORDER:
            self.macro_region_combo.setCurrentText(region)
        rebuild_macro_buttons(self, region)

        # Load templates
        load_templates(self)
        if last_template := self.settings.get("last_template"):
            idx = self.template_combo.findText(last_template)
            if idx >= 0:
                self.template_combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("F5"), self, partial(on_start_recording, self))
        QShortcut(QKeySequence("F6"), self, partial(on_stop_recording, self))
        # One key that starts and stops: what a dictation mic's record button
        # sends in keyboard mode. Configurable, like the field keys below.
        QShortcut(
            QKeySequence(self.settings.get("key_record_toggle")), self,
            self._toggle_recording,
        )
        self._field_keys = FieldKeys(
            self.editor,
            self.settings.get("key_next_field"),
            self.settings.get("key_previous_field"),
            # Off only while recording: after Stop the report is the
            # radiologist's, and Tab must move between fields, not type a tab.
            enabled=lambda: not self.recorder.is_recording,
        )

    def _toggle_recording(self) -> None:
        if self.recorder.is_recording:
            on_stop_recording(self)
        elif self.btn_record.isEnabled():
            on_start_recording(self)

    # ------------------------------------------------------------------
    # Background services (cloud training + report analysis)
    # ------------------------------------------------------------------

    def _start_background_services(self) -> None:
        """Launch the cloud job monitor and a one-shot report analysis pass.

        Both degrade silently: the monitor is inert unless cloud training is
        enabled, and report analysis is skipped if disabled in settings. Failures
        here must never block app startup.
        """
        # Pre-build the slow spelling index and speech model in the background
        # so the radiologist's first spoken chunk is instant, not a ~1.3 s stall.
        try:
            from src.dictation.warmup import (
                postprocess_warmers,
                asr_warmer,
                warm_up_async,
            )

            from src.dictation.asr.models import resolve_model

            model_size = resolve_model(self.settings.get("model_size"))
            # Warm the model the recording worker will actually load: an active
            # fine-tuned directory overrides model_size (recording_session
            # passes it to LiveTranscribeWorker), and the process-wide model
            # cache is keyed on that reference: warming the stock model while
            # a fine-tune is active would miss the cache AND pin an unused
            # model in RAM.
            from src.cloud.framework.registry import active_voice_model_path

            active_model = active_voice_model_path()
            warmers = postprocess_warmers() + [
                asr_warmer(
                    model_size,
                    model_path=str(active_model) if active_model else None,
                )
            ]
            # The live model writes the first words on screen: leaving it lazy
            # makes the first dictation after every launch stall on its load.
            live_model_size = resolve_model(self.settings.get("live_model_size"))
            if live_model_size != model_size or active_model:
                warmers.append(asr_warmer(live_model_size))
            warm_up_async(warmers)
        except Exception as exc:  # warm-up is an optimisation, never fatal
            logger.debug("Could not start dictation warm-up: %s", exc)

        if self.settings.get("cloud_enabled"):
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

    def _on_model_available(self, version: str) -> None:
        """A fine-tuned model finished training: offer to activate it."""
        try:
            from src.ui.dialogs import show_model_update_notification
            show_model_update_notification(self, version)
        except Exception as exc:
            logger.warning("Model update notification failed: %s", exc)

    # ------------------------------------------------------------------
    # Auto-save timer
    # ------------------------------------------------------------------

    def _setup_autosave_timer(self) -> None:
        interval_seconds = self.settings.get("auto_save_interval")
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
        # Extra selections and the gutter marks are painted in code, so the
        # stylesheet cannot reach them: both are told the theme explicitly.
        self.term_marks.set_theme(theme)
        self.finding_gutter.set_theme(theme)
        if self.settings.get("theme") != theme:
            self.settings.set("theme", theme)

    def on_toggle_theme(self) -> None:
        current = self.settings.get("theme")
        self._apply_theme("light" if current == "dark" else "dark")

    # ------------------------------------------------------------------
    # Term marks, and telling the radiologist they are there
    # ------------------------------------------------------------------

    def _on_marks_changed(self, count: int) -> None:
        """Report how many words are marked, and how to use them.

        The instruction half retires itself: a permanent "how to use" line is
        what the One Surface work removed, because it teaches once and then
        costs a row on every report forever. Once the lookup has actually been
        used a few times, the count alone remains.
        """
        if count <= 0:
            self._info_marks.hide()
            return
        noun = "word" if count == 1 else "words"
        if lookup_hint_active(self.settings.get):
            text = f"{count} {noun} to check: highlight one to see alternatives"
        else:
            text = f"{count} {noun} to check"
        self._info_marks.setText(text)
        self._info_marks.show()

    def _on_findings_changed(self, count: int) -> None:
        """Say, always and without opening anything, what this report contains.

        Three readings, and only one of them is loud: a clean report is a quiet
        "No findings" rather than a red zero, because a warning shown every
        session is a warning nobody reads by the time it matters.
        """
        self._findings_pill.setText(self.findings.label)
        set_findings_state(self._findings_pill, self.findings.state)

    def _on_lookup_used(self) -> None:
        """One more taken suggestion; the hint stops explaining after a few."""
        note_lookup_used(self.settings)

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
    # inverts the wrong value before the window is first shown: and the setting
    # is the single writer for this state anyway.
    def on_toggle_patient_panel(self) -> None:
        self._set_patient_panel_visible(not self.settings.get("patient_info_visible"))

    def on_toggle_macros_panel(self) -> None:
        self._set_macros_panel_visible(not self.settings.get("macros_panel_visible"))

    # ------------------------------------------------------------------
    # Font size
    # ------------------------------------------------------------------

    def on_font_increase(self) -> None:
        self._change_font_size(+1)

    def on_font_decrease(self) -> None:
        self._change_font_size(-1)

    def _change_font_size(self, delta: int) -> None:
        size = self.settings.get("font_size") + delta
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

    def on_insert_template(self) -> None:
        name = self.template_combo.currentText()
        if not name:
            return
        path = templates_dir() / name
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if self.recorder.is_recording:
                # Replacing the document mid-recording would destroy the
                # dictation the radiologist is in the middle of. Put the
                # template above it instead.
                insert_before_dictation(self, content if content.endswith("\n") else content + "\n")
                self._show_status(f"Template inserted above dictation: {name}", 2000, state="ok")
            else:
                self.replace_report(content)
                self._start_new_report_findings()
                self._show_status(f"Template loaded: {name}  (Ctrl+Z to undo)", 3000, state="ok")
            self.settings.set("last_template", name)
            audit_log.log_template_loaded(name)
        except Exception as exc:
            QMessageBox.warning(self, "Template Error", str(exc))

    # ------------------------------------------------------------------
    # Macro management
    # ------------------------------------------------------------------

    def _insert_macro(self, text: str) -> None:
        if self.recorder.is_recording:
            # Anything dropped inside the dictated region is overwritten by the
            # next live cycle, so a macro goes in immediately before it.
            insert_before_dictation(self, f"{text}\n")
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

    def replace_report(self, text: str) -> None:
        """Replace the whole report as one undoable edit.

        ``setPlainText`` and ``clear`` throw the undo history away, so a
        template loaded or Clear pressed over a finished report lost it for
        good. While a polish is still running this is an edit like any other:
        the session sees the report changed and stops writing.
        """
        cursor = QTextCursor(self.editor.document())
        cursor.beginEditBlock()
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.insertText(text)
        cursor.endEditBlock()

    def _refuse_while_recording(self, what: str) -> bool:
        if self.recorder.is_recording:
            self._show_status(f"Stop recording before {what}", 3000, state="warn")
            return True
        return False

    def on_new_report(self) -> None:
        if self._refuse_while_recording("starting a new report"):
            return
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
        self.replace_report("")
        self._start_new_report_findings()
        for widget in self.patient_inputs.values():
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(0)
            else:
                widget.clear()
        self._show_status("New report started.")

    def _load_report_from_path(self, path: str) -> None:
        if self._refuse_while_recording("opening a report"):
            return
        with open(path, "r", encoding="utf-8") as fh:
            self.replace_report(fh.read())
        self._start_new_report_findings()
        self._show_status(f"Opened: {os.path.basename(path)}", 2000, state="ok")

    def _start_new_report_findings(self) -> None:
        """Forget the previous report's findings and its acknowledgements.

        Acknowledgement answers "has this been phoned through for *this*
        patient". Carrying them into the next report would show the next
        patient's identical finding as already communicated.
        """
        self.findings.reset()
        self.finding_gutter.refresh()

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
        text = self.editor.toPlainText()
        QApplication.clipboard().setText(text)
        audit_log.log_report_copied(self._get_patient_info().get("id", ""), len(text.split()))
        self._show_status("Copied to clipboard.", 1500, state="ok")

    def _release_selection(self, text: str) -> bool:
        """Copy/cut/drag of a selection the gate would ask about (ReportEditor)."""
        if not confirm_release(self):
            return False
        audit_log.log_report_copied(self._get_patient_info().get("id", ""), len(text.split()))
        return True

    def _post_save(self, path: str, verb: str, log_fn) -> None:  # type: ignore[type-arg]
        self.settings.add_recent_report(path)
        rebuild_recent_menu(self)
        self._show_status(f"{verb}: {os.path.basename(path)}", 2000, state="ok")
        log_fn(path, self._get_patient_info().get("id", ""))

    def on_save_txt(self) -> None:
        # Before the file dialog: these are the questions worth full attention,
        # and asking them once someone has already picked a filename catches them
        # in "just save it" mode. Acknowledging and then cancelling the dialog
        # only over-records: the acknowledgement did happen, and the release
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
        if not confirm_release(self):  # before the dialog: see on_save_txt
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
        if self._refuse_while_recording("clearing the report"):
            return
        if self.editor.toPlainText().strip():
            audit_log.log_report_cleared(self._get_patient_info().get("id", ""))
        self.replace_report("")
        self._start_new_report_findings()
        self._show_status("Cleared  (Ctrl+Z to undo)", 3000, state="ok")

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
    # invokes it in the *emitting* thread: which for the transcription and
    # post-process workers means touching QTextEdit from a background thread.
    # Binding them here gives Qt a UI-thread receiver, so it queues the call.
    # ------------------------------------------------------------------

    @Slot(str, int)
    def _on_partial_text(self, transcript: str, committed_len: int) -> None:
        on_partial_text(self, transcript, committed_len)

    @Slot(str)
    def _on_handback(self, raw: str) -> None:
        on_handback(self, raw)

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
        return normalize_patient_info({
            key: widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            for key, widget in self.patient_inputs.items()
        })

    def _default_filename(self, ext: str) -> str:
        return report_filename(self._get_patient_info(), ext)

    def _show_status(self, message: str, timeout: int = 0, state: str = "idle") -> None:
        """Say what the app is doing, in the status bar and on the state pill.

        ``timeout`` reverts to Ready: but only if nothing newer has been shown
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
        """Revert a timed message: to Ready, or back to the work still running.

        A short message shown mid-dictation (a clipping warning, a correction
        count) must not leave the app claiming to be idle while the worker is
        still transcribing, so while a session is live it reverts to whatever
        the worker last reported instead.
        """
        revert = self._status.revert(generation, self.dictation_active())
        if revert is not None:
            self._show_status(revert[0], state=revert[1])

    def _on_text_changed(self, by_dictation: bool = False) -> None:
        text = self.editor.toPlainText()
        word_count = len(text.split()) if text.strip() else 0
        line_count = text.count("\n") + 1 if text else 0
        self._info_words.setText(f"Words: {word_count}  |  Lines: {line_count}")
        self._wordcount_label.setText(f"Words: {word_count}")

        # The single "the report changed" funnel: editor.textChanged reaches
        # here, and so do the dictation writes that block that signal and call
        # this method by hand. Both overlays hang off it rather than off
        # textChanged, so neither can miss a dictated report.
        self.finding_gutter.schedule_refresh()
        self.term_marks.schedule_rescan()

        self._last_editor_text = text

    def flush_dictation_edits(self) -> None:
        """Log how the last dictation's output differs from the delivered text.

        Called at every commit point (export, save, clear, open, close, and the
        start of the next recording). Compares the snapshot taken when dictation
        finished against the current editor text and records the radiologist's
        edits: the signal for whether dictation itself was mistaken. Cleared
        after flushing so each session is logged once.
        """
        snapshot = self._post_dictation_snapshot
        if snapshot is None:
            return
        self._post_dictation_snapshot = None
        if not self.settings.get("learning_enabled"):
            return
        with contextlib.suppress(Exception):
            from src.features.edit_tracking import record_session_edits
            final = self.editor.toPlainText()
            record_session_edits(snapshot, final)
            learn_from_session(snapshot, final)

    # ------------------------------------------------------------------
    # Window close – persist settings
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Shut down in steps that cannot skip each other.

        A dictation still running owns two threads; destroying the window while
        either runs aborts the process (and used to, losing everything since
        the last autosave). So the report is autosaved first, the session is
        retired the same way a new recording retires it, and its threads are
        given a few seconds to stop. Each step is its own ``suppress``: one
        failing must never cost the learning data or the training capture.
        """
        with contextlib.suppress(Exception):
            self.settings.batch_set({
                "window_width": self.width(),
                "window_height": self.height(),
                "splitter_sizes": self.splitter.sizes(),
            })
        with contextlib.suppress(Exception):
            self._do_autosave()
        with contextlib.suppress(Exception):
            if self.recorder.is_recording:
                self.recorder.stop()
            _abandon_unfinished_session(self)
        for thread in (*self._retiring_threads, self.live_thread, self.pp_thread):
            with contextlib.suppress(RuntimeError):  # already finished and deleted
                if thread is not None and thread.isRunning():
                    thread.quit()
                    thread.wait(5000)

        # Stop the cloud job monitor thread if running.
        if self._cloud_monitor is not None:
            with contextlib.suppress(Exception):
                self._cloud_monitor.stop()
                if self._cloud_monitor_thread is not None:
                    self._cloud_monitor_thread.quit()
                    self._cloud_monitor_thread.wait(2000)

        # Persist any open training-capture session (review-time corrections).
        with contextlib.suppress(Exception):
            from src.ui.training_capture import finalize_training_capture
            finalize_training_capture()
        with contextlib.suppress(Exception):
            get_adaptive_learning().force_save()
        super().closeEvent(event)


def main() -> None:
    """Application entry point."""
    settings = Settings()
    retention_days = settings.get("autosave_retention_days")
    startup_cleanup(retention_days)

    app = QApplication(sys.argv)
    app.setApplicationName("MSK Radiology Dictation")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
