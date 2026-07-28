"""Dialog windows and modal interactions."""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

from src.features import clinical_disclaimer
from src.features.adaptive_learning import get_adaptive_learning

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)

_UNFILLED_FIELD_RE = re.compile(r"\[([A-Z][A-Z0-9 _/-]{1,40})\]|\{\{([^}]{1,40})\}\}")

_Yes = QMessageBox.StandardButton.Yes
_No = QMessageBox.StandardButton.No


def show_learning_consent_if_needed(window: MainWindow) -> None:
    """Show adaptive learning consent dialog on first launch."""
    if window.settings.get("learning_consent_shown", False):
        return
    reply = QMessageBox.question(
        window,
        "Adaptive Learning — Your Consent",
        "This application can learn from your corrections to improve future "
        "transcription accuracy.\n\n"
        "All learning data stays on this device and is never transmitted.\n\n"
        "Allow adaptive learning from your corrections?",
        _Yes | _No,
        _Yes,
    )
    enabled = reply == _Yes
    window.settings.set("learning_enabled", enabled)
    window.settings.set("learning_consent_shown", True)
    from src.features import audit_log
    audit_log.log_learning_consent(enabled)


def show_cloud_training_dialog(window: MainWindow) -> None:
    """Settings panel for cloud voice training: consent, credentials, status.

    This is the single reachable entry point for the whole cloud subsystem.
    Without it the feature stays dormant — collection is gated on consent and
    uploads need a Lightning AI key, neither of which can be set elsewhere.
    Everything is opt-in: closing with the box unchecked keeps the app offline.
    """
    from PySide6.QtWidgets import (
        QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
        QLineEdit, QPushButton, QVBoxLayout,
    )
    from src.cloud.framework.client import clear_api_key, get_api_key, store_api_key

    s = window.settings
    dlg = QDialog(window)
    dlg.setWindowTitle("Cloud Voice Training")
    dlg.setMinimumWidth(460)
    layout = QVBoxLayout(dlg)

    blurb = QLabel(
        "Build a personalised voice model on Lightning AI from your own "
        "corrections. De-identified audio and text are uploaded; patient "
        "identifiers are removed and spoken identifiers silenced before upload, "
        "and any record that fails the privacy check is dropped.\n\n"
        "The app always works fully offline — this is entirely optional."
    )
    blurb.setWordWrap(True)
    layout.addWidget(blurb)

    enable_box = QCheckBox("Enable cloud voice training (upload my corrections)")
    enable_box.setChecked(bool(s.get("cloud_enabled", False))
                          and bool(s.get("cloud_training_consent", False)))
    layout.addWidget(enable_box)

    form = QFormLayout()
    key_edit = QLineEdit()
    key_edit.setEchoMode(QLineEdit.EchoMode.Password)
    key_edit.setPlaceholderText(
        "•••• stored in OS keychain" if get_api_key() else "Lightning AI API key")
    form.addRow("API key:", key_edit)

    project_edit = QLineEdit(s.get("lightning_project_id", "") or "")
    project_edit.setPlaceholderText("Lightning AI project ID")
    form.addRow("Project ID:", project_edit)
    layout.addLayout(form)

    # Status: how many corrections are staged, and the active model.
    try:
        from src.training.collector import get_correction_collector
        pending = get_correction_collector().get_pending_count()
    except Exception:
        pending = 0
    active = s.get("active_model_version") or "base (default)"
    status = QLabel(f"Staged corrections awaiting upload: {pending}\n"
                    f"Active model: {active}")
    status.setWordWrap(True)
    layout.addWidget(status)

    revert_btn = QPushButton("Revert to base model")
    revert_btn.setEnabled(bool(s.get("active_model_version")))

    def _revert() -> None:
        try:
            from src.cloud.framework.registry import ModelRegistry
            ModelRegistry().rollback_to_base()
            revert_btn.setEnabled(False)
            status.setText(f"Staged corrections awaiting upload: {pending}\n"
                           "Active model: base (default)")
            window._show_status("Reverted to base model", 3000)
        except Exception as exc:
            QMessageBox.warning(window, "Error", f"Could not revert: {exc}")

    revert_btn.clicked.connect(_revert)
    layout.addWidget(revert_btn)

    _Save = QDialogButtonBox.StandardButton.Save
    _Cancel = QDialogButtonBox.StandardButton.Cancel
    buttons = QDialogButtonBox(_Save | _Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    enabled = enable_box.isChecked()
    s.batch_set({
        "cloud_enabled": enabled,
        "cloud_training_consent": enabled,
        "lightning_project_id": project_edit.text().strip(),
    })
    # Only touch the keychain when the user typed a new key.
    new_key = key_edit.text().strip()
    if new_key:
        store_api_key(new_key)
    elif not enabled and get_api_key():
        # Disabling cloud training removes the stored credential.
        clear_api_key()

    from src.features import audit_log
    audit_log.log_cloud_consent(enabled)
    window._show_status(
        "Cloud voice training enabled" if enabled else "Cloud voice training disabled",
        3000,
    )


def show_model_update_notification(window: MainWindow, version: str) -> None:
    """Offer to activate a freshly-downloaded fine-tuned model."""
    reply = QMessageBox.question(
        window,
        "Personalised Model Ready",
        f"A new personalised voice model ({version}) has finished training and "
        "downloaded successfully.\n\n"
        "Activate it for future dictation? You can revert to the base model "
        "any time from settings.",
        _Yes | _No,
        _Yes,
    )
    if reply == _Yes:
        try:
            from src.cloud.framework.registry import ModelRegistry
            if ModelRegistry().activate_model(version):
                window._show_status(f"Activated personalised model {version}", 4000)
            else:
                QMessageBox.warning(window, "Activation Failed",
                                    f"Could not activate model {version}.")
        except Exception as exc:
            QMessageBox.warning(window, "Error", f"Activation failed: {exc}")


def show_ai_cleanup_settings_dialog(window: MainWindow) -> None:
    """Settings panel for the optional Groq report-cleanup feature.

    Lets the user enable AI cleanup, store the Groq API key (OS keychain), and
    pick the model. Off by default; enabling also requires cloud consent, which
    is granted in the Cloud Voice Training dialog.
    """
    from PySide6.QtWidgets import (
        QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
        QLineEdit, QVBoxLayout,
    )
    from src.dictation.postprocess.llm_cleanup import (
        clear_groq_key, get_groq_key, store_groq_key, _DEFAULT_MODEL,
    )

    s = window.settings
    dlg = QDialog(window)
    dlg.setWindowTitle("AI Cleanup (Groq)")
    dlg.setMinimumWidth(460)
    layout = QVBoxLayout(dlg)

    blurb = QLabel(
        "Polish a finished report with a fast cloud LLM (Groq) — fixing grammar, "
        "punctuation, and obvious speech-to-text slips only. Clinical content is "
        "never changed. Patient identifiers are removed before any text is sent, "
        "and this requires cloud consent (granted in Cloud Voice Training).\n\n"
        "Optional and off by default; the app works fully offline without it."
    )
    blurb.setWordWrap(True)
    layout.addWidget(blurb)

    enable_box = QCheckBox("Enable AI cleanup (sends de-identified report text to Groq)")
    enable_box.setChecked(bool(s.get("groq_cleanup_enabled", False)))
    layout.addWidget(enable_box)

    form = QFormLayout()
    key_edit = QLineEdit()
    key_edit.setEchoMode(QLineEdit.EchoMode.Password)
    key_edit.setPlaceholderText(
        "•••• stored in OS keychain" if get_groq_key() else "Groq API key")
    form.addRow("API key:", key_edit)

    model_edit = QLineEdit(s.get("groq_model", _DEFAULT_MODEL) or _DEFAULT_MODEL)
    form.addRow("Model:", model_edit)
    layout.addLayout(form)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return

    enabled = enable_box.isChecked()
    s.batch_set({
        "groq_cleanup_enabled": enabled,
        "groq_model": model_edit.text().strip() or _DEFAULT_MODEL,
    })
    if new_key := key_edit.text().strip():
        store_groq_key(new_key)
    elif not enabled and get_groq_key():
        clear_groq_key()
    window._show_status(
        "AI cleanup enabled" if enabled else "AI cleanup disabled", 3000)


def on_run_ai_cleanup(window: MainWindow) -> None:
    """Run Groq cleanup on the current report and apply the result in-editor."""
    from src.dictation.postprocess.llm_cleanup import clean_with_llm, is_enabled

    text = window.editor.toPlainText().strip()
    if not text:
        window._show_status("Nothing to clean up", 2000)
        return
    if not is_enabled():
        QMessageBox.information(
            window, "AI Cleanup Disabled",
            "Enable AI cleanup first in Settings → AI Cleanup (Groq).")
        return

    window._show_status("Running AI cleanup…", 0)
    try:
        cleaned, changes = clean_with_llm(text, window._get_patient_info())
    except Exception as exc:   # clean_with_llm is already best-effort; belt & braces
        QMessageBox.warning(window, "AI Cleanup Failed", f"Could not clean up: {exc}")
        return

    if cleaned == text:
        window._show_status("AI cleanup made no changes", 3000)
        return

    summary = "\n".join(f"  • {c}" for c in changes[:10]) or "  • (formatting only)"
    reply = QMessageBox.question(
        window, "Apply AI Cleanup?",
        f"The AI suggested {len(changes)} change(s):\n\n{summary}\n\n"
        "Apply them to the report? Review carefully before signing off.",
        _Yes | _No, _Yes,
    )
    if reply == _Yes:
        window.editor.setPlainText(cleaned)
        window._show_status("AI cleanup applied", 3000)


def show_scan_assistant_dialog(window: MainWindow) -> None:
    """Open a chest X-ray, analyse it, and show localised, confident findings.

    Assistive only: findings are surfaced solely when the model is confident AND
    can point to a region. A prominent disclaimer states this is not a diagnosis.
    """
    from PySide6.QtWidgets import QFileDialog

    path, _ = QFileDialog.getOpenFileName(
        window, "Open Chest X-ray", "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
    if not path:
        return

    window._show_status("Analysing scan…", 0)
    try:
        from src.imaging.analyzer import analyze_scan
        result = analyze_scan(path)
    except Exception as exc:
        QMessageBox.warning(
            window, "Scan Analysis Unavailable",
            f"Could not analyse the scan: {exc}\n\n"
            "The imaging feature needs the optional imaging dependencies "
            "(see scripts/lightning/requirements_imaging.txt).")
        return

    _show_scan_result(window, path, result)


def _show_scan_result(window: MainWindow, image_path: str, result) -> None:
    """Render the ImagingResult in a dialog with the overlay and disclaimer."""
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QDialog, QDialogButtonBox, QLabel, QScrollArea, QVBoxLayout,
    )

    dlg = QDialog(window)
    dlg.setWindowTitle("Scan Assistant — Suggestions")
    dlg.setMinimumWidth(520)
    layout = QVBoxLayout(dlg)

    disclaimer = QLabel(result.disclaimer)
    disclaimer.setWordWrap(True)
    disclaimer.setStyleSheet("color: #b00; font-weight: bold;")
    layout.addWidget(disclaimer)

    if result.has_findings:
        lines = "\n".join(
            f"  • {f.label} — confidence {f.probability:.0%}"
            for f in result.findings)
        summary = QLabel(f"Findings flagged for your review:\n{lines}")
    else:
        summary = QLabel("No findings met the confidence threshold. "
                         "This does NOT rule out disease — review the image yourself.")
    summary.setWordWrap(True)
    layout.addWidget(summary)

    overlay = result.overlay_png_path or image_path
    pix = QPixmap(overlay)
    if not pix.isNull():
        img_label = QLabel()
        img_label.setPixmap(pix.scaledToWidth(480))
        scroll = QScrollArea()
        scroll.setWidget(img_label)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(360)
        layout.addWidget(scroll)

    _append_reference_cases(layout, result)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dlg.reject)
    buttons.accepted.connect(dlg.accept)
    buttons.clicked.connect(lambda _b: dlg.accept())
    layout.addWidget(buttons)
    dlg.exec()


def _retrieve_reference_matches(result) -> list:
    """Best-effort similar-reference-case retrieval for a scan result.

    Returns an empty list (never raises) when imaging/retrieval deps are missing,
    no reference datasets are registered, or no query embedding is available — so
    the panel simply doesn't render and the rest of the dialog is unaffected.
    """
    embedding = getattr(result, "embedding", None)
    if embedding is None:
        return []
    try:
        from src.imaging.retrieval import get_reference_index, retrieve_reference_cases

        index = get_reference_index()
        if index is None:
            return []
        # Filter the reference pool to cases sharing any of this scan's findings.
        labels = [f.label for f in result.findings]
        filters = {"required_labels": labels} if labels else None
        return retrieve_reference_cases(embedding, index, filters=filters, top_k=5)
    except Exception as exc:   # optional deps / no data — degrade silently
        logger.debug("Reference retrieval unavailable: %s", exc)
        return []


def _append_reference_cases(layout, result) -> None:
    """Add a 'Similar reference cases' thumbnail strip to the scan dialog.

    Renders nothing when retrieval yields no matches (see
    :func:`_retrieve_reference_matches`).
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget,
    )

    matches = _retrieve_reference_matches(result)
    if not matches:
        return

    header = QLabel("Similar reference cases (for visual comparison — not a diagnosis):")
    header.setWordWrap(True)
    header.setStyleSheet("font-weight: bold;")
    layout.addWidget(header)

    strip = QWidget()
    row = QHBoxLayout(strip)
    for match in matches:
        case = match.case
        cell = QWidget()
        cell_layout = QVBoxLayout(cell)
        thumb = QLabel()
        pix = QPixmap(case.path)
        if not pix.isNull():
            thumb.setPixmap(pix.scaledToWidth(150, Qt.TransformationMode.SmoothTransformation))
        else:
            thumb.setText("(image unavailable)")
        cell_layout.addWidget(thumb)

        positives = [name for name, val in case.labels.items() if val == 1]
        attrs = case.attributes
        attr_bits = [f"{k}: {attrs[k]}" for k in ("age", "sex", "view") if attrs.get(k) is not None]
        caption = QLabel(
            f"{match.similarity:.0%} match\n"
            + (", ".join(positives) or "—")
            + ("\n" + " · ".join(attr_bits) if attr_bits else ""))
        caption.setWordWrap(True)
        caption.setStyleSheet("font-size: 11px;")
        cell_layout.addWidget(caption)
        row.addWidget(cell)

    scroll = QScrollArea()
    scroll.setWidget(strip)
    scroll.setWidgetResizable(True)
    scroll.setMinimumHeight(220)
    layout.addWidget(scroll)


def show_disclaimer_if_needed(window: MainWindow) -> None:
    """Show the clinical disclaimer on first launch.

    The desktop shape of the shared statement in
    ``features/clinical_disclaimer.py`` — the wording and the "have they seen
    it" decision live there, alongside the web front-end's modal. This function
    owns only the message box.
    """
    if not clinical_disclaimer.needs_showing(window.settings):
        return
    QMessageBox.information(
        window,
        clinical_disclaimer.DISCLAIMER_TITLE,
        clinical_disclaimer.DISCLAIMER_TEXT,
    )
    clinical_disclaimer.mark_shown(window.settings)


def validate_template_fields(window: MainWindow) -> bool:
    """Return True if safe to proceed; show warning and return False otherwise."""
    text = window.editor.toPlainText()
    matches = _UNFILLED_FIELD_RE.findall(text)
    if not matches:
        return True
    fields = [m[0] or m[1] for m in matches]
    unique_fields = list(dict.fromkeys(fields))
    reply = QMessageBox.warning(
        window,
        "Unfilled Template Fields",
        f"The report contains {len(unique_fields)} unfilled field(s):\n\n"
        + "\n".join(f"  • {f}" for f in unique_fields[:10])
        + ("\n  …" if len(unique_fields) > 10 else "")
        + "\n\nDo you want to proceed anyway?",
        _Yes | _No,
        _No,
    )
    return reply == _Yes


def on_show_learning_stats(window: MainWindow) -> None:
    """Show adaptive learning statistics."""
    try:
        learning = get_adaptive_learning()
        stats = learning.get_stats()
        corrections = learning.export_corrections()

        msg = (
            f"Adaptive Learning Statistics\n"
            f"{'='*40}\n\n"
            f"Learned corrections: {stats['corrections']}\n"
            f"Custom terms: {stats['custom_terms']}\n"
            f"Tracked term frequencies: {stats['tracked_terms']}\n"
            f"Accent hints: {stats['accent_hints']}\n"
        )

        # Post-dictation edits — the "was dictation itself wrong?" signal.
        try:
            from src.features.edit_tracking import edit_stats, load_edits
            ed = edit_stats()
            if ed["total"]:
                msg += (
                    f"\nPost-dictation edits logged: {ed['total']}\n"
                    f"  (replaced: {ed.get('replace', 0)}, "
                    f"inserted: {ed.get('insert', 0)}, deleted: {ed.get('delete', 0)})\n"
                )
                if recent := [
                    e for e in load_edits() if e.get("op") == "replace"
                ][-5:]:
                    msg += "Recent dictation fixes:\n"
                    for e in recent:
                        msg += f"  • {e['before']} → {e['after']}\n"
                msg += "Run scripts/mine_corrections.py to turn these into rules.\n"
        except Exception:
            pass

        if corrections:
            msg += "\nRecent corrections:\n"
            for wrong, correct in list(corrections.items())[-10:]:
                msg += f"  • {wrong} → {correct}\n"

        QMessageBox.information(window, "Learning Statistics", msg)
    except Exception as exc:
        QMessageBox.warning(window, "Error", f"Could not retrieve stats: {exc}")


def on_reset_learning(window: MainWindow) -> None:
    """Reset all adaptive learning data."""
    reply = QMessageBox.question(
        window,
        "Reset Learning Data",
        "This will delete all learned corrections and custom vocabulary.\n\n"
        "This cannot be undone. Continue?",
        _Yes | _No,
    )
    if reply == _Yes:
        try:
            get_adaptive_learning().reset()
            from src.features import audit_log
            audit_log.log_learning_reset()
            window._show_status("Learning data reset", 2000)
        except Exception as exc:
            QMessageBox.warning(window, "Error", f"Reset failed: {exc}")


def show_correction_rules_dialog(window: MainWindow) -> None:
    """Editor for site-added spelling-correction rules (``user_corrections.yaml``).

    A radiologist can add ``misheard → correct`` rules without a developer. The
    optional "only before" guard makes real-word homophones safe (e.g. only
    rewrite *plural* → *pleural* before "effusion"). Rules persist to the user
    file and apply on the next dictation; shipped rules are never touched.
    """
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
        QTableWidgetItem, QLabel, QHeaderView,
    )
    from src.dictation.postprocess import rules as correction_rules

    dialog = QDialog(window)
    dialog.setWindowTitle("Correction Rules")
    dialog.resize(640, 420)
    layout = QVBoxLayout(dialog)

    layout.addWidget(QLabel(
        "Add words the dictation mishears. 'Only before' is optional — use it for\n"
        "real words (e.g. correct “plural” to “pleural” only before “effusion, space”)."
    ))

    table = QTableWidget(0, 3, dialog)
    table.setHorizontalHeaderLabels(["Misheard", "Correct to", "Only before (comma-sep, optional)"])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    layout.addWidget(table)

    def _add_row(pattern: str = "", replacement: str = "", after: str = "") -> None:
        r = table.rowCount()
        table.insertRow(r)
        table.setItem(r, 0, QTableWidgetItem(pattern))
        table.setItem(r, 1, QTableWidgetItem(replacement))
        table.setItem(r, 2, QTableWidgetItem(after))

    for rule in correction_rules.load_user_rules():
        _add_row(
            str(rule.get("pattern", "")),
            str(rule.get("replacement", "")),
            ", ".join(rule.get("context_after") or []),
        )
    if table.rowCount() == 0:
        _add_row()

    btns = QHBoxLayout()
    add_btn = QPushButton("Add row")
    del_btn = QPushButton("Delete selected")
    save_btn = QPushButton("Save")
    cancel_btn = QPushButton("Cancel")
    add_btn.clicked.connect(lambda: _add_row())
    del_btn.clicked.connect(lambda: table.removeRow(table.currentRow()) if table.currentRow() >= 0 else None)
    cancel_btn.clicked.connect(dialog.reject)
    for b in (add_btn, del_btn):
        btns.addWidget(b)
    btns.addStretch(1)
    btns.addWidget(save_btn)
    btns.addWidget(cancel_btn)
    layout.addLayout(btns)

    def _save() -> None:
        new_rules = []
        for r in range(table.rowCount()):
            item0 = table.item(r, 0)
            item1 = table.item(r, 1)
            item2 = table.item(r, 2)
            pattern = (item0.text() if item0 else "").strip()
            replacement = (item1.text() if item1 else "").strip()
            after_raw = (item2.text() if item2 else "").strip()
            if not pattern or not replacement:
                continue
            rule: dict = {"pattern": pattern, "replacement": replacement}
            if after_raw:
                rule["context_after"] = [w.strip() for w in after_raw.split(",") if w.strip()]
            new_rules.append(rule)
        try:
            correction_rules.save_user_rules(new_rules)
            window._show_status(f"Saved {len(new_rules)} correction rule(s)", 2000)
            dialog.accept()
        except Exception as exc:
            QMessageBox.warning(dialog, "Error", f"Could not save rules: {exc}")

    save_btn.clicked.connect(_save)
    dialog.exec()


def show_image_label_dialog(window: MainWindow) -> None:
    """Dialog for manually labeling a chest X-ray image.

    User selects an image file (PNG/JPG), assigns confidence labels to each
    pathology, and consents to upload de-identified data for training.
    Emits 'image_label_ready' signal on OK with the ImageLabelRecord.
    """
    from datetime import datetime, timezone
    from pathlib import Path
    from PySide6.QtWidgets import (
        QCheckBox, QDialog, QFileDialog, QLabel,
        QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
    )
    from PySide6.QtCore import Qt
    from src.training.schemas import ImageLabelRecord
    from src.training.collector import CorrectionCollector

    dialog = QDialog(window)
    dialog.setWindowTitle("Label Chest X-Ray Image")
    dialog.setMinimumSize(560, 600)
    layout = QVBoxLayout(dialog)

    # File picker
    file_layout = QVBoxLayout()
    file_label = QLabel("No image selected")
    file_label.setWordWrap(True)
    file_layout.addWidget(QLabel("Image file:"))
    file_layout.addWidget(file_label)

    def pick_file():
        path, _ = QFileDialog.getOpenFileName(
            dialog, "Select X-Ray Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if path:
            file_label.setText(Path(path).name)
            dialog._image_path = path  # type: ignore

    pick_btn = QPushButton("Browse...")
    pick_btn.clicked.connect(pick_file)
    file_layout.addWidget(pick_btn)
    layout.addLayout(file_layout)

    # Pathology labels with sliders
    layout.addWidget(QLabel("Pathology confidence (0–100%):"))
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll_widget = QWidget()
    scroll_layout = QVBoxLayout(scroll_widget)

    _PATHOLOGIES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Effusion", "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis",
        "Fracture", "Hernia", "Infiltration", "Lung Lesion", "Lung Opacity",
        "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
    ]

    dialog._sliders = {}  # type: ignore
    dialog._checkboxes = {}  # type: ignore

    for path in _PATHOLOGIES:
        hbox = QVBoxLayout()
        cb = QCheckBox(path)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.setEnabled(False)
        cb.stateChanged.connect(lambda checked, s=slider: s.setEnabled(checked))
        slider.valueChanged.connect(lambda: None)  # updates live
        hbox.addWidget(cb)
        hbox.addWidget(slider)
        scroll_layout.addLayout(hbox)
        dialog._checkboxes[path] = cb  # type: ignore
        dialog._sliders[path] = slider  # type: ignore

    scroll.setWidget(scroll_widget)
    layout.addWidget(scroll)

    # Consent checkbox
    consent_cb = QCheckBox(
        "I consent to upload de-identified images for model training"
    )
    consent_cb.setChecked(False)
    layout.addWidget(consent_cb)

    # Buttons
    ok_btn = QPushButton("OK")
    ok_btn.setEnabled(False)
    cancel_btn = QPushButton("Cancel")

    def update_ok_button():
        has_image = hasattr(dialog, "_image_path") and dialog._image_path  # type: ignore
        has_labels = any(cb.isChecked() for cb in dialog._checkboxes.values())  # type: ignore
        has_consent = consent_cb.isChecked()
        ok_btn.setEnabled(has_image and has_labels and has_consent)

    for cb in dialog._checkboxes.values():  # type: ignore
        cb.stateChanged.connect(update_ok_button)
    consent_cb.stateChanged.connect(update_ok_button)

    def _on_ok():
        if not hasattr(dialog, "_image_path"):
            return
        labels = {
            path: float(dialog._sliders[path].value()) / 100.0  # type: ignore
            for path in _PATHOLOGIES if dialog._checkboxes[path].isChecked()  # type: ignore
        }
        record = ImageLabelRecord(
            image_path=dialog._image_path,  # type: ignore
            labels=labels,
            timestamp=datetime.now(timezone.utc).isoformat(),
            consent_flags={"image_labeling_consent": True},
        )
        try:
            collector = CorrectionCollector()
            success, msg = collector.capture_image_label(record)
            if success:
                window._show_status(msg, 3000)
                dialog.accept()
            else:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(dialog, "Error", msg)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(dialog, "Error", f"Failed to save: {exc}")

    ok_btn.clicked.connect(_on_ok)
    cancel_btn.clicked.connect(dialog.reject)

    btn_layout = QVBoxLayout()
    btn_layout.addStretch(1)
    btn_layout.addWidget(ok_btn)
    btn_layout.addWidget(cancel_btn)
    layout.addLayout(btn_layout)

    dialog._image_path = None  # type: ignore
    dialog.exec()
