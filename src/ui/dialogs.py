"""Dialog windows and modal interactions."""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

from src.features.adaptive_learning import get_adaptive_learning

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)

_UNFILLED_FIELD_RE = re.compile(r"\[([A-Z][A-Z0-9 _/-]{1,40})\]|\{\{([^}]{1,40})\}\}")


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
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    enabled = reply == QMessageBox.Yes
    window.settings.set("learning_enabled", enabled)
    window.settings.set("learning_consent_shown", True)
    from src.features import audit_log
    audit_log.log_learning_consent(enabled)


def show_cloud_training_consent_dialog(window: MainWindow) -> bool:
    """Ask the user to opt in to cloud fine-tuning. Returns the granted state.

    Unlike on-device learning, this uploads de-identified audio + text to
    Lightning AI, so consent is explicit and revocable. Persists both the
    consent flag and the master ``cloud_enabled`` switch.
    """
    reply = QMessageBox.question(
        window,
        "Cloud Voice Training — Your Consent",
        "To build a personalised voice model, this app can upload your "
        "corrections — de-identified audio snippets and text — to Lightning AI "
        "for fine-tuning.\n\n"
        "• Patient identifiers (names, IDs, dates) are removed before upload.\n"
        "• Spoken patient identifiers are silenced in audio clips.\n"
        "• Records that fail the privacy check are dropped, never uploaded.\n"
        "• You can disable this at any time; the app always works offline.\n\n"
        "Enable cloud voice training?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    enabled = reply == QMessageBox.Yes
    window.settings.batch_set({
        "cloud_enabled": enabled,
        "cloud_training_consent": enabled,
    })
    from src.features import audit_log
    audit_log.log_cloud_consent(enabled)
    return enabled


def show_model_update_notification(window: MainWindow, version: str) -> None:
    """Offer to activate a freshly-downloaded fine-tuned model."""
    reply = QMessageBox.question(
        window,
        "Personalised Model Ready",
        f"A new personalised voice model ({version}) has finished training and "
        "downloaded successfully.\n\n"
        "Activate it for future dictation? You can revert to the base model "
        "any time from settings.",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    if reply == QMessageBox.Yes:
        try:
            from src.cloud.model_registry import ModelRegistry
            if ModelRegistry().activate_model(version):
                window._show_status(f"Activated personalised model {version}", 4000)
            else:
                QMessageBox.warning(window, "Activation Failed",
                                    f"Could not activate model {version}.")
        except Exception as exc:
            QMessageBox.warning(window, "Error", f"Activation failed: {exc}")


def show_disclaimer_if_needed(window: MainWindow) -> None:
    """Show clinical disclaimer dialog on first launch."""
    if window.settings.get("disclaimer_shown", False):
        return
    QMessageBox.information(
        window,
        "Clinical Disclaimer",
        "DISCLAIMER — IMPORTANT\n\n"
        "This tool uses OpenAI Whisper for speech recognition. Whisper is a "
        "general-purpose model and is not FDA-cleared or CE-marked for clinical "
        "medical documentation.\n\n"
        "All transcriptions MUST be reviewed and verified by a qualified "
        "radiologist before clinical use or patient record entry.\n\n"
        "The software author accepts no liability for transcription errors.",
    )
    window.settings.set("disclaimer_shown", True)


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
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    return reply == QMessageBox.Yes


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

        if corrections:
            msg += "\nRecent corrections:\n"
            for i, (wrong, correct) in enumerate(list(corrections.items())[-10:]):
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
        QMessageBox.Yes | QMessageBox.No,
    )
    if reply == QMessageBox.Yes:
        try:
            get_adaptive_learning().reset()
            from src.features import audit_log
            audit_log.log_learning_reset()
            window._show_status("Learning data reset", 2000)
        except Exception as exc:
            QMessageBox.warning(window, "Error", f"Reset failed: {exc}")
