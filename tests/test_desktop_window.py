"""The real desktop window, driven offscreen (fixture: conftest.desktop)."""

import pytest

pytest.importorskip("PySide6")


def test_every_menu_has_its_items_and_shortcuts(desktop):
    menus = {m.title().replace("&", ""): [a.text() for a in m.actions() if not a.isSeparator()]
             for m in (a.menu() for a in desktop.menuBar().actions()) if m is not None}
    for name in ("File", "Report", "View", "Tools", "Settings"):
        assert menus.get(name), f"{name} menu is empty: {menus}"
    shortcuts = {a.shortcut().toString() for m in desktop.menuBar().findChildren(type(desktop.menuBar().actions()[0]))
                 for a in [m] if not a.shortcut().isEmpty()}
    assert {"Ctrl+S", "Ctrl+N", "Ctrl+T"} <= shortcuts


from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui import main_window as mw  # noqa: E402
from src.ui import recording_session as rs  # noqa: E402


def _dictate(w, *partials):
    for text in partials:
        w.worker().partial.emit(text, 0)
        assert w.wait_for(lambda: text.lower() in w.editor.toPlainText().lower())


def _stop_and_hand_back(w):
    rs.on_stop_recording(w)
    assert w.wait_for(w.btn_record.isEnabled)


def test_record_again_during_the_polish_retires_the_old_session(desktop):
    rs.on_start_recording(desktop)
    old = desktop.worker()
    _dictate(desktop, "Small left pleural effusion.")
    _stop_and_hand_back(desktop)
    rs.on_start_recording(desktop)               # used to abort the process
    new = desktop.worker()
    assert old.cancelled and new is not old
    old.partial.emit("OLD POLISH OVERWRITES", 0)
    old.finished.emit()
    _dictate(desktop, "No acute abnormality.")
    desktop.pump()
    assert "OLD POLISH" not in desktop.editor.toPlainText()
    assert desktop.recorder.is_recording and desktop.btn_stop.isEnabled()


def test_record_right_after_stop_does_not_crash(desktop):
    rs.on_start_recording(desktop)
    _dictate(desktop, "No fracture.")
    rs.on_stop_recording(desktop)
    rs.on_start_recording(desktop)               # F5 before the hand-back landed
    _dictate(desktop, "Normal study.")
    assert desktop.recorder.is_recording


def test_a_polish_landing_does_not_move_the_caret(desktop):
    rs.on_start_recording(desktop)
    _dictate(desktop, "No fracture.")
    _stop_and_hand_back(desktop)
    desktop.editor.insertPlainText("")           # untouched report
    cursor = desktop.editor.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(2, cursor.MoveMode.KeepAnchor)
    desktop.editor.setTextCursor(cursor)
    desktop.worker().partial.emit("No fracture seen.", 0)   # the polish's update
    assert desktop.wait_for(lambda: "seen" in desktop.editor.toPlainText())
    selection = desktop.editor.textCursor()
    assert (selection.anchor(), selection.position()) == (0, 2)


def test_a_quick_phrase_during_recording_survives_the_next_update(desktop):
    rs.on_start_recording(desktop)
    _dictate(desktop, "No fracture.")
    desktop._insert_macro("NORMAL KNEE")
    _dictate(desktop, "No fracture. Small effusion.")
    text = desktop.editor.toPlainText()
    assert "NORMAL KNEE" in text and "Small effusion." in text


def test_nothing_leaves_while_recording(desktop, monkeypatch):
    QApplication.clipboard().setText("before")
    rs.on_start_recording(desktop)
    _dictate(desktop, "No fracture.")
    desktop.on_copy()
    assert QApplication.clipboard().text() == "before"


def test_copying_a_finding_out_of_the_editor_goes_through_the_gate(desktop, monkeypatch):
    asked = []
    monkeypatch.setattr(mw, "confirm_release", lambda w: asked.append(True) or False)
    QApplication.clipboard().setText("before")
    desktop.replace_report("Tension pneumothorax.")
    desktop.editor.selectAll()
    desktop.editor.copy()
    assert asked and "pneumothorax" not in QApplication.clipboard().text()
    desktop.replace_report("Normal study.")       # nothing to ask about: a plain copy
    desktop.editor.selectAll()
    desktop.editor.copy()
    assert QApplication.clipboard().text() == "Normal study." and len(asked) == 1


def test_clear_and_template_load_can_be_undone(desktop):
    desktop.replace_report("A report that took ten minutes.")
    desktop.on_clear()
    assert desktop.editor.toPlainText() == ""
    desktop.editor.undo()
    assert desktop.editor.toPlainText() == "A report that took ten minutes."


def test_closing_mid_recording_keeps_the_report(desktop, tmp_path):
    rs.on_start_recording(desktop)
    _dictate(desktop, "Tension pneumothorax on the right.")
    desktop.close()
    saved = list((tmp_path / "data").rglob("*.txt"))
    assert any("Tension pneumothorax" in p.read_text() for p in saved)


def test_a_finding_already_communicated_is_not_asked_again(desktop, monkeypatch, tmp_path):
    from datetime import datetime
    from types import SimpleNamespace

    from PySide6.QtWidgets import QDialog

    from src.features import audit_log
    from src.features.report_release import Communication
    monkeypatch.setattr(audit_log, "_log_path", tmp_path / "audit.log")
    asked = []

    class Dialog:
        def __init__(self, _parent, check, _name):
            asked.append([f.term for f in check.findings])
            self.add_to_report = SimpleNamespace(isChecked=lambda: True)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def communication(self):
            return Communication("Dr Patel", "telephone", datetime(2026, 10, 24, 10, 0), "")

    monkeypatch.setattr(rs, "CommunicationDialog", Dialog)
    monkeypatch.setattr(rs, "confirm_report_checks", lambda _w: True)
    desktop.editor.setPlainText("Large right pneumothorax.")
    assert rs.confirm_release(desktop) and rs.confirm_release(desktop)
    desktop.editor.setPlainText(desktop.editor.toPlainText() + "\nPneumoperitoneum.")
    assert rs.confirm_release(desktop)
    assert asked == [["pneumothorax"], ["Pneumoperitoneum"]]
    assert desktop.editor.toPlainText().count("communicated to Dr Patel") == 2
