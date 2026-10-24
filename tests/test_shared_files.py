"""The desktop app and the web app share one data folder; neither undoes the other.

Each used to read a file, change its copy and write the whole thing back, so
whichever saved second silently threw away what the first had just saved.
"""

import json
import multiprocessing
import subprocess
import sys
import textwrap
from pathlib import Path

from src.core import json_store
from src.core.file_lock import FileLock
from src.features import adaptive_learning as al

WRITERS, WRITES = 6, 15


def _write_many(path, writer):
    for i in range(WRITES):
        json_store.update_json(path, {}, lambda d: d.update({f"{writer}-{i}": i}))


def test_concurrent_processes_lose_no_update(tmp_path):
    path = tmp_path / "shared.json"
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_write_many, args=(path, w)) for w in range(WRITERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert len(json.loads(path.read_text())) == WRITERS * WRITES


def test_a_lock_is_exclusive_and_dies_with_its_holder(tmp_path):
    path = tmp_path / "x.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys, time
            from src.core.file_lock import FileLock
            FileLock({str(path)!r}).acquire()
            print("held", flush=True)
            time.sleep(60)
        """)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert not FileLock(path).acquire(timeout=0)
    finally:
        holder.kill()
        holder.wait()
    lock = FileLock(path)
    assert lock.acquire(timeout=5)  # killed, not released: the OS let go
    lock.release()


def _instance(tmp_path, monkeypatch):
    monkeypatch.setattr(al.AdaptiveLearning, "_instance", None)
    instance = al.AdaptiveLearning(data_dir=tmp_path)
    monkeypatch.setattr(al.AdaptiveLearning, "_instance", None)
    return instance


def test_two_apps_learning_side_by_side_keep_both(tmp_path, monkeypatch):
    desktop, web = _instance(tmp_path, monkeypatch), _instance(tmp_path, monkeypatch)
    desktop.learn_correction("menscus", "meniscus")
    desktop.record_term_usage("effusion")
    web.learn_correction("labrom", "labrum")
    web.record_term_usage("effusion")
    desktop.force_save()
    web.force_save()
    saved = json.loads((tmp_path / "learned_corrections.json").read_text())
    assert saved["word_corrections"] == {"menscus": "meniscus", "labrom": "labrum"}
    assert saved["term_frequency"] == {"effusion": 2}
    # The second to save now knows what the first learned.
    assert web.export_corrections()["menscus"] == "meniscus"


def test_a_failed_save_keeps_what_was_learned(tmp_path, monkeypatch):
    learning = _instance(tmp_path, monkeypatch)
    learning.learn_correction("menscus", "meniscus")
    write = json_store.write_json
    monkeypatch.setattr(json_store, "write_json", lambda *_a: False)
    learning.force_save()
    monkeypatch.setattr(json_store, "write_json", write)
    learning.force_save()
    saved = json.loads((tmp_path / "learned_corrections.json").read_text())
    assert saved["word_corrections"] == {"menscus": "meniscus"}


def test_two_settings_instances_keep_both_changes(tmp_path, monkeypatch):
    from src.core import settings as core_settings
    monkeypatch.setattr(core_settings, "settings_file", lambda: tmp_path / "s.json")
    desktop, web = core_settings.Settings(), core_settings.Settings()
    desktop.set("theme", "dark")
    web.set("accent", "south_asian")
    on_disk = json.loads((tmp_path / "s.json").read_text())
    assert on_disk == {"theme": "dark", "accent": "south_asian"}


def test_a_settings_save_that_fails_stays_pending(tmp_path, monkeypatch):
    from src.core import settings as core_settings
    monkeypatch.setattr(core_settings, "settings_file", lambda: tmp_path / "s.json")
    s = core_settings.Settings()
    write = json_store.write_json
    monkeypatch.setattr(json_store, "write_json", lambda *_a: False)
    s.set("theme", "dark")
    monkeypatch.setattr(json_store, "write_json", write)
    s.set("accent", "south_asian")
    on_disk = json.loads((tmp_path / "s.json").read_text())
    assert on_disk == {"theme": "dark", "accent": "south_asian"}


def _start_app(data_dir):
    """Another app over the same data folder: takes its temp folder, then waits."""
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import pathlib, time
            from src.core import paths
            paths._data_dir = lambda: pathlib.Path({str(data_dir)!r})
            (paths.temp_dir() / "recording.wav").write_bytes(b"RIFF")
            print(paths.temp_dir(), flush=True)
            time.sleep(60)
        """)], stdout=subprocess.PIPE, text=True)


def test_starting_an_app_never_deletes_another_apps_recording(tmp_path, monkeypatch):
    from src.core import paths
    monkeypatch.setattr(paths, "_data_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "_temp_session", None)
    desktop = _start_app(tmp_path)
    try:
        recording = Path(desktop.stdout.readline().strip()) / "recording.wav"
        mine = paths.temp_dir() / "mine.wav"
        mine.write_bytes(b"RIFF")
        paths.cleanup_temp_files()  # what the web app does when it starts
        assert recording.exists() and mine.exists()
    finally:
        desktop.kill()
        desktop.wait()
    paths.cleanup_temp_files()  # the desktop has exited: its folder goes
    assert not recording.parent.exists() and mine.exists()
