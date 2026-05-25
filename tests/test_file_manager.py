"""Path resolution and cleanup behaviour for src.features.file_manager."""

from __future__ import annotations

import time
from pathlib import Path


import src.features.file_manager as fm


class TestPathResolution:
    """Path helpers should anchor at the real project root, not src/."""

    def test_project_root_contains_src_directory(self) -> None:
        root = fm._project_root()
        assert (root / "src").is_dir()

    def test_medical_wordlist_path_points_at_bundled_resource(self) -> None:
        path = fm.medical_wordlist_path()
        assert path.name == "medical_terms.txt"
        assert path.parent.name == "resources"
        assert path.parent.parent.name == "src"

    def test_templates_dir_resolves_under_src(self) -> None:
        path = fm.templates_dir()
        assert path.name == "templates"
        assert path.parent.name == "src"


class TestDirectoryCreation:
    """Lazy directory helpers should create their target on demand."""

    def test_temp_dir_is_created(self) -> None:
        path = fm.temp_dir()
        assert path.is_dir()
        assert path.name == "temp"

    def test_autosave_dir_is_created(self) -> None:
        path = fm.autosave_dir()
        assert path.is_dir()
        assert path.name == "autosave"

    def test_resources_dir_is_created(self) -> None:
        path = fm.resources_dir()
        assert path.is_dir()
        assert path.name == "resources"


class TestTempWavLifecycle:
    """create_temp_wav and cleanup_temp_files should manage their own folder."""

    def test_create_temp_wav_returns_path_under_temp_dir(self) -> None:
        path = Path(fm.create_temp_wav())
        assert path.parent == fm.temp_dir()
        assert path.name.startswith("recording_")
        assert path.suffix == ".wav"

    def test_cleanup_temp_files_removes_existing_files(self) -> None:
        path = fm.temp_dir() / "leftover.wav"
        path.write_text("data")
        assert path.exists()

        fm.cleanup_temp_files()

        assert not path.exists()


class TestAutosaveCleanup:
    """Old autosave files should be deleted, recent ones kept."""

    def test_old_files_are_deleted(self) -> None:
        old = fm.autosave_dir() / "old.txt"
        old.write_text("old")
        # Backdate to 60 days ago
        ts = time.time() - 60 * 86400
        import os
        os.utime(old, (ts, ts))

        fm.cleanup_old_autosaves(retention_days=30)

        assert not old.exists()

    def test_recent_files_are_kept(self) -> None:
        new = fm.autosave_dir() / "fresh.txt"
        new.write_text("fresh")

        fm.cleanup_old_autosaves(retention_days=30)

        assert new.exists()
