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

    def test_cache_dir_is_created(self) -> None:
        path = fm.cache_dir()
        assert path.is_dir()
        assert path.name == "cache"

    def test_cache_paths_live_under_cache_dir(self) -> None:
        cache = fm.cache_dir()
        assert fm.medical_dict_cache_path().parent == cache
        assert fm.whisper_cache_dir().parent == cache
        assert fm.imaging_embeddings_dir().parent == cache


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


class TestLegacyCacheCleanup:
    """Startup cleanup removes pre-data/cache/ homes and nothing else."""

    def test_legacy_dirs_removed_and_siblings_untouched(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        data = tmp_path / "data"
        legacy_resources = data / "resources"
        legacy_embeddings = data / "imaging" / "embeddings"
        legacy_resources.mkdir(parents=True)
        legacy_embeddings.mkdir(parents=True)
        (legacy_resources / "medical_symspell.pkl").write_bytes(b"stale")
        (legacy_embeddings / "embeddings.npz").write_bytes(b"stale")
        # Non-derived neighbours that must survive.
        datasets = data / "imaging" / "datasets.json"
        datasets.write_text("{}")
        autosave = data / "autosave"
        autosave.mkdir()
        (autosave / "report.txt").write_text("keep")

        monkeypatch.setattr(fm, "_data_dir", lambda: data)
        fm._remove_legacy_cache_locations()

        assert not legacy_resources.exists()
        assert not legacy_embeddings.exists()
        assert datasets.read_text() == "{}"
        assert (autosave / "report.txt").read_text() == "keep"

    def test_noop_when_no_legacy_dirs(self, tmp_path: Path, monkeypatch) -> None:
        data = tmp_path / "data"
        data.mkdir()
        monkeypatch.setattr(fm, "_data_dir", lambda: data)
        fm._remove_legacy_cache_locations()  # must not raise
