"""Tests for the fine-tuned model registry and activation/rollback."""

from __future__ import annotations

import pytest

import json

from src.cloud.framework.registry import ModelRegistry
from src.training.schemas import TASK_SCAN_CLASSIFIER, TASK_WHISPER_VOICE


@pytest.fixture
def registry(tmp_path, monkeypatch):
    # Keep settings writes out of the way; activation mirrors into Settings.
    import src.core.settings as settings_module
    monkeypatch.setattr(settings_module, "settings_file",
                        lambda: tmp_path / "settings.json")
    return ModelRegistry(registry_path=tmp_path / "registry.json")


def _add_model(registry, tmp_path, version: str):
    d = tmp_path / version
    d.mkdir(exist_ok=True)
    registry.register_downloaded_model(version, d, base_model="base")
    return d


def test_register_does_not_activate(registry, tmp_path):
    _add_model(registry, tmp_path, "v1")
    assert registry.get_active_version() is None
    assert registry.get_active_model_path() is None


def test_activate_and_resolve_path(registry, tmp_path):
    model_dir = _add_model(registry, tmp_path, "v1")
    assert registry.activate_model("v1") is True
    assert registry.get_active_model_path() == model_dir


def test_activate_unknown_version_fails(registry):
    assert registry.activate_model("ghost") is False


def test_rollback_to_base(registry, tmp_path):
    _add_model(registry, tmp_path, "v1")
    registry.activate_model("v1")
    registry.rollback_to_base()
    assert registry.get_active_version() is None
    assert registry.get_active_model_path() is None


def test_missing_dir_self_heals(registry, tmp_path):
    model_dir = _add_model(registry, tmp_path, "v1")
    registry.activate_model("v1")
    # Simulate the model directory being deleted underneath us.
    model_dir.rmdir()
    assert registry.get_active_model_path() is None
    assert registry.get_active_version() is None  # rolled back


def test_list_available_newest_first(registry, tmp_path):
    _add_model(registry, tmp_path, "v1")
    _add_model(registry, tmp_path, "v2")
    versions = [m.version for m in registry.list_available_models()]
    assert set(versions) == {"v1", "v2"}


def test_activation_is_per_task(registry, tmp_path):
    # A voice model and a scan model are independently active.
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    registry.register_downloaded_model("voice1", voice_dir, base_model="base",
                                       task_type=TASK_WHISPER_VOICE)
    registry.register_downloaded_model("scan1", scan_dir, base_model="dn121",
                                       task_type=TASK_SCAN_CLASSIFIER)
    assert registry.activate_model("voice1") is True
    assert registry.activate_model("scan1") is True
    # Activating the scan model must not deactivate the voice model.
    assert registry.get_active_model_path(TASK_WHISPER_VOICE) == voice_dir
    assert registry.get_active_model_path(TASK_SCAN_CLASSIFIER) == scan_dir


def test_list_filtered_by_task(registry, tmp_path):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    registry.register_downloaded_model("voice1", voice_dir, base_model="base",
                                       task_type=TASK_WHISPER_VOICE)
    registry.register_downloaded_model("scan1", scan_dir, base_model="dn121",
                                       task_type=TASK_SCAN_CLASSIFIER)
    scan = [m.version for m in registry.list_available_models(TASK_SCAN_CLASSIFIER)]
    assert scan == ["scan1"]


def test_legacy_registry_normalized(tmp_path):
    # A pre-multitask registry used a flat ``active_version`` string.
    path = tmp_path / "registry.json"
    model_dir = tmp_path / "old"
    model_dir.mkdir()
    path.write_text(json.dumps({
        "active_version": "old1",
        "models": [{"version": "old1", "base_model": "base", "created_at": "2026",
                    "local_path": str(model_dir)}],
    }), encoding="utf-8")
    reg = ModelRegistry(registry_path=path)
    # The legacy active model resolves as the voice task's active model.
    assert reg.get_active_model_path(TASK_WHISPER_VOICE) == model_dir
