"""Tests for the fine-tuned model registry and activation/rollback."""

from __future__ import annotations

import pytest

from src.cloud.model_registry import ModelRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    # Keep settings writes out of the way; activation mirrors into Settings.
    import src.core.settings as settings_module
    monkeypatch.setattr(settings_module, "settings_file",
                        lambda: tmp_path / "settings.json")
    return ModelRegistry(registry_path=tmp_path / "registry.json")


def test_register_does_not_activate(registry, tmp_path):
    model_dir = tmp_path / "v1"
    model_dir.mkdir()
    registry.register_downloaded_model("v1", model_dir, base_model="base")
    assert registry.get_active_version() is None
    assert registry.get_active_model_path() is None


def test_activate_and_resolve_path(registry, tmp_path):
    model_dir = tmp_path / "v1"
    model_dir.mkdir()
    registry.register_downloaded_model("v1", model_dir, base_model="base")
    assert registry.activate_model("v1") is True
    assert registry.get_active_model_path() == model_dir


def test_activate_unknown_version_fails(registry):
    assert registry.activate_model("ghost") is False


def test_rollback_to_base(registry, tmp_path):
    model_dir = tmp_path / "v1"
    model_dir.mkdir()
    registry.register_downloaded_model("v1", model_dir, base_model="base")
    registry.activate_model("v1")
    registry.rollback_to_base()
    assert registry.get_active_version() is None
    assert registry.get_active_model_path() is None


def test_missing_dir_self_heals(registry, tmp_path):
    model_dir = tmp_path / "v1"
    model_dir.mkdir()
    registry.register_downloaded_model("v1", model_dir, base_model="base")
    registry.activate_model("v1")
    # Simulate the model directory being deleted underneath us.
    model_dir.rmdir()
    assert registry.get_active_model_path() is None
    assert registry.get_active_version() is None  # rolled back


def test_list_available_newest_first(registry, tmp_path):
    for v in ("v1", "v2"):
        d = tmp_path / v
        d.mkdir()
        registry.register_downloaded_model(v, d, base_model="base")
    versions = [m.version for m in registry.list_available_models()]
    assert set(versions) == {"v1", "v2"}
