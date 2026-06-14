"""Shared pytest fixtures for isolated, deterministic regression tests."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from runtime_stubs import install_test_runtime_stubs

adaptive_learning, medical_dict = install_test_runtime_stubs()

import src.dictation.postprocess.medical_dict_match as _postprocess_dict  # noqa: E402
import src.core.settings as settings_module  # noqa: E402
import src.features.file_manager as file_manager  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset global singleton state and redirect filesystem writes to a tmp dir."""
    temp_data_dir = Path.cwd() / ".pytest_temp" / uuid4().hex
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    # Redirect data/ and settings file writes away from real project paths.
    # settings.py imports settings_file by name, so patch the binding inside
    # the settings module too.
    settings_path = temp_data_dir / "dictation_settings.json"
    monkeypatch.setattr(file_manager, "_data_dir", lambda: temp_data_dir)
    monkeypatch.setattr(file_manager, "settings_file", lambda: settings_path)
    monkeypatch.setattr(settings_module, "settings_file", lambda: settings_path)

    monkeypatch.setattr(
        adaptive_learning.AdaptiveLearning,
        "_default_data_dir",
        staticmethod(lambda: temp_data_dir),
    )
    adaptive_learning.AdaptiveLearning._instance = None
    adaptive_learning._adaptive_learning = None  # type: ignore

    medical_dict._TERMS = set()  # type: ignore
    medical_dict._COMMON_TERMS = []  # type: ignore
    medical_dict._FULL_TERMS_LIST = []  # type: ignore
    medical_dict._CORRECTION_TARGETS = []  # type: ignore
    _postprocess_dict._MEDICAL_TERMS_CACHE = set()  # type: ignore
