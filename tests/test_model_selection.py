"""Which model the app actually loads for a given `model_size` setting.

This had a silent failure that cost real accuracy: `SUPPORTED_MODELS` listed only
the multilingual sizes, so a settings file naming `small.en` matched nothing. The
web app reassigned it to a hardcoded fallback and the desktop combo box ignored
`setCurrentText` for a value it had no item for, staying on its first entry —
`tiny`. Neither said a word. These tests pin both halves: the English-only models
are selectable, and an unknown name falls back *loudly*.
"""

from __future__ import annotations

import logging

import pytest

from src.core.settings import _DEFAULTS
from src.dictation.transcriber import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    resolve_model,
)


class TestSupportedModels:
    @pytest.mark.parametrize("name", ["tiny.en", "base.en", "small.en", "medium.en"])
    def test_english_only_models_are_selectable(self, name: str) -> None:
        # They are faster *and* more accurate than the same-size multilingual model
        # for English dictation, so they are the ones worth defaulting to.
        assert name in SUPPORTED_MODELS
        assert resolve_model(name) == name

    def test_the_default_is_itself_supported(self) -> None:
        assert DEFAULT_MODEL in SUPPORTED_MODELS

    def test_the_shipped_setting_is_a_model_that_exists(self) -> None:
        # The exact defect: dictation_settings.json named a model no picker offered.
        assert _DEFAULTS["model_size"] in SUPPORTED_MODELS


class TestResolveModel:
    def test_an_unknown_model_falls_back_and_says_so(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            resolved = resolve_model("small.en-turbo-v9")
        assert resolved == DEFAULT_MODEL
        assert "small.en-turbo-v9" in caplog.text

    @pytest.mark.parametrize("missing", [None, "", "  "])
    def test_a_missing_setting_falls_back(self, missing) -> None:
        assert resolve_model(missing) == DEFAULT_MODEL


class TestDesktopPicker:
    def test_the_combo_reports_the_model_the_settings_asked_for(self) -> None:
        # The original bug in one assertion: build the picker exactly as views.py
        # does, hand it the saved setting, and read back what recording_session
        # would pass to the worker.
        pytest.importorskip("PySide6.QtWidgets")
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QComboBox

        app = QApplication.instance() or QApplication([])
        combo = QComboBox()
        combo.addItems(SUPPORTED_MODELS)
        combo.setCurrentText(resolve_model("small.en"))

        assert combo.currentText() == "small.en"
        assert app is not None  # keep the instance alive for the assertion above
