"""Which Parakeet runs, and the files the hotword engine derives for itself.

No model is loaded here: the factory's choice and the two derived files are
the parts that can be wrong without anything failing loudly (a wrong
``bpe.vocab`` splits hotwords into the wrong pieces and the boost silently
does nothing).
"""

import importlib.util

import pytest

from src.core.settings import get_default
from src.dictation.asr import factory
from src.dictation.asr.engines import parakeet_engine
from src.dictation.asr.engines.parakeet_engine import ParakeetEngine
from src.dictation.asr.engines.parakeet_hotword_engine import (
    ParakeetHotwordEngine,
    write_bpe_vocab,
    write_hotwords,
)


def test_bpe_vocab_scores_are_the_pieces_ranks(tmp_path):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<unk> 0\n▁t 1\n▁th 2\nin 3\n<blk> 4\n", encoding="utf-8")
    write_bpe_vocab(tokens, tmp_path / "bpe.vocab")
    assert (tmp_path / "bpe.vocab").read_text(encoding="utf-8").splitlines() == [
        "<unk>\t0", "▁t\t-0", "▁th\t-1", "in\t-2",
    ]


def test_hotwords_are_lower_case_and_unique(tmp_path):
    n = write_hotwords(["Pneumothorax", "pneumothorax", "Bone  marrow oedema", ""], tmp_path / "h.txt")
    assert n == 2
    assert (tmp_path / "h.txt").read_text().splitlines() == ["bone marrow oedema", "pneumothorax"]


class _Settings(dict):
    def get(self, key, default=None):
        return super().get(key, get_default(key))


def _a_fine_tune(monkeypatch, tmp_path):
    """One fine-tuned Parakeet installed, named "my-tune"."""
    (tmp_path / "my-tune").mkdir()
    (tmp_path / "my-tune" / "encoder-model.onnx").write_bytes(b"")
    monkeypatch.setattr(parakeet_engine, "fine_tuned_dir", lambda: tmp_path)
    return "my-tune"


def _tier(monkeypatch, installed, **settings):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        factory.importlib.util, "find_spec",
        lambda name: object() if name in installed else (None if name in ("onnx_asr", "sherpa_onnx") else real(name)),
    )
    return factory.parakeet_tier(_Settings({"asr_hotwords": True, "hotword_score": 1.0, **settings}))


def test_hotwords_engine_when_sherpa_is_installed(monkeypatch):
    name, engine = _tier(monkeypatch, {"onnx_asr", "sherpa_onnx"})
    assert (name, type(engine)) == ("parakeet-hotwords", ParakeetHotwordEngine)


def test_greedy_engine_when_hotwords_are_off_or_unavailable(monkeypatch):
    assert _tier(monkeypatch, {"onnx_asr", "sherpa_onnx"}, asr_hotwords=False)[0] == "parakeet"
    assert _tier(monkeypatch, {"onnx_asr"})[0] == "parakeet"


def test_a_fine_tuned_export_wins(monkeypatch, tmp_path):
    tune = _a_fine_tune(monkeypatch, tmp_path)
    name, engine = _tier(monkeypatch, {"onnx_asr", "sherpa_onnx"}, parakeet_model=tune)
    assert name == "parakeet" and isinstance(engine, ParakeetEngine) and engine.model_dir == tmp_path / tune


def test_no_parakeet_without_either_runtime(monkeypatch):
    assert _tier(monkeypatch, set()) is None


def test_engine_label_names_what_actually_answers(monkeypatch, tmp_path):
    real = importlib.util.find_spec
    monkeypatch.setattr(factory.importlib.util, "find_spec",
                        lambda name: object() if name in ("onnx_asr", "sherpa_onnx") else real(name))
    base = {"asr_engine": "local", "asr_hotwords": True, "hotword_score": 1.0, "model_size": "base.en"}
    assert factory.describe_engine(_Settings(base)) == "Parakeet + radiology terms"
    assert factory.describe_engine(_Settings({**base, "asr_hotwords": False})) == "Parakeet"
    tune = _a_fine_tune(monkeypatch, tmp_path)
    assert factory.describe_engine(_Settings({**base, "parakeet_model": tune})) == "Parakeet, tuned: my-tune"
    assert factory.describe_engine(_Settings({**base, "asr_engine": "deepgram"})) == "Deepgram (cloud)"


def test_a_voice_that_is_no_longer_installed_falls_back_and_says_so(monkeypatch, tmp_path):
    # A deleted fine-tune folder must not stop dictation or the page loading.
    monkeypatch.setattr(parakeet_engine, "fine_tuned_dir", lambda: tmp_path)
    name, _ = _tier(monkeypatch, {"onnx_asr", "sherpa_onnx"}, parakeet_model="deleted-tune")
    assert name == "parakeet-hotwords"
    base = {"asr_engine": "local", "asr_hotwords": True, "model_size": "base.en", "parakeet_model": "deleted-tune"}
    assert factory.describe_engine(_Settings(base)) == "Parakeet + radiology terms (deleted-tune not installed)"


class _Saved(_Settings):
    def set(self, key, value):
        self[key] = value


def test_a_voice_that_fails_to_load_leaves_the_previous_one_chosen(monkeypatch, tmp_path):
    from src.dictation.asr import voice_model

    tune = _a_fine_tune(monkeypatch, tmp_path)

    class _Broken:
        def preload(self):
            raise RuntimeError("corrupt weights")

    monkeypatch.setattr(voice_model, "parakeet_tier", lambda settings: ("parakeet", _Broken()))
    settings = _Saved({"parakeet_model": "standard"})
    with pytest.raises(RuntimeError):
        voice_model.switch(settings, tune)
    assert settings["parakeet_model"] == "standard"
def test_a_word_confidence_must_be_a_probability():
    from src.dictation.asr.types import Word
    import pytest as _pytest
    with _pytest.raises(ValueError):
        Word("pneumothorax", 0.0, 1.0, 1.67)       # a hotword-boosted beam score
    assert Word("pneumothorax", 0.0, 1.0, None).confidence is None


def test_pieces_without_probabilities_give_words_without_confidence():
    from src.dictation.asr.engines.parakeet_engine import merge_pieces
    from src.dictation.asr.types import AsrSegment
    words = merge_pieces([" no", " eff", "usion"], [0.0, 0.4, 0.6], None)
    assert [w.text for w in words] == ["no", "effusion"]
    assert all(w.confidence is None for w in words)
    assert AsrSegment("no effusion", 0.0, 1.0, words).confidence is None


def test_the_hotword_engine_claims_no_word_confidence():
    assert ParakeetHotwordEngine().capabilities().word_confidence is False


def test_audio_shorter_than_the_encoder_input_decodes_to_nothing():
    import numpy as np
    from src.dictation.asr.engines.parakeet_engine import MIN_SAMPLES, _as_input
    assert _as_input(np.zeros(MIN_SAMPLES - 1, np.float32)) is None
    assert _as_input(np.zeros(MIN_SAMPLES, np.float32)) is not None
