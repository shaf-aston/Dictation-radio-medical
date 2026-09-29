"""The polish after Stop only runs when the accurate engine is a different model."""

import numpy as np

from src.dictation.asr.engines import fallback_engine
from src.dictation.asr.engines.deepgram_engine import DeepgramEngine
from src.dictation.asr.engines.fallback_engine import ChainEngine
from src.dictation.asr.engines.faster_whisper_engine import FasterWhisperEngine
from src.dictation.asr.port import engine_identity
from src.dictation.asr.types import AsrResult, AsrSegment, EngineCaps, Word
from src.dictation.stream.live_session import LiveSession
from src.dictation.stream.segmenter import Chunk


def _chain(size):
    return ChainEngine([("deepgram", DeepgramEngine()), ("faster-whisper", FasterWhisperEngine(model_size=size))])


def test_two_deepgram_led_chains_are_one_model(monkeypatch):
    fallback_engine.reset_provider_health()
    monkeypatch.setattr(DeepgramEngine, "usable", lambda self: True)
    assert engine_identity(_chain("tiny.en")) == engine_identity(_chain("small.en"))


def test_without_a_key_the_local_tiers_differ(monkeypatch):
    fallback_engine.reset_provider_health()
    monkeypatch.setattr(DeepgramEngine, "usable", lambda self: False)
    assert engine_identity(_chain("tiny.en")) != engine_identity(_chain("small.en"))
    fallback_engine.reset_provider_health()


class _Engine:
    def __init__(self, name, network=True):
        self.name, self.network, self.calls = name, network, 0

    def capabilities(self):
        return EngineCaps(word_confidence=True, hotwords=False, network=self.network)

    def identity(self):
        return self.name

    def transcribe(self, audio, ctx):
        self.calls += 1
        w = (Word("better", 0.0, 1.0, 0.99),)
        return AsrResult("better", (AsrSegment("better", 0.0, 1.0, w),))


def _session(live, final):
    s = LiveSession(live, final, cleanup_level="soft")
    s.feed(np.zeros(16000 * 4, dtype=np.float32))
    # One committed chunk the live model was unsure of.
    s._ledger.commit(Chunk(0, 16000 * 4, closed=True), "worse", 0.3)
    return s


def test_same_model_skips_the_polish():
    live = _Engine("deepgram")
    final = _Engine("deepgram")
    s = _session(live, final)
    s.finalize()
    assert final.calls == 0
    assert "worse" in s._ledger.committed_text


def test_better_model_still_polishes_weak_chunks():
    live, final = _Engine("tiny"), _Engine("small")
    s = _session(live, final)
    s.finalize()
    assert final.calls == 1
    assert s._ledger.committed_text == "better"


def test_same_local_model_still_polishes_at_the_wider_beam():
    live, final = _Engine("tiny", network=False), _Engine("tiny", network=False)
    s = _session(live, final)
    s.finalize()
    assert final.calls == 1
