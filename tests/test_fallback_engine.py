"""The ASR chain: a provider that can never work is dropped, a flaky one is retried."""

import pytest

from src.dictation.asr.engines import fallback_engine
from src.dictation.asr.engines.fallback_engine import ChainEngine
from src.dictation.asr.types import AsrResult, ProviderUnavailable, TranscribeContext


@pytest.fixture(autouse=True)
def _reset_dead_providers():
    # _DEAD_PROVIDERS is process-lifetime by design (that's the point: one
    # rejection anywhere skips the provider everywhere); tests need a clean
    # slate so one test's dead key can't leak into the next.
    fallback_engine._DEAD_PROVIDERS.clear()
    yield
    fallback_engine._DEAD_PROVIDERS.clear()


class _Engine:
    def __init__(self, error=None):
        self.error, self.calls = error, 0

    def transcribe(self, audio, ctx):
        self.calls += 1
        if self.error:
            raise self.error
        return AsrResult(text="ok")


def test_dead_key_is_tried_once_then_skipped():
    dead, local = _Engine(ProviderUnavailable("401")), _Engine()
    chain = ChainEngine([("deepgram", dead), ("whisper", local)])
    for _ in range(3):
        assert chain.transcribe(None, TranscribeContext()).text == "ok"
    assert dead.calls == 1 and local.calls == 3


def test_network_blip_is_retried_next_decode():
    flaky, local = _Engine(ConnectionError("reset")), _Engine()
    chain = ChainEngine([("deepgram", flaky), ("whisper", local)])
    chain.transcribe(None, TranscribeContext())
    chain.transcribe(None, TranscribeContext())
    assert flaky.calls == 2


def test_last_provider_error_still_reaches_caller():
    chain = ChainEngine([("deepgram", _Engine(ProviderUnavailable("401")))])
    for _ in range(2):
        with pytest.raises(ProviderUnavailable):
            chain.transcribe(None, TranscribeContext())


def test_dead_key_skipped_across_separate_chain_instances():
    # web_app builds one ChainEngine per model name (live tier, polish tier);
    # a rejection learned by one instance must not be re-paid by the other.
    dead1, local1 = _Engine(ProviderUnavailable("401")), _Engine()
    live_chain = ChainEngine([("deepgram", dead1), ("whisper", local1)])
    assert live_chain.transcribe(None, TranscribeContext()).text == "ok"
    assert dead1.calls == 1

    dead2, local2 = _Engine(ProviderUnavailable("401")), _Engine()
    polish_chain = ChainEngine([("deepgram", dead2), ("whisper", local2)])
    assert polish_chain.transcribe(None, TranscribeContext()).text == "ok"
    assert dead2.calls == 0  # never called: learned dead by the other instance
