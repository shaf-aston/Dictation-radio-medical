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
    fallback_engine.reset_provider_health()
    yield
    fallback_engine.reset_provider_health()


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


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_outage_trips_the_breaker_then_probes(monkeypatch):
    # A network outage must not put a deadline in front of every decode:
    # after TRIP_AFTER failures in a row the provider is skipped for a while.
    clock = _Clock()
    monkeypatch.setattr(fallback_engine, "_clock", clock)
    down, local = _Engine(TimeoutError("read timed out")), _Engine()
    chain = ChainEngine([("deepgram", down), ("whisper", local)])
    for _ in range(5):
        assert chain.transcribe(None, TranscribeContext()).text == "ok"
    assert down.calls == fallback_engine.TRIP_AFTER
    assert local.calls == 5

    # Window over: one probe. Still down, so it re-trips on that one failure.
    clock.now += fallback_engine.TRIP_SEC + 1
    chain.transcribe(None, TranscribeContext())
    chain.transcribe(None, TranscribeContext())
    assert down.calls == fallback_engine.TRIP_AFTER + 1

    # Back up: the next probe succeeds and the provider is used again.
    clock.now += fallback_engine.TRIP_SEC + 1
    down.error = None
    chain.transcribe(None, TranscribeContext())
    chain.transcribe(None, TranscribeContext())
    assert down.calls == fallback_engine.TRIP_AFTER + 3


def test_success_resets_the_failure_count():
    flaky, local = _Engine(ConnectionError("reset")), _Engine()
    chain = ChainEngine([("deepgram", flaky), ("whisper", local)])
    for _ in range(4):
        flaky.error = ConnectionError("reset")
        chain.transcribe(None, TranscribeContext())
        flaky.error = None
        chain.transcribe(None, TranscribeContext())
    # Never two failures in a row, so it was never skipped.
    assert flaky.calls == 8


class _CapsEngine(_Engine):
    def __init__(self, caps, preload_error=None):
        super().__init__()
        self.caps, self.preload_error = caps, preload_error

    def capabilities(self):
        return self.caps

    def preload(self):
        if self.preload_error:
            raise self.preload_error


def test_missing_key_at_preload_describes_the_engine_that_will_answer():
    cloud = _CapsEngine("cloud-caps", preload_error=ProviderUnavailable("no key"))
    local = _CapsEngine("local-caps")
    chain = ChainEngine([("deepgram", cloud), ("whisper", local)])
    assert chain.capabilities() == "cloud-caps"
    chain.preload()
    assert chain.capabilities() == "local-caps"
    chain.transcribe(None, TranscribeContext())
    assert cloud.calls == 0


def test_deepgram_deadline_grows_with_the_clip():
    from src.dictation.asr.engines.deepgram_engine import call_timeout

    short, long = call_timeout(2.0), call_timeout(25.0)
    assert short.connect == long.connect <= 2.0
    assert short.read < long.read <= 10.0
    assert short.read >= 2.0
