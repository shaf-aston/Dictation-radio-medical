"""Streaming mode: Deepgram's live socket does the live work, the ledger keeps it.

Driven against scripts/eval/fake_deepgram.py, a local server speaking the same
protocol, so the real client (asr/engines/deepgram_stream.py), the real
LiveSession and a real socket are all exercised with no key and no network.
"""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("websockets")

from scripts.eval import simulate_lag as sim  # noqa: E402
from scripts.eval.fake_deepgram import FakeDeepgram, FakeDeepgramConfig, words_of  # noqa: E402
from src.dictation.asr.engines import deepgram_engine, deepgram_stream  # noqa: E402
from src.dictation.asr.engines import fallback_engine  # noqa: E402
from src.dictation.asr.engines.deepgram_engine import DeepgramEngine  # noqa: E402
from src.dictation.asr.engines.fallback_engine import ChainEngine  # noqa: E402
from src.dictation.asr.types import CostModel, StreamFinal, StreamInterim  # noqa: E402
from src.dictation.stream import live_session, polish  # noqa: E402
from src.dictation.stream.live_session import LiveSession  # noqa: E402

SR = 16000


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    fallback_engine.reset_provider_health()
    monkeypatch.setattr(live_session, "detect_speech", sim.energy_vad)
    monkeypatch.setattr(polish, "detect_speech", sim.energy_vad)
    monkeypatch.setattr(deepgram_engine, "_boosted_keywords", lambda: ())
    yield
    fallback_engine.reset_provider_health()


class _Batch(sim.SimEngine):
    """The chain's local tier: decodes chunks when the socket is not there."""

    def __init__(self):
        super().__init__(CostModel(fixed_sec=0.0), serial=False)

    def identity(self):
        return ("local", "tiny")


def _chain(url, key="good-key", monkeypatch=None):
    monkeypatch.setattr(deepgram_engine, "get_api_key", lambda: key)
    cloud = DeepgramEngine(live_url=url)
    local = _Batch()
    # The REST path is not what these tests are about: a batch decode that
    # reaches Deepgram fails over to the local tier at once.
    monkeypatch.setattr(cloud, "transcribe", lambda *_a: (_ for _ in ()).throw(ConnectionError("no REST here")))
    return ChainEngine([("deepgram", cloud), ("local", local)]), local


def _dictate(session, audio, pace=0.01):
    block = SR // 10
    updates = []
    for i in range(0, len(audio), block):
        session.feed(audio[i : i + block])
        if (u := session.cycle()) is not None:
            updates.append(u)
        time.sleep(pace)
    return updates


def test_streaming_commits_every_word_once_in_order(monkeypatch):
    audio, spoken = sim.build_dictation(15.0, seed=11)
    with FakeDeepgram(FakeDeepgramConfig(latency_sec=0.02)) as server:
        chain, local = _chain(server.url, monkeypatch=monkeypatch)
        session = LiveSession(chain, chain, cleanup_level="soft", postprocess=False)
        updates = _dictate(session, audio)
        assert session.streaming, "never opened the live socket"
        # Kept text grows while speaking, not only at Stop.
        assert any(words_of(u.committed) for u in updates[: len(updates) // 2])
        session.close_open_tail_fast()
        final = session.finalize()
    assert words_of(final) == [w.index for w in spoken]
    # Nothing was decoded locally: the socket did all of it.
    assert local.calls == 0
    assert server.connections == 1


def test_interims_show_before_the_final(monkeypatch):
    audio, _ = sim.build_dictation(6.0, seed=2)
    with FakeDeepgram(FakeDeepgramConfig(latency_sec=0.0, endpointing_ms=5000)) as server:
        chain, _ = _chain(server.url, monkeypatch=monkeypatch)
        session = LiveSession(chain, chain, postprocess=False)
        updates = _dictate(session, audio)
        session.close()
    assert any(u.preview for u in updates)


def test_a_dropped_socket_falls_back_to_chunks_without_losing_a_word(monkeypatch):
    audio, spoken = sim.build_dictation(15.0, seed=11)
    with FakeDeepgram(FakeDeepgramConfig(latency_sec=0.0, drop_after_sec=6.0)) as server:
        chain, local = _chain(server.url, monkeypatch=monkeypatch)
        session = LiveSession(chain, chain, postprocess=False)
        _dictate(session, audio)
        assert not session.streaming
        session.close_open_tail_fast()
        final = session.finalize()
    assert words_of(final) == [w.index for w in spoken]
    assert local.calls > 0


def test_a_rejected_key_is_learned_and_chunks_are_decoded(monkeypatch):
    audio, spoken = sim.build_dictation(6.0, seed=4)
    with FakeDeepgram() as server:
        chain, local = _chain(server.url, key="wrong", monkeypatch=monkeypatch)
        session = LiveSession(chain, chain, postprocess=False)
        _dictate(session, audio, pace=0)
        session.close_open_tail_fast()
        final = session.finalize()
    assert not session.streaming
    assert "deepgram" in fallback_engine._DEAD_PROVIDERS
    assert words_of(final) == [w.index for w in spoken]


def test_streaming_off_by_setting(monkeypatch):
    with FakeDeepgram() as server:
        chain, _ = _chain(server.url, monkeypatch=monkeypatch)
        session = LiveSession(chain, chain, postprocess=False, streaming=False)
        _dictate(session, np.zeros(SR, dtype=np.float32), pace=0)
    assert server.connections == 0


def test_parses_a_real_shaped_deepgram_result():
    # Field names and nesting as Deepgram's live API documents them.
    payload = {
        "type": "Results", "channel_index": [0, 1], "duration": 1.98, "start": 3.02,
        "is_final": True, "speech_final": True,
        "channel": {"alternatives": [{
            "transcript": "No pneumothorax.", "confidence": 0.99,
            "words": [
                {"word": "no", "start": 3.1, "end": 3.3, "confidence": 0.99, "punctuated_word": "No"},
                {"word": "pneumothorax", "start": 3.3, "end": 4.1, "confidence": 0.62,
                 "punctuated_word": "pneumothorax."},
            ],
        }]},
    }

    class _Ws:
        """A socket that stays open, silent, until closed."""

        def __init__(self):
            self.closed = threading.Event()

        def __iter__(self):
            self.closed.wait()
            return iter(())

        def send(self, _):
            pass

        def close(self):
            self.closed.set()

    stream = deepgram_stream.DeepgramStream(_Ws())
    stream._handle(payload)
    stream._handle({**payload, "is_final": False, "channel": {"alternatives": [{"transcript": "No pneumo"}]}})
    final, interim = stream.poll()[:2]
    stream.close()
    assert isinstance(final, StreamFinal) and isinstance(interim, StreamInterim)
    assert (final.text, final.start, round(final.end, 2)) == ("No pneumothorax.", 3.02, 5.0)
    assert [w.text for w in final.words] == ["No", "pneumothorax."]
    assert round(final.words[0].start, 2) == 0.08  # relative to the final's start
    assert interim.text == "No pneumo"
