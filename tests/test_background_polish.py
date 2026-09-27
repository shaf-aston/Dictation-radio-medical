"""The accurate engine re-decodes weak chunks while the dictation is still going.

Driven by the synthetic dictation from scripts/eval/simulate_lag.py: words are
encoded in the audio, the live engine hears every third group of four words
badly, and the accurate engine writes its words in capitals so the test can
see exactly which text came from which model.
"""

import time

import pytest

from scripts.eval import simulate_lag as sim
from src.dictation.asr.types import AsrResult, AsrSegment, CostModel, Word
from src.dictation.stream import live_session, polish
from src.dictation.stream.live_session import LiveSession

SR = 16000


@pytest.fixture(autouse=True)
def _fake_vad(monkeypatch):
    monkeypatch.setattr(live_session, "detect_speech", sim.energy_vad)
    monkeypatch.setattr(polish, "detect_speech", sim.energy_vad)


class _Accurate(sim.SimEngine):
    def __init__(self):
        super().__init__(CostModel(fixed_sec=0.0), serial=False, name="small")
        self.clips = []

    def transcribe(self, audio, ctx):
        self.clips.append(len(audio))
        r = super().transcribe(audio, ctx)
        words = tuple(Word(w.text.upper(), w.start, w.end, 0.97) for s in r.segments for w in s.words)
        text = " ".join(w.text for w in words)
        return AsrResult(text, (AsrSegment(text, 0.0, len(audio) / SR, words),) if words else ())


def _live():
    return sim.SimEngine(CostModel(fixed_sec=0.0), serial=False, name="tiny", weak_every=4)


def _dictate(session, audio):
    for i in range(0, len(audio), SR // 10):
        session.feed(audio[i : i + SR // 10])
        session.cycle()
        time.sleep(0.002)


def _upper(text):
    return [t for t in text.split() if t.startswith("W")]


def test_weak_chunks_are_upgraded_before_stop():
    audio, spoken = sim.build_dictation(40.0, seed=3)
    accurate = _Accurate()
    session = LiveSession(_live(), accurate, postprocess=False)
    _dictate(session, audio)
    time.sleep(0.2)
    session.cycle()
    # Upgraded while recording: the accurate model's words are already in.
    assert _upper(session.committed_raw)
    calls_before_stop = len(accurate.clips)
    assert calls_before_stop > 0

    session.close_open_tail_fast()
    final = session.finalize()
    # Every word, once, in order, whichever model wrote it.
    assert sim._seen(final) == [w.index for w in spoken]
    # Stop did not ask again for what was already upgraded: the accurate
    # engine's calls at Stop are at most the last runs and the stop tail.
    assert len(accurate.clips) - calls_before_stop <= 3


def test_the_newest_chunk_is_left_alone():
    audio, _ = sim.build_dictation(20.0, seed=5)
    accurate = _Accurate()
    session = LiveSession(_live(), accurate, postprocess=False)
    _dictate(session, audio)
    time.sleep(0.2)
    session.cycle()
    newest = len(session._ledger.committed) - 1
    assert newest not in session._bg_claimed


def test_no_background_decode_when_the_accurate_engine_is_the_live_one():
    audio, _ = sim.build_dictation(20.0, seed=5)
    live = _live()
    live.capabilities = lambda: sim.EngineCaps(word_confidence=True, hotwords=False, network=True)
    session = LiveSession(live, live, postprocess=False)
    _dictate(session, audio)
    time.sleep(0.2)
    assert session._bg_thread is None


def test_switched_off():
    audio, _ = sim.build_dictation(20.0, seed=5)
    accurate = _Accurate()
    session = LiveSession(_live(), accurate, postprocess=False, background_polish=False)
    _dictate(session, audio)
    time.sleep(0.2)
    assert accurate.clips == []
