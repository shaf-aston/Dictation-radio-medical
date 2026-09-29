"""The live loop's shape comes from what the engine says a call costs."""

from src.core.settings import get_default
from src.dictation.asr.engines.deepgram_engine import DeepgramEngine
from src.dictation.asr.engines.faster_whisper_engine import FasterWhisperEngine
from src.dictation.asr.types import CostModel, EngineCaps
from src.dictation.stream.policy import (
    CHEAP_POLICY,
    LOCAL_POLICY,
    plan_for,
    plan_from_settings,
)


def _caps(fixed):
    return EngineCaps(word_confidence=True, hotwords=False, cost=CostModel(fixed))


def test_cloud_engine_gets_short_chunks():
    plan = plan_for(DeepgramEngine().capabilities())
    assert plan.row == "cheap"
    assert (plan.policy.min_sec, plan.policy.soft_max_sec, plan.policy.force_cut_sec) == CHEAP_POLICY
    assert plan.preview_min_tail_sec < 1.0


def test_whisper_gets_the_measured_local_policy():
    for size in ("tiny.en", "small.en", "base.en"):
        plan = plan_for(FasterWhisperEngine(model_size=size).capabilities())
        assert plan.row == "local", size
        assert (plan.policy.min_sec, plan.policy.soft_max_sec, plan.policy.force_cut_sec) == LOCAL_POLICY


def test_old_persisted_numbers_are_ignored_on_auto():
    # An install that predates auto has 6 / 15 / 20 written into its file.
    stored = {"chunk_policy": "auto", "chunk_min_sec": 6.0, "chunk_soft_max_sec": 15.0,
              "chunk_force_cut_sec": 20.0, "chunk_trailing_silence_sec": 0.6}
    plan = plan_from_settings(stored.get, _caps(0.2))
    assert plan.policy.min_sec == CHEAP_POLICY[0]


def test_manual_uses_the_numbers_verbatim():
    stored = {"chunk_policy": "manual", "chunk_min_sec": 6.0, "chunk_soft_max_sec": 15.0,
              "chunk_force_cut_sec": 20.0, "chunk_trailing_silence_sec": 0.7,
              "preview_min_tail_sec": 1.0}
    plan = plan_from_settings(stored.get, _caps(0.2))
    assert plan.row == "manual"
    assert (plan.policy.min_sec, plan.policy.soft_max_sec, plan.policy.trailing_silence_sec) == (6.0, 15.0, 0.7)


def test_shipped_defaults():
    # The measured 2 / 5 is what a fresh install gets, and auto is the default.
    assert get_default("chunk_policy") == "auto"
    assert (get_default("chunk_min_sec"), get_default("chunk_soft_max_sec")) == (2.0, 5.0)
    assert plan_from_settings(get_default, None).policy.min_sec == 2.0
