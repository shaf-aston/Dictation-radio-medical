"""The per-dictation run log: what it records, and what it refuses to grow into.

The log exists to answer "did that long dictation behave differently from this
short one" with numbers instead of assertion. That makes two properties matter
more than the contents: it must stay bounded (it holds a full report per run,
on the one machine that can least afford an unbounded file), and it must never
be able to break the dictation it is describing.
"""

from __future__ import annotations

import json

import pytest

from src.core import perf
from src.features import run_log


class DummySettings(dict):
    """Settings stand-in: the log only ever reads two keys."""

    def get(self, key, default=None):
        return super().get(key, default)


@pytest.fixture(autouse=True)
def log_in_tmp(tmp_path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(run_log, "run_log_path", lambda: path)
    perf.reset()
    yield path
    perf.reset()


@pytest.fixture
def settings():
    return DummySettings({"run_log_max": 200, "run_log_store_text": True})


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_finished_run_is_recorded(log_in_tmp, settings):
    record = run_log.start("desktop", model="small.en", engine="faster-whisper")
    record.audio_sec = 61.5
    run_log.finish(record, "Small right pneumothorax.", settings)

    row = _rows(log_in_tmp)[0]
    assert row["front_end"] == "desktop"
    assert row["model"] == "small.en"
    assert row["audio_sec"] == 61.5
    assert row["word_count"] == 3
    assert row["started"]


def test_the_stage_timings_are_captured_before_the_next_reset(log_in_tmp, settings):
    """perf resets every run, so a snapshot taken later is a snapshot of nothing."""
    perf.record("dictation.postprocess", 0.25)
    perf.set_gauge("stream.decode_ratio", 1.4)

    record = run_log.start("desktop")
    run_log.finish(record, "text", settings)
    perf.reset()                      # the next run starts

    row = _rows(log_in_tmp)[0]
    assert "dictation.postprocess" in row["stages"]
    assert row["decode_ratio"] == 1.4


def test_the_report_text_is_kept_when_asked(log_in_tmp, settings):
    run_log.finish(run_log.start("web"), "Impression: normal.", settings)
    assert _rows(log_in_tmp)[0]["text"] == "Impression: normal."


def test_the_report_text_can_be_withheld(log_in_tmp):
    """Numbers always; the body only by consent. The timings must survive."""
    settings = DummySettings({"run_log_store_text": False})
    record = run_log.start("web")
    record.audio_sec = 12.0
    run_log.finish(record, "Impression: normal.", settings)

    row = _rows(log_in_tmp)[0]
    assert row["text"] == ""
    assert row["word_count"] == 2      # counted before it was dropped
    assert row["audio_sec"] == 12.0


def test_the_log_stays_bounded(log_in_tmp):
    """A record holds a whole report; an uncapped log is a disk-filler."""
    settings = DummySettings({"run_log_max": 5})
    for i in range(80):
        run_log.finish(run_log.start("web"), f"report {i}", settings)

    rows = _rows(log_in_tmp)
    assert len(rows) <= 5 + run_log._TRIM_SLACK
    # Trimming keeps the newest, which is what a diagnostics page wants.
    assert rows[-1]["text"] == "report 79"


def test_trimming_drops_the_oldest_not_the_newest(log_in_tmp):
    """Which end is discarded matters: the newest runs are the diagnostic ones."""
    total = 3 + run_log._TRIM_SLACK + 2
    settings = DummySettings({"run_log_max": 3})
    for i in range(total):
        run_log.finish(run_log.start("web"), f"report {i}", settings)

    kept = [row["text"] for row in _rows(log_in_tmp)]
    # The run just recorded is always present, and the survivors are the tail
    # of the sequence in order — never a hole in the middle.
    assert kept[-1] == f"report {total - 1}"
    numbers = [int(text.split()[1]) for text in kept]
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers)))


def test_recent_returns_newest_first(log_in_tmp, settings):
    for i in range(3):
        run_log.finish(run_log.start("web"), f"report {i}", settings)
    assert [row["text"] for row in run_log.recent()] == [
        "report 2", "report 1", "report 0",
    ]
    assert len(run_log.recent(limit=2)) == 2


def test_a_broken_log_never_breaks_the_dictation(monkeypatch, settings):
    """Logging must not be able to fail the thing it is recording."""
    monkeypatch.setattr(run_log, "run_log_path", lambda: (_ for _ in ()).throw(OSError("nope")))
    run_log.finish(run_log.start("desktop"), "text", settings)   # must not raise


def test_a_missing_log_reads_as_empty(log_in_tmp):
    assert run_log.recent() == []
