"""The developer console is only worth having if its diary is honest.

Three things have to hold, and each one is a way the panel could lie without
looking broken: a poll must never re-send a line it already printed, the buffer
must stay bounded however long a session runs, and a block that fails must still
be recorded with the time it burned before failing: a stage that dies slowly is
exactly the one you opened the console to find.
"""

from __future__ import annotations

import logging

import pytest

from src.core import event_log


@pytest.fixture(autouse=True)
def clean_diary():
    event_log.reset()
    event_log.configure(event_log.DEFAULT_CAPACITY)
    yield
    event_log.detach_from_logging()
    event_log.reset()


def test_a_poll_only_ever_receives_what_it_has_not_seen():
    event_log.emit("live", "one")
    event_log.emit("live", "two")
    seen = event_log.events()
    assert [e["message"] for e in seen] == ["one", "two"]

    event_log.emit("live", "three")
    fresh = event_log.events(after=seen[-1]["seq"])
    assert [e["message"] for e in fresh] == ["three"]


def test_the_buffer_is_bounded_however_long_the_session_runs():
    event_log.configure(5)
    for index in range(50):
        event_log.emit("live", f"cycle {index}")

    kept = event_log.events()
    assert len(kept) == 5
    assert kept[-1]["message"] == "cycle 49"
    # Sequence numbers keep counting even though the old lines are gone, so a
    # panel that was polling across the overflow cannot be handed a number it
    # has already printed.
    assert kept[-1]["seq"] == 50


def test_a_capacity_of_zero_is_refused_rather_than_silently_dropping_everything():
    with pytest.raises(ValueError):
        event_log.configure(0)


def test_a_block_that_fails_is_still_timed_and_marked_as_an_error():
    with pytest.raises(RuntimeError):
        with event_log.timed("asr", "decode chunk", index=2):
            raise RuntimeError("model died")

    (entry,) = event_log.events()
    assert entry["message"] == "decode chunk"
    assert entry["level"] == "error"
    assert entry["ms"] is not None
    assert entry["fields"]["index"] == 2


def test_what_the_block_learns_on_the_way_out_is_recorded_with_it():
    with event_log.timed("asr", "decode chunk", clip_sec=6.2) as note:
        note["words"] = 11

    (entry,) = event_log.events()
    assert entry["fields"] == {"clip_sec": 6.2, "words": 11}


def test_ordinary_log_lines_land_in_the_same_stream():
    # Without this the panel would show the timings and miss the warning that
    # explains them -- two halves of one story, in two places.
    event_log.attach_to_logging()
    logging.getLogger("src.dictation.stream.live_session").warning("Decode failed (%s)", "tail")

    messages = [e["message"] for e in event_log.events()]
    assert "Decode failed (tail)" in messages
    assert event_log.events()[-1]["level"] == "warning"


def test_running_shows_an_open_timed_block_and_only_while_open():
    from src.core import event_log
    assert event_log.running() == []
    with event_log.timed("asr", "live.chunk decode"):
        (open_block,) = event_log.running()
        assert open_block["message"] == "live.chunk decode"
        assert open_block["started"] > 0
    assert event_log.running() == []
