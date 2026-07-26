"""Tests for src/dictation/stream/ — the chunk-once streaming machine.

No Qt, no audio, no model: segmenter/ledger/tail are pure state, exactly like
window_state.py's existing test coverage. These are the tests that catch the
M2 pre-mortem's most likely failure (a mis-cut landing mid-word) before it
ever reaches real audio.
"""

from __future__ import annotations

from src.dictation.stream.ledger import ChunkLedger
from src.dictation.stream.segmenter import Chunk, ChunkPolicy, cut_chunks
from src.dictation.stream.tail import LocalAgreement2, agreeing_prefix
from src.dictation.stream.vad import SAMPLE_RATE, SpeechMark

SR = SAMPLE_RATE


def marks(*pairs: tuple[float, float]) -> list[SpeechMark]:
    """Build SpeechMarks from (start_sec, end_sec) pairs."""
    return [SpeechMark(int(a * SR), int(b * SR)) for a, b in pairs]


# ---------------------------------------------------------------------------
# segmenter.cut_chunks
# ---------------------------------------------------------------------------

class TestCutChunks:
    def test_no_audio_returns_nothing(self):
        assert cut_chunks(0, []) == []

    def test_short_audio_is_all_open_tail(self):
        # Below min_sec — nothing to cut yet.
        chunks = cut_chunks(int(3 * SR), marks((0, 1), (1.5, 2.5)))
        assert chunks == [Chunk(0, int(3 * SR), closed=False)]

    def test_cuts_at_pause_within_soft_max(self):
        # Speech 0-7s, pause, speech 7.5-9s — the 7s pause is within
        # [min_sec=6, soft_max_sec=15], so it becomes the cut point.
        chunks = cut_chunks(int(9 * SR), marks((0, 7), (7.5, 9)))
        assert len(chunks) == 2
        assert chunks[0] == Chunk(0, int(7 * SR), closed=True)
        assert chunks[1] == Chunk(int(7 * SR), int(9 * SR), closed=False)

    def test_prefers_the_latest_pause_within_soft_max(self):
        # Two candidate pauses at 7s and 10s, both <= soft_max (15s) — the
        # later one wins (fewer, longer chunks over many short ones).
        m = marks((0, 7), (7.5, 10), (10.5, 20))
        chunks = cut_chunks(int(20 * SR), m)
        assert chunks[0] == Chunk(0, int(10 * SR), closed=True)

    def test_no_pause_before_soft_max_uses_earliest_pause_after(self):
        # No silence until 17s (past soft_max=15s) but before force_cut=20s.
        m = marks((0, 17), (17.5, 20))
        chunks = cut_chunks(int(20 * SR), m)
        assert chunks[0] == Chunk(0, int(17 * SR), closed=True)

    def test_force_cuts_a_single_unbroken_speech_run(self):
        # One continuous mark with no internal silence at all, well past
        # force_cut_sec (20s) — the accepted-risk safety valve.
        m = marks((0, 30))
        chunks = cut_chunks(int(30 * SR), m)
        assert chunks[0] == Chunk(0, int(20 * SR), closed=True)
        assert chunks[1] == Chunk(int(20 * SR), int(30 * SR), closed=False)

    def test_never_cuts_before_min_sec(self):
        # A pause at 2s is too early (< min_sec=6s) to use.
        m = marks((0, 2), (2.5, 8), (8.5, 9))
        chunks = cut_chunks(int(9 * SR), m)
        assert chunks[0].start_sample == 0
        assert chunks[0].end_sample >= int(6 * SR)

    def test_empty_marks_still_force_cuts_long_silence(self):
        # No speech at all detected, but the buffer keeps growing — treated
        # as one long open tail until force_cut_sec, same safety valve.
        chunks = cut_chunks(int(25 * SR), [])
        assert chunks[0] == Chunk(0, int(20 * SR), closed=True)
        assert chunks[1].closed is False

    def test_custom_policy_is_respected(self):
        policy = ChunkPolicy(min_sec=1.0, soft_max_sec=2.0, force_cut_sec=3.0)
        m = marks((0, 1.5), (1.6, 5))
        chunks = cut_chunks(int(5 * SR), m, policy=policy)
        assert chunks[0] == Chunk(0, int(1.5 * SR), closed=True)


# ---------------------------------------------------------------------------
# ledger.ChunkLedger
# ---------------------------------------------------------------------------

class TestChunkLedger:
    def test_commit_advances_open_start_and_freezes_text(self):
        ledger = ChunkLedger()
        ledger.commit(Chunk(0, int(7 * SR), closed=True), "left kidney normal", 0.9)
        assert ledger.open_start_sample == int(7 * SR)
        assert ledger.committed_text == "left kidney normal"
        assert ledger.committed[0].mean_confidence == 0.9

    def test_cannot_commit_an_open_chunk(self):
        ledger = ChunkLedger()
        try:
            ledger.commit(Chunk(0, 100, closed=False), "x", None)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_cannot_commit_out_of_order(self):
        # A chunk that doesn't start exactly at the current frontier would
        # silently corrupt the sample bookkeeping — must raise instead.
        ledger = ChunkLedger()
        try:
            ledger.commit(Chunk(int(5 * SR), int(10 * SR), closed=True), "x", None)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_pending_cuts_are_rebased_to_absolute_samples(self):
        ledger = ChunkLedger()
        ledger.commit(Chunk(0, int(7 * SR), closed=True), "first chunk", 0.9)
        # Total audio is now 15s; tail marks are relative to the open tail
        # itself (0 = the 7s open_start) — pending_cuts must re-base them.
        tail_marks = marks((0, 6), (6.5, 8))
        chunks = ledger.pending_cuts(int(15 * SR), tail_marks)
        assert chunks[0] == Chunk(int(7 * SR), int(13 * SR), closed=True)

    def test_multiple_chunks_from_one_pending_cuts_commit_correctly(self):
        # Regression: committing chunk N used to shift open_start out from
        # under chunk N+1's coordinates when pending_cuts returned
        # tail-relative offsets. Absolute coordinates make this a non-issue.
        # A 30s burst with two well-spaced pauses forces two closed chunks
        # out of a single pending_cuts() call (a catch-up cycle).
        ledger = ChunkLedger()
        tail_marks = marks((0, 14), (14.5, 22), (22.5, 30))
        chunks = ledger.pending_cuts(int(30 * SR), tail_marks)
        closed = [c for c in chunks if c.closed]
        assert len(closed) == 2
        for chunk in closed:
            ledger.commit(chunk, "text", 0.9)
        assert ledger.open_start_sample == closed[-1].end_sample

    def test_committed_text_joins_with_space_on_short_gap(self):
        ledger = ChunkLedger(pause_threshold=2.5)
        ledger.commit(Chunk(0, int(5 * SR), closed=True), "findings are normal", 0.9)
        ledger.commit(Chunk(int(5 * SR), int(8 * SR), closed=True), "impression follows", 0.9)
        assert ledger.committed_text == "findings are normal impression follows"

    def test_committed_text_joins_with_newline_on_long_gap(self):
        ledger = ChunkLedger(pause_threshold=2.5)
        ledger.commit(Chunk(0, int(5 * SR), closed=True), "findings are normal", 0.9)
        # 3s of silence between the two chunks of real text.
        gap_chunk = Chunk(int(5 * SR), int(8 * SR), closed=True)
        ledger.commit(gap_chunk, "", None)  # silence-only chunk, no text
        ledger.commit(Chunk(int(8 * SR), int(11 * SR), closed=True), "impression follows", 0.9)
        assert ledger.committed_text == "findings are normal\nimpression follows"

    def test_low_confidence_indices_ignores_none(self):
        ledger = ChunkLedger()
        ledger.commit(Chunk(0, int(5 * SR), closed=True), "sure text", 0.95)
        ledger.commit(Chunk(int(5 * SR), int(10 * SR), closed=True), "unsure text", 0.4)
        ledger.commit(Chunk(int(10 * SR), int(15 * SR), closed=True), "no signal", None)
        assert ledger.low_confidence_indices(ceiling=0.75) == [1]

    def test_replace_overwrites_text_and_confidence_only(self):
        ledger = ChunkLedger()
        ledger.commit(Chunk(0, int(5 * SR), closed=True), "unsure text", 0.4)
        ledger.replace(0, "corrected text", 0.92)
        assert ledger.committed[0].text == "corrected text"
        assert ledger.committed[0].mean_confidence == 0.92
        assert ledger.committed[0].start_sample == 0
        assert ledger.committed[0].end_sample == int(5 * SR)


# ---------------------------------------------------------------------------
# tail.LocalAgreement2
# ---------------------------------------------------------------------------

class TestLocalAgreement2:
    def test_agreeing_prefix_stops_at_first_divergence(self):
        assert agreeing_prefix("the left kidney shows", "the left kidney is") == "the left kidney"

    def test_agreeing_prefix_full_match(self):
        assert agreeing_prefix("no acute findings", "no acute findings") == "no acute findings"

    def test_agreeing_prefix_empty_previous(self):
        assert agreeing_prefix("", "anything here") == ""

    def test_update_returns_growing_stable_prefix_across_cycles(self):
        la = LocalAgreement2()
        assert la.update("the left") == ""  # nothing to agree with yet
        assert la.update("the left kidney") == "the left"
        assert la.update("the left kidney shows") == "the left kidney"

    def test_reset_clears_agreement_state(self):
        la = LocalAgreement2()
        la.update("some words here")
        la.reset()
        assert la.update("completely different") == ""
