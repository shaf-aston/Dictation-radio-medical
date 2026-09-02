"""The two pure rules this change added, and the edge case each one has.

Run: python -m pytest tests/test_stream_rules.py
"""

from src.dictation.asr.types import AsrResult, AsrSegment, Word
from src.dictation.stream.ledger import ChunkLedger, close_sentence
from src.dictation.stream.rules import low_confidence_words
import pytest

from src.dictation.stream.segmenter import Chunk, ChunkPolicy, cut_chunks
from src.dictation.stream.vad import SpeechMark


def test_close_sentence_adds_a_stop_only_where_one_is_missing():
    assert close_sentence("no acute fracture") == "no acute fracture."
    assert close_sentence("the spine is intact.") == "the spine is intact."
    assert close_sentence("he said \"no.\"") == "he said \"no.\""
    # Mid-list: the pause does not tell us the sentence ended.
    assert close_sentence("findings include,") == "findings include,"
    assert close_sentence("   ") == "   "


def test_a_long_pause_ends_the_sentence_and_breaks_the_paragraph():
    sr = 16000
    ledger = ChunkLedger(pause_threshold=2.0, sr=sr)
    ledger.commit(Chunk(0, sr * 5, closed=True), "no acute fracture", 0.9)
    # A five-second silence, then the next chunk.
    ledger.commit(Chunk(sr * 5, sr * 10, closed=True), "", None)
    ledger.commit(Chunk(sr * 10, sr * 15, closed=True), "the spine is intact", 0.9)
    assert ledger.committed_text == "no acute fracture.\nthe spine is intact"


def test_a_short_pause_leaves_the_sentence_open():
    sr = 16000
    ledger = ChunkLedger(pause_threshold=2.0, sr=sr)
    ledger.commit(Chunk(0, sr * 5, closed=True), "no acute", 0.9)
    ledger.commit(Chunk(sr * 5, sr * 10, closed=True), "fracture", 0.9)
    assert ledger.committed_text == "no acute fracture"


def _result(*pairs):
    words = tuple(Word(t, 0.0, 0.0, c) for t, c in pairs)
    return AsrResult(" ".join(w.text for w in words), (AsrSegment("", 0.0, 0.0, words),))


def test_low_confidence_words_are_normalised_and_gated():
    result = _result(("Pneumothorax,", 0.31), ("no", 0.99), ("Effusion", 0.6))
    # 0.6 is the ceiling itself, so it is confident enough to leave unmarked.
    assert low_confidence_words(result, 0.6) == {"pneumothorax"}


def test_no_word_timestamps_means_no_marks():
    result = AsrResult("some text", (AsrSegment("some text", 0.0, 1.0, ()),))
    assert low_confidence_words(result, 0.9) == set()


def test_the_preview_is_dropped_only_when_its_own_call_is_too_slow():
    from src.dictation.stream.rules import should_skip_preview

    assert should_skip_preview(1.4, 3.0) is False   # measured tiny.en cost
    assert should_skip_preview(4.2, 3.0) is True
    assert should_skip_preview(0.0, 3.0) is False   # not measured yet
    assert should_skip_preview(99.0, 0.0) is False  # the skip is switched off


# --- The trailing pause: a chunk closes while the speaker is still thinking ---
#
# Before this, a cut point had to be a speech mark with ANOTHER mark after it,
# so the last thing said before a pause waited for the speaker to resume. A
# radiologist who stops to read the film saw nothing committed until they
# spoke again, with no upper bound short of the 20s force cut.

SR = 16000


def _marks(*spans_sec):
    return [SpeechMark(int(a * SR), int(b * SR)) for a, b in spans_sec]


def test_a_trailing_pause_closes_the_chunk_without_waiting_for_the_speaker():
    # 7s of speech, then 1s of silence and nothing more. The pause is real and
    # long enough, so the chunk closes on it.
    chunks = cut_chunks(int(8.0 * SR), _marks((0.0, 7.0)))
    closed = [c for c in chunks if c.closed]
    assert len(closed) == 1
    assert closed[0].end_sample == int(7.0 * SR)
    # What is left over is the silence, still open.
    assert chunks[-1].closed is False
    assert chunks[-1].start_sample == int(7.0 * SR)


def test_a_pause_too_short_to_be_a_thought_does_not_close_it():
    # Same speech, but only 0.3s of silence has arrived: under
    # trailing_silence_sec, so this could still be a breath mid-sentence.
    chunks = cut_chunks(int(7.3 * SR), _marks((0.0, 7.0)))
    assert [c for c in chunks if c.closed] == []


def test_the_minimum_chunk_length_still_wins_over_a_trailing_pause():
    # 3s of speech then 2s of silence. The pause is long enough, but the chunk
    # would be under chunk_min_sec, and short chunks cost a full decode each
    # for very little text: the floor is the whole reason it exists.
    chunks = cut_chunks(int(5.0 * SR), _marks((0.0, 3.0)))
    assert [c for c in chunks if c.closed] == []


def test_a_long_silence_is_never_force_cut_into_a_chunk_of_nothing():
    # Once the chunk closes at the mark end, the rest is pure silence. The
    # force cut is the safety valve for an unbroken monologue, so it must not
    # fire here: a silent chunk costs a whole decode and returns no words.
    chunks = cut_chunks(int(30.0 * SR), _marks((0.0, 7.0)))
    assert len([c for c in chunks if c.closed]) == 1
    assert len(chunks) == 2


def test_the_force_cut_still_fires_on_an_unbroken_monologue():
    # The case it exists for: 45s of continuous speech, no silence anywhere.
    chunks = cut_chunks(int(45.0 * SR), _marks((0.0, 45.0)))
    closed = [c for c in chunks if c.closed]
    assert [c.end_sample for c in closed] == [int(20.0 * SR), int(40.0 * SR)]


def test_the_live_loop_still_cuts_at_the_same_sample_when_the_speaker_resumes():
    # The live loop calls this every live_cycle_sec on the audio that has
    # ARRIVED, so at the moment the pause is confirmed there is no later mark
    # to prefer. Cuts at 7s, exactly as it did before the change.
    chunks = cut_chunks(int(8.0 * SR), _marks((0.0, 7.0), (7.4, 8.0)))
    closed = [c for c in chunks if c.closed]
    assert closed and closed[0].end_sample == int(7.0 * SR)


def test_the_trailing_pause_never_moves_a_cut_that_would_have_happened_anyway():
    # The load-bearing property. A trailing pause is a FALLBACK, not another
    # candidate: the policy prefers the latest pause up to soft_max_sec, and a
    # trailing candidate is always the latest, so as a peer it would have
    # pushed every cut later than an ordinary pause would have. Measured on
    # pause-heavy audio, that cancelled the whole gain.
    #
    # Two marks with a gap, whole file. The first cut stays at 7s, exactly
    # where it was before this feature existed. What is new is the SECOND
    # chunk: it closes at 14s on the trailing pause instead of staying open.
    chunks = cut_chunks(int(20.0 * SR), _marks((0.0, 7.0), (9.0, 14.0)))
    assert [c.end_sample for c in chunks if c.closed] == [int(7.0 * SR), int(14.0 * SR)]


def test_a_trailing_pause_is_refused_when_the_knob_is_not_positive():
    with pytest.raises(ValueError):
        ChunkPolicy(trailing_silence_sec=0.0)
