"""The two pure rules this change added, and the edge case each one has.

Run: python -m pytest tests/test_stream_rules.py
"""

from src.dictation.asr.types import AsrResult, AsrSegment, Word
from src.dictation.stream.ledger import ChunkLedger, close_sentence
from src.dictation.stream.rules import low_confidence_words
from src.dictation.stream.segmenter import Chunk


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
