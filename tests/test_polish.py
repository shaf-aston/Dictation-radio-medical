"""The shared polish after Stop: which chunks are redone, and that each is redone once."""

import numpy as np

from src.dictation.asr.types import AsrResult, AsrSegment, Word
from src.dictation.stream.ledger import ChunkLedger
from src.dictation.stream.polish import polish
from src.dictation.stream.segmenter import Chunk, ChunkPolicy

SR = 16000


def _ledger(confidences):
    ledger = ChunkLedger(ChunkPolicy(), pause_threshold=2.5, sr=SR)
    for n, conf in enumerate(confidences):
        ledger.commit(Chunk(n * SR, (n + 1) * SR, closed=True), f"live{n}", conf)
    return ledger


def _run(ledger, results=None, **kw):
    """Polish *ledger*. Returns the count, the decode stages, and which chunk
    index each replacement landed on, so "replaced exactly once" is assertable.
    """
    calls = []
    replaced = []

    def decode(clip, stage):
        calls.append(stage)
        if results is not None:
            return results(len(calls))
        return AsrResult(text=f"accurate{len(calls)}", segments=[])

    # Audio ends exactly at the committed frontier: no open tail to decode.
    audio = np.zeros(len(ledger.committed) * SR, dtype=np.float32)
    count = polish(
        ledger,
        audio,
        decode,
        ceiling=0.8,
        on_decoded=lambda _r, i, _s, _e: replaced.append(i),
        **kw,
    )
    return count, calls, replaced


def _word(text, start, end):
    return Word(text=text, start=start, end=end, confidence=0.9)


def test_only_unsure_and_forced_chunks_are_redone_once():
    ledger = _ledger([0.95, 0.5, 0.95])
    count, calls, replaced = _run(ledger, force=(2, 1))  # 1 is also low: no second decode
    # Chunks 1 and 2 are neighbours, so one batch decode is attempted first;
    # this fake engine reports no word timings, so the run falls back to one
    # decode per chunk. Two chunks improved, each replaced exactly once.
    assert count == 2
    assert calls == ["final.polish", "final.polish", "final.polish"]
    assert replaced == [1, 2]
    assert "live0" in ledger.committed_text  # confident chunk kept as-is


def test_no_timings_engine_falls_back_to_one_decode_per_chunk():
    ledger = _ledger([0.1, 0.1, 0.1])
    count, calls, replaced = _run(ledger)
    # One probe decode for the run of three, then three per-chunk decodes.
    assert count == 3 and len(calls) == 4
    assert replaced == [0, 1, 2]
    assert ledger.committed_text == "accurate2 accurate3 accurate4"


def test_paragraph_break_inside_a_batch_is_not_flattened():
    ledger = _ledger([0.1, 0.1])
    para = AsrResult(
        text="one\ntwo",
        segments=(
            AsrSegment("one", 0.0, 0.5, (_word("one", 0.0, 0.5),)),
            AsrSegment("two", 1.2, 1.5, (_word("two", 1.2, 1.5),)),
        ),
    )

    # Call 1 is the batch probe; the per-chunk decodes that follow say so.
    def results(n):
        return para if n == 1 else AsrResult(text=f"chunk{n}", segments=())

    count, calls, replaced = _run(ledger, results=results)
    # The batch's words would rebuild as "one two", losing the break, so the
    # run is decoded chunk by chunk: probe plus one decode each, and the
    # flattened text is never stored.
    assert count == 2 and len(calls) == 3
    assert replaced == [0, 1]
    assert ledger.committed_text == "chunk2 chunk3"


def test_a_blank_word_does_not_blank_a_chunk():
    ledger = _ledger([0.1, 0.1])
    # Every real word falls in chunk 0; chunk 1 holds only a blank one, which
    # spells nothing, so the run must fall back instead of storing "".
    blank = AsrResult(
        text="one two",
        segments=(
            AsrSegment("one two", 0.0, 0.8, (_word(" one", 0.0, 0.4), _word(" two", 0.4, 0.8))),
            AsrSegment("", 1.2, 1.5, (_word(" ", 1.2, 1.5),)),
        ),
    )

    def results(n):
        return blank if n == 1 else AsrResult(text=f"chunk{n}", segments=())

    count, calls, replaced = _run(ledger, results=results)
    assert count == 2 and len(calls) == 3
    assert replaced == [0, 1]
    assert ledger.committed_text == "chunk2 chunk3"


def test_cancel_stops_before_any_decode():
    count, calls, _replaced = _run(_ledger([0.1, 0.1]), cancelled=lambda: True)
    assert count == 0 and calls == []
