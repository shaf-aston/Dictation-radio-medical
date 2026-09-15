"""The shared polish after Stop: which chunks are redone, and that each is redone once."""

import numpy as np

from src.dictation.asr.types import AsrResult
from src.dictation.stream.ledger import ChunkLedger
from src.dictation.stream.polish import polish
from src.dictation.stream.segmenter import Chunk, ChunkPolicy

SR = 16000


def _ledger(confidences):
    ledger = ChunkLedger(ChunkPolicy(), pause_threshold=2.5, sr=SR)
    for n, conf in enumerate(confidences):
        ledger.commit(Chunk(n * SR, (n + 1) * SR, closed=True), f"live{n}", conf)
    return ledger


def _run(ledger, **kw):
    calls = []

    def decode(clip, stage):
        calls.append(stage)
        return AsrResult(text=f"accurate{len(calls)}", segments=[])

    # Audio ends exactly at the committed frontier: no open tail to decode.
    audio = np.zeros(len(ledger.committed) * SR, dtype=np.float32)
    count = polish(ledger, audio, decode, ceiling=0.8, **kw)
    return count, calls


def test_only_unsure_and_forced_chunks_are_redone_once():
    ledger = _ledger([0.95, 0.5, 0.95])
    count, calls = _run(ledger, force=(2, 1))  # 1 is also low-confidence: no second decode
    assert count == 2 and calls == ["final.polish", "final.polish"]
    assert "live0" in ledger.committed_text  # confident chunk kept as-is


def test_cancel_stops_before_any_decode():
    count, calls = _run(_ledger([0.1, 0.1]), cancelled=lambda: True)
    assert count == 0 and calls == []
