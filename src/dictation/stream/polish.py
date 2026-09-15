"""The polish after Stop: one rule for both front-ends.

Re-decodes, with the accurate engine, only what is worth it: committed chunks
the live model was unsure about, any chunk the caller forces in, and whatever
audio never closed into a chunk before Stop. Everything the fast model got
confidently right is kept, which is why this costs seconds and not a full
re-transcribe.

Owns no engine, thread, or signal. The caller hands in how to decode and how to
report progress, so the desktop worker and the web session cannot drift apart
on which chunks get redone or how the last section is closed.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

from src.dictation.asr.types import AsrResult
from src.dictation.stream.ledger import ChunkLedger
from src.dictation.stream.rules import mean_confidence
from src.dictation.stream.segmenter import Chunk
from src.dictation.stream.vad import detect_speech

#: ``decode(clip, stage)``: ``None`` on failure, never raises.
Decode = Callable[[np.ndarray, str], Optional[AsrResult]]
#: ``on_decoded(result, index, start_sample, end_sample)``; index is ``None``
#: for the last section, which is committed as a new chunk.
OnDecoded = Callable[[AsrResult, Optional[int], int, int], None]


def polish(
    ledger: ChunkLedger,
    audio: np.ndarray,
    decode: Decode,
    *,
    ceiling: float,
    force: Iterable[int] = (),
    on_progress: Callable[[str], None] = lambda _msg: None,
    on_decoded: OnDecoded = lambda *_args: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> int:
    """Polish ``ledger`` in place against ``audio``. Returns chunks re-decoded."""
    targets = list(ledger.low_confidence_indices(ceiling))
    targets += [i for i in force if i not in targets]
    for done, i in enumerate(targets, start=1):
        if cancelled():
            return done - 1
        on_progress(f"Improving section {done} of {len(targets)}")
        c = ledger.committed[i]
        clip = audio[c.start_sample:c.end_sample]
        if not len(clip):
            continue
        result = decode(clip, "final.polish")
        if result is not None:
            ledger.replace(i, result.text, mean_confidence(result))
            on_decoded(result, i, c.start_sample, c.end_sample)

    start, end = ledger.open_start_sample, len(audio)
    tail = audio[start:end]
    if cancelled() or not len(tail) or not detect_speech(tail):
        return len(targets)
    on_progress("Improving the last section")
    result = decode(tail, "final.tail")
    if result is not None and (text := result.text.strip()):
        ledger.commit(Chunk(start, end, closed=True), text, mean_confidence(result))
        on_decoded(result, None, start, end)
    return len(targets)
