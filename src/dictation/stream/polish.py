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

import logging
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.dictation.asr.types import AsrResult, AsrSegment, Word
from src.dictation.stream.ledger import ChunkLedger
from src.dictation.stream.rules import mean_confidence
from src.dictation.stream.segmenter import Chunk
from src.dictation.stream.vad import detect_speech

logger = logging.getLogger(__name__)

#: ``decode(clip, stage)``: ``None`` on failure, never raises.
Decode = Callable[[np.ndarray, str], Optional[AsrResult]]
#: ``on_decoded(result, index, start_sample, end_sample)``; index is ``None``
#: for the last section, which is committed as a new chunk.
OnDecoded = Callable[[AsrResult, Optional[int], int, int], None]

#: Default for ``polish(batch_max_sec=...)``: the longest audio one polish
#: decode may cover, in seconds. Whisper pads every call out to a 30-second mel
#: window, so a decode costs about the same whatever it is handed: the price is
#: per call, not per second of audio. Batching neighbouring chunks up to just
#: under that pad therefore buys back whole calls, and lets the accurate model
#: read a paragraph instead of context-free fragments.
#:
#: How much it buys depends entirely on how long the chunks are, which is a
#: setting and not a constant. With the shipped defaults (``chunk_min_sec``
#: 6.0, ``chunk_soft_max_sec`` 15.0) a chunk is most of this ceiling on its
#: own, so a run is usually one or two chunks and the saving is small. Where
#: the settings file lowers them (2.0 / 5.0, the policy this project's eval
#: measured as best) chunks are a few seconds each, a run holds five to ten of
#: them, and the saving is large: a three-minute dictation goes from tens of
#: full encoder passes to a handful.
BATCH_MAX_SEC = 25.0

#: Why :func:`_split` could not divide a batch. The first two are statements
#: about the engine and hold for every later run, so batching latches off; the
#: last two are about this run's audio (a silent chunk the engine returned no
#: words for, routine once faster-whisper has dropped a hallucinated segment,
#: or a paragraph break inside the run), so only that run falls back.
_NO_TIMINGS = "no-timings"
_MISMATCH = "mismatch"
_EMPTY_CHUNK = "empty-chunk"
_PARAGRAPH = "paragraph"
_LATCHING = (_NO_TIMINGS, _MISMATCH)

_WHY = {
    _NO_TIMINGS: (
        "Polish batch of %d chunks came back with no word timings at all: this "
        "engine's chunks are decoded one at a time from here on."
    ),
    _MISMATCH: (
        "Polish batch of %d chunks came back with words that do not spell out "
        "the transcript, so the split would drop its punctuation: this "
        "engine's chunks are decoded one at a time from here on."
    ),
    _EMPTY_CHUNK: (
        "Polish batch of %d chunks had a chunk no word fell into (silence, or "
        "a dropped segment): this one run is decoded chunk by chunk, batching "
        "stays on."
    ),
    _PARAGRAPH: (
        "Polish batch of %d chunks holds a paragraph break the words cannot "
        "say where to put back: this one run is decoded chunk by chunk, "
        "batching stays on."
    ),
}


def _runs(
    targets: List[int], ledger: ChunkLedger, sr: int, batch_max_sec: float
) -> List[List[int]]:
    """Target indices grouped into consecutive runs of at most *batch_max_sec*.

    Only neighbours are grouped: consecutive ledger chunks are contiguous in
    the audio by construction, so a run is one clip. A gap ends the run, and a
    single chunk longer than the ceiling stays a run of its own.
    """
    runs: List[List[int]] = []
    for i in targets:
        if runs and runs[-1][-1] == i - 1:
            first = ledger.committed[runs[-1][0]]
            span = (ledger.committed[i].end_sample - first.start_sample) / sr
            if span <= batch_max_sec:
                runs[-1].append(i)
                continue
        runs.append([i])
    return runs


def _chunk_at(ledger: ChunkLedger, run: List[int], sample: int) -> int:
    """Which chunk of *run* owns *sample*; the ends clamp, so nothing is lost."""
    for i in run[:-1]:
        if sample < ledger.committed[i].end_sample:
            return i
    return run[-1]


def _spelled(words: Iterable[Word]) -> str:
    """The words written out as text, one space between any that do not bring
    their own. ``Word.text`` spacing is per engine: faster-whisper keeps the
    decoder's leading space, Deepgram and Parakeet hand back bare tokens.
    """
    return "".join(
        w.text if w.text[:1].isspace() else " " + w.text for w in words
    ).strip()


def _flat(text: str) -> str:
    """*text* with every run of whitespace, newlines included, reduced to one
    space: two texts that differ only in their paragraph breaks are equal here.
    """
    return " ".join(text.split())


def _collapsed(text: str) -> str:
    """*text* with spaces and tabs collapsed but newlines kept.

    A newline is not whitespace noise in this project: the engine writes one
    wherever the gap between two segments reached ``pause_threshold``
    (``faster_whisper_engine``), and that is how the report gets its
    paragraphs. Collapsing it away would let a rebuilt text that has lost a
    break compare equal to the one that kept it.
    """
    return "\n".join(
        piece for line in text.split("\n") if (piece := _flat(line))
    )


def _split(
    result: AsrResult, ledger: ChunkLedger, run: List[int], clip_start: int, sr: int
) -> Tuple[Optional[List[AsrResult]], str]:
    """One result per chunk of *run*, plus ``""``; or ``(None, why)`` when the
    words cannot say where the chunks divide.

    Every decoded word lands in exactly one chunk: the one whose sample window
    holds the word's midpoint. A ``None`` means the caller must decode the
    run's chunks one at a time instead, because an engine that gave no word
    timings, a chunk no word fell into, or words that do not spell out the
    transcript, would otherwise have its text guessed at. *why* says which of
    those it was, because the caller treats them differently: two are what this
    engine always does, one is what this run's audio happened to be.

    The last rule is why the rebuilt text is checked against ``result.text``
    before it is returned. Only some engines hand back words that are the
    transcript's own pieces: Deepgram asks for punctuation and capitalisation,
    and they live in the transcript alone, its word tokens being raw and
    lowercase. Splitting those would hand the radiologist a report whose
    batched half has no full stops while the rest of it has. The same check
    catches a lost paragraph break, because it compares the newlines too, and
    that one is about this run's audio rather than the engine.
    """
    words = [w for seg in result.segments for w in seg.words]
    if not words:
        return None, _NO_TIMINGS
    held: Dict[int, List[Word]] = {i: [] for i in run}
    for w in words:
        # A word whose text is blank spells nothing, so a chunk holding only
        # those would rebuild as an empty string and blank a chunk that had
        # text. faster-whisper keeps a segment's words even when its text is
        # empty, so this happens. Skipping them here sends such a chunk down
        # the empty-chunk path, which decodes it on its own instead.
        if not w.text.strip():
            continue
        at = clip_start + int((w.start + w.end) / 2 * sr)
        held[_chunk_at(ledger, run, at)].append(w)
    if any(not held[i] for i in run):
        return None, _EMPTY_CHUNK
    out: List[AsrResult] = []
    for i in run:
        mine = tuple(held[i])
        text = _spelled(mine)
        out.append(
            AsrResult(text, (AsrSegment(text, mine[0].start, mine[-1].end, mine),))
        )
    rebuilt = " ".join(r.text for r in out)
    if _collapsed(rebuilt) != _collapsed(result.text):
        # Same words, different paragraphs: the batch's audio holds a pause the
        # engine turned into a newline, and words joined with spaces cannot say
        # where it went. Decoding this run's chunks one at a time keeps the
        # engine's own text, break and all.
        why = _PARAGRAPH if _flat(rebuilt) == _flat(result.text) else _MISMATCH
        return None, why
    return out, ""


def _decode_one(
    ledger: ChunkLedger, audio: np.ndarray, decode: Decode, i: int, on_decoded: OnDecoded
) -> None:
    """Re-decode one committed chunk on its own and freeze the result."""
    c = ledger.committed[i]
    clip = audio[c.start_sample:c.end_sample]
    if not len(clip):
        return
    result = decode(clip, "final.polish")
    if result is not None:
        ledger.replace(i, result.text, mean_confidence(result))
        on_decoded(result, i, c.start_sample, c.end_sample)


def polish(
    ledger: ChunkLedger,
    audio: np.ndarray,
    decode: Decode,
    *,
    ceiling: float,
    batch_max_sec: float = BATCH_MAX_SEC,
    force: Iterable[int] = (),
    on_progress: Callable[[str], None] = lambda _msg: None,
    on_decoded: OnDecoded = lambda *_args: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> int:
    """Polish ``ledger`` in place against ``audio``. Returns chunks re-decoded.

    *batch_max_sec* is the tuning knob for how much audio one decode may
    cover; callers that know better than :data:`BATCH_MAX_SEC` pass their own.
    """
    targets = list(ledger.low_confidence_indices(ceiling))
    targets += [i for i in force if i not in targets]
    targets.sort()
    # The ledger's own sample rate: a word time is in seconds, a chunk window
    # is in samples, and the two only line up through it.
    sr = ledger.sample_rate
    done = 0
    batching = True
    for run in _runs(targets, ledger, sr, batch_max_sec):
        if cancelled():
            return done
        # Sections are counted in chunks, not batches: the number the
        # radiologist watches is how much of the report is being improved.
        on_progress(f"Improving section {done + 1} of {len(targets)}")
        parts: Optional[List[AsrResult]] = None
        if batching and len(run) > 1:
            first, last = ledger.committed[run[0]], ledger.committed[run[-1]]
            clip = audio[first.start_sample:last.end_sample]
            result = decode(clip, "final.polish") if len(clip) else None
            if result is not None:
                parts, why = _split(result, ledger, run, first.start_sample, sr)
                if parts is None:
                    # Only an engine's own habits latch batching off. A silent
                    # chunk or a paragraph break is this run's audio, and the
                    # next run may be fine.
                    batching = why not in _LATCHING
                    logger.warning(_WHY[why], len(run))
        if parts is None:
            for i in run:
                if cancelled():
                    return done
                _decode_one(ledger, audio, decode, i, on_decoded)
        else:
            for i, part in zip(run, parts):
                if cancelled():
                    return done
                c = ledger.committed[i]
                ledger.replace(i, part.text, mean_confidence(part))
                on_decoded(part, i, c.start_sample, c.end_sample)
        done += len(run)

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
