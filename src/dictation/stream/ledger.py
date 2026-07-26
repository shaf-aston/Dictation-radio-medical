"""Append-only committed-chunk ledger — the chunk-once guarantee.

Once a chunk closes (a VAD silence boundary was found, or the segmenter's
force-cut safety valve fired), its decoded text is frozen: nothing re-decodes
it and nothing rewrites it, except the one-time confidence-targeted polish
after recording stops. This is what turns "each second of audio decoded ~8
times" (the old sliding-window re-decode in window_state.py) into "each
second decoded once" — the actual fix for the slow half of this rebuild.

Owns the absolute-sample bookkeeping; :mod:`src.dictation.stream.segmenter`
stays pure (0-based, no I/O) by operating only on the still-open tail each
cycle, which this class is responsible for slicing out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.dictation.stream.segmenter import Chunk, ChunkPolicy, cut_chunks
from src.dictation.stream.vad import SAMPLE_RATE, SpeechMark


@dataclass(frozen=True)
class CommittedChunk:
    """One permanently-decoded chunk. Replaced only by :meth:`ChunkLedger.replace`
    (the confidence-targeted polish), never mutated in place."""

    start_sample: int
    end_sample: int
    text: str
    mean_confidence: Optional[float]


class ChunkLedger:
    """Tracks which audio has been decoded-and-frozen versus still open.

    Every sample offset this class hands out or accepts — including
    :meth:`pending_cuts`'s return value — is **absolute** (from the start of
    the recording), never tail-relative. A cycle can close more than one
    chunk from a single VAD pass (catching up after a burst); if
    :meth:`commit` accepted tail-relative offsets, committing chunk N would
    shift ``open_start`` out from under chunk N+1's still tail-relative
    coordinates. Working in absolute samples throughout — and
    :meth:`commit` validating that a chunk starts exactly at the current
    frontier — makes that class of bug impossible instead of easy to
    reintroduce.
    """

    def __init__(
        self,
        policy: ChunkPolicy = ChunkPolicy(),
        pause_threshold: float = 2.5,
        sr: int = SAMPLE_RATE,
    ) -> None:
        self.policy = policy
        # Mirrors transcriber.py's own segment-join rule (gap >= pause_threshold
        # -> newline) so a pause that spans a chunk boundary still reads as a
        # paragraph break, not just one that happens to land inside one chunk.
        self._pause_threshold = pause_threshold
        self._sr = sr
        self._committed: List[CommittedChunk] = []
        self._open_start: int = 0

    @property
    def committed(self) -> List[CommittedChunk]:
        return list(self._committed)

    @property
    def open_start_sample(self) -> int:
        """Where the still-open (undecided) tail begins."""
        return self._open_start

    @property
    def committed_text(self) -> str:
        pieces: List[str] = []
        prev_end: Optional[int] = None
        for c in self._committed:
            if not c.text:
                continue
            if prev_end is not None:
                gap_sec = (c.start_sample - prev_end) / self._sr
                pieces.append("\n" if gap_sec >= self._pause_threshold else " ")
            pieces.append(c.text)
            prev_end = c.end_sample
        return "".join(pieces)

    def pending_cuts(self, total_samples: int, tail_marks: List[SpeechMark]) -> List[Chunk]:
        """Chunk boundaries for the current open tail, in absolute samples.

        *tail_marks* must be VAD marks computed over just the open tail
        (samples ``[open_start_sample, total_samples)``), not the whole
        recording — recomputing VAD over the whole growing buffer every cycle
        would itself become O(n^2) over a long dictation. They are relative to
        the open tail (0 = ``open_start_sample``); this method re-bases the
        segmenter's tail-relative output back to absolute samples before
        returning it, so every chunk a caller ever sees from this class is in
        the same coordinate frame.
        """
        base = self._open_start
        tail_len = max(0, total_samples - base)
        return [
            Chunk(base + c.start_sample, base + c.end_sample, c.closed)
            for c in cut_chunks(tail_len, tail_marks, self.policy)
        ]

    def commit(self, chunk: Chunk, text: str, mean_confidence: Optional[float]) -> None:
        """Freeze one chunk. *chunk* offsets are absolute (as returned by
        :meth:`pending_cuts`); it must start exactly at the current open
        frontier and must be closed — committing chunks out of order or
        committing the open tail is a caller bug, not a recoverable state."""
        if not chunk.closed:
            raise ValueError("Cannot commit an open (still-growing) chunk")
        if chunk.start_sample != self._open_start:
            raise ValueError(
                f"Chunk starts at {chunk.start_sample}, expected the open "
                f"frontier {self._open_start} — commits must be in order"
            )
        self._committed.append(
            CommittedChunk(chunk.start_sample, chunk.end_sample, text.strip(), mean_confidence)
        )
        self._open_start = chunk.end_sample

    def low_confidence_indices(self, ceiling: float) -> List[int]:
        """Indices of committed chunks worth a confidence-targeted re-decode.

        A chunk with ``mean_confidence is None`` (the engine gave no word
        timestamps for that call) is never flagged — there is no signal to
        target the polish at, and guessing would defeat the point of gating
        on confidence at all.
        """
        return [
            i for i, c in enumerate(self._committed)
            if c.mean_confidence is not None and c.mean_confidence < ceiling
        ]

    def replace(self, index: int, text: str, mean_confidence: Optional[float]) -> None:
        """Overwrite one committed chunk's text — the polish pass only."""
        old = self._committed[index]
        self._committed[index] = CommittedChunk(
            old.start_sample, old.end_sample, text.strip(), mean_confidence
        )
