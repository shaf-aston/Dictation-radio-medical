"""Pure chunk-cut policy — no Qt, no I/O, no clock, no audio decoding.

Turns VAD speech marks (relative to some audio range) into a list of chunk
boundaries. Every non-final boundary sits at a VAD-confirmed silence gap —
never at an arbitrary sample offset — so a closed chunk is never cut mid-word
except via the ``force_cut_sec`` safety valve, which exists only for a single
unbroken speech run with no silence at all (a long monologue). That is the
plan's known accepted risk for this milestone (see the M2 pre-mortem in
CLAUDE.md's linked plan); the confidence-targeted polish after recording
stops is the safety net for it.

Trivially unit-testable: feed it marks, check the returned cut points. Takes
sample counts relative to whatever range the caller is segmenting (the ledger
passes tail-relative offsets so this function never needs to know about the
committed prefix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.dictation.stream.vad import SAMPLE_RATE, SpeechMark


@dataclass(frozen=True)
class ChunkPolicy:
    """Tunable chunk-length targets, in seconds."""

    min_sec: float = 6.0          # never cut a chunk shorter than this at a pause
    soft_max_sec: float = 15.0    # prefer cutting by here if a pause is available
    force_cut_sec: float = 20.0   # hard ceiling — cut here even mid-speech


@dataclass(frozen=True)
class Chunk:
    """One candidate chunk. ``closed=False`` marks the still-open tail — the
    caller must not decode-and-freeze it, only re-decode it for live display.
    """

    start_sample: int
    end_sample: int
    closed: bool

    @property
    def length_samples(self) -> int:
        return self.end_sample - self.start_sample


def cut_chunks(
    total_samples: int,
    marks: List[SpeechMark],
    policy: ChunkPolicy = ChunkPolicy(),
    sr: int = SAMPLE_RATE,
) -> List[Chunk]:
    """Cut ``[0, total_samples)`` into closed chunks plus one open tail.

    A cut point is the end of a speech mark that is followed by real silence
    (i.e. the next mark, if any, starts later) — that sample is guaranteed to
    be silence, so cutting there cannot split a word. Among the pauses that
    fall in ``[min_sec, soft_max_sec]`` after the current chunk start, the
    latest one is chosen (fewer, longer chunks amortise decode overhead
    better than many short ones); if none exists there, the earliest pause in
    ``(soft_max_sec, force_cut_sec)`` is used instead; if no pause exists at
    all before ``force_cut_sec``, the chunk is force-cut exactly there.

    The final region — from the last cut to ``total_samples`` — is always
    returned as the open tail (``closed=False``), even if empty-length is
    impossible (it is omitted only when ``total_samples <= 0``).
    """
    if total_samples <= 0:
        return []

    min_s = int(policy.min_sec * sr)
    soft_s = int(policy.soft_max_sec * sr)
    force_s = int(policy.force_cut_sec * sr)

    cut_candidates = [
        marks[i].end_sample
        for i in range(len(marks) - 1)
        if marks[i].end_sample < marks[i + 1].start_sample
    ]

    chunks: List[Chunk] = []
    chunk_start = 0
    while True:
        min_pt = chunk_start + min_s
        soft_pt = chunk_start + soft_s
        force_pt = chunk_start + force_s

        in_range = [c for c in cut_candidates if min_pt <= c < force_pt]
        within_soft = [c for c in in_range if c <= soft_pt]

        if within_soft:
            cut = within_soft[-1]
        elif in_range:
            cut = in_range[0]
        elif force_pt < total_samples:
            cut = force_pt
        else:
            break

        chunks.append(Chunk(chunk_start, cut, closed=True))
        chunk_start = cut
        cut_candidates = [c for c in cut_candidates if c > chunk_start]

    chunks.append(Chunk(chunk_start, total_samples, closed=False))
    return chunks
