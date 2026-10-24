"""Append-only committed-chunk ledger: the chunk-once guarantee.

Once a chunk closes (a VAD silence boundary was found, or the segmenter's
force-cut safety valve fired), its decoded text is frozen: nothing re-decodes
it and nothing rewrites it, except the one-time confidence-targeted polish
after recording stops. This is what turns "each second of audio decoded ~8
times" (the old sliding-window re-decode in window_state.py) into "each
second decoded once": the actual fix for the slow half of this rebuild.

Owns the absolute-sample bookkeeping; :mod:`src.dictation.stream.segmenter`
stays pure (0-based, no I/O) by operating only on the still-open tail each
cycle, which this class is responsible for slicing out.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

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
    #: Where the VAD heard speech in it (absolute samples); the chunk's own
    #: bounds when unknown. Pauses are measured between these.
    speech: Optional[Tuple[int, int]] = None
    #: This chunk ended at a force cut, not at a silence.
    forced_end: bool = False

    @property
    def speech_bounds(self) -> Tuple[int, int]:
        return self.speech or (self.start_sample, self.end_sample)


#: Characters that already end a sentence.
_SENTENCE_END = ".!?:;"
#: Closing marks a stop can hide behind: he said "no."
_TRAILING = '"\')]}”’'


def close_sentence(text: str) -> str:
    """End *text* with a full stop unless it already ends a sentence.

    A pause long enough to break the paragraph is the speaker finishing a
    thought. Whisper decodes each chunk in isolation and never hears the
    silence that followed, so it routinely leaves that last word bare --
    which is why a dictation reads back as one run-on line per paragraph,
    and why the capitalisation stage downstream has no sentence boundary to
    work from. A trailing comma is left alone: the speaker was mid-list, and
    a full stop there would claim more than the pause supports.
    """
    stripped = text.rstrip()
    if not stripped or stripped[-1] == ",":
        return stripped or text
    if stripped.rstrip(_TRAILING)[-1:] in _SENTENCE_END:
        return stripped
    return stripped + "."


class ChunkLedger:
    """Tracks which audio has been decoded-and-frozen versus still open.

    Every sample offset this class hands out or accepts, including
    :meth:`pending_cuts`'s return value, is **absolute** (from the start of
    the recording), never tail-relative. A cycle can close more than one
    chunk from a single VAD pass (catching up after a burst); if
    :meth:`commit` accepted tail-relative offsets, committing chunk N would
    shift ``open_start`` out from under chunk N+1's still tail-relative
    coordinates. Working in absolute samples throughout, and
    :meth:`commit` validating that a chunk starts exactly at the current
    frontier, makes that class of bug impossible instead of easy to
    reintroduce.
    """

    def __init__(
        self,
        policy: ChunkPolicy = ChunkPolicy(),
        pause_threshold: float = 2.5,
        sr: int = SAMPLE_RATE,
    ) -> None:
        self.policy = policy
        # Mirrors faster_whisper_engine.py's own segment-join rule (gap >= pause_threshold
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
    def sample_rate(self) -> int:
        """Samples per second of the audio these chunk windows count in.

        Public because a word time is in seconds and a chunk window is in
        samples: anything lining the two up (the after-Stop polish) needs this
        exact number, and a wrong one puts words in the wrong chunk while
        still looking plausible.
        """
        return self._sr

    @property
    def open_start_sample(self) -> int:
        """Where the still-open (undecided) tail begins."""
        return self._open_start

    @property
    def committed_text(self) -> str:
        """The committed chunks' text, a long pause between two of them read
        as the end of a sentence and a paragraph break.

        The pause is the silence between one chunk's last speech and the next
        one's first. Chunks tile the audio with no gaps, and cuts fall where
        speech ends, so measured between chunk boundaries every pause was 0
        (and a chunk with no text in the middle of a sentence read as one).
        """
        pieces: List[str] = []
        prev_end: Optional[int] = None
        for c in self._committed:
            if not c.text:
                # Heard but not words (a cough, a breath): not a pause either.
                if c.speech is not None and prev_end is not None:
                    prev_end = max(prev_end, c.speech[1])
                continue
            start, end = c.speech_bounds
            if prev_end is not None:
                if (start - prev_end) / self._sr >= self._pause_threshold:
                    pieces[-1] = close_sentence(pieces[-1])
                    pieces.append("\n")
                else:
                    pieces.append(" ")
            pieces.append(c.text)
            prev_end = end
        return "".join(pieces)

    def pending_cuts(self, total_samples: int, tail_marks: List[SpeechMark]) -> List[Chunk]:
        """Chunk boundaries for the current open tail, in absolute samples.

        *tail_marks* must be VAD marks computed over just the open tail
        (samples ``[open_start_sample, total_samples)``), not the whole
        recording: recomputing VAD over the whole growing buffer every cycle
        would itself become O(n^2) over a long dictation. They are relative to
        the open tail (0 = ``open_start_sample``); this method re-bases the
        segmenter's tail-relative output back to absolute samples before
        returning it, so every chunk a caller ever sees from this class is in
        the same coordinate frame.
        """
        base = self._open_start
        tail_len = max(0, total_samples - base)
        return [
            Chunk(base + c.start_sample, base + c.end_sample, c.closed, c.forced)
            for c in cut_chunks(tail_len, tail_marks, self.policy)
        ]

    def commit(
        self, chunk: Chunk, text: str, mean_confidence: Optional[float],
        speech: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Freeze one chunk. *chunk* offsets are absolute (as returned by
        :meth:`pending_cuts`); it must start exactly at the current open
        frontier and must be closed: committing chunks out of order or
        committing the open tail is a caller bug, not a recoverable state."""
        if not chunk.closed:
            raise ValueError("Cannot commit an open (still-growing) chunk")
        if chunk.start_sample != self._open_start:
            raise ValueError(
                f"Chunk starts at {chunk.start_sample}, expected the open "
                f"frontier {self._open_start}: commits must be in order"
            )
        self._committed.append(CommittedChunk(
            chunk.start_sample, chunk.end_sample, text.strip(), mean_confidence,
            speech=speech, forced_end=chunk.forced,
        ))
        self._open_start = chunk.end_sample

    def forced_cut_indices(self) -> List[int]:
        """Both chunks on each side of every force cut.

        A boundary not at a silence is never trusted: on the real engine the
        words at a 20-second force cut came out wrong in 5 of 12 clips
        ("impression" -> "compression") while both halves read as confident.
        """
        out = set()
        for i, c in enumerate(self._committed):
            if c.forced_end:
                out.add(i)
                if i + 1 < len(self._committed):
                    out.add(i + 1)
        return sorted(out)

    def low_confidence_indices(self, ceiling: float) -> List[int]:
        """Indices of committed chunks worth a confidence-targeted re-decode.

        A chunk with ``mean_confidence is None`` (the engine gave no word
        timestamps for that call) is never flagged: there is no signal to
        target the polish at, and guessing would defeat the point of gating
        on confidence at all.
        """
        return [
            i for i, c in enumerate(self._committed)
            if c.mean_confidence is not None and c.mean_confidence < ceiling
        ]

    def replace(self, index: int, text: str, mean_confidence: Optional[float]) -> bool:
        """Overwrite one committed chunk's text: the polish pass only.

        Refused (False) when it would empty a chunk that had text: an empty
        re-decode of audio already heard as speech is a failed decode, not
        silence. Applied, it deleted the words from the report, and with the
        background polish it took words back while the radiologist was still
        dictating.
        """
        old = self._committed[index]
        if old.text and not text.strip():
            return False
        self._committed[index] = replace(old, text=text.strip(), mean_confidence=mean_confidence)
        return True
