"""Pure chunk-cut policy: no Qt, no I/O, no clock, no audio decoding.

Turns VAD speech marks (relative to some audio range) into a list of chunk
boundaries. Every non-final boundary sits at a VAD-confirmed silence gap,
never at an arbitrary sample offset, so a closed chunk is never cut mid-word
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
from typing import List, Optional

from src.dictation.stream.vad import SAMPLE_RATE, SpeechMark


@dataclass(frozen=True)
class ChunkPolicy:
    """Tunable chunk-length targets, in seconds."""

    min_sec: float = 6.0          # never cut a chunk shorter than this at a pause
    soft_max_sec: float = 15.0    # prefer cutting by here if a pause is available
    force_cut_sec: float = 20.0   # hard ceiling: cut here even mid-speech
    # How much silence must sit after the LAST speech mark before that mark's
    # end counts as a cut point. Without this the final mark can never be a
    # cut (there is no following mark to prove the silence is real), so a
    # radiologist who stops to read the film gets nothing committed until they
    # speak again. Measured: 7s of speech then 1s of continuing silence closed
    # no chunk at all; 32ms of resumed speech closed it instantly.
    # 0.6s, not the 0.3s the VAD splits on: detect_speech pads every mark end
    # by speech_pad_ms, so apparent trailing silence understates the real gap,
    # and a breath mid-sentence must not read as the end of a thought.
    trailing_silence_sec: float = 0.6

    def __post_init__(self) -> None:
        """Reject lengths that are out of order or not positive.

        These come straight from user-editable settings, so they are checked
        here rather than trusted. Out-of-order values break the "never shorter
        than min_sec" promise below, and a force_cut_sec of 0 makes
        cut_chunks() loop forever on a zero-length chunk. Raising at
        record-start with a readable message beats a hung recording.
        """
        if not 0 < self.min_sec <= self.soft_max_sec <= self.force_cut_sec:
            raise ValueError(
                "chunk lengths must be 0 < min_sec <= soft_max_sec <= force_cut_sec, "
                f"got min={self.min_sec} soft_max={self.soft_max_sec} "
                f"force_cut={self.force_cut_sec}"
            )
        if self.trailing_silence_sec <= 0:
            raise ValueError(
                "trailing_silence_sec must be positive, got "
                f"{self.trailing_silence_sec}"
            )


@dataclass(frozen=True)
class Chunk:
    """One candidate chunk. ``closed=False`` marks the still-open tail: the
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

    A cut point is the end of a speech mark that is followed by real silence:
    that sample is guaranteed to be silence, so cutting there cannot split a
    word. Among the pauses that fall in ``[min_sec, soft_max_sec]`` after the
    current chunk start, the latest one is chosen (fewer, longer chunks
    amortise decode overhead better than many short ones); if none exists
    there, the earliest pause in ``(soft_max_sec, force_cut_sec)`` is used
    instead.

    Only if none of those can close the chunk does the *trailing* pause apply:
    the last mark has no successor to prove the silence after it is real, so
    the clock proves it instead, once ``policy.trailing_silence_sec`` of audio
    has arrived with no speech in it. That is what lets a chunk close while
    the speaker is still thinking; without it the last thing said before a
    pause waits for them to start talking again. It is deliberately a fallback
    and not another candidate: being always the latest, as a peer it won the
    "prefer the latest pause" rule every time and pushed cuts later.

    If nothing at all can close a chunk before ``force_cut_sec``, and the
    region contains speech, it is force-cut exactly there.

    The final region, from the last cut to ``total_samples``, is always
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

    # The last mark has no successor to prove the silence after it is real, so
    # the clock proves it instead: enough audio has arrived with no speech in
    # it. Kept SEPARATE from the ordinary candidates, and used only when none
    # of them can close the chunk. Measured on pause-heavy audio: as a peer it
    # is always the latest candidate, so the "prefer the latest pause" rule
    # picked it every time and pushed the cut later than an ordinary pause
    # would have. As a fallback it can only ever close a chunk that would
    # otherwise have stayed open, which is the whole point of it.
    trailing_cut: Optional[int] = None
    if marks:
        last_end = marks[-1].end_sample
        if total_samples - last_end >= int(policy.trailing_silence_sec * sr):
            trailing_cut = last_end

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
        elif trailing_cut is not None and min_pt <= trailing_cut < force_pt:
            # Nothing else can close this chunk, and the speaker has stopped.
            cut = trailing_cut
        elif force_pt < total_samples and any(m.end_sample > chunk_start for m in marks):
            # The force cut is the safety valve for an unbroken monologue with
            # no silence in it, so it may only fire on a region that actually
            # contains speech. Without the guard, closing on a trailing pause
            # leaves a silent tail that gets force-cut into a silent chunk and
            # decoded for nothing.
            cut = force_pt
        else:
            break

        chunks.append(Chunk(chunk_start, cut, closed=True))
        chunk_start = cut
        cut_candidates = [c for c in cut_candidates if c > chunk_start]
        if trailing_cut is not None and trailing_cut <= chunk_start:
            trailing_cut = None

    chunks.append(Chunk(chunk_start, total_samples, closed=False))
    return chunks
