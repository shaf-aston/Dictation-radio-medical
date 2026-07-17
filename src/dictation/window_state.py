"""Pure sliding-window / commit state machine for live transcription.

This is the heart of the live-speed design (see CLAUDE.md → "Live-speed
design"), extracted out of the Qt worker so it can be reasoned about and
unit-tested **without** PySide6, audio, or a Whisper model.

It owns nothing but plain Python state:

* ``committed_text`` / ``committed_samples`` — the frozen prefix and how far
  into the recording (in samples) it reaches.  Text at or before the commit
  frontier is never revised again.
* window geometry — where the next transcription window starts.
* commit advancement — which freshly-decoded segments are safely behind the
  frontier and can be frozen.

The Qt worker (:mod:`src.dictation.worker`) does the I/O and threading and
delegates every window/commit decision here.  Keeping this class pure is what
makes the commit-frontier arithmetic testable in isolation.
"""

from __future__ import annotations

from typing import List

from src.dictation.text_diff import trim_committed_tail

# ---------------------------------------------------------------------------
# Window geometry constant (shared with the worker's tuning block)
# ---------------------------------------------------------------------------
_OVERLAP_SEC = 3.0  # context overlap when sliding the window; the commit lag
#                     must exceed this so the window always re-covers the
#                     committed tail for lossless overlap dedup.


class WindowState:
    """Sliding-window bookkeeping for one live-transcription session.

    All timestamps are in seconds relative to the start of the recording;
    ``*_samples`` values are absolute sample offsets at the recording's
    samplerate.  The class performs no I/O.
    """

    def __init__(
        self,
        window_sec: float,
        commit_lag_sec: float,
        overlap_sec: float = _OVERLAP_SEC,
        pause_threshold: float = 2.5,
    ) -> None:
        self._overlap_sec = overlap_sec
        # Mirrors the transcriber's segment-join rule (src/dictation/
        # transcriber.py, transcribe(): gap >= pause_threshold -> newline) so
        # frozen text keeps the radiologist's paragraphing.
        self._pause_threshold = pause_threshold
        # The window ceiling must leave room for the overlap; the commit lag
        # must exceed the overlap so the window always re-covers the committed
        # tail (no word loss), and stay within the ceiling minus the overlap.
        self.window_sec = max(float(window_sec), overlap_sec * 2)
        self.commit_lag_sec = min(
            max(float(commit_lag_sec), overlap_sec), self.window_sec - overlap_sec
        )

        self.committed_text: str = ""
        self.committed_samples: int = 0

        # Bootstrap-commit state — the previous cycle's segments and where its
        # window started, used to freeze segments that slid out of view.
        self._prev_segments: List[dict] = []
        self._prev_chunk_start_sec: float = 0.0

    # ------------------------------------------------------------------
    # Window geometry
    # ------------------------------------------------------------------

    def window_start(self, total_samples: int, sr: int) -> int:
        """First sample of the window to transcribe this cycle.

        The window starts just before the commit frontier (with
        ``overlap_sec`` of context) so only un-committed audio plus a small
        overlap is re-transcribed each cycle — keeping the steady-state chunk
        near ``commit_lag + overlap`` rather than a full window.
        ``window_sec`` is only a hard ceiling: it caps the chunk when no text
        has committed yet, or when a long un-committable stretch (e.g. one
        unbroken monologue) would otherwise grow the chunk without bound.
        """
        ceiling_samples = int(self.window_sec * sr)
        if self.committed_samples > 0:
            overlap_samples = int(self._overlap_sec * sr)
            anchor = max(0, self.committed_samples - overlap_samples)
        else:
            anchor = 0
        earliest = max(0, total_samples - ceiling_samples)
        return max(anchor, earliest)

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def build_output(self, chunk_text: str, chunk_start_sec: float) -> str:
        """Combine the committed prefix with the new portion of this chunk."""
        if not self.committed_text or chunk_start_sec == 0.0:
            return chunk_text
        return trim_committed_tail(self.committed_text, chunk_text)

    def committed_prefix_len(self, output: str) -> int:
        """Length of the frozen prefix inside *output*, or 0 if it isn't one."""
        committed = self.committed_text
        if committed and output.startswith(committed):
            return len(committed)
        return 0

    # ------------------------------------------------------------------
    # Commit advancement
    # ------------------------------------------------------------------

    def _join_segments(self, parts: List[tuple], prev_end: float | None) -> str:
        """Join ``(abs_start, abs_end, text)`` parts, preserving paragraphs.

        A gap of ``pause_threshold`` or more between consecutive segments joins
        with a newline instead of a space — the same rule the transcriber uses
        so committed text keeps the radiologist's paragraphing.

        *prev_end* seeds the gap check for the FIRST part: the absolute end of
        the previously committed batch (or ``None`` when nothing is committed
        yet).  Without it the rule would never fire across commit batches —
        exactly the boundaries most likely to be paragraph breaks, since a
        pause longer than the per-cycle frontier advance always splits the two
        sides into separate ``advance_commit`` calls.  A leading ``"\\n"`` on
        the returned string is preserved by ``trim_committed_tail``.
        """
        pieces: List[str] = []
        for abs_start, abs_end, text in parts:
            if pieces:
                sep = "\n" if abs_start - prev_end >= self._pause_threshold else " "
                pieces.append(sep)
            elif (
                prev_end is not None
                and abs_start - prev_end >= self._pause_threshold
            ):
                pieces.append("\n")
            pieces.append(text)
            prev_end = abs_end
        return "".join(pieces)

    def advance_commit(
        self,
        segments: List[dict],
        chunk_start_sec: float,
        total_sec: float,
        sr: int,
    ) -> None:
        """Freeze segments that are safely behind the transcription frontier."""
        safe_abs = total_sec - self.commit_lag_sec
        current_end = self.committed_samples / sr
        if safe_abs <= current_end:
            return

        new_parts: List[tuple] = []
        new_committed_end = current_end
        for seg in segments:
            abs_end = chunk_start_sec + float(seg.get("end", 0))
            if abs_end > current_end and abs_end <= safe_abs:
                if text := (seg.get("text") or "").strip():
                    abs_start = chunk_start_sec + float(seg.get("start", 0))
                    new_parts.append((abs_start, abs_end, text))
                new_committed_end = abs_end

        if not new_parts:
            return

        addition = self._join_segments(
            new_parts, current_end if self.committed_text else None
        )
        # A boundary-spanning segment (start < frontier < end) re-covers already
        # committed words; dedup via trim_committed_tail so the frozen prefix
        # doesn't accumulate duplicates at every window slide.
        self.committed_text = (
            trim_committed_tail(self.committed_text, addition)
            if self.committed_text
            else addition
        )
        self.committed_samples = int(new_committed_end * sr)

    def record_segments(self, segments: List[dict], chunk_start_sec: float) -> None:
        """Remember this cycle's segments for a later bootstrap commit."""
        self._prev_segments = segments
        self._prev_chunk_start_sec = chunk_start_sec

    def maybe_bootstrap(self, new_chunk_start_sec: float, sr: int) -> None:
        """Seed the commit frontier from the previous window when it slides.

        Called before :meth:`advance_commit` has fired but the window has
        moved forward.  Commits prior-cycle segments whose absolute end
        timestamp is now before the new window's start, so the committed prefix
        keeps advancing even when VAD silenced the early audio in the new chunk.
        """
        if (
            new_chunk_start_sec <= self._prev_chunk_start_sec + 0.5
            or not self._prev_segments
        ):
            return

        parts: List[tuple] = []
        committed_end = self.committed_samples / sr
        new_end = committed_end
        for seg in self._prev_segments:
            abs_end = self._prev_chunk_start_sec + float(seg.get("end", 0))
            if abs_end <= new_chunk_start_sec and abs_end > new_end:
                if text := (seg.get("text") or "").strip():
                    abs_start = self._prev_chunk_start_sec + float(seg.get("start", 0))
                    parts.append((abs_start, abs_end, text))
                new_end = abs_end

        if not parts:
            return

        addition = self._join_segments(
            parts, committed_end if self.committed_text else None
        )
        self.committed_text = (
            trim_committed_tail(self.committed_text, addition)
            if self.committed_text
            else addition
        )
        self.committed_samples = int(new_end * sr)
