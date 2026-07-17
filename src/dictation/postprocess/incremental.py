"""Incremental post-processing for live dictation.

The live worker re-emits the *whole* transcript every cycle (committed prefix +
the current window). Running the ten-stage pipeline over that whole document
each time is quadratic over a session: by minute ten, every cycle re-scans ten
minutes of already-frozen text through ~180 terminology regexes and a 98k-term
fuzzy dictionary, only to produce the same output for it as last cycle.

:class:`IncrementalPostprocessor` fixes that by caching the processed form of
the frozen prefix and running the pipeline only over the *un-committed tail*.

Correctness is preserved by where it splits. Several stages are
context-sensitive at their edges — capitalisation depends on being at a
sentence start, terminology phrases can span words — so the text is only ever
split at a **sentence boundary at or before the commit frontier**. Each piece
handed to the pipeline therefore begins exactly where a sentence begins, which
is the same condition it would have met inside the full document. When no such
boundary exists yet (a long unbroken monologue), it falls back to processing
the whole document, so behaviour is never *wrong*, only unaccelerated.

The final post-recording pass still reprocesses the whole document from
scratch, so any drift is corrected before the report is saved.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from src.core import perf
from src.dictation.postprocess.pipeline import (
    postprocess_transcript,
    postprocess_transcript_with_changes,
)

logger = logging.getLogger(__name__)

# End of a sentence: terminal punctuation followed by whitespace, or a newline.
# The split index is the START of the next sentence.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _last_sentence_start(text: str, limit: int) -> int:
    """Index of the last sentence start at or before *limit*, else 0."""
    split = 0
    for match in _SENTENCE_END_RE.finditer(text, 0, max(limit, 0)):
        split = match.end()
    return split


class IncrementalPostprocessor:
    """Stateful post-processor that only re-runs the pipeline on new text.

    One instance per dictation session, per consumer (the desktop postprocess
    thread and the web live handler each hold their own). Not thread-safe: it
    is owned by the single thread that processes transcripts.
    """

    def __init__(self, accent: str = "neutral", cleanup_level: str = "medium") -> None:
        self.accent = accent
        self.cleanup_level = cleanup_level
        self._raw_prefix: str = ""
        self._processed_prefix: str = ""

    def reset(self) -> None:
        """Drop the cache (new recording, or settings changed mid-session)."""
        self._raw_prefix = ""
        self._processed_prefix = ""

    def configure(self, accent: str, cleanup_level: str) -> None:
        """Update the pipeline options, resetting the cache if they changed."""
        if accent != self.accent or cleanup_level != self.cleanup_level:
            self.accent = accent
            self.cleanup_level = cleanup_level
            self.reset()

    def process(self, text: str, committed_len: int = 0) -> Tuple[str, List[str]]:
        """Post-process *text*, reusing cached work for the committed prefix.

        Args:
            text: The full raw transcript emitted by the worker.
            committed_len: Length, in characters of *text*, of the frozen
                prefix the worker will never revise. ``0`` disables the
                optimisation for this call.

        Returns:
            ``(processed_text, changes)`` — same contract as
            :func:`postprocess_transcript_with_changes`. ``changes`` covers the
            newly-processed tail only; callers already de-duplicate the
            cumulative list for the corrections banner.
        """
        split = _last_sentence_start(text, committed_len)
        if split <= 0:
            # Nothing safely frozen yet — process the whole document.
            self.reset()
            return self._process_with_changes(text)

        raw_prefix = text[:split]
        if not self._extend_prefix(raw_prefix):
            # The document was rewritten under us (final pass, manual edit):
            # the cache no longer describes it, so start clean.
            self.reset()
            return self._process_with_changes(text)

        tail = text[split:]
        processed_tail, changes = self._process_with_changes(tail)
        return self._join(self._processed_prefix, processed_tail), changes

    # -- internals ------------------------------------------------------

    def _extend_prefix(self, raw_prefix: str) -> bool:
        """Bring the cached prefix up to *raw_prefix*. False if incompatible.

        Only the *newly frozen* span is pushed through the pipeline, so the
        committed text is processed exactly once over the whole session.
        """
        if raw_prefix == self._raw_prefix:
            return True
        if not raw_prefix.startswith(self._raw_prefix):
            return False

        addition = raw_prefix[len(self._raw_prefix):]
        processed_addition = self._process(addition)
        self._processed_prefix = self._join(
            self._processed_prefix, processed_addition
        )
        self._raw_prefix = raw_prefix
        return True

    @staticmethod
    def _join(left: str, right: str) -> str:
        """Concatenate two processed pieces across the sentence boundary."""
        if not left:
            return right
        if not right:
            return left
        return f"{left.rstrip()} {right.lstrip()}"

    def _process(self, text: str) -> str:
        if not text.strip():
            return ""
        with perf.stage("postprocess.incremental_piece"):
            return postprocess_transcript(
                text, self.accent, self.cleanup_level, live=True
            )

    def _process_with_changes(self, text: str) -> Tuple[str, List[str]]:
        if not text.strip():
            return "", []
        with perf.stage("postprocess.incremental_tail"):
            return postprocess_transcript_with_changes(
                text, self.accent, self.cleanup_level, live=True
            )
