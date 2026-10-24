"""Incremental post-processing for live dictation.

The live worker re-emits the *whole* transcript every cycle (committed prefix +
the current window). Running the pipeline over that whole document
each time is quadratic over a session: by minute ten, every cycle re-scans ten
minutes of already-frozen text through ~180 terminology regexes and a 98k-term
fuzzy dictionary, only to produce the same output for it as last cycle.

:class:`IncrementalPostprocessor` fixes that by caching the processed form of
the frozen prefix and running the pipeline only over the *un-committed tail*.

Correctness is a property, tested (tests/test_live_postprocess.py): for every
text, the incremental result equals :func:`postprocess_transcript` on the whole
text. That holds because the text is only ever split at a **run of line
breaks**, the one place every stage already treats as a hard edge (a
hallucination line is ``^...$``, a cue window stops at a clause break,
capitalisation restarts), and because the line breaks are kept exactly as
spoken. Splitting at sentence ends, the rule this replaced, broke that: a
piece starting mid-line read ``^`` as a line start, and joining pieces with one
space lost every paragraph break, so the text handed back at Stop was not the
report the final pass would have made. A command that reaches back into
earlier text ("scratch that", "correct the word") sends the whole document
through, since only a whole-document pass can see what it reaches.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from src.core import perf
from src.dictation.postprocess.hallucinations import REMOVED_PATTERNS
from src.dictation.postprocess.voice_commands import SCRATCH_PATTERN, SPOKEN_COMMANDS
from src.dictation.postprocess.pipeline import (
    postprocess_transcript,
    postprocess_transcript_with_changes,
)

logger = logging.getLogger(__name__)

#: Where the pipeline is separable: a run of line breaks.
_PARAGRAPH_BREAK = re.compile(r"\n+")
#: Commands that edit text before themselves.
_REACHES_BACK = re.compile(
    rf"{SCRATCH_PATTERN.pattern}|\bcorrect(?:\s+the)?\s+word\b", re.IGNORECASE
)


#: What a spoken command writes (a break, a stop) joins with the text on
#: either side, and a command can even span a line break ("new\nparagraph");
#: a line the hallucination filter removes takes a break with it. No split
#: sits beside either.
_COMMAND = re.compile(SPOKEN_COMMANDS, re.IGNORECASE)


def _last_paragraph_start(text: str, limit: int) -> int:
    """Index just past the last run of line breaks ending at or before *limit*
    that no spoken command touches or spans; 0 when there is none."""
    commands = [m.span() for m in _COMMAND.finditer(text)] + [
        m.span() for pattern in REMOVED_PATTERNS for m in pattern.finditer(text)
    ]
    split = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if match.end() > limit:  # never inside a run: "\n" + "\n" is not "\n\n"
            break
        # The break and the punctuation or spaces around it: a command whose
        # span reaches into that is beside the break.
        left, right = match.start(), match.end()
        while left > 0 and not text[left - 1].isalnum():
            left -= 1
        while right < len(text) and not text[right].isalnum():
            right += 1
        if any(start <= right and end >= left for start, end in commands):
            continue
        split = match.end()
    return split


def _process_piece(text: str, process) -> str:
    """*process* the text of a piece, keeping its edge line breaks verbatim."""
    body = text.strip("\n")
    if not body.strip():
        return text
    lead = text[: len(text) - len(text.lstrip("\n"))]
    trail = text[len(text.rstrip("\n")):]
    return f"{lead}{process(body)}{trail}"


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
            ``(processed_text, changes)``: same contract as
            :func:`postprocess_transcript_with_changes`. ``changes`` covers the
            newly-processed tail only; callers already de-duplicate the
            cumulative list for the corrections banner.
        """
        split = _last_paragraph_start(text, committed_len)
        if _REACHES_BACK.search(text):
            split = 0
        if split <= 0:
            # Nothing safely frozen yet: process the whole document.
            self.reset()
            return self._process_with_changes(text)

        raw_prefix = text[:split]
        if not self._extend_prefix(raw_prefix):
            # The document was rewritten under us (final pass, manual edit):
            # the cache no longer describes it, so start clean.
            self.reset()
            return self._process_with_changes(text)

        processed_tail, changes = self._process_with_changes(text[split:])
        # The pipeline strips the document's outer whitespace; so does this.
        return (self._processed_prefix + processed_tail).strip(), changes

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
        self._processed_prefix += _process_piece(addition, self._process)
        self._raw_prefix = raw_prefix
        return True

    def _process(self, text: str) -> str:
        if not text.strip():
            return ""
        with perf.stage("postprocess.incremental_piece"):
            return postprocess_transcript(
                text, self.accent, self.cleanup_level
            )

    def _process_with_changes(self, text: str) -> Tuple[str, List[str]]:
        if not text.strip():
            return "", []
        with perf.stage("postprocess.incremental_tail"):
            return postprocess_transcript_with_changes(
                text, self.accent, self.cleanup_level
            )
