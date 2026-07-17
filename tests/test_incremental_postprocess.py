"""Incremental post-processing must match whole-document processing.

The live path only re-processes the un-committed tail. That is only safe if the
result is the same text the old whole-document path produced — these tests are
the guard on that equivalence.
"""

from __future__ import annotations

import pytest

from src.dictation.postprocess.incremental import (
    IncrementalPostprocessor,
    _last_sentence_start,
)
from src.dictation.postprocess.pipeline import postprocess_transcript

TRANSCRIPT = (
    "there is a small pleural effusion on the right. "
    "the cardiomediastinal silhouette is normal. "
    "no pneumothorax is identified. "
    "there is a fracture of the fifth rib."
)


def whole(text: str) -> str:
    return postprocess_transcript(text, "neutral", "medium", live=True)


class TestSentenceSplit:
    def test_finds_the_last_boundary_before_the_limit(self) -> None:
        text = "one. two. three."
        assert _last_sentence_start(text, len("one. two.")) == len("one. ")

    def test_no_boundary_yields_zero(self) -> None:
        assert _last_sentence_start("an unbroken monologue with no stop", 20) == 0


class TestEquivalence:
    def test_matches_whole_document_for_a_growing_transcript(self) -> None:
        proc = IncrementalPostprocessor()
        # Simulate the worker: the transcript grows, and the commit frontier
        # trails behind the newest words.
        for end in range(20, len(TRANSCRIPT) + 1, 17):
            text = TRANSCRIPT[:end]
            committed_len = max(0, end - 40)  # frontier lags the live tail
            processed, _ = proc.process(text, committed_len)
            assert processed == whole(text)

    def test_matches_whole_document_with_no_commit(self) -> None:
        proc = IncrementalPostprocessor()
        processed, _ = proc.process(TRANSCRIPT, 0)
        assert processed == whole(TRANSCRIPT)

    def test_recovers_when_the_document_is_rewritten(self) -> None:
        # The final high-beam pass replaces the whole transcript; the cached
        # prefix no longer describes it and must be discarded, not concatenated.
        proc = IncrementalPostprocessor()
        proc.process(TRANSCRIPT, len(TRANSCRIPT) - 20)

        rewritten = "a completely different report. with new sentences."
        processed, _ = proc.process(rewritten, len(rewritten) - 10)

        assert processed == whole(rewritten)

    def test_reset_clears_the_cached_prefix(self) -> None:
        proc = IncrementalPostprocessor()
        proc.process(TRANSCRIPT, len(TRANSCRIPT) - 20)
        proc.reset()

        tail = "no acute abnormality."
        processed, _ = proc.process(tail, 0)

        assert processed == whole(tail)

    def test_configure_change_resets_the_cache(self) -> None:
        proc = IncrementalPostprocessor(accent="neutral", cleanup_level="medium")
        proc.process(TRANSCRIPT, len(TRANSCRIPT) - 20)

        proc.configure("neutral", "soft")
        processed, _ = proc.process(TRANSCRIPT, len(TRANSCRIPT) - 20)

        assert processed == postprocess_transcript(
            TRANSCRIPT, "neutral", "soft", live=True
        )

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_blank_input_is_handled(self, text: str) -> None:
        proc = IncrementalPostprocessor()
        processed, changes = proc.process(text, 0)
        assert processed == ""
        assert changes == []
