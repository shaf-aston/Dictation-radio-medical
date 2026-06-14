"""Data-driven regression net for the correction pipeline.

Each ``input -> expected`` pair in ``tests/corpus/corrections.yaml`` is run
through the full post-processing pipeline. Adding a correction rule means
adding a pair here — the corpus then guarantees the fix holds forever and that
homophone rules never fire on the negative ("must not change") cases.

See docs/CORRECTIONS.md for the add-a-correction workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest
import yaml

from src.dictation.postprocess.pipeline import postprocess_transcript

_CORPUS = Path(__file__).parent / "corpus" / "corrections.yaml"


def _load_cases() -> List[Tuple[str, str, str]]:
    raw = yaml.safe_load(_CORPUS.read_text(encoding="utf-8")) or []
    cases = []
    for entry in raw:
        note = entry.get("note", "")
        label = entry.get("id") or note or entry["in"][:40]
        cases.append(pytest.param(entry["in"], entry["out"], id=label))
    return cases


@pytest.mark.parametrize("text_in, expected", _load_cases())
def test_correction_corpus(text_in: str, expected: str) -> None:
    assert postprocess_transcript(text_in) == expected
