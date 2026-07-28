"""The shipped Whisper prompt must fit the slot faster-whisper actually gives it.

faster-whisper keeps only the LAST ``448 // 2 - 1`` = 223 tokens of
``initial_prompt`` (``faster_whisper/transcribe.py::get_prompt`` slices
``previous_tokens[-(max_length // 2 - 1):]``) and discards the rest without a
word. The prompt file was ~1150 tokens, so 81% of it — the whole
musculoskeletal section, which sat at the front — never reached the decoder.
This is the assertion that would have caught that.

Tokenising needs the real Whisper BPE vocabulary, and two things get in the
way of just importing it:

* ``tests/conftest.py`` stubs out ``faster_whisper``, so importing the
  tokenizer through that package inside the suite yields the stub.
* Building it from a ``WhisperModel`` downloads the model, and tests must not
  touch the network.

So this reads the ``tokenizer.json`` of whichever Whisper model is already in
the project's own download cache (``data/cache/whisper/``) with the
``tokenizers`` package — a hard dependency of faster-whisper, and not stubbed.
Same vocabulary, same ids, no network. With an empty cache there is no honest
way to count tokens, so the test SKIPS loudly rather than passing: run any
transcription once (or ``python scripts/verify_setup.py``) to populate it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dictation.transcriber import RADIOLOGY_PROMPT

# faster_whisper/transcribe.py::get_prompt — max_length // 2 - 1, max_length=448.
PROMPT_TOKEN_BUDGET = 448 // 2 - 1

# Resolved here rather than via features/file_manager.whisper_cache_dir(),
# because conftest redirects the data dir to an empty tmp folder for isolation.
# tests/ sits at the project root, so parents[1] is the root.
_WHISPER_CACHE = Path(__file__).resolve().parents[1] / "data" / "cache" / "whisper"


def _whisper_tokenizer():
    """The real Whisper BPE from the local model cache, or skip."""
    tokenizers = pytest.importorskip(
        "tokenizers", reason="tokenizers (a faster-whisper dependency) is not installed"
    )
    vocab_files = sorted(_WHISPER_CACHE.glob("**/tokenizer.json"))
    if not vocab_files:
        pytest.skip(
            f"No Whisper tokenizer.json under {_WHISPER_CACHE} — the prompt budget "
            "cannot be measured. Run a transcription once to populate the cache."
        )
    return tokenizers.Tokenizer.from_file(str(vocab_files[0]))


def test_shipped_prompt_fits_the_decoder_budget() -> None:
    """The whole prompt reaches the decoder; nothing is silently truncated."""
    tokenizer = _whisper_tokenizer()
    # get_prompt encodes " " + text.strip(), so measure exactly that.
    n_tokens = len(
        tokenizer.encode(" " + RADIOLOGY_PROMPT.strip(), add_special_tokens=False).ids
    )
    assert n_tokens <= PROMPT_TOKEN_BUDGET, (
        f"radiology_prompt.txt is {n_tokens} tokens; faster-whisper keeps only the "
        f"last {PROMPT_TOKEN_BUDGET}, so the first {n_tokens - PROMPT_TOKEN_BUDGET} "
        "would be dropped without warning. Delete terms to pay for additions."
    )


def test_learned_terms_never_push_the_shipped_dictionary_out(monkeypatch) -> None:
    """A full custom vocabulary costs learned terms, never the curated ones.

    The prompt the worker actually sends is the shipped vocabulary plus whatever
    the user's own corrections have added, and the two share this one 223-token
    slot. `adaptive_learning` caps its suffix at 80 terms, which on real radiology
    words is ~216 tokens — enough to evict nearly the whole shipped dictionary on
    its own. Since Whisper keeps the LAST 223 tokens, the survivor is whichever
    text is written last, so this pins that order.
    """
    tokenizer = _whisper_tokenizer()
    from src.dictation import worker

    # Deliberately larger than the whole budget, so something must be dropped.
    monkeypatch.setattr(
        worker, "get_custom_prompt_suffix", lambda: " ".join(["supraspinatus"] * 300)
    )
    composed = worker.LiveTranscribeWorker._build_context_prompt(object())

    kept_ids = tokenizer.encode(" " + composed.strip(), add_special_tokens=False).ids[
        -PROMPT_TOKEN_BUDGET:
    ]
    kept_text = tokenizer.decode(kept_ids)

    opening = " ".join(RADIOLOGY_PROMPT.split()[:6])
    assert opening in kept_text, (
        "the shipped radiology prompt was truncated away by the learned terms — "
        f"the decoder would only have seen: {kept_text[:120]!r}"
    )


def test_comment_lines_are_stripped_from_the_prompt() -> None:
    """The budget is only respected if the header never becomes prompt text.

    The file's explanatory header is roughly as long as its content, so a
    regression in the loader's comment stripping would blow the budget while
    the file itself still looked fine.
    """
    assert "#" not in RADIOLOGY_PROMPT
    assert "BUDGET" not in RADIOLOGY_PROMPT
