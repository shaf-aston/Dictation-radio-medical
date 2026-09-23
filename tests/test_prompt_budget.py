"""faster-whisper keeps only the last 223 tokens of initial_prompt and silently
drops the rest (faster_whisper/transcribe.py::get_prompt:
previous_tokens[-(448 // 2 - 1):]). This file exceeding that budget is a silent
regression: the front of the prompt (whatever was added last) would never
reach the decoder. See src/dictation/resources/radiology_prompt.txt's header.
"""

import pytest

from src.dictation.asr.prompt import RADIOLOGY_PROMPT

_BUDGET = 223


def _token_count(text: str) -> int:
    tokenizers = pytest.importorskip("tokenizers")
    from faster_whisper.tokenizer import Tokenizer

    hf = tokenizers.Tokenizer.from_pretrained("openai/whisper-tiny.en")
    tok = Tokenizer(hf, multilingual=False, task="transcribe", language=None)
    return len(tok.encode(text))


def test_radiology_prompt_fits_token_budget():
    try:
        count = _token_count(RADIOLOGY_PROMPT)
    except Exception as exc:  # offline env with no cached tokenizer files
        pytest.skip(f"whisper tokenizer unavailable: {exc}")
    assert count <= _BUDGET, (
        f"radiology_prompt.txt is {count} tokens, over the {_BUDGET}-token "
        "budget faster-whisper keeps; the front of the prompt would be "
        "silently truncated. Trim a term."
    )
