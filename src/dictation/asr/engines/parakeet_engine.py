"""AsrEngine adapter over NVIDIA Parakeet TDT, running on onnxruntime (M3).

The second engine behind the swap-seam, and the reason the seam exists: it is
a different architecture (a Transducer, not an encoder-decoder), so nothing it
does resembles Whisper except the shape of what comes out.

Three differences the caller has to know about, because they are not defects to
be papered over:

* **No prompt.** Whisper is primed with the 223-token radiology vocabulary
  (``RADIOLOGY_PROMPT``); a Transducer has no prompt slot at all, so
  ``ctx.initial_prompt`` and ``ctx.hotwords`` are ignored and
  :meth:`capabilities` says ``hotwords=False``. Parakeet meets medical
  vocabulary with nothing but what it learned in training.
* **Greedy decoding only.** ``ctx.beam_size`` is ignored — onnx-asr ships
  greedy search for every architecture it supports.
* **Its own VAD is not used.** ``ctx.vad_filter`` is ignored; the app's
  streaming layer (``dictation/stream/``) already owns segmentation.

Word confidence *is* real here. Parakeet emits sub-word pieces with a log
probability each, which merge into per-word probabilities — the same quantity
faster-whisper reports — so :meth:`capabilities` honestly says
``word_confidence=True``.

Everything heavy is imported lazily: with ``onnx-asr`` absent the rest of the
app is unaffected, and asking for this engine fails with an install hint rather
than an ImportError at startup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from src.dictation.asr.types import AsrResult, AsrSegment, EngineCaps, TranscribeContext, Word
from src.features.file_manager import onnx_asr_cache_dir

logger = logging.getLogger(__name__)

#: The onnx-asr model id this engine defaults to. Pinned here (with the pinned
#: ``onnx-asr`` version in requirements.txt) so a download is reproducible:
#: the id resolves to one HuggingFace repo, and the copy under data/cache/ is
#: fetched exactly once and reused offline from then on.
DEFAULT_MODEL = "nemo-parakeet-tdt-0.6b-v3"

#: int8 by default: the fp32 encoder alone is 2.4 GB against 652 MB quantised,
#: and faster-whisper already runs int8 on this CPU — matching the quantisation
#: is what makes an engine comparison measure the engine.
DEFAULT_QUANTIZATION = "int8"

#: Parakeet's fixed input rate. Anything else is resampled by onnx-asr.
SAMPLE_RATE = 16000

#: onnx-asr has already turned SentencePiece's U+2581 word marker into a plain
#: leading space by the time tokens reach us, so a piece that starts with a
#: space is the start of a word (" Ch", "est", " radi", "ogra", "ph", ".").
_WORD_START = " "

_MISSING_HINT = (
    "The 'parakeet' ASR engine needs the onnx-asr package: pip install onnx-asr"
)


class ParakeetEngine:
    """Runs Parakeet TDT through onnx-asr to satisfy the :class:`AsrEngine` port."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        quantization: Optional[str] = DEFAULT_QUANTIZATION,
    ) -> None:
        self.model_name = model_name
        self.quantization = quantization
        self._model: Any = None

    # -- port ---------------------------------------------------------------

    def preload(self) -> None:
        """Download (once) and load the model, so no transcribe() pays for it."""
        if self._model is None:
            self._model = self._load()

    def capabilities(self) -> EngineCaps:
        return EngineCaps(word_confidence=True, hotwords=False)

    def transcribe(self, audio: Any, ctx: TranscribeContext) -> AsrResult:
        source = _as_input(audio)
        if source is None:  # empty waveform - nothing to decode
            return AsrResult(text="")

        self.preload()
        result = self._model.recognize(source, sample_rate=SAMPLE_RATE)

        text = (result.text or "").strip()
        if not text:
            return AsrResult(text="")

        words = _merge_pieces(result.tokens, result.timestamps, result.logprobs)
        end = words[-1].end if words else 0.0
        # One segment: a Transducer emits a continuous token stream with no
        # segment concept of its own, and inventing boundaries here would be a
        # guess dressed up as engine output.
        return AsrResult(
            text=text,
            segments=(AsrSegment(text=text, start=0.0, end=end, words=words),),
        )

    # -- internals ----------------------------------------------------------

    def _load(self) -> Any:
        try:
            import onnx_asr  # noqa: PLC0415 - lazy: the app runs without it
        except ImportError as exc:
            raise RuntimeError(_MISSING_HINT) from exc

        path = onnx_asr_cache_dir(f"{self.model_name}-{self.quantization or 'fp32'}")
        if not path.exists():
            logger.info("Downloading %s into %s (one time)", self.model_name, path)
        return onnx_asr.load_model(
            self.model_name, path, quantization=self.quantization
        ).with_timestamps()


def _as_input(audio: Any) -> Any:
    """Normalise the port's audio argument into something onnx-asr accepts.

    Returns ``None`` for empty audio so the caller can skip the model
    entirely — a zero-length waveform crashes the log-mel preprocessor rather
    than transcribing to nothing.
    """
    if isinstance(audio, (str, Path)):
        return str(audio)

    import numpy as np  # noqa: PLC0415 - kept out of module import for symmetry

    samples = np.asarray(audio)
    if samples.size == 0:
        return None
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    return samples


def _merge_pieces(
    tokens: Optional[Sequence[str]],
    timestamps: Optional[Sequence[float]],
    logprobs: Optional[Sequence[float]],
) -> Tuple[Word, ...]:
    """Merge sub-word pieces into words, each with its own probability.

    A word's confidence is the geometric mean of its pieces' probabilities
    (``exp`` of the mean log probability), which is the same quantity
    faster-whisper reports per word, so the two engines' numbers mean the same
    thing to the correction gate downstream.

    Parakeet timestamps a piece's *start* only. A word therefore ends where the
    next one starts; the last word ends at its own start. Good enough to order
    and locate words, not to measure the pause between them — which is why the
    streaming layer keeps owning segmentation.
    """
    if not tokens or not timestamps or not logprobs:
        return ()
    if not (len(tokens) == len(timestamps) == len(logprobs)):
        # Fail visibly rather than zip-truncating into silently misaligned
        # confidences, which would mislabel which word the decoder doubted.
        logger.warning(
            "Parakeet returned %d tokens, %d timestamps, %d logprobs - "
            "dropping word confidence for this utterance",
            len(tokens), len(timestamps), len(logprobs),
        )
        return ()

    import math  # noqa: PLC0415

    groups: List[List[int]] = []
    for i, token in enumerate(tokens):
        if token.startswith(_WORD_START) or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)

    words: List[Word] = []
    for group in groups:
        text = "".join(tokens[i] for i in group).strip()
        if not text:
            continue
        mean_logprob = sum(logprobs[i] for i in group) / len(group)
        words.append(Word(
            text=text,
            start=float(timestamps[group[0]]),
            end=float(timestamps[group[0]]),
            confidence=math.exp(mean_logprob),
        ))

    for i in range(len(words) - 1):
        words[i] = Word(
            text=words[i].text,
            start=words[i].start,
            end=words[i + 1].start,
            confidence=words[i].confidence,
        )
    return tuple(words)
