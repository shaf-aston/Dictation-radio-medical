"""AsrEngine adapter: Parakeet TDT with radiology hotwords, on sherpa-onnx.

The same model as :mod:`parakeet_engine` (NVIDIA Parakeet TDT 0.6B v2, int8),
run by a different runtime for one reason: onnx-asr decodes greedily and has
no way to bias the decoder, while sherpa-onnx can boost a list of phrases
during beam search. The list is the curated radiology lexicon, so the model
leans toward "steatosis" when it is unsure between that and "stenosis",
instead of a text rule guessing afterwards.

Measured 2026-10-23 on the whole piper set (54 min), against the same model
greedy: medical-term error 17.0% -> 11.9% raw, 9.7% -> 7.7% after the
post-processing pipeline, WER 5.3% -> 4.5%; decode time 0.11 -> 0.15 x real time. A higher score (2.0)
found more terms but inserted them where they were not said (WER rose), which
is why ``hotword_score`` defaults to 1.0.

What it needs, all fetched once into ``data/cache/sherpa/`` and offline after:
the int8 encoder / decoder / joiner and ``tokens.txt`` from the sherpa-onnx
export. ``bpe.vocab`` (which sherpa needs to split a hotword into the model's
pieces) is derived from ``tokens.txt``: the tokenizer's scores are its pieces'
ranks, so nothing larger has to be downloaded.

A fine-tuned export (picked with ``parakeet_model``) is in onnx-asr's layout and still
runs through :mod:`parakeet_engine`; the factory picks. ``model_dir`` can also name a
full-precision folder in sherpa's layout (``encoder.onnx``, ``decoder.onnx``,
``joiner.onnx``; experiments/remote-gpu/sherpa_view.py rebuilds a fine-tune that way).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from src.dictation.asr.engines.parakeet_engine import MIN_SAMPLES, merge_pieces
from src.dictation.asr.types import (
    LOCAL_DECODE_COST,
    AsrResult,
    AsrSegment,
    EngineCaps,
    ProviderUnavailable,
    TranscribeContext,
)

logger = logging.getLogger(__name__)

#: The sherpa-onnx export of the app's default Parakeet (see parakeet_engine).
REPO_ID = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
#: The upstream commit and file checksums this engine was measured with. A download is
#: pinned to the commit and every file is checked before use: a file that differs is
#: deleted and the engine refuses to load (downloaded artifacts are validated).
_REVISION = "1ab9323565ddb038682214b292f588070a538ce2"
_FILES = {
    "encoder.int8.onnx": "a32b12d17bbbc309d0686fbbcc2987b5e9b8333a7da83fa6b089f0a2acd651ab",
    "decoder.int8.onnx": "b6bb64963457237b900e496ee9994b59294526439fbcc1fecf705b31a15c6b4e",
    "joiner.int8.onnx": "7946164367946e7f9f29a122407c3252b680dbae9a51343eb2488d057c3c43d2",
    "tokens.txt": "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
}
SAMPLE_RATE = 16000
#: Beam width for modified beam search. Hotwords only act during beam search;
#: 4 measured the same speed as greedy on this CPU.
_BEAM = 4

_RECOGNISERS: dict = {}
_LOCK = threading.Lock()


def _download(root: Path) -> None:
    """Fetch the pinned sherpa export into *root*; delete and refuse any file whose checksum differs."""
    import hashlib  # noqa: PLC0415

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    logger.info("Downloading %s into %s (one time)", REPO_ID, root)
    snapshot_download(REPO_ID, revision=_REVISION, local_dir=root, allow_patterns=list(_FILES))
    for name, want in _FILES.items():
        with open(root / name, "rb") as fh:
            got = hashlib.file_digest(fh, "sha256").hexdigest()
        if got != want:
            (root / name).unlink()
            raise ProviderUnavailable(f"{name} from {REPO_ID} does not match its pinned checksum; deleted it")


def write_bpe_vocab(tokens_txt: Path, out: Path) -> None:
    """Write sherpa's ``bpe.vocab`` (piece TAB score) from ``tokens.txt``.

    The SentencePiece vocab of this model scores each piece by its rank:
    ``<unk>`` 0, then 0, -1, -2 ... in id order. ``tokens.txt`` holds the
    same pieces in id order (plus ``<blk>``, which is not a piece), so the
    scores can be rebuilt exactly.
    """
    lines = []
    for raw in tokens_txt.read_text(encoding="utf-8").splitlines():
        piece, _, index = raw.rpartition(" ")
        if not piece or piece == "<blk>":
            continue
        i = int(index)
        lines.append(f"{piece}\t{0 if i == 0 else f'-{i - 1}'}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hotwords(terms: Iterable[str], out: Path) -> int:
    """One lower-case phrase per line, as the model writes them mid-sentence."""
    unique = sorted({" ".join(t.lower().split()) for t in terms if t and t.strip()})
    out.write_text("\n".join(unique) + "\n", encoding="utf-8")
    return len(unique)


class ParakeetHotwordEngine:
    """Parakeet TDT through sherpa-onnx, beam search boosted by radiology terms."""

    def __init__(self, hotword_score: float = 1.0, model_dir: Optional[Path] = None,
                 threads: int = 4) -> None:
        self.hotword_score = hotword_score
        self.model_dir = model_dir
        self.threads = threads
        self._recogniser: Any = None

    # -- port ---------------------------------------------------------------

    def identity(self) -> tuple:
        # The folder is part of it: two engines on different models must not share a recogniser.
        return ("parakeet-hotwords", str(self.model_dir or REPO_ID), self.hotword_score)

    def capabilities(self) -> EngineCaps:
        # Beam-search scores here include the hotword bonus, so they are not
        # probabilities: words carry timings, and no confidence.
        return EngineCaps(word_confidence=False, hotwords=True, cost=LOCAL_DECODE_COST)

    def preload(self) -> None:
        if self._recogniser is not None:
            return
        key = self.identity()
        with _LOCK:
            if key not in _RECOGNISERS:
                try:
                    _RECOGNISERS[key] = self._load()
                except Exception as exc:
                    # Offline with nothing cached, retrying would re-attempt a
                    # download every live cycle: give up until restart.
                    raise ProviderUnavailable(f"Parakeet (hotwords) failed to load: {exc}") from exc
            self._recogniser = _RECOGNISERS[key]

    def transcribe(self, audio: Any, ctx: TranscribeContext) -> AsrResult:
        import numpy as np  # noqa: PLC0415

        if isinstance(audio, (str, Path)):
            import soundfile as sf  # noqa: PLC0415
            samples, rate = sf.read(str(audio), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
        else:
            samples, rate = np.asarray(audio, dtype=np.float32), SAMPLE_RATE
        if samples.size < MIN_SAMPLES:
            return AsrResult(text="")

        self.preload()
        stream = self._recogniser.create_stream()
        stream.accept_waveform(rate, samples)
        self._recogniser.decode_stream(stream)
        result = stream.result
        text = (result.text or "").strip()
        if not text:
            return AsrResult(text="")
        words = merge_pieces(result.tokens, result.timestamps, None)
        end = words[-1].end if words else 0.0
        return AsrResult(text=text, segments=(AsrSegment(text=text, start=0.0, end=end, words=words),))

    # -- internals ----------------------------------------------------------

    def _load(self) -> Any:
        import sherpa_onnx  # noqa: PLC0415 - optional dependency

        from src.core.paths import cache_dir  # noqa: PLC0415
        from src.medical.medical_dict import get_correction_targets  # noqa: PLC0415

        root = self.model_dir or cache_dir() / "sherpa" / REPO_ID.split("/")[-1]
        if self.model_dir is None and not all((root / f).exists() for f in _FILES):
            _download(root)
        write_bpe_vocab(root / "tokens.txt", root / "bpe.vocab")
        count = write_hotwords(get_correction_targets(), root / "hotwords.txt")
        logger.info("Parakeet (sherpa-onnx) with %d radiology hotwords, score %.1f",
                    count, self.hotword_score)
        part = ".onnx" if (root / "encoder.onnx").exists() else ".int8.onnx"  # full precision when present
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(root / f"encoder{part}"),
            decoder=str(root / f"decoder{part}"),
            joiner=str(root / f"joiner{part}"),
            tokens=str(root / "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=self.threads,
            decoding_method="modified_beam_search",
            max_active_paths=_BEAM,
            hotwords_file=str(root / "hotwords.txt"),
            hotwords_score=self.hotword_score,
            modeling_unit="bpe",
            bpe_vocab=str(root / "bpe.vocab"),
        )
