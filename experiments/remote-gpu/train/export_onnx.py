"""Runs ON Kaggle (shipped with the training data): a NeMo Parakeet model as the files onnx-asr loads."""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def export(folder: Path, net, onnx_repo: str, cache: Path) -> None:
    """Write the four files onnx-asr reads, in fp32 and int8, the way the published model was built.

    NeMo's export leaves `net` unfit to train (every layer frozen, the decoder's start token
    dropped), so never pass the model that is about to be trained."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    folder.mkdir(parents=True)
    net.cpu().eval().export(str(folder / "model.onnx"))  # writes encoder-model.onnx (+ .data) and decoder_joint-model.onnx
    for part in ("encoder-model", "decoder_joint-model"):
        # Unsigned weights, as the published int8 files have; signed cost 1.4 to 3.8 points of WER.
        quantize_dynamic(folder / f"{part}.onnx", folder / f"{part}.int8.onnx", weight_type=QuantType.QUInt8)
    vocab = "".join(f"{tok} {i}\n" for i, tok in enumerate([*net.tokenizer.vocab, "<blk>"]))
    published = Path(hf_hub_download(onnx_repo, "vocab.txt", local_dir=cache)).read_text(encoding="utf-8")
    if vocab.split() != published.split():
        raise SystemExit("token list differs from the published model's; onnx-asr would decode the wrong words")
    (folder / "vocab.txt").write_text(vocab, encoding="utf-8")
    shutil.copy(hf_hub_download(onnx_repo, "config.json", local_dir=cache), folder / "config.json")


def copy_precision(src: Path, dest: Path, quantization) -> list:
    """Only the files for the chosen precision go to `dest` (None = full precision); returns their names."""
    dest.mkdir()
    for f in src.iterdir():
        # fp32 keeps every non-int8 file, so the encoder's separate weights file comes along whatever it is named
        if f.name in ("vocab.txt", "config.json") or (".int8." in f.name) == bool(quantization):
            shutil.copy(f, dest / f.name)
    return sorted(p.name for p in dest.iterdir())
