"""A tuned Parakeet folder (onnx-asr layout) rebuilt in the layout sherpa-onnx loads, so the
radiology hotwords (sherpa's beam search) can run on a tuned model, not only on stock.

    python experiments/remote-gpu/sherpa_view.py <model folder>     # prints the view folder

onnx-asr keeps the prediction network and the joint in one file (decoder_joint-model.onnx);
sherpa wants them apart (decoder.onnx, joiner.onnx), cut where NeMo's own separate export
cuts them: the decoder hands the joint its output as [batch, 640, steps]. The encoder is
the same network: only sherpa's metadata is added, and its weight files are hard-linked,
not copied. Output goes to cache_dir()/sherpa/<name>/ (derived, safe to delete).
"""
import os
import shutil
import sys
from pathlib import Path

import onnx
from onnx import helper, utils

# What sherpa reads from the encoder to set up features and the decoder (the same
# values its published Parakeet TDT exports carry; vocab_size is per model).
SHERPA_META = {"normalize_type": "per_feature", "pred_rnn_layers": "2", "pred_hidden": "640",
               "subsampling_factor": "8", "model_type": "EncDecRNNTBPEModel", "version": "2",
               "model_author": "NeMo", "feat_dim": "128",
               # sherpa treats a model as TDT (joiner = tokens + durations) only when this url contains "tdt"
               "url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"}
PRED_OUT = "/joint/Transpose_1_output_0"  # prediction-network output as the joint reads it: [batch, steps, 640]
FILES = ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")


def build(src: Path, dest: Path | None = None) -> Path:
    src = Path(src)
    if dest is None:  # the Kaggle kernel passes its own; it has no src/
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.core.paths import cache_dir
        # experiments keep a version's weights in <version>/model, the app in fine_tuned/<version>
        dest = cache_dir() / "sherpa" / (src.parent.name if src.name == "model" else src.name)
    if all((dest / f).exists() for f in FILES):
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    tokens = (src / "vocab.txt").read_text(encoding="utf-8")
    (dest / "tokens.txt").write_text(tokens, encoding="utf-8")  # same "piece id" lines, <blk> last

    enc = onnx.load(src / "encoder-model.onnx", load_external_data=False)
    # The encoder's weights sit beside it as separate files, some named by graph constants rather than
    # the weight list, so every file that is not a model, token list or config comes along.
    for f in src.iterdir():
        if f.suffix not in (".onnx", ".txt", ".json") and not (dest / f.name).exists():
            try:
                os.link(f, dest / f.name)
            except OSError:  # another drive: copy instead
                shutil.copy(f, dest / f.name)
    helper.set_model_props(enc, {**SHERPA_META, "vocab_size": str(len(tokens.splitlines()) - 1)})
    onnx.save(enc, dest / "encoder.onnx")

    # shape inference fills in the inner tensors the extractor needs to name as inputs/outputs
    dj = onnx.shape_inference.infer_shapes(onnx.load(src / "decoder_joint-model.onnx"))
    ex = utils.Extractor(dj)
    dec = ex.extract_model(["targets", "target_length", "input_states_1", "input_states_2"],
                           [PRED_OUT, "prednet_lengths", "output_states_1", "output_states_2"])
    dec.graph.node.append(helper.make_node("Transpose", [PRED_OUT], ["outputs"], perm=[0, 2, 1]))
    dec.graph.output[0].CopyFrom(helper.make_tensor_value_info("outputs", onnx.TensorProto.FLOAT, ["B", 640, "U"]))
    onnx.save(dec, dest / "decoder.onnx")

    joi = ex.extract_model(["encoder_outputs", PRED_OUT], ["outputs"])
    joi.graph.input[1].CopyFrom(helper.make_tensor_value_info("decoder_outputs", onnx.TensorProto.FLOAT, ["B", 640, "U"]))
    joi.graph.node.insert(0, helper.make_node("Transpose", ["decoder_outputs"], [PRED_OUT], perm=[0, 2, 1]))
    onnx.save(joi, dest / "joiner.onnx")
    for m in ("decoder.onnx", "joiner.onnx"):
        onnx.checker.check_model(str(dest / m))
    return dest


if __name__ == "__main__":
    print(build(Path(sys.argv[1])))
