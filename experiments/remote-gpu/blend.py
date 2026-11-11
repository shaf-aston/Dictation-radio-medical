"""Blend a tuned Parakeet with the stock one, weight by weight (WiSE-FT, Wortsman et al. 2022):
    blended = stock + alpha * (tuned - stock)
alpha 1 is the tuned model, 0 is stock. A fine-tune often gains on its own kind of speech and loses a little
elsewhere; a blend part-way back usually keeps most of the gain and wins some of the loss back. No GPU needed.

    python experiments/remote-gpu/blend.py parakeet-rad-v8 0.75

Writes models/<tuned>-w<alpha*100>/ (parakeet.json + model/), which compare.py scores like any version.
The two exports name most matrices by number ("onnx::MatMul_6662"), and the numbers differ, so those pair
by the order the graph first uses them; named weights pair by name. A frozen layer must come out identical,
so a wrong pairing fails loud.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx

from config import MODELS
from src.dictation.asr.engines.parakeet_engine import model_dir_for
from src.core.paths import onnx_asr_cache_dir

STOCK = "nemo-parakeet-tdt-0.6b-v3-fp32"
FROZEN_PROBE = "layers.3."  # parakeet.json freezes the first 12 encoder layers
TYPES = {1: np.float32, 10: np.float16, 7: np.int64, 6: np.int32}


def values(t, base: Path) -> np.ndarray:
    ext = {e.key: e.value for e in t.external_data}
    if not ext:
        return onnx.numpy_helper.to_array(t)
    with open(base / ext["location"], "rb") as f:  # read directly: onnx refuses a file with two hard links
        f.seek(int(ext.get("offset", 0)))
        return np.frombuffer(f.read(int(ext["length"])), dtype=TYPES[t.data_type]).reshape(tuple(t.dims))


def first_use(graph) -> list[str]:
    inits, seen = {i.name for i in graph.initializer}, {}
    for node in graph.node:
        for x in node.input:
            if x in inits:
                seen.setdefault(x, None)
    return list(seen)


def pairing(stock, tuned) -> dict[str, str]:
    """tuned initializer name -> stock initializer name."""
    anon = lambda g: [n for n in first_use(g) if n.startswith("onnx::")]  # noqa: E731
    pairs = {i.name: i.name for i in tuned.graph.initializer if not i.name.startswith("onnx::")}
    a, b = anon(stock.graph), anon(tuned.graph)
    if len(a) != len(b):
        raise SystemExit(f"unnamed weights differ in count: stock {len(a)}, tuned {len(b)}")
    pairs.update(zip(b, a))
    return pairs


def blend_file(name: str, stock_dir: Path, tuned_dir: Path, out_dir: Path, alpha: float) -> None:
    stock = onnx.load(stock_dir / name, load_external_data=False)
    tuned = onnx.load(tuned_dir / name, load_external_data=False)
    s_init = {i.name: i for i in stock.graph.initializer}
    data_name, pairs = f"{name}.data", pairing(stock, tuned)
    with open(out_dir / data_name, "wb") as out:
        for t in tuned.graph.initializer:
            s_name = pairs[t.name]
            new, old = values(t, tuned_dir), values(s_init[s_name], stock_dir)
            if new.shape != old.shape:
                raise SystemExit(f"{name}: {t.name} {new.shape} paired with {s_name} {old.shape}")
            if t.name.startswith(FROZEN_PROBE) and not np.array_equal(new, old):
                raise SystemExit(f"{name}: frozen {t.name} differs from stock, pairing is wrong")
            mixed = new if new.dtype.kind != "f" else (old + alpha * (new.astype(np.float64) - old)).astype(new.dtype)
            if t.external_data:
                del t.external_data[:]
                t.data_location = onnx.TensorProto.EXTERNAL
                for k, v in (("location", data_name), ("offset", str(out.tell())), ("length", str(mixed.nbytes))):
                    t.external_data.add(key=k, value=v)
                out.write(mixed.tobytes())
            else:
                t.CopyFrom(onnx.numpy_helper.from_array(mixed, t.name))
    onnx.save_model(tuned, out_dir / name)
    if (out_dir / data_name).stat().st_size == 0:
        os.remove(out_dir / data_name)


def main(tuned_name: str, alpha: float) -> None:
    if not 0 < alpha < 1:
        raise SystemExit("alpha must be between 0 and 1 (0 = stock, 1 = the tuned model)")
    tuned_dir, stock_dir = model_dir_for(tuned_name), onnx_asr_cache_dir(STOCK)
    name = f"{tuned_name}-w{round(alpha * 100)}"
    out = MODELS / name / "model"
    out.mkdir(parents=True, exist_ok=False)
    for part in ("encoder-model.onnx", "decoder_joint-model.onnx"):
        blend_file(part, stock_dir, tuned_dir, out, alpha)
    for f in ("config.json", "vocab.txt"):
        shutil.copy(tuned_dir / f, out / f)
    cfg = json.loads((MODELS / tuned_name / "parakeet.json").read_text())
    (MODELS / name / "parakeet.json").write_text(json.dumps(cfg, indent=2))
    (MODELS / name / "README.md").write_text(
        f"# {name}\n\nWeight blend: stock + {alpha} x ({tuned_name} - stock), made by blend.py. Not trained.\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
