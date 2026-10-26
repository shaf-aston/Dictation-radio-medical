"""One command trains a new model version on free Kaggle GPU.

    python train/run_train.py --engine parakeet --name parakeet-rad [--note "why"] [--data data/train] [--smoke]

pack.py bundles the clips, kaggle_io.py runs them through train/<engine>_kernel.py on a GPU
with the settings in train/<engine>.json, and the result lands in a NEW folder
models/<name>-vN/. An existing version is never touched. --smoke runs a few steps to prove
the kernel works end to end; its output stays in train/_out and no version is saved.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HERE, MODELS, SETTINGS, load_env  # noqa: E402
from kaggle_io import fetch_output, push_dataset, run_kernel  # noqa: E402
from pack import PACK, pack_data  # noqa: E402

OUT = HERE / "train" / "_out"
LOAD_LINE = {
    "whisper": "FasterWhisperEngine(model_path=r'{model}')",
    "parakeet": "onnx_asr.load_model('nemo-conformer-tdt', r'{model}', quantization={quantization!r})",
}
def check_smoke(log: dict, cfg: dict) -> None:
    """A smoke run learns almost nothing, so our export must match the published model, the trained model
    must match the untouched one at full precision, and the BatchNorm averages must not have moved.
    The tuned int8 score is printed, not gated: rounding magnifies even a 1e-7 change on 24 clips."""
    if log.get("training", "finished") != "finished":
        raise SystemExit(f"smoke run failed: training {log['training']}")
    scores, gap = log.get("eval", {}), SETTINGS["smoke_max_wer_gap_pp"]
    # score label -> the label it must match; stock is scored at the export's precision
    pairs = {f"untouched_{cfg.get('quantization') or 'fp32'}": "stock", "tuned_fp32": "untouched_fp32"}
    for label, got in scores.items():
        print(label, got)
    print("max change", log.get("max_change"))
    for label, ref in pairs.items():
        off = {s: scores[label][s] for s in scores[ref] if abs(scores[label][s] - scores[ref][s]) > gap}
        if off:
            raise SystemExit(f"smoke run failed: {label} differs from {ref} by more than {gap} pp: {off}")
    moved = log.get("max_change", {}).get("buffers")
    if moved != 0.0:
        raise SystemExit(f"smoke run failed: BatchNorm averages moved by {moved} (None = not measured)")


def next_version(name: str) -> Path:
    n = 1
    while (MODELS / f"{name}-v{n}").exists():
        n += 1
    return MODELS / f"{name}-v{n}"


def save_version(dest: Path, engine: str, cfg: dict, clips: int, data: Path, note: str) -> None:
    data_label = os.path.relpath(data, HERE)  # before anything moves: a failure here must not strand the model
    dest.mkdir(parents=True)  # fails loudly if it somehow exists: never overwrite
    shutil.move(str(OUT / cfg["output"]), str(dest / "model"))
    if (OUT / "train_log.json").exists():
        shutil.move(str(OUT / "train_log.json"), dest / "train_log.json")
    (dest / f"{engine}.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy(PACK / "manifest.jsonl", dest / "train_manifest.jsonl")
    (dest / "README.md").write_text(
        f"# {dest.name}\n\n- date: {date.today()}\n- engine: {engine}\n- base model: {cfg['base_model']}\n"
        f"- data: {clips} local clips from {data_label} (list: train_manifest.jsonl)"
        f" plus Kaggle sources {cfg['kaggle_sources'] or 'none'}\n"
        f"- settings: {engine}.json\n- note: {note}\n"
        f"- measured error rate: not yet. Run `python compare.py` and paste the table here.\n"
        f"\nLoad with: {LOAD_LINE[engine].format(model=dest / 'model', quantization=cfg.get('quantization'))}\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=sorted(LOAD_LINE), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--data", type=Path, default=HERE / "data" / "train")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = json.loads((HERE / "train" / f"{a.engine}.json").read_text())
    smoke = cfg.pop("smoke", None)
    if a.smoke and smoke is None:
        raise SystemExit(f"train/{a.engine}.json has no smoke block, so --smoke would start a full run")
    if a.smoke:
        cfg.update(smoke)
    load_env()
    dest = next_version(a.name)
    clips = pack_data(a.data.resolve())
    print(f"packed {clips} clips -> {'smoke run' if a.smoke else dest.name}", flush=True)
    push_dataset(PACK)
    run_kernel(a.engine, cfg)
    fetch_output(OUT, cfg)
    if a.smoke:
        check_smoke(json.loads((OUT / "train_log.json").read_text()), cfg)
        print("smoke run passed; output in", OUT)
        return
    save_version(dest, a.engine, cfg, clips, a.data.resolve(), a.note)
    shutil.rmtree(OUT, ignore_errors=True)
    print("saved", dest)


if __name__ == "__main__":
    main()
