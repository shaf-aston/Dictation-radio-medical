"""Install a tuned Parakeet into the app: python install_model.py <name> [--remove-source]

Packs models/<name>/model/ (one file per tensor, as exported) into the app's
layout (encoder-model.onnx + one encoder-model.onnx.data) at
data/models/fine_tuned/<name>/, then transcribes clips from the protocol and
eka eval sets with both copies and stops unless every output is identical.
With --remove-source the experiment copy is deleted after that check
(compare.py reads an installed version from the app's folder).

Packing needs `onnx` (C:\\Python312 has it); the check runs onnx_asr, which
the same interpreter must have: run with the one that has both, or pass
--check-python to run the check with the app's .venv.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
from config import MODELS, REPO  # noqa: E402

CHECK = r"""
import glob, json, sys, soundfile as sf, onnx_asr
a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
A = onnx_asr.load_model("nemo-conformer-tdt", a)
B = onnx_asr.load_model("nemo-conformer-tdt", b)
files = sorted(glob.glob(r"data/eval/protocol/*.wav"))[:n] + sorted(glob.glob(r"data/eval/eka/**/*.wav", recursive=True))[:n]
diff = []
for f in files:
    x, sr = sf.read(f, dtype="float32")
    x = x if x.ndim == 1 else x.mean(1)
    if A.recognize(x, sample_rate=sr) != B.recognize(x, sample_rate=sr):
        diff.append(f)
print(json.dumps({"clips": len(files), "different": diff}))
"""


def pack(src: Path, dest: Path) -> None:
    import onnx

    dest.mkdir(parents=True)
    model = onnx.load(str(src / "encoder-model.onnx"), load_external_data=True)
    onnx.save(model, str(dest / "encoder-model.onnx"), save_as_external_data=True,
              all_tensors_to_one_file=True, location="encoder-model.onnx.data")
    for f in src.iterdir():
        if f.suffix in {".json", ".txt"} or f.name.startswith("decoder_joint"):
            shutil.copy2(f, dest / f.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--clips", type=int, default=10, help="clips per eval set for the identical-output check")
    ap.add_argument("--check-python", default=str(REPO / ".venv" / "Scripts" / "python.exe"))
    ap.add_argument("--remove-source", action="store_true")
    a = ap.parse_args()

    src = MODELS / a.name / "model"
    dest = REPO / "data" / "models" / "fine_tuned" / a.name
    if not (src / "encoder-model.onnx").exists():
        raise SystemExit(f"no exported model at {src}")
    if dest.exists():
        raise SystemExit(f"{dest} already exists; remove it first to reinstall")
    try:  # any failure (onnx missing, disk full, bad --check-python) must not leave a half install
        pack(src, dest)
        out = subprocess.run([a.check_python, "-c", CHECK, str(src), str(dest), str(a.clips)],
                             cwd=REPO, capture_output=True, text=True)
        try:
            r = json.loads(out.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            raise SystemExit(f"check failed to run; install removed\n{out.stderr[-2000:]}")
        if r["different"] or not r["clips"]:
            raise SystemExit(f"packed copy differs on {r['different'] or 'no clips'}; install removed")
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    print(f"installed {dest} ({r['clips']} clips, identical output)")
    if a.remove_source:
        shutil.rmtree(src)
        print(f"removed {src}")


if __name__ == "__main__":
    main()
