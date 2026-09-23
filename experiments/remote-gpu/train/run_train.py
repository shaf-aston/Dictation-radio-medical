"""One command trains a new model version on free Kaggle GPU.

    python train/run_train.py --name small-en-rad [--note "why this run"]

Packs data/train + train/config.json into a private Kaggle dataset, runs
train_kernel.py on a GPU, waits, then downloads the result into a NEW folder
models/<name>-vN/. An existing version folder is never touched.
"""
import argparse, json, os, shutil, subprocess, sys, time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, KAGGLE_USER, MODELS, load_env  # noqa: E402

DATASET, KERNEL = "rd-train-data", "rd-train"
PACK, KDIR = HERE / "train" / "_pack", HERE / "train" / "_kernel"


def kaggle(*args, check=True):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if check and r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r.stdout + r.stderr


def next_version(name: str) -> Path:
    n = 1
    while (MODELS / f"{name}-v{n}").exists():
        n += 1
    return MODELS / f"{name}-v{n}"


def pack_data() -> int:
    import soundfile as sf
    from scipy.signal import resample_poly
    src = HERE / "data" / "train"
    rows = [json.loads(l) for l in (src / "manifest.jsonl").read_text().splitlines() if l.strip()]
    shutil.rmtree(PACK, ignore_errors=True); (PACK / "_flac").mkdir(parents=True)
    for r in rows:
        a, sr = sf.read(src / r["audio"], dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            a = resample_poly(a, 16000, sr)
        r["audio"] = r["audio"].replace(".wav", ".flac")
        sf.write(PACK / "_flac" / r["audio"], a, 16000)
    # One archive, not 900 files: the Kaggle CLI uploads files one request at a time.
    import tarfile
    with tarfile.open(PACK / "audio.tar", "w") as tf:
        tf.add(PACK / "_flac", arcname=".")
    shutil.rmtree(PACK / "_flac")
    (PACK / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    shutil.copy(HERE / "train" / "config.json", PACK / "config.json")
    (PACK / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET, "id": f"{KAGGLE_USER}/{DATASET}", "licenses": [{"name": "CC0-1.0"}]}))
    return len(rows)


def _windows_upload_state_dir() -> None:
    """Kaggle CLI on Windows fails ("No such file") unless its resume-state folder
    for the uploaded path already exists. Make it, or the upload hangs."""
    import os
    if os.name != "nt":
        return
    tail = str(PACK.resolve()).replace(":", "_").replace("\\", "/")
    base = Path(os.environ.get("TEMP", "")) / ".kaggle" / "uploads" / tail
    base.mkdir(parents=True, exist_ok=True)


def push_dataset() -> None:
    _windows_upload_state_dir()
    exists = "rd-train-data" in kaggle("datasets", "list", "--mine", check=False)
    if exists:
        kaggle("datasets", "version", "-p", str(PACK), "-m", f"data {date.today()}", "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(PACK), "--dir-mode", "zip")
    for _ in range(40):  # dataset must finish processing before the kernel can mount it
        if "ready" in kaggle("datasets", "status", f"{KAGGLE_USER}/{DATASET}", check=False).lower():
            return
        time.sleep(15)
    raise SystemExit("dataset never became ready")


def run_kernel() -> None:
    shutil.rmtree(KDIR, ignore_errors=True); KDIR.mkdir(parents=True)
    shutil.copy(HERE / "train" / "train_kernel.py", KDIR)
    (KDIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{KERNEL}", "title": KERNEL, "code_file": "train_kernel.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": [f"{KAGGLE_USER}/{DATASET}"]}))
    kaggle("kernels", "push", "-p", str(KDIR))
    started = time.time()
    time.sleep(90)  # right after a push the status still shows the PREVIOUS run's result
    while True:
        s = kaggle("kernels", "status", f"{KAGGLE_USER}/{KERNEL}", check=False)
        if "COMPLETE" in s:
            return
        if "ERROR" in s or "CANCEL" in s:
            raise SystemExit(f"Kaggle run failed: {s.strip()}\nlog: kaggle kernels output {KAGGLE_USER}/{KERNEL}")
        if time.time() - started > 6 * 3600:
            raise SystemExit("gave up after 6 hours")
        time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    load_env()
    dest = next_version(a.name)
    n = pack_data()
    print(f"packed {n} clips -> {dest.name}", flush=True)
    push_dataset()
    run_kernel()
    tmp = HERE / "train" / "_out"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
    kaggle("kernels", "output", f"{KAGGLE_USER}/{KERNEL}", "-p", str(tmp))
    if not (tmp / "model_ct2" / "model.bin").exists():
        raise SystemExit(f"no model_ct2/model.bin in output {tmp} (see rd-train.log there)")
    dest.mkdir(parents=True)  # fails loudly if it somehow exists: never overwrite
    shutil.move(str(tmp / "model_ct2"), str(dest / "model"))
    for f in ("train_log.json",):
        if (tmp / f).exists():
            shutil.move(str(tmp / f), dest / f)
    shutil.copy(HERE / "train" / "config.json", dest / "config.json")
    shutil.copy(PACK / "manifest.jsonl", dest / "train_manifest.jsonl")
    cfg = json.loads((HERE / "train" / "config.json").read_text())
    (dest / "README.md").write_text(
        f"# {dest.name}\n\n- date: {date.today()}\n- base model: {cfg['base_model']}\n"
        f"- data: {n} synthetic clips (Windows David + Zira voices; Hazel held out), "
        f"see train_manifest.jsonl\n- settings: config.json\n- note: {a.note}\n"
        f"- measured error rate: not yet. Run `python compare.py` and paste the table here.\n"
        f"\nLoad with: FasterWhisperEngine(model_path=r'{dest / 'model'}')\n", encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print("saved", dest)


if __name__ == "__main__":
    main()
