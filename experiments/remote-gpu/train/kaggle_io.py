"""Everything that talks to the Kaggle CLI: upload data, run the GPU kernel, fetch the result."""
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import date
from pathlib import Path

from config import HERE, SETTINGS

USER, DATASET = SETTINGS["kaggle_user"], SETTINGS["dataset"]
KDIR = HERE / "train" / "_kernel"


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if check and r.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r.stdout + r.stderr


def _windows_upload_state_dir(pack: Path) -> None:
    """Kaggle CLI on Windows fails ("No such file") unless its resume-state folder
    for the uploaded path already exists. Make it, or the upload hangs."""
    if os.name != "nt":
        return
    tail = str(pack.resolve()).replace(":", "_").replace("\\", "/")
    (Path(tempfile.gettempdir()) / ".kaggle" / "uploads" / tail).mkdir(parents=True, exist_ok=True)


def push_dataset(pack: Path) -> None:
    """Create or version the private dataset, then wait until Kaggle has processed it."""
    _windows_upload_state_dir(pack)
    (pack / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET, "id": f"{USER}/{DATASET}", "licenses": [{"name": "CC0-1.0"}], "isPrivate": True}))
    mine = kaggle("datasets", "list", "--mine", "--csv", check=False).splitlines()
    if any(ln.split(",", 1)[0] == f"{USER}/{DATASET}" for ln in mine):
        kaggle("datasets", "version", "-p", str(pack), "-m", f"data {date.today()}", "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(pack), "--dir-mode", "zip")
    # "ready" right after a version describes the PREVIOUS version, so wait a
    # while before believing it. The kernel also checks the manifest's hash
    # (cfg["data_sha256"]) and stops if Kaggle mounts stale data anyway.
    time.sleep(180)
    for _ in range(40):
        if "ready" in kaggle("datasets", "status", f"{USER}/{DATASET}", check=False).lower():
            return
        time.sleep(15)
    raise SystemExit("dataset never became ready")


def run_kernel(engine: str, cfg: dict) -> None:
    """Push train/<engine>_kernel.py, with `cfg` written into it, to a GPU and block until it finishes."""
    code = (HERE / "train" / f"{engine}_kernel.py").read_text(encoding="utf-8")
    placeholder = "\ncfg = {}"
    if placeholder not in code:
        raise SystemExit(f"{engine}_kernel.py has no `cfg = {{}}` line to fill in")
    shutil.rmtree(KDIR, ignore_errors=True)
    KDIR.mkdir(parents=True)
    kernel, code_file = cfg["kernel"], f"{engine}_kernel.py"
    (KDIR / code_file).write_text(code.replace(placeholder, f"\ncfg = {cfg!r}", 1), encoding="utf-8")
    (KDIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{kernel}", "title": kernel, "code_file": code_file,
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true", "machine_shape": SETTINGS["machine"],
        "dataset_sources": [f"{USER}/{DATASET}", *cfg["kaggle_sources"]]}))
    kaggle("kernels", "push", "-p", str(KDIR))
    started = time.time()
    time.sleep(SETTINGS["kernel_start_wait_sec"])  # right after a push, status still shows the PREVIOUS run
    while True:
        s = kaggle("kernels", "status", f"{USER}/{kernel}", check=False)
        if "COMPLETE" in s:
            return
        if "ERROR" in s or "CANCEL" in s:
            raise SystemExit(f"Kaggle run failed: {s.strip()}\nlog: kaggle kernels output {USER}/{kernel}")
        if time.time() - started > SETTINGS["max_wait_hours"] * 3600:
            raise SystemExit(f"gave up after {SETTINGS['max_wait_hours']} hours")
        time.sleep(SETTINGS["poll_sec"])


def fetch_output(dest: Path, cfg: dict) -> None:
    """Download the finished kernel's output into `dest` and check the model is in it."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir()
    kaggle("kernels", "output", f"{USER}/{cfg['kernel']}", "-p", str(dest))
    if not (dest / cfg["output"] / cfg["check_file"]).exists():
        raise SystemExit(f"no {cfg['output']}/{cfg['check_file']} in {dest} (see the .log file there)")
