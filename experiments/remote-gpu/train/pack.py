"""Turn a folder of wav clips + manifest.jsonl into the upload bundle (16 kHz flac in one tar)."""
import json
import shutil
import tarfile
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly

from config import HERE

PACK = HERE / "train" / "_pack"


def pack_data(src: Path) -> int:
    """Fill PACK from `src`; returns the clip count.

    Only folders under data/ are packed: the bundle goes to Kaggle, and the app's own
    eval sets (the user's voice among them) must never be uploaded by a mistyped --data."""
    src = src.resolve()
    if not src.is_relative_to(HERE / "data"):
        raise SystemExit(f"--data must be a folder under {HERE / 'data'}, got {src}")
    rows = [json.loads(ln) for ln in (src / "manifest.jsonl").read_text().splitlines() if ln.strip()]
    shutil.rmtree(PACK, ignore_errors=True)
    (PACK / "_flac").mkdir(parents=True)
    seen = set()
    for r in rows:
        audio = (src / r["audio"]).resolve()
        if not audio.is_relative_to(src):
            raise SystemExit(f"manifest audio path leaves {src}: {r['audio']}")
        a, sr = sf.read(audio, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            a = resample_poly(a, 16000, sr)
        r["audio"] = Path(r["audio"]).with_suffix(".flac").name
        if r["audio"] in seen:  # flattened names would overwrite each other and mislabel a clip
            raise SystemExit(f"two clips pack to the same name: {r['audio']}")
        seen.add(r["audio"])
        sf.write(PACK / "_flac" / r["audio"], a, 16000)
    # One archive, not hundreds of files: the Kaggle CLI uploads files one request at a time.
    with tarfile.open(PACK / "audio.tar", "w") as tf:
        tf.add(PACK / "_flac", arcname=".")
    shutil.rmtree(PACK / "_flac")
    (PACK / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return len(rows)
