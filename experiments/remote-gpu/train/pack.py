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
    """Fill PACK from `src`; returns the clip count."""
    rows = [json.loads(ln) for ln in (src / "manifest.jsonl").read_text().splitlines() if ln.strip()]
    shutil.rmtree(PACK, ignore_errors=True)
    (PACK / "_flac").mkdir(parents=True)
    for r in rows:
        a, sr = sf.read(src / r["audio"], dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            a = resample_poly(a, 16000, sr)
        r["audio"] = Path(r["audio"]).with_suffix(".flac").name
        sf.write(PACK / "_flac" / r["audio"], a, 16000)
    # One archive, not hundreds of files: the Kaggle CLI uploads files one request at a time.
    with tarfile.open(PACK / "audio.tar", "w") as tf:
        tf.add(PACK / "_flac", arcname=".")
    shutil.rmtree(PACK / "_flac")
    (PACK / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return len(rows)
