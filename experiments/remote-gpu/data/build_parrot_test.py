"""Build the local `parrot` eval set: python data/build_parrot_test.py models/parakeet-rad-v4/train_texts.txt [count]

PARROT radiology reports (MSK included) read by Kokoro TTS voices; its test split, scored by compare.py --set parrot.
The split is a random cut of report pieces, not by report, so a piece is left out if any of its sentences is
a training line's (the same rule the kernel uses for parrot_test). Synthetic voices that also read the training
split: the number says how well report wording is learned, not how the model hears a real person.
Licence CC-BY-NC-SA 4.0 (non-commercial). Audio stays under data/ (gitignored).
"""
import io
import re
import sys
import urllib.request
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPO  # noqa: E402

sys.path.insert(0, str(REPO))
from scripts.eval.corpus import Clip, audio_duration, write_manifest  # noqa: E402
from src.core.paths import eval_set_dir  # noqa: E402

URL = "https://huggingface.co/datasets/ysdede/parrot-radiology-asr-en/resolve/main/data/test-00000-of-00001.parquet"


def parrot_text(t: str) -> str:
    return " ".join(re.sub(r"(NEWLINE|PARAGRAPH)", " ", t).split())


def normalise(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", t.lower()).split())


def sentences(t: str) -> set:
    return {s for s in (normalise(x) for x in re.split(r"(?<!\d)[.;:?!](?!\d)", t)) if len(s.split()) >= 4}


def main() -> None:
    import pyarrow.parquet as pq

    texts = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    train = set().union(*(sentences(t) for t in texts))
    out = eval_set_dir("parrot")
    out.mkdir(parents=True, exist_ok=True)
    shard = out / "source" / "test-00000-of-00001.parquet"
    if not shard.exists():
        shard.parent.mkdir(exist_ok=True)
        part = shard.with_suffix(".part")  # an interrupted download must not look finished
        urllib.request.urlretrieve(URL, part)  # noqa: S310 - fixed https URL
        part.replace(shard)
    clips, shared = [], 0
    for i, row in enumerate(pq.read_table(shard, columns=["audio", "transcription"]).to_pylist()):
        text = parrot_text(row["transcription"])
        if sentences(text) & train:
            shared += 1
            continue
        a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr != 16000 or a.ndim != 1:
            continue
        dest = out / f"parrot_{i:04d}.wav"
        sf.write(dest, a, 16000)
        clips.append(Clip(set_name="parrot", audio_path=dest, reference=text, source="ysdede/parrot-radiology-asr-en test",
                          licence="CC-BY-NC-SA-4.0", synthetic=True, duration_sec=audio_duration(dest)))
        if len(clips) >= count:
            break
    write_manifest("parrot", clips)
    print(f"parrot set: {len(clips)} clips; {shared} test pieces left out for sharing a sentence with training")


if __name__ == "__main__":
    main()
