"""Build the local `protocol` eval set: python data/build_protocol_test.py [count]

MRI protocol dictation spoken with symbols ("right slash left", "open bracket ...
close bracket", "one point five tesla"), read only by settings.json's held-out
voice, which no training run uses. Two parts:

- HAND: lines written like real reports, not from the training templates
  (tests whether the wording generalises).
- generated: the training templates with a separate seed; any line that is in
  a training set (data/train manifest, or any models/*/train_texts.txt) is
  skipped (tests whether the trained wording is heard in an unseen voice).

The reference is what the report should finally say ("right/left", "(4 mm)",
"1.5 T"), so compare.py's "+fix" column and --features score what the user sees.
WER ignores punctuation; use compare.py --features for slash and brackets.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyttsx3  # noqa: E402
from config import MODELS, REPO, SETTINGS  # noqa: E402
from gen_text import _protocol_line, to_target  # noqa: E402

sys.path.insert(0, str(REPO))
from scripts.eval.corpus import Clip, audio_duration, write_manifest  # noqa: E402
from src.core.paths import eval_set_dir  # noqa: E402

TRAIN_MANIFEST = Path(__file__).resolve().parent / "train" / "manifest.jsonl"
SEED = 4242

HAND = [
    "Multiplanar, multisequence MRI of the right slash left elbow performed on a one point five tesla magnet.",
    "Sequences colon coronal PD fat saturated, coronal T one, sagittal PD fat saturated, axial PD fat saturated.",
    "MRI of the left shoulder on a three tesla scanner.",
    "The anterior slash posterior labrum is intact.",
    "There is a full thickness tear of the supraspinatus tendon open bracket twelve millimetres retraction close bracket.",
    "Bone marrow oedema in the lateral femoral condyle open bracket image fourteen close bracket.",
    "No ligamentous injury slash tear.",
    "Mild tendinosis of the common flexor slash pronator origin.",
    "Comparison colon radiograph from last month open bracket no fracture close bracket.",
    "The ulnar collateral ligament is intact on T two fat saturated images.",
    "Examination performed on a one point five T system.",
    "Small ganglion cyst open bracket six millimetres close bracket at the dorsal wrist.",
    "The ACL slash PCL are intact.",
    "Low signal on T one slash T two weighted images is in keeping with sclerosis.",
    "Imaging of the right knee on a three T magnet, sagittal STIR and axial T two.",
    "Findings colon moderate joint effusion open bracket see sagittal STIR close bracket.",
    "Medial slash lateral menisci are intact.",
    "Grade two strain of the medial head of gastrocnemius.",
    "No fracture, dislocation or bone bruise.",
    "Sequences colon sagittal T one, sagittal STIR, axial gradient echo.",
]


def written(spoken: str) -> str:
    """What the finished report should say for a spoken protocol line."""
    t = to_target(spoken)
    t = re.sub(r"\s*\bopen bracket\s+(.*?)\s+close bracket\b", r" (\1)", t)
    t = re.sub(r"\s+(?:forward )?slash\s+", "/", t)
    t = re.sub(r"\s+colon\b", ":", t)
    t = re.sub(r"\s+comma\b", ",", t)
    t = re.sub(r"\b(\d+(?:\.\d+)?) (?:tesla|T)\b", r"\1 T", t)
    return t


def trained_lines() -> set:
    seen = set()
    if TRAIN_MANIFEST.exists():
        for ln in TRAIN_MANIFEST.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                seen.update({r.get("spoken", ""), r["text"]})
    for f in MODELS.glob("*/train_texts.txt"):
        seen.update(f.read_text(encoding="utf-8").splitlines())
    return {s.strip() for s in seen if s.strip()}


def lines(count: int) -> list:
    banned, rng, out = trained_lines(), random.Random(SEED), list(HAND)
    sentences = {s for line in banned for s in re.split(r"(?<=\.)\s+", line)}
    tries = 0
    while len(out) < count and tries < count * 200:
        tries += 1
        s = _protocol_line(rng)
        if s not in sentences and to_target(s) not in sentences and s not in out:
            out.append(s)
    return out


def main(count: int) -> None:
    out = eval_set_dir("protocol")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("protocol_*.wav"):
        old.unlink()
    eng = pyttsx3.init()
    voice = [v.id for v in eng.getProperty("voices") if SETTINGS["held_out_voice"] in v.id.lower()]
    if not voice:
        raise SystemExit(f"held-out voice {SETTINGS['held_out_voice']!r} is not installed")
    eng.setProperty("voice", voice[0])
    clips = []
    for i, spoken in enumerate(lines(count)):
        dest = out / f"protocol_{i:04d}.wav"
        eng.save_to_file(spoken, str(dest))
        eng.runAndWait()
        clips.append(Clip(set_name="protocol", audio_path=dest, reference=written(spoken),
                          source="hand" if i < len(HAND) else f"gen_text seed {SEED}",
                          licence="own", synthetic=True, duration_sec=audio_duration(dest)))
    write_manifest("protocol", clips)
    print(f"protocol set: {len(clips)} clips ({len(HAND)} hand-written), voice {voice[0].rsplit(chr(92), 1)[-1]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
