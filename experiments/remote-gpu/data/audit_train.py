"""Check the synthetic training set BEFORE uploading: python data/audit_train.py

Applies the kernel's own keep rules (duration window from parakeet.json, and
text_rules.usable, the same file the kernel imports) and counts, per feature,
how many clips were generated and how many will train. Exits non-zero if any
feature except "number WORDS" (dropped by design) ends up with fewer than
MIN_KEPT clips.
"""
import json
import re
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from text_rules import usable  # noqa: E402

MANIFEST = HERE / "data" / "train" / "manifest.jsonl"
CFG = json.loads((HERE / "train" / "parakeet.json").read_text())
MIN_KEPT = 30

FEATURES = {
    "slash": r"\bslash\b",
    "brackets": r"\bopen bracket\b",
    "tesla": r"\btesla\b",
    "field T": r"\b\d+(\.\d+)? T\b",
    "sequences": r"\bsequences?\b",
    "T1/T2": r"\bT[12]\b",
    "numbers (digits)": r"\d",
    "number WORDS": r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty)\b",
}


def duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def audit(manifest: Path = MANIFEST) -> dict:
    lo, hi = CFG["min_duration_sec"], CFG["max_duration_sec"]
    stats = {k: {"made": 0, "kept": 0} for k in ["ALL", *FEATURES]}
    reasons = {"too long": 0, "too short": 0, "text rule": 0}
    for ln in manifest.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        dur, text = duration(manifest.parent / row["audio"]), row["text"]
        ok_len, ok_text = lo <= dur <= hi, usable(text)
        if dur > hi:
            reasons["too long"] += 1
        elif dur < lo:
            reasons["too short"] += 1
        elif not ok_text:
            reasons["text rule"] += 1
        hit = ["ALL"] + [k for k, p in FEATURES.items() if re.search(p, text)]
        for k in hit:
            stats[k]["made"] += 1
            stats[k]["kept"] += ok_len and ok_text
    return {"features": stats, "dropped": reasons}


def main() -> None:
    r = audit()
    print(f"{'feature':18} {'made':>6} {'kept':>6}")
    for k, s in r["features"].items():
        print(f"{k:18} {s['made']:6} {s['kept']:6}")
    print("dropped:", r["dropped"])
    short = [k for k, s in r["features"].items()
             if k not in {"ALL", "number WORDS"} and s["kept"] < MIN_KEPT]
    if short:
        raise SystemExit(f"too few training clips for: {', '.join(short)}")


if __name__ == "__main__":
    main()
