"""Build a synthetic radiology speech dataset for fine-tuning Parakeet.

    python -m scripts.finetune.build_dataset --voices <dir of Piper voices> \\
        --out data/finetune/radiology-v1 [--text my_old_reports.txt] [--limit N]

Writes 16 kHz WAVs and NeMo manifests (``train.jsonl``, ``val.jsonl``,
``test_heldout.jsonl``), plus ``heldout_terms.txt`` and ``stats.json``. Zip the
folder and upload it to Kaggle as a dataset; the notebook in this folder trains
on it. Nothing here touches the network.

What the text is, and what it is not
------------------------------------
* **Not** ``scripts/eval/resources/report_corpus.txt``. That corpus is what the
  ``tts``/``piper`` gold sets read; training on it would grade the model on its
  own homework. The evaluation stays honest only if it never appears here.
* The templates' own sentences, with each pick-list resolved to one option.
* Carrier sentences for every radiology lexicon term ("Note is made of
  {term}."): the approach SynthASR used to teach a model drug names it had
  never heard, which is exactly the gap here (the stock model hears
  "oblique tear" as "a leak tear").
* Optional ``--text`` files: the radiologist's own past reports, one sentence
  or report per line. They stay on this machine until the zip is uploaded.

Leakage guards
--------------
* 20% of lexicon terms are **held out**: no training sentence contains one.
  ``test_heldout.jsonl`` is made of them, so a fine-tune that only memorised
  its training terms shows up as no better on these.
* One voice is held out of training entirely and only reads the test set.

Synthetic speech teaches vocabulary and pronunciation-to-spelling, not
acoustics: a model fine-tuned only on it can get *worse* on real voices
(see README). The gate is always the radiologist's own recorded voice.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "src" / "templates"
LEXICON = ROOT / "src" / "resources" / "radiology_lexicon.txt"

_CARRIERS = (
    "Note is made of {t}.",
    "Appearances are in keeping with {t}.",
    "There is no evidence of {t}.",
    "Findings suggest {t}.",
    "{T} is noted.",
    "Comment: {t}.",
)
#: For terms that read as adjectives ("symmetric", "hypoechoic"): a noun
#: carrier would make "Note is made of symmetric.", which no one dictates.
_ADJECTIVE_CARRIERS = (
    "The appearance is {t}.",
    "There is a {t} lesion.",
    "This is {t} in appearance.",
)
_ADJECTIVE_ENDINGS = ("ic", "al", "ous", "ar", "ive", "ent", "ant", "oid", "ed", "ary", "ory")
_FIELD = re.compile(r"\[([^\]\n]*)\]")


def lexicon_terms() -> List[str]:
    lines = LEXICON.read_text(encoding="utf-8").splitlines()
    return sorted({ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")})


def template_sentences(rng: random.Random) -> List[str]:
    """Each template line with content, its fields resolved to one option."""
    out = []
    for path in sorted(TEMPLATES.glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            def pick(m: re.Match) -> str:
                options = [o.strip() for o in m.group(1).split("/") if o.strip()]
                if options == ["X"] or not options:
                    return str(rng.choice([2, 3, 4, 5, 8, 12]))
                return rng.choice(options)
            text = _FIELD.sub(pick, line).strip()
            # A bare "X" outside brackets is a number placeholder too.
            text = re.sub(r"\bX\b", lambda _: str(rng.choice([2, 3, 4, 5, 8, 12])), text)
            # Headings alone ("FINDINGS:") are not worth a clip.
            if len(text.split()) >= 3 and not text.endswith(":"):
                out.append(text)
    return sorted(set(out))


def carrier_sentences(terms: Sequence[str], rng: random.Random) -> List[str]:
    out = []
    for term in terms:
        adjective = " " not in term and term.lower().endswith(_ADJECTIVE_ENDINGS)
        carrier = rng.choice(_ADJECTIVE_CARRIERS if adjective else _CARRIERS)
        out.append(carrier.format(t=term, T=term[:1].upper() + term[1:]))
    return out


def user_sentences(paths: Sequence[Path]) -> List[str]:
    out = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if len(line.split()) >= 3 and not line.startswith("#"):
                out.append(line)
    return out


def split(
    sentences: Sequence[str], terms: Sequence[str], rng: random.Random, hold_frac: float = 0.2
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """(train, val, test_heldout, heldout_terms); no train sentence has a held-out term."""
    shuffled = list(terms)
    rng.shuffle(shuffled)
    held = sorted(shuffled[: int(len(shuffled) * hold_frac)])
    held_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in held) + r")\b", re.IGNORECASE)
    seen, test = [], []
    for s in sentences:
        (test if held_re.search(s) else seen).append(s)
    rng.shuffle(seen)
    n_val = max(1, len(seen) // 20)
    return seen[n_val:], seen[:n_val], test, held


def synthesise(
    voices: Sequence[Path], items: Sequence[Tuple[str, str]], out_dir: Path,
    rng: random.Random, augment: bool,
) -> List[Dict]:
    """Speak (split, text) items; returns manifest rows. *voices* per item are drawn at random."""
    import numpy as np
    import soundfile as sf
    from faster_whisper import decode_audio
    from piper import PiperVoice, SynthesisConfig

    loaded = {v: PiperVoice.load(str(v)) for v in voices}
    rows = []
    for i, (part, text) in enumerate(items):
        voice_path = rng.choice(list(voices))
        pace = rng.choice((0.9, 1.0, 1.1)) if augment else 1.0
        raw = out_dir / "tmp.wav"
        with wave.open(str(raw), "wb") as wav:
            loaded[voice_path].synthesize_wav(text, wav, syn_config=SynthesisConfig(length_scale=pace))
        audio = np.asarray(decode_audio(str(raw), sampling_rate=16000), dtype=np.float32)
        if augment:
            audio *= 10 ** (rng.uniform(-12, 0) / 20)          # mic level varies
            if rng.random() < 0.5:                              # a quiet room is not silent
                noise = np.random.default_rng(i).standard_normal(len(audio)).astype(np.float32)
                snr_db = rng.uniform(25, 40)
                scale = np.sqrt(np.mean(audio ** 2) / (10 ** (snr_db / 10))) / (np.std(noise) or 1)
                audio = audio + noise * scale
        audio = np.clip(audio, -1.0, 1.0)
        rel = Path("audio") / part / f"{i:06d}.wav"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_dir / rel), audio, 16000, subtype="PCM_16")
        rows.append({"audio_filepath": str(rel), "duration": round(len(audio) / 16000, 3),
                     "text": text, "voice": voice_path.stem, "split": part})
    (out_dir / "tmp.wav").unlink(missing_ok=True)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voices", type=Path, required=True, help="dir of Piper *.onnx voices")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--text", type=Path, action="append", default=[],
                        help="extra sentences, e.g. your own past reports (repeatable)")
    parser.add_argument("--copies", type=int, default=2,
                        help="times each training sentence is spoken (different voice/pace)")
    parser.add_argument("--limit", type=int, default=0, help="cap sentences, for a smoke run")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    voices = sorted(args.voices.glob("*.onnx"))
    if len(voices) < 2:
        parser.error("need at least two voices: one is held out for the test set")
    test_voice, train_voices = voices[0], voices[1:]

    terms = lexicon_terms()
    sentences = template_sentences(rng) + carrier_sentences(terms, rng) + user_sentences(args.text)
    if args.limit:
        rng.shuffle(sentences)
        sentences = sentences[: args.limit]
    train, val, test, held = split(sentences, terms, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = synthesise(train_voices, [("train", s) for s in train for _ in range(args.copies)],
                      args.out, rng, augment=True)
    rows += synthesise(train_voices, [("val", s) for s in val], args.out, rng, augment=False)
    rows += synthesise([test_voice], [("test_heldout", s) for s in test], args.out, rng, augment=False)

    for part in ("train", "val", "test_heldout"):
        with open(args.out / f"{part}.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                if row["split"] == part:
                    fh.write(json.dumps({k: row[k] for k in ("audio_filepath", "duration", "text")}) + "\n")
    (args.out / "heldout_terms.txt").write_text("\n".join(held) + "\n", encoding="utf-8")
    stats = {
        part: {"clips": sum(r["split"] == part for r in rows),
               "hours": round(sum(r["duration"] for r in rows if r["split"] == part) / 3600, 2)}
        for part in ("train", "val", "test_heldout")
    }
    stats.update(test_voice=test_voice.stem, train_voices=[v.stem for v in train_voices],
                 heldout_terms=len(held), seed=args.seed)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
