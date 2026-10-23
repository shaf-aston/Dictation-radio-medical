"""Re-run post-processing and scoring on a past run's raw decodes, in seconds.

    python -m scripts.eval.rescore data/eval/reports/<run>.json [--raw]

Decoding 54 minutes of audio takes minutes; post-processing it takes under a
second. Every post-processing change is a text-only change, so it is judged by
replaying the stored raw recogniser output through the *current* pipeline and
scoring it against the *current* references in each set's manifest (which may
have been corrected since the run). ``--raw`` scores the raw decode with no
post-processing at all: the baseline any stage has to beat.

Prints the same headline numbers as ``run_eval``. It cannot measure anything
that changes the audio or the decoder: those need a real ``run_eval``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

from scripts.eval.corpus import load_set
from scripts.eval.metrics import correction_effect, term_error_rate, word_error_rate


def _lexicon() -> list:
    from src.medical.medical_dict import get_correction_targets
    return list(get_correction_targets())


def rescore(report: dict, raw_only: bool = False, accent: str = "neutral") -> Dict[str, dict]:
    from src.dictation.postprocess.pipeline import postprocess_transcript

    lexicon = _lexicon()
    out = {}
    for set_name, block in report["sets"].items():
        refs = {c.clip_id: c.reference for c in load_set(set_name)}
        errors = words = term_errors = terms = fixes = breaks = 0
        for clip in block["clip_results"]:
            raw = clip.get("raw")
            if raw is None or clip["clip_id"] not in refs:
                continue
            ref = refs[clip["clip_id"]]
            hyp = raw if raw_only else postprocess_transcript(raw, accent=accent)
            wer = word_error_rate(ref, hyp)
            ter = term_error_rate(ref, hyp, lexicon)
            errors += wer.substitutions + wer.deletions + wer.insertions
            words += wer.ref_words
            term_errors += ter.term_errors
            terms += ter.term_tokens
            if not raw_only:
                effect = correction_effect(ref, raw, hyp)
                fixes += effect.true_fixes
                breaks += effect.false_corrections
        out[set_name] = {
            "wer": errors / words if words else 0.0,
            "term_error_rate": term_errors / terms if terms else 0.0,
            "true_fixes": fixes, "false_corrections": breaks,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--raw", action="store_true", help="score the raw decode, no pipeline")
    parser.add_argument("--accent", default="neutral")
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    for set_name, s in rescore(report, args.raw, args.accent).items():
        mode = "raw decode" if args.raw else "current pipeline"
        print(f"{set_name} ({mode}): WER {s['wer']:.2%}  term error {s['term_error_rate']:.2%}"
              + ("" if args.raw else f"  fixes/breaks {s['true_fixes']}/{s['false_corrections']}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
