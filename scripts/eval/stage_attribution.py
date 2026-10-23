"""Which post-processing stage earns its place, measured per stage.

    python -m scripts.eval.stage_attribution data/eval/reports/<run>.json

Reads a ``run_eval`` report (it needs the raw text, so the run must not use
``--no-text``), replays the pipeline one stage at a time on each clip's raw
recogniser output, and scores every stage on its own: words it fixed, correct
words it broke, and time it took. A stage that breaks as much as it fixes on
the engine actually in use is a stage to remove, whatever it was written for.

The whole-pipeline numbers in the report are the sum of these; this splits
them so a decision can be made per stage instead of per pipeline. It answers
"what does the recogniser typically get wrong, and which rule fixes it", the
question that should decide what post-processing exists at all.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from scripts.eval.metrics import correction_effect, normalize_words


def attribute(clips: Sequence[dict], accent: str = "neutral") -> Dict[str, dict]:
    """Per stage: true fixes, false corrections, neutral edits, seconds, examples."""
    from src.dictation.postprocess.pipeline import pipeline_stages

    totals: Dict[str, dict] = defaultdict(
        lambda: {"fixes": 0, "breaks": 0, "neutral": 0, "sec": 0.0, "examples": []}
    )
    for clip in clips:
        text, reference = clip["raw"], clip["reference"]
        for name, func, extra in pipeline_stages(accent):
            start = time.perf_counter()
            after = func(text, extra) if extra else func(text)
            row = totals[name]
            row["sec"] += time.perf_counter() - start
            if after != text:
                effect = correction_effect(reference, text, after)
                row["fixes"] += effect.true_fixes
                row["breaks"] += effect.false_corrections
                row["neutral"] += effect.neutral_changes
                row["examples"].extend(effect.examples[:2])
            text = after
        clip["processed_now"] = text  # what the current pipeline makes of it
    return dict(totals)


def residual_errors(clips: Sequence[dict], top: int = 25) -> List[tuple]:
    """The reference words most often still wrong after the whole pipeline.

    This is the list post-processing *should* be written from: what the
    recogniser typically gets wrong, ranked by how often, rather than what
    someone guessed it might.
    """
    import difflib

    missed: Counter = Counter()
    for clip in clips:
        ref = normalize_words(clip["reference"])
        hyp = normalize_words(clip.get("processed_now", clip["processed"]))
        matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op in ("replace", "delete"):
                heard = " ".join(hyp[j1:j2]) or "-"
                missed[(" ".join(ref[i1:i2]), heard)] += 1
    return missed.most_common(top)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="a run_eval report JSON")
    parser.add_argument("--accent", default="neutral")
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    clips = [c for s in report["sets"].values() for c in s["clip_results"] if c.get("raw")]
    # References as the manifests hold them now: a corrected reference must
    # not leave a stage charged with "breaking" what it actually fixed.
    from scripts.eval.corpus import load_set
    current = {c.clip_id: c.reference for name in report["sets"] for c in load_set(name)}
    for clip in clips:
        clip["reference"] = current.get(clip["clip_id"], clip["reference"])
    if not clips:
        print("No raw text in this report: re-run run_eval without --no-text.")
        return 1

    print(f"{len(clips)} clips from {args.report.name}\n")
    print(f"{'stage':<38}{'fixed':>7}{'broke':>7}{'net':>6}{'ms/clip':>9}")
    for name, row in attribute(clips, args.accent).items():
        net = row["fixes"] - row["breaks"]
        print(f"{name:<38}{row['fixes']:>7}{row['breaks']:>7}{net:>+6}"
              f"{1000 * row['sec'] / len(clips):>9.1f}")
        for ref_word, before, after in row["examples"][:3]:  # words it broke
            print(f"{'':>6}broke {before!r} -> {after!r}  (reference {ref_word!r})")

    print("\nMost frequent errors left after the pipeline (reference -> heard):")
    for (ref_words, heard), n in residual_errors(clips):
        print(f"  {n:>3}  {ref_words!r} -> {heard!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
