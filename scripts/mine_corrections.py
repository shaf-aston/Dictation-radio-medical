#!/usr/bin/env python
"""Mine real dictation logs for the correction rules worth adding next.

The app already records every pipeline run (``data/analysis/analysis_*.json``)
and every passively-learned single-word user edit
(``data/learned_corrections.json``). Nothing read that data — so gaps were only
found by stumbling on them. This script closes the loop: it ranks the
highest-frequency uncorrected mishearings and prints ready-to-paste YAML stubs
for ``corrections.yaml``.

Three signals:

1. **Post-dictation edits** — every word the radiologist changed *after*
   dictation finished (``data/analysis/dictation_edits.jsonl``, written by
   ``features/edit_tracking.py``). A clinician fixing the delivered text is the
   strongest evidence the pipeline was wrong, so these rank first.
2. **Survivors** — words present in a transcript's *output* that are neither in
   the medical dictionary nor ordinary English-looking, i.e. likely
   mis-transcriptions the pipeline failed to fix. rapidfuzz suggests the nearest
   medical term as a candidate correction.
3. **User edits** — the ``word_corrections`` a radiologist already made by hand
   via the single-word passive learner, which have no shipped rule yet.

Output is a ranked report plus a YAML block. Nothing is written automatically —
a human reviews and pastes. Run::

    python scripts/mine_corrections.py
    python scripts/mine_corrections.py --top 40 --min-count 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.file_manager import analysis_dir, _data_dir  # noqa: E402

_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'\-]{3,}\b")
_LEARNED_FILE = "learned_corrections.json"


def _english_filter():
    """Return ``is_ordinary_english(word) -> bool``.

    Uses pyspellchecker if installed (best signal); otherwise falls back to a
    compact stoplist of common words so the report isn't drowned in plain
    English. Either way the miner only *triages* — a human reviews every stub.
    """
    try:
        from spellchecker import SpellChecker  # optional dep
        sc = SpellChecker()
        return lambda w: w in sc
    except Exception:
        common = {
            "there", "their", "they", "them", "then", "than", "this", "that",
            "these", "those", "with", "without", "within", "amount", "seen",
            "noun", "agree", "about", "above", "after", "again", "being",
            "could", "would", "should", "where", "which", "while", "small",
            "large", "given", "shows", "noted", "right", "left", "upper",
            "lower", "both", "also", "from", "into", "some", "more", "most",
            "such", "very", "have", "been", "were", "your", "form", "verb",
            "verbs", "value", "values", "finding", "findings",
        }
        return lambda w: w in common


def _load_medical_terms() -> Set[str]:
    try:
        from src.medical import medical_dict
        return medical_dict.get_medical_terms()
    except Exception:
        return set()


def _suggest(word: str) -> Optional[str]:
    try:
        from src.medical import medical_dict
        return medical_dict.suggest_correction(word, cutoff=0.80)
    except Exception:
        return None


def _iter_outputs() -> Iterable[str]:
    """Yield the ``output`` text of every saved analysis run."""
    for f in analysis_dir().glob("analysis_*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if out := data.get("output"):
            yield out


def _load_user_edits() -> Dict[str, str]:
    path = _data_dir() / _LEARNED_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("word_corrections", {})
    except Exception:
        return {}


def _load_dictation_edits() -> "Counter[tuple[str, str]]":
    """Count ``(before -> after)`` word-level fixes the radiologist made.

    Reads the post-dictation edit log. Only one-to-one single-word ``replace``
    edits are turned into correction candidates (multi-word and structural edits
    are recorded for context but aren't auto-suggestable as a ``misheard ->
    correct`` rule). Frequency = how often the same fix recurs.
    """
    counts: "Counter[tuple[str, str]]" = Counter()
    try:
        from src.features.file_manager import dictation_edits_path
        path = dictation_edits_path()
    except Exception:
        return counts
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("op") != "replace":
            continue
        before, after = rec.get("before", ""), rec.get("after", "")
        # Single word -> single word only, and an actual change.
        if before and after and " " not in before and " " not in after and before != after:
            counts[(before.lower(), after.lower())] += 1
    return counts


def _slug(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", word.lower()).strip("-") or "rule"


def mine(top: int, min_count: int) -> str:
    terms = _load_medical_terms()
    known = {t.lower() for t in terms}
    try:
        from src.dictation.postprocess.medical_dict_match import PROTECTED_TERMS, _ACRONYMS
        known |= {t.lower() for t in PROTECTED_TERMS} | {a.lower() for a in _ACRONYMS}
    except Exception:
        pass
    is_english = _english_filter()

    # Signal 1: survivors in outputs.
    survivors: Counter[str] = Counter()
    for text in _iter_outputs():
        for m in _WORD_RE.finditer(text):
            w = m.group(0).lower()
            if w in known or is_english(w) or w.endswith(("s", "ed", "ing")):
                continue
            survivors[w] += 1

    # Signal 2: user edits with no candidate yet.
    user_edits = _load_user_edits()

    # Signal 0 (ranked first): post-dictation edits — the clinician's own fixes.
    dictation_edits = _load_dictation_edits()

    lines = ["# ---- Mined correction candidates ----", ""]
    stubs = []

    if dictation_edits:
        if ranked_edits := [
            (pair, c)
            for pair, c in dictation_edits.most_common()
            if c >= min_count
        ]:
            lines.append(
                f"## Post-dictation edits by the radiologist ({len(ranked_edits)}) "
                "— strongest signal"
            )
            for (before, after), c in ranked_edits:
                lines.append(f"  {c:4d}x  {before!r} -> {after!r}")
                stubs.append((
                    f"edit-{_slug(before)}", before, after,
                    f"radiologist changed this {c}x after dictation",
                ))
            lines.append("")

    if user_edits:
        lines.append(f"## Hand corrections by users ({len(user_edits)}) — highest confidence")
        for wrong, right in sorted(user_edits.items()):
            lines.append(f"  {wrong!r} -> {right!r}")
            stubs.append((f"user-{_slug(wrong)}", wrong, right, "from a user's manual edit"))
        lines.append("")

    ranked = [(w, c) for w, c in survivors.most_common(top) if c >= min_count]
    if ranked:
        lines.append(f"## Unrecognised words surviving the pipeline (top {len(ranked)})")
        for w, c in ranked:
            sug = _suggest(w)
            hint = f" -> maybe {sug!r}" if sug and sug != w else "  (no close term)"
            lines.append(f"  {c:4d}x  {w!r}{hint}")
            if sug and sug != w:
                stubs.append((_slug(w), w, sug, f"mined: {c} occurrences, rapidfuzz suggestion"))
        lines.append("")

    if not user_edits and not ranked and not dictation_edits:
        lines.append("No candidates found. (No analysis logs yet, or all words recognised.)")

    if stubs:
        lines.append("# ---- Paste-ready YAML for corrections.yaml (REVIEW before adding) ----")
        for rid, pattern, repl, note in stubs:
            lines += [
                f"- id: {rid}",
                f'  pattern: "{pattern}"',
                f'  replacement: "{repl}"',
                f'  note: "{note}"',
                "",
            ]
        lines.append("# After pasting, add an input->expected pair to "
                     "tests/corpus/corrections.yaml and run pytest.")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=30, help="max survivor words to report")
    ap.add_argument("--min-count", type=int, default=1, help="min occurrences to report a survivor")
    args = ap.parse_args()
    print(mine(args.top, args.min_count))


if __name__ == "__main__":
    main()
