"""A labelled corpus for the report consistency checks, and their scores.

    python -m scripts.eval.report_checks_corpus

Builds reports from the real templates with known answers: clean ones, ones
with one injected error (wrong side, wrong-sex organ, missing or empty
impression), and clean ones written to tempt a false alarm (the other side
named for comparison, a side word in passing). Prints precision and recall per
check; ``tests/test_report_checks.py`` holds them to a floor.

**Read before quoting a number.** The corpus is written by the same hand as the
rules, so it measures whether the rules do what they were designed to do, not
how they fare on real reports. Real reports will find phrasings this corpus
does not have. The honest number comes from running the checks over a sample
of the radiologist's own signed reports and counting by hand.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.core.paths import templates_dir
from src.features.report_checks import check_report


@dataclass(frozen=True)
class Case:
    text: str
    sex: str
    #: The kinds of issue the case truly contains.
    truth: frozenset
    #: What the case is for: clean, tempting (other side named on purpose),
    #: passing (a side word in passing), injected, or unrecorded (no sex given).
    category: str = "clean"


_CLEAN_SIDE = [
    "There is an oblique tear of the posterior horn of the medial meniscus.",
    "Small joint effusion.",
    "The {side} {part} demonstrates normal alignment.",
    "No fracture of the {side} {part}.",
]
_TEMPTING_SIDE = [  # clean: the other side named on purpose
    "Appearances are similar to the contralateral {other} {part}.",
    "Compared with the {other} side, there is mild asymmetry.",
    "Unchanged from the prior study of the {other} {part}.",
]
_PASSING_SIDE = [  # clean, but a side word in passing: an honest false alarm
    "The patient lies {other} side down for comfort.",
]
_WRONG_SIDE = [
    "Marrow oedema in the {other} {part}.",
    "There is a full-thickness tear on the {other}.",
]

_SEX_TEXT = {
    "F": ["The uterus is normal in size.", "Both ovaries are unremarkable."],
    "M": ["The prostate is not enlarged.", "The seminal vesicles are normal."],
}
_NEUTRAL = ["The bladder is unremarkable.", "No free fluid.", "Bones are normal."]


def _sided_templates() -> List[Tuple[str, str]]:
    out = []
    for path in sorted(templates_dir().glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        first = text.splitlines()[0] if text else ""
        if "[Right / Left]" in first or "[right/left]" in first:
            part = path.stem.split("_")[-1].replace("joint", "joint")
            out.append((text, part))
    return out


def _fill(template: str, side: str) -> str:
    text = template.replace("[Right / Left / Bilateral]", side.title())
    text = text.replace("[Right / Left]", side.title()).replace("[right/left]", side)
    # Every other field gets a neutral value, so only the injected error differs.
    import re
    return re.sub(r"\[[^\]\n]*\]", "3", text)


def _with_impression(body: str, impression: str) -> str:
    head, _, _ = body.partition("IMPRESSION:")
    return f"{head.rstrip()}\n\nIMPRESSION:\n{impression}\n"


def build_corpus(seed: int = 7) -> List[Case]:
    rng = random.Random(seed)
    cases: List[Case] = []
    for template, part in _sided_templates():
        for side in ("right", "left"):
            other = "left" if side == "right" else "right"
            base = _fill(template, side)
            fmt = dict(side=side, other=other, part=part)

            def finding(pool):
                return rng.choice(pool).format(**fmt)

            clean = base.replace("FINDINGS:", "FINDINGS:\n" + finding(_CLEAN_SIDE), 1)
            cases.append(Case(_with_impression(clean, "1. Meniscal tear."), "", frozenset()))
            for sentence in _TEMPTING_SIDE:
                tempt = base.replace("FINDINGS:", "FINDINGS:\n" + sentence.format(**fmt), 1)
                cases.append(Case(_with_impression(tempt, "1. No acute abnormality."), "",
                                  frozenset(), "tempting"))
            passing = base.replace("FINDINGS:", "FINDINGS:\n" + finding(_PASSING_SIDE), 1)
            cases.append(Case(_with_impression(passing, "1. Normal."), "", frozenset(), "passing"))
            for sentence in _WRONG_SIDE:
                wrong = base.replace("FINDINGS:", "FINDINGS:\n" + sentence.format(**fmt), 1)
                cases.append(Case(_with_impression(wrong, "1. Marrow oedema."), "",
                                  frozenset({"side"}), "injected"))
            # Impression missing / empty, on an otherwise clean report.
            cases.append(Case(clean.split("IMPRESSION:")[0], "", frozenset({"impression"}), "injected"))
            cases.append(Case(_with_impression(clean, "1."), "", frozenset({"impression"}), "injected"))

    for i in range(60):
        sex = "F" if i % 2 else "M"
        other = "M" if sex == "F" else "F"
        lines = rng.sample(_NEUTRAL, 2)
        clean = "EXAMINATION: CT Pelvis\nFINDINGS:\n" + " ".join(lines + [rng.choice(_SEX_TEXT[sex])])
        cases.append(Case(clean + "\nIMPRESSION:\n1. Normal.\n", sex, frozenset()))
        wrong = "EXAMINATION: CT Pelvis\nFINDINGS:\n" + " ".join(lines + [rng.choice(_SEX_TEXT[other])])
        cases.append(Case(wrong + "\nIMPRESSION:\n1. Normal.\n", sex, frozenset({"sex"}), "injected"))
        # Sex not recorded: never checked, so never a flag.
        cases.append(Case(wrong + "\nIMPRESSION:\n1. Normal.\n", "", frozenset(), "unrecorded"))
    return cases


def false_alarms(cases: List[Case]) -> Dict[str, Tuple[int, int]]:
    """Per category of clean case: (reports flagged, reports in the category).

    Reported this way rather than as one precision figure, because a single
    precision depends on how many of each kind of clean report the corpus
    happens to hold: that mix is a choice made here, not a fact about reports.
    """
    out: Dict[str, Tuple[int, int]] = {}
    for case in cases:
        if case.truth:
            continue
        flagged = bool(check_report(case.text, case.sex))
        hit, total = out.get(case.category, (0, 0))
        out[case.category] = (hit + flagged, total + 1)
    return out


def score(cases: List[Case]) -> Dict[str, Dict[str, float]]:
    """Precision and recall per kind, counted per report."""
    out = {}
    for kind in ("side", "sex", "impression"):
        tp = fp = fn = 0
        for case in cases:
            flagged = any(i.kind == kind for i in check_report(case.text, case.sex))
            truly = kind in case.truth
            tp += flagged and truly
            fp += flagged and not truly
            fn += truly and not flagged
        out[kind] = {
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "cases": tp + fn,
            "false_alarms": fp,
        }
    return out


def main() -> None:
    cases = build_corpus()
    print(f"{len(cases)} reports (synthetic: see the module docstring before quoting)")
    print("caught (recall):")
    for kind, s in score(cases).items():
        print(f"  {kind:<11} {s['recall']:.0%} of {s['cases']} reports with the error")
    print("false alarms on clean reports, by kind of clean report:")
    for category, (hit, total) in false_alarms(cases).items():
        print(f"  {category:<11} {hit} of {total}")


if __name__ == "__main__":
    main()
