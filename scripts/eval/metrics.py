"""Accuracy metrics for dictation evaluation — pure functions, no I/O.

Four numbers matter for this product, and only one of them is the usual WER:

* :func:`word_error_rate` — the standard yardstick. Catches "did I break general
  English" but says nothing about whether the *medical* words came out right.
* :func:`term_error_rate` — WER restricted to reference words that are curated
  radiology-lexicon terms. A report can score a respectable overall WER while
  mangling every anatomical term in it; this is the number that reflects what a
  radiologist actually cares about.
* :func:`correction_effect` — the honest scoreboard for the post-processing
  pipeline. It needs three texts (reference, raw ASR, post-processed) and splits
  every change the pipeline made into **true fixes** (a wrong word made right)
  and **false corrections** (a *right* word made wrong). A correction layer that
  fixes 10 words and breaks 12 is worse than no correction layer at all, and
  nothing in the codebase could previously tell you which side of that line it
  sat on.
* real-time factor — computed by the caller (decode seconds / audio seconds); it
  needs no alignment so it does not live here.

Alignment is Levenshtein over *word sequences* via ``rapidfuzz`` (already a
project dependency — no need for ``jiwer``, and the raw opcodes are required
for the term-restricted and false-correction metrics anyway).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Set, Tuple

from rapidfuzz.distance import Levenshtein

# Keep intra-word hyphens and apostrophes ("T2-weighted", "patient's"); strip
# every other punctuation mark. Scoring must not punish an engine for emitting
# a comma the reference happens to lack.
_STRIP_PUNCT = re.compile(r"[^\w\s'-]", re.UNICODE)
_EDGE_PUNCT = re.compile(r"^['-]+|['-]+$")

#: Spoken unit words map to the symbol the pipeline deliberately standardises
#: them to (``measurements.py``). Without this, every "twelve millimetres"
#: dictation scores as an error against a pipeline that correctly wrote "12 mm"
#: — the harness would be marking the product down for working as designed.
#:
#: Deliberately narrow. Spelling variants such as calibre/caliber are NOT
#: canonicalised: a British-spelling report silently Americanised is a real
#: change to the radiologist's text, and it should stay visible as one.
_UNIT_CANON = {
    "millimetre": "mm", "millimetres": "mm",
    "millimeter": "mm", "millimeters": "mm",
    "centimetre": "cm", "centimetres": "cm",
    "centimeter": "cm", "centimeters": "cm",
    "milliliter": "ml", "millilitre": "ml",
    "milliliters": "ml", "millilitres": "ml",
    "percent": "%",
}


def normalize_words(text: str) -> List[str]:
    """Lower-case, de-punctuate and split *text* into comparable word tokens.

    Applied identically to reference and hypothesis so no engine is scored on
    its punctuation or casing habits. This is the single normalisation point —
    every metric in this module consumes its output, so a change here changes
    all metrics consistently.

    Hyphens split into separate tokens on *both* sides, so the pipeline joining
    "full thickness" into "full-thickness" (a deliberate terminology rule) does
    not register as two word errors. Units are canonicalised for the same
    reason; see :data:`_UNIT_CANON`.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _STRIP_PUNCT.sub(" ", text)
    words: List[str] = []
    for token in text.split():
        for part in token.split("-"):
            part = _EDGE_PUNCT.sub("", part)
            if part:
                words.append(_UNIT_CANON.get(part, part))
    return words


# ---------------------------------------------------------------------------
# Word error rate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WerResult:
    """Word-error-rate breakdown. ``wer`` is (S+D+I) / len(reference)."""

    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_words: int
    wer: float

    def as_dict(self) -> dict:
        return {
            "wer": round(self.wer, 5),
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "ref_words": self.ref_words,
        }


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    """Standard WER between two texts.

    An empty reference yields ``wer`` 0.0 when the hypothesis is also empty and
    1.0 otherwise — reporting a divide-by-zero as a perfect score would be the
    kind of silent lie this harness exists to prevent.
    """
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    if not ref:
        return WerResult(0, 0, len(hyp), 0, 0, 0.0 if not hyp else 1.0)

    sub = dele = ins = hits = 0
    for op in Levenshtein.editops(ref, hyp).as_opcodes():
        src_n, dst_n = op.src_end - op.src_start, op.dest_end - op.dest_start
        if op.tag == "equal":
            hits += src_n
        elif op.tag == "replace":
            sub += src_n
        elif op.tag == "delete":
            dele += src_n
        elif op.tag == "insert":
            ins += dst_n

    return WerResult(sub, dele, ins, hits, len(ref), (sub + dele + ins) / len(ref))


# ---------------------------------------------------------------------------
# Medical-term error rate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TermErrorResult:
    """Error rate over reference words that are curated lexicon terms."""

    term_tokens: int
    term_hits: int
    term_errors: int
    error_rate: float
    missed_terms: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "term_error_rate": round(self.error_rate, 5),
            "term_tokens": self.term_tokens,
            "term_hits": self.term_hits,
            "term_errors": self.term_errors,
            # Capped: a report list is for diagnosis, not a data dump.
            "missed_terms": self.missed_terms[:40],
        }


def term_error_rate(
    reference: str, hypothesis: str, terms: Iterable[str]
) -> TermErrorResult:
    """Error rate restricted to reference words present in *terms*.

    Only reference-side occurrences count, so an engine cannot improve this
    score by sprinkling extra lexicon words into its output (that would show up
    in :func:`word_error_rate` as insertions instead).
    """
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    lexicon = {t.strip().lower() for t in terms if t and t.strip()}
    term_idx = {i for i, w in enumerate(ref) if w in lexicon}
    if not term_idx:
        return TermErrorResult(0, 0, 0, 0.0, [])

    hit_idx = _matched_ref_indices(ref, hyp)
    hits = len(term_idx & hit_idx)
    missed = sorted({ref[i] for i in term_idx - hit_idx})
    errors = len(term_idx) - hits
    return TermErrorResult(
        len(term_idx), hits, errors, errors / len(term_idx), missed
    )


# ---------------------------------------------------------------------------
# Correction-layer effect (true fixes vs false corrections)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrectionEffect:
    """What the post-processing pipeline did, split by whether it helped.

    ``false_correction_rate`` is false / (false + true): the share of the
    pipeline's *meaningful* edits that made the transcript worse. ``net_gain``
    is the plain word count the pipeline is up or down on the raw ASR output —
    a negative number means the correction layer is a liability.
    """

    changes: int
    true_fixes: int
    false_corrections: int
    neutral_changes: int
    false_correction_rate: float
    net_gain: int
    examples: List[Tuple[str, str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "false_correction_rate": round(self.false_correction_rate, 5),
            "changes": self.changes,
            "true_fixes": self.true_fixes,
            "false_corrections": self.false_corrections,
            "neutral_changes": self.neutral_changes,
            "net_gain": self.net_gain,
            # (reference, raw ASR, post-processed) for the worst offenders.
            "false_correction_examples": [list(e) for e in self.examples[:25]],
        }


def correction_effect(reference: str, raw: str, processed: str) -> CorrectionEffect:
    """Score every edit the post-processing pipeline made against the truth.

    Args:
        reference: Ground-truth transcript.
        raw: The ASR engine's output, before post-processing.
        processed: The same text after the post-processing pipeline.

    Each altered word position is classified by comparing whether it aligned to
    the reference *before* the pipeline touched it and whether it does *after*:

    ==================  ================  ==================
    was right (raw)     is right (post)   classification
    ==================  ================  ==================
    no                  yes               true fix
    yes                 no                false correction
    otherwise           --                neutral
    ==================  ================  ==================

    A "neutral" change is one that swapped one wrong word for another, or that
    only altered casing/punctuation the normaliser discards.
    """
    ref = normalize_words(reference)
    raw_w, post_w = normalize_words(raw), normalize_words(processed)

    raw_ok = _matched_hyp_indices(ref, raw_w)
    post_ok = _matched_hyp_indices(ref, post_w)

    true_fixes = false_corrections = neutral = 0
    examples: List[Tuple[str, str, str]] = []

    for raw_i, post_i in _changed_positions(raw_w, post_w):
        was_right = raw_i is not None and raw_i in raw_ok
        now_right = post_i is not None and post_i in post_ok
        if now_right and not was_right:
            true_fixes += 1
        elif was_right and not now_right:
            false_corrections += 1
            examples.append((
                _ref_word_near(ref, raw_w, raw_i),
                raw_w[raw_i] if raw_i is not None else "",
                post_w[post_i] if post_i is not None else "<deleted>",
            ))
        else:
            neutral += 1

    meaningful = true_fixes + false_corrections
    return CorrectionEffect(
        changes=true_fixes + false_corrections + neutral,
        true_fixes=true_fixes,
        false_corrections=false_corrections,
        neutral_changes=neutral,
        false_correction_rate=(false_corrections / meaningful) if meaningful else 0.0,
        net_gain=true_fixes - false_corrections,
        examples=examples,
    )


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _matched_ref_indices(ref: Sequence[str], hyp: Sequence[str]) -> Set[int]:
    """Reference positions the hypothesis got exactly right."""
    return {
        i
        for op in Levenshtein.editops(ref, hyp).as_opcodes()
        if op.tag == "equal"
        for i in range(op.src_start, op.src_end)
    }


def _matched_hyp_indices(ref: Sequence[str], hyp: Sequence[str]) -> Set[int]:
    """Hypothesis positions that align to the reference — i.e. correct words."""
    return {
        i
        for op in Levenshtein.editops(ref, hyp).as_opcodes()
        if op.tag == "equal"
        for i in range(op.dest_start, op.dest_end)
    }


def _changed_positions(
    before: Sequence[str], after: Sequence[str]
) -> List[Tuple[int | None, int | None]]:
    """Pair up every position an edit touched as ``(before_idx, after_idx)``.

    A replace of n words by n words pairs positionally; a pure delete pairs
    against ``None`` on the after side, an insert against ``None`` before.
    """
    pairs: List[Tuple[int | None, int | None]] = []
    for op in Levenshtein.editops(before, after).as_opcodes():
        if op.tag == "equal":
            continue
        src = list(range(op.src_start, op.src_end))
        dst = list(range(op.dest_start, op.dest_end))
        for i in range(max(len(src), len(dst))):
            pairs.append((
                src[i] if i < len(src) else None,
                dst[i] if i < len(dst) else None,
            ))
    return pairs


def _ref_word_near(ref: Sequence[str], hyp: Sequence[str], hyp_i: int | None) -> str:
    """Best-guess reference word for a hypothesis position, for example output.

    Only ever used to make a failure human-readable, never to compute a score,
    so an approximate positional map is good enough.
    """
    if hyp_i is None or not ref:
        return ""
    return ref[min(hyp_i, len(ref) - 1)] if hyp else ""
