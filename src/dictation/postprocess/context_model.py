"""Offline n-gram context model for real-word ("confusion set") correction.

The fuzzy dictionary stage fixes *misspellings*. This model exists to answer a
different question the spell-checker cannot: given two real words that sound
alike ("cord" vs "chord", "coarse" vs "course"), *which one fits here?* It does
that the way a reader does — by looking at the neighbouring words — using plain
word-adjacency statistics learned from radiology text.

Design (loose coupling / SRP):

* **Pure statistics, no vocabulary opinions.** This module knows nothing about
  which words are confusable — that lives in ``confusion_sets.yaml``. It only
  answers "how likely is word B to follow word A in radiology prose?". The
  corrector (:mod:`context_correct`) composes the two.
* **Fully offline, tiny, fast.** A bigram/unigram count table over a few
  thousand sentences is a few hundred KB and scores a word in microseconds — no
  model download, no network, no GPU. It runs on the same machine as the
  dictation, so no report text ever leaves the device (the reports are PHI).
* **Learns with use.** Cold-started from a bundled seed corpus, it also folds in
  the radiologist's own finalized reports (:func:`append_learned_text`) so the
  disambiguation gets sharper for *their* case mix over time.

The built counts are cached on disk keyed by a signature of the source files, so
the (cheap) build is paid once per corpus change, not per launch.
"""

from __future__ import annotations

import logging
import math
import pickle
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.features.file_manager import (
    context_model_cache_path,
    context_seed_corpus_path,
    learned_context_corpus_path,
)

logger = logging.getLogger(__name__)

# Word = run of letters (with internal hyphen/apostrophe), lowercased. Numbers
# and punctuation are context boundaries, not tokens — "L4-L5" or "3.2 cm"
# carries no disambiguating signal for a confusable *word*.
_TOKEN_RE = re.compile(r"[a-z][a-z'\-]*")

# Add-k smoothing constant. Small so real adjacency evidence dominates, non-zero
# so an unseen neighbour pair still gets a finite (low) probability rather than
# -inf. Physical calibration knob — kept here, not inlined at the call site.
_SMOOTHING_K = 0.4


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class ContextModel:
    """Bigram + unigram counts with add-k smoothed conditional log-probabilities.

    Immutable once built. :meth:`score` returns ``log P(word | prev)`` and
    :meth:`log_prob` the raw conditional; the corrector uses the *difference*
    between two candidates in the same context, so absolute scale is irrelevant.
    """

    def __init__(self, unigram: Dict[str, int], bigram: Dict[Tuple[str, str], int]):
        self._uni = unigram
        self._bi = bigram
        self._total = sum(unigram.values()) or 1
        self._vocab = len(unigram) or 1

    def log_prob(self, prev: Optional[str], word: str) -> float:
        """Add-k smoothed ``log P(word | prev)``.

        With no usable *prev* (sentence start or an unknown previous word) this
        backs off to the smoothed unigram ``log P(word)`` so the term still
        contributes, just without adjacency information.
        """
        k, v = _SMOOTHING_K, self._vocab
        if prev is None or prev not in self._uni:
            return math.log((self._uni.get(word, 0) + k) / (self._total + k * v))
        num = self._bi.get((prev, word), 0) + k
        den = self._uni[prev] + k * v
        return math.log(num / den)

    def score(self, left: Optional[str], word: str, right: Optional[str]) -> float:
        """Context fit of *word* between *left* and *right*: sum of both directions.

        ``log P(word | left) + log P(right | word)`` — how well *word* follows
        the previous token and precedes the next. Comparing this across the
        members of a confusion set, in the *same* (left, right) context, is what
        picks the intended word.
        """
        s = self.log_prob(left, word)
        if right is not None:
            s += self.log_prob(word, right)
        return s


# ---------------------------------------------------------------------------
# Build / cache / singleton
# ---------------------------------------------------------------------------

_MODEL: Optional[ContextModel] = None
_MODEL_SIG: Optional[tuple] = None
_LOCK = threading.Lock()


def _source_signature() -> tuple:
    """Cheap fingerprint of the corpus sources — (path, size, mtime_ns) each.

    Changes whenever the seed corpus ships an update or the learned corpus
    grows, so the on-disk cache is transparently invalidated.
    """
    sig = []
    for p in (context_seed_corpus_path(), learned_context_corpus_path()):
        try:
            st = p.stat()
            sig.append((str(p), st.st_size, st.st_mtime_ns))
        except OSError:
            sig.append((str(p), 0, 0))  # missing is fine — learned corpus starts absent
    return tuple(sig)


def _count_file(path: Path, unigram: Dict[str, int], bigram: Dict[Tuple[str, str], int]) -> None:
    """Accumulate unigram/bigram counts from *path*, per line (a line is a
    sentence — bigrams never cross line boundaries, so end-of-report words don't
    spuriously predict the next report's first word)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                toks = _tokenize(s)
                prev: Optional[str] = None
                for t in toks:
                    unigram[t] = unigram.get(t, 0) + 1
                    if prev is not None:
                        key = (prev, t)
                        bigram[key] = bigram.get(key, 0) + 1
                    prev = t
    except OSError as exc:
        logger.warning("Context corpus unreadable at %s: %s", path, exc)


def _build_model() -> ContextModel:
    unigram: Dict[str, int] = {}
    bigram: Dict[Tuple[str, str], int] = {}
    _count_file(context_seed_corpus_path(), unigram, bigram)
    learned = learned_context_corpus_path()
    if learned.exists():
        _count_file(learned, unigram, bigram)
    logger.info(
        "Built context model: %d unigrams, %d bigrams", len(unigram), len(bigram)
    )
    return ContextModel(unigram, bigram)


def _load_cache(path: Path, signature: tuple) -> Optional[ContextModel]:
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            cached_sig, model = pickle.load(f)
    except Exception as exc:
        logger.warning("Could not load context-model cache: %s", exc)
        return None
    return model if cached_sig == signature else None


def _save_cache(path: Path, signature: tuple, model: ContextModel) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((signature, model), f)
    except OSError as exc:
        logger.warning("Could not write context-model cache: %s", exc)


def get_context_model() -> ContextModel:
    """The process-wide context model, built (or unpickled) once and cached.

    Rebuilds only when the seed or learned corpus changes (size/mtime
    signature). Thread-safe; safe to call from the warmup threads.
    """
    global _MODEL, _MODEL_SIG
    signature = _source_signature()
    if _MODEL is not None and _MODEL_SIG == signature:
        return _MODEL
    with _LOCK:
        if _MODEL is not None and _MODEL_SIG == signature:
            return _MODEL
        cache_path = context_model_cache_path()
        model = _load_cache(cache_path, signature)
        if model is None:
            model = _build_model()
            _save_cache(cache_path, signature, model)
        _MODEL, _MODEL_SIG = model, signature
    return _MODEL


# Sentences already written to the learned corpus, so repeated calls (the UI's
# adaptive-learning hook is debounced and fires with the growing draft many
# times per report) never append a duplicate — which would both bloat the file
# and skew the bigram counts toward whatever the radiologist happened to retype.
# Loaded from disk once, then kept in sync in-memory.
_LEARNED_SEEN: Optional[set] = None
_LEARNED_LOCK = threading.Lock()
# Only learn from lines with enough words to carry real adjacency signal (a
# bare "Normal." or a stray token teaches the model nothing useful).
_MIN_LEARN_WORDS = 4
# Cap the on-device learned corpus so it can't grow without bound over years of
# use (which would also slow every model rebuild). ~5 MB is on the order of tens
# of thousands of report sentences — far past the point of diminishing returns
# for n-gram disambiguation. Once reached, learning stops appending (the seed +
# accumulated corpus already covers the vocabulary); it never deletes.
_MAX_LEARNED_BYTES = 5 * 1024 * 1024


def _load_seen() -> set:
    global _LEARNED_SEEN
    if _LEARNED_SEEN is not None:
        return _LEARNED_SEEN
    seen: set = set()
    path = learned_context_corpus_path()
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        seen.add(s.lower())
    except OSError as exc:
        logger.warning("Could not read learned context corpus: %s", exc)
    _LEARNED_SEEN = seen
    return seen


def append_learned_text(text: str) -> None:
    """Fold a corrected report's text into the on-device learned corpus.

    Feeds the context model the radiologist's own finished sentences so
    disambiguation sharpens for their vocabulary and case mix over time. Stays
    entirely on-device; the report text is never uploaded (it is PHI).

    De-duplicated: a sentence already in the corpus is skipped, so this is safe
    to call from the debounced editor hook that fires repeatedly with the same
    (growing) draft — only genuinely new sentences are recorded, exactly once.

    Best-effort and non-fatal: a write failure just means the model doesn't
    learn from this text. The in-memory model refreshes on the next
    :func:`get_context_model` once the file signature changes.
    """
    if not text or not text.strip():
        return
    # One sentence per line so bigrams don't cross sentence boundaries.
    sentences = re.split(r"(?<=[.:;!?])\s+", text.strip())
    with _LEARNED_LOCK:
        seen = _load_seen()
        fresh = []
        for s in sentences:
            s = s.strip()
            if not s or len(_tokenize(s)) < _MIN_LEARN_WORDS:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            fresh.append(s)
        if not fresh:
            return
        try:
            path = learned_context_corpus_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            # Stop growing once the corpus is large enough — bounded rebuild cost
            # and disk. The seed + accumulated learning already covers the
            # vocabulary; further reports add negligible disambiguation power.
            if path.exists() and path.stat().st_size >= _MAX_LEARNED_BYTES:
                return
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(fresh) + "\n")
        except OSError as exc:
            logger.warning("Could not append to learned context corpus: %s", exc)


def reset_cache() -> None:
    """Drop the in-memory model + learned-seen cache so the next call rebuilds.
    For tests."""
    global _MODEL, _MODEL_SIG, _LEARNED_SEEN
    with _LOCK:
        _MODEL, _MODEL_SIG = None, None
    with _LEARNED_LOCK:
        _LEARNED_SEEN = None
