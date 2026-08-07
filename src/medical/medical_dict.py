"""Medical terminology dictionary with fuzzy-match suggestions.

Two wordlists with two distinct jobs (keeping them separate is what fixes the
"dictation keeps misspelling medical terms" problem):

* **Membership** — :func:`get_medical_terms` answers *"is this already a real
  word, leave it alone?"*. Broad is good here, so it is the union of the generic
  medical wordlist (``src/resources/medical_terms.txt``, ~98k terms) and the
  curated radiology lexicon. The generic list ships in-repo; if it is missing on
  first use (e.g. a fresh checkout without LFS), a one-time download refills it.

* **Correction targets** — :func:`get_correction_targets` returns the curated
  radiology lexicon (``src/resources/radiology_lexicon.txt``), the terms a typo
  should preferentially snap to.

:func:`get_symspell` builds a SymSpell index over the *membership* set (so any
of the ~98k known terms is a valid correction, not just the curated radiology
lexicon — terms like "esophageal" or "thyroid" are only in the generic list),
with correction-target entries given a large frequency boost so a tied edit
distance still prefers the clean radiology spelling (e.g. "efusion" ->
"effusion", not "fusion").

After first load everything is cached in-memory (the SymSpell index is also
cached to disk — see :func:`get_symspell`).
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from symspellpy import SymSpell

try:
    from symspellpy import SymSpell  # type: ignore[assignment]
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False

from src.features.file_manager import (
    medical_dict_cache_path,
    medical_wordlist_path,
    radiology_lexicon_path,
)

logger = logging.getLogger(__name__)

# rapidfuzz is only needed by suggest_correction()'s fallback path, which runs
# only when the SymSpell index is unavailable. Importing it eagerly cost ~1.77s
# at every cold start (its C-extension pulls in a large module tree) for a path
# that never executes in the fully-installed config — so it is loaded lazily on
# first use. `_rapidfuzz()` caches the result; None means "not installed".
_RAPIDFUZZ: "Optional[object]" = None  # (process, fuzz) once loaded; False if absent


def _rapidfuzz():  # type: ignore[no-untyped-def]
    """Lazy-load rapidfuzz. Returns (process, fuzz) or None if not installed."""
    global _RAPIDFUZZ
    if _RAPIDFUZZ is None:
        try:
            from rapidfuzz import process, fuzz  # noqa: PLC0415
            _RAPIDFUZZ = (process, fuzz)
        except ImportError:
            _RAPIDFUZZ = False
    return _RAPIDFUZZ or None

# Lazy-loaded singleton state
_TERMS: Optional[Set[str]] = None                # Membership: generic ∪ lexicon
_COMMON_TERMS: Optional[List[str]] = None        # Top 10K membership, fastest lookup
_FULL_TERMS_LIST: Optional[List[str]] = None     # Sorted membership, fallback search
_CORRECTION_TARGETS: Optional[List[str]] = None  # Curated lexicon: the snap targets
_LOCK = threading.Lock()

# Free medical wordlist (plaintext, one term per line)
# Source: https://github.com/glutanimate/wordlist-medicalterms-en
_WORDLIST_URL = (
    "https://raw.githubusercontent.com/glutanimate/wordlist-medicalterms-en/master/wordlist.txt"
)

# Below this count, the on-disk file is considered incomplete and refetched
_MIN_VALID_TERMS = 1000


def _download_wordlist(path: Path) -> bool:
    import urllib.request  # noqa: PLC0415 — lazy: only the rare one-time refill

    try:
        with urllib.request.urlopen(_WORDLIST_URL, timeout=10) as resp:
            content = resp.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return True
    except Exception as exc:
        logger.warning("Medical wordlist download failed: %s", exc)
        return False


def _load_terms_from_disk(path: Path) -> Optional[Set[str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return {line.strip().lower() for line in f if line.strip()}
    except Exception as exc:
        logger.warning("Failed to load medical terms from %s: %s", path, exc)
        return None


def _load_lexicon_from_disk(path: Path) -> List[str]:
    """Read the curated radiology lexicon, sorted shortest-first then alpha.

    One term per line; ``#`` comments and blank lines are ignored. A missing
    file is not fatal — the corrector simply has no snap targets and falls back
    to leaving unknown words alone (safe under-correction, never junk).
    """
    if not path.exists():
        logger.warning(
            "Radiology lexicon missing at %s — spelling correction has no snap "
            "targets and will leave mis-transcribed terms uncorrected.", path,
        )
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            terms = {
                line.strip().lower()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            }
    except Exception as exc:
        logger.warning("Failed to load radiology lexicon from %s: %s", path, exc)
        return []
    return sorted(terms, key=lambda t: (len(t), t))


def _load_and_cache_terms() -> Set[str]:
    global _TERMS, _COMMON_TERMS, _FULL_TERMS_LIST, _CORRECTION_TARGETS
    # Broad generic wordlist — the membership net (one-time refill if missing).
    path = medical_wordlist_path()
    logger.info("Loading medical terms from %s", path)
    generic = _load_terms_from_disk(path) if path.exists() else None
    if generic is None or len(generic) < _MIN_VALID_TERMS:
        logger.warning("Medical terms missing or incomplete, downloading...")
        if _download_wordlist(path):
            generic = _load_terms_from_disk(path)
    generic = generic or set()

    # Curated radiology lexicon — the clean snap targets (always bundled in-repo).
    lexicon = _load_lexicon_from_disk(radiology_lexicon_path())
    _CORRECTION_TARGETS = lexicon

    # Membership = broad net ∪ curated lexicon, so every curated radiology term
    # is also recognised as "already correct" and left untouched.
    _TERMS = generic | set(lexicon)
    logger.info(
        "Loaded %d medical terms (%d generic + %d radiology lexicon)",
        len(_TERMS), len(generic), len(lexicon),
    )
    if _TERMS:
        sorted_terms = sorted(_TERMS, key=len)
        _COMMON_TERMS = sorted_terms[:10000]
        _FULL_TERMS_LIST = sorted_terms
    else:
        _COMMON_TERMS = []
        _FULL_TERMS_LIST = []
    return _TERMS


def get_medical_terms() -> Set[str]:
    """Broad membership set (generic ∪ radiology lexicon): words to leave alone."""
    global _TERMS
    if _TERMS is not None:
        return _TERMS
    with _LOCK:
        return _TERMS if _TERMS is not None else _load_and_cache_terms()


# ---------------------------------------------------------------------------
# The English-word guard
# ---------------------------------------------------------------------------
# A genuine typo is a *non-word*. Nothing may treat a word that is already valid
# English ("there", "around", "again") as a mistake just because a real medical
# term ("marrow" vs "narrow") sits one edit away. pyspellchecker bundles an
# OFFLINE frequency dictionary (no network), so it is the guard.
#
# It lives here, beside the two wordlists, because "is this a real word?" is one
# question with one answer: the fuzzy corrector
# (``dictation/postprocess/medical_dict_match.py``) and the marking scan
# (``medical/term_lookup.py``) must never disagree about it. ``medical/`` is also
# the layer both of those can import from without inverting the module map.
_ENGLISH = None  # None = not yet loaded; False = unavailable; else a SpellChecker


def is_english_word(word: str) -> Optional[bool]:
    """True/False if *word* is/isn't standard English, or None if no checker.

    ``None`` is a third answer, not a failure: callers must decide what to do
    without the guard, and the safe choice is always the conservative one.
    """
    global _ENGLISH
    if _ENGLISH is None:
        try:
            from spellchecker import SpellChecker  # noqa: PLC0415 — optional dep, lazy
            _ENGLISH = SpellChecker()
        except Exception:  # not installed / failed to load — guard unavailable
            _ENGLISH = False
            # Loud, once: without this guard the corrector drops to the
            # conservative ratio path and silently leaves the whole class of
            # one-letter medical misspellings uncorrected. A silent degradation
            # here is exactly how "the dictation keeps misspelling things" goes
            # undiagnosed.
            logger.warning(
                "Spelling corrector degraded: pyspellchecker is not installed, so "
                "the English-word guard is off and one-letter medical misspellings "
                "(e.g. 'atelactasis'->'atelectasis', 'vertabra'->'vertebra') will "
                "NOT be corrected. Install it: pip install pyspellchecker"
            )
    return None if _ENGLISH is False else bool(_ENGLISH.known([word]))


def get_correction_targets() -> List[str]:
    """Curated radiology lexicon used as the fuzzy corrector's snap targets.

    Kept separate from :func:`get_medical_terms` on purpose: a typo is only ever
    snapped to a genuine radiology term from this clean list, never to the
    generic wordlist's chemistry/drug/obscure-procedure entries that previously
    caused over-correction. Loads on first use; empty if the lexicon is absent.
    """
    global _CORRECTION_TARGETS
    if _CORRECTION_TARGETS is not None:
        return _CORRECTION_TARGETS
    with _LOCK:
        if _CORRECTION_TARGETS is None:
            _load_and_cache_terms()
        return _CORRECTION_TARGETS or []


# ---------------------------------------------------------------------------
# SymSpell index — fast nearest-term lookup over the full membership wordlist
# ---------------------------------------------------------------------------
# Lexicon entries get a large frequency boost so that when a typo is
# equidistant from a curated radiology term and a generic-wordlist term
# ("efusion" -> "effusion" vs "fusion"), SymSpell's frequency-ranked
# suggestions prefer the clean radiology spelling.
_LEXICON_BOOST = 10_000

_SYMSPELL: Optional["SymSpell"] = None
_SYMSPELL_SIGNATURE: Optional[tuple] = None


def _build_symspell(terms: Set[str], lexicon: List[str]) -> "SymSpell":
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for term in terms:
        sym.create_dictionary_entry(term, 1)
    for term in lexicon:
        sym.create_dictionary_entry(term, _LEXICON_BOOST)
    return sym


def _load_symspell_cache(path: Path, signature: tuple) -> Optional["SymSpell"]:
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            cached_signature, sym = pickle.load(f)
    except Exception as exc:
        logger.warning("Could not load SymSpell cache: %s", exc)
        return None
    return sym if cached_signature == signature else None


def _save_symspell_cache(path: Path, signature: tuple, sym: "SymSpell") -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((signature, sym), f)
    except OSError as exc:
        logger.warning("Could not write SymSpell cache: %s", exc)


def get_symspell() -> Optional["SymSpell"]:
    """SymSpell index over the full membership wordlist (generic ∪ curated
    lexicon), used to find the nearest known term to a non-word within a
    length-scaled edit distance.

    Building this from ~98k terms takes a few seconds, so the result is cached
    on disk (:func:`~src.features.file_manager.medical_dict_cache_path`) and
    rebuilt only when the term counts change (e.g. the bundled wordlist or
    lexicon is updated). Returns None if symspellpy is not installed — callers
    fall back to the slower rapidfuzz ratio path.
    """
    global _SYMSPELL, _SYMSPELL_SIGNATURE
    if not SYMSPELL_AVAILABLE:
        return None
    terms = get_medical_terms()
    lexicon = get_correction_targets()
    signature = (len(terms), len(lexicon))
    if _SYMSPELL is not None and _SYMSPELL_SIGNATURE == signature:
        return _SYMSPELL
    with _LOCK:
        if _SYMSPELL is not None and _SYMSPELL_SIGNATURE == signature:
            return _SYMSPELL
        cache_path = medical_dict_cache_path()
        sym = _load_symspell_cache(cache_path, signature)
        if sym is None:
            sym = _build_symspell(terms, lexicon)
            _save_symspell_cache(cache_path, signature, sym)
        _SYMSPELL, _SYMSPELL_SIGNATURE = sym, signature
    return _SYMSPELL


def suggest_correction(word: str, cutoff: float = 0.9) -> Optional[str]:
    """
    Suggest a close medical term using rapidfuzz (fast) or fallback to set lookup.
    Returns the top suggestion if similarity >= cutoff, else None.
    """
    terms = get_medical_terms()
    if not terms:
        return None

    word_lower = word.lower()
    if word_lower in terms:
        return word_lower

    rf = _rapidfuzz()
    if rf and _FULL_TERMS_LIST:
        process, fuzz = rf
        result = process.extractOne(
            word_lower,
            _FULL_TERMS_LIST,
            scorer=fuzz.ratio,
            score_cutoff=cutoff * 100,
        )
        return result[0] if result else None

    return None
