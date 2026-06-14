"""Medical terminology dictionary with fuzzy-match suggestions.

Two wordlists with two distinct jobs (keeping them separate is what fixes the
"dictation keeps misspelling medical terms" problem):

* **Membership** — :func:`get_medical_terms` answers *"is this already a real
  word, leave it alone?"*. Broad is good here, so it is the union of the generic
  medical wordlist (``src/resources/medical_terms.txt``, ~98k terms) and the
  curated radiology lexicon. The generic list ships in-repo; if it is missing on
  first use (e.g. a fresh checkout without LFS), a one-time download refills it.

* **Correction targets** — :func:`get_correction_targets` answers *"what real
  radiology term should this typo become?"*. This MUST be clean, so it is only
  the curated radiology lexicon (``src/resources/radiology_lexicon.txt``). The
  generic list is deliberately excluded here: its chemistry/drug/obscure-procedure
  entries are exactly what used to pull a misspelling toward junk.

After first load everything is cached in-memory.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from rapidfuzz import fuzz, process

try:
    from rapidfuzz import process, fuzz  # type: ignore[assignment]
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from src.features.file_manager import medical_wordlist_path, radiology_lexicon_path

logger = logging.getLogger(__name__)

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

    if RAPIDFUZZ_AVAILABLE and _COMMON_TERMS:
        result = process.extractOne(
            word_lower,
            _COMMON_TERMS,
            scorer=fuzz.ratio,
            score_cutoff=cutoff * 100
        )
        if result:
            return result[0]
        result = process.extractOne(
            word_lower,
            _FULL_TERMS_LIST or list(terms),
            scorer=fuzz.ratio,
            score_cutoff=cutoff * 100,
        )
        return result[0] if result else None

    return None
