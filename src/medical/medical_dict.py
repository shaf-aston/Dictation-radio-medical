"""Medical terminology dictionary with fuzzy-match suggestions.

The wordlist ships at `src/resources/medical_terms.txt`. If it is missing
on first use (e.g. fresh checkout without LFS), a one-time download from
the upstream GitHub repo refills it. After that, lookups are in-memory.
"""

import logging
import threading
import urllib.request
from pathlib import Path
from typing import List, Optional, Set

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from src.features.file_manager import medical_wordlist_path

logger = logging.getLogger(__name__)

# Lazy-loaded singleton state
_TERMS: Optional[Set[str]] = None
_COMMON_TERMS: Optional[List[str]] = None       # Top 10K for fastest lookup
_FULL_TERMS_LIST: Optional[List[str]] = None    # Sorted list for fallback search
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


def get_medical_terms() -> Set[str]:
    global _TERMS, _COMMON_TERMS, _FULL_TERMS_LIST
    if _TERMS is not None:
        return _TERMS
    with _LOCK:
        if _TERMS is not None:
            return _TERMS
        path = medical_wordlist_path()
        logger.info("Loading medical terms from %s", path)
        terms = _load_terms_from_disk(path) if path.exists() else None
        if terms is None or len(terms) < _MIN_VALID_TERMS:
            logger.warning("Medical terms missing or incomplete, downloading...")
            if _download_wordlist(path):
                terms = _load_terms_from_disk(path)
        _TERMS = terms or set()
        logger.info("Loaded %d medical terms", len(_TERMS))
        if _TERMS:
            sorted_terms = sorted(_TERMS, key=len)
            _COMMON_TERMS = sorted_terms[:10000]
            _FULL_TERMS_LIST = sorted_terms
            logger.info("Cached %d common terms for fast lookup", len(_COMMON_TERMS))
        return _TERMS


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
