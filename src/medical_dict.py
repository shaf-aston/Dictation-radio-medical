import os
import threading
import urllib.request
import logging
from typing import Optional, Set, List

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

logger = logging.getLogger(__name__)

# Lazy-loaded singleton state
_TERMS: Optional[Set[str]] = None
_COMMON_TERMS: Optional[List[str]] = None  # Top 5000 for fastest lookup
_LOCK = threading.Lock()

# Default URL to a free medical wordlist (plaintext, one term per line)
# Source: https://github.com/glutanimate/wordlist-medicalterms-en
_WORDLIST_URL = (
    "https://raw.githubusercontent.com/glutanimate/wordlist-medicalterms-en/master/wordlist.txt"
)


def _resource_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "resources")
    os.makedirs(path, exist_ok=True)
    return path


def _local_wordlist_path() -> str:
    return os.path.join(_resource_dir(), "medical_terms.txt")


def _download_wordlist(path: str) -> bool:
    try:
        with urllib.request.urlopen(_WORDLIST_URL, timeout=10) as resp:
            content = resp.read()
        with open(path, "wb") as f:
            f.write(content)
        return True
    except Exception:
        return False


def _load_terms_from_disk(path: str) -> Optional[Set[str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            terms = {line.strip().lower() for line in f if line.strip()}
        return terms
    except Exception:
        return None


def get_medical_terms() -> Set[str]:
    global _TERMS, _COMMON_TERMS
    if _TERMS is not None:
        return _TERMS
    with _LOCK:
        if _TERMS is not None:
            return _TERMS
        path = _local_wordlist_path()
        logger.info("Loading medical terms from %s", path)
        terms = _load_terms_from_disk(path)
        if terms is None or len(terms) < 1000:
            logger.warning("Medical terms not found or incomplete, downloading...")
            if _download_wordlist(path):
                terms = _load_terms_from_disk(path)
        _TERMS = terms or set()
        logger.info("Loaded %d medical terms", len(_TERMS))
        # Cache top 10K most common medical terms (shorter = more likely in radiology)
        if _TERMS:
            sorted_terms = sorted(_TERMS, key=len)
            _COMMON_TERMS = sorted_terms[:10000]
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
    # Fast path: exact match
    if word_lower in terms:
        return word_lower
    
    # Use rapidfuzz if available (100x faster than difflib)
    if RAPIDFUZZ_AVAILABLE and _COMMON_TERMS:
        # Search common terms first (most medical words)
        result = process.extractOne(
            word_lower, 
            _COMMON_TERMS, 
            scorer=fuzz.ratio, 
            score_cutoff=cutoff * 100
        )
        if result:
            return result[0]
        # Fallback to full set if not found in common terms
        result = process.extractOne(
            word_lower,
            list(terms),
            scorer=fuzz.ratio,
            score_cutoff=cutoff * 100,
        )
        return result[0] if result else None
    
    # Slow fallback: no fuzzy matching without rapidfuzz
    return None
