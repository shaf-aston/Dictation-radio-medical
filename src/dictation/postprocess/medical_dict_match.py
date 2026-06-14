"""Stage 7 — fuzzy medical-dictionary correction.

Replaces a misspelled word with the closest entry in the curated **radiology
lexicon** (``medical_dict.get_correction_targets``) when it is within a small,
length-scaled **edit distance** of that term. Acronyms, protected terms, and
short words are never touched. Snapping to the curated lexicon — not the broad
98k generic wordlist used for membership — is what keeps a typo from being
pulled toward chemistry/drug junk that merely sits one edit away.

Why edit distance and not a flat similarity ratio: a single typo in a
normal-length word ("vertabra" → "vertebra", "atelactasis" → "atelectasis")
only scores ~88–93% on a character ratio, so a flat 94% floor silently left
the entire class of one-letter medical misspellings uncorrected. Edit
distance separates "1 typo away from a real term" (a fix) from "a genuinely
different word" (leave alone) the way a ratio cannot. This is the SymSpell
acceptance model, scoped to non-dictionary words only.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Set

from src.medical import medical_dict

logger = logging.getLogger(__name__)

try:  # rapidfuzz is a core dep, but mirror medical_dict's guard for stub envs
    from rapidfuzz import process  # type: ignore
    from rapidfuzz.distance import DamerauLevenshtein  # type: ignore
    _DL_AVAILABLE = True
except ImportError:
    process = None  # type: ignore
    DamerauLevenshtein = None  # type: ignore
    _DL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Floor used in the fallback path when no English guard is available: without a
# way to tell a real word from a non-word we keep the old conservative ratio so
# the stage never demotes ordinary English.
_LEGACY_CUTOFF = 0.94
_MIN_LEN = 5                            # ignore words shorter than this
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'\-]{2,}\b")


def _max_edits(n: int) -> int:
    """Edits we'll forgive as a typo, scaled by word length.

    One edit is enough for ordinary words; long medical terms ("lymphade-
    nopathy", "choledocholithiasis") routinely take two transcription slips,
    so they get a little more room. Short words get none — at <5 chars a
    single edit too easily lands on an unrelated real word.
    """
    if n < 5:
        return 0
    return 1 if n <= 9 else 2


def _shared_prefix_len(a: str, b: str) -> int:
    """Number of leading characters *a* and *b* have in common."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _nearest_term(word: str, max_distance: int) -> Optional[str]:
    """Closest medical term to *word* within *max_distance* edits, or None.

    Retrieval and acceptance use the SAME metric (edit distance), so we never
    reject a fixable typo just because some other term scored higher on a
    different (ratio) metric. Searches the curated radiology lexicon for a
    deterministic, truly-nearest result.

    When several terms are equidistant, raw edit distance alone is a coin toss
    that can pick a shorter unrelated word ("efusion" is one edit from both
    "effusion" and "fusion"). A real misspelling almost always keeps the start
    of the intended word, so ties break first on the longest shared prefix and
    then on the closest length — steering "efusion" to "effusion".
    """
    if max_distance < 1:
        return None
    terms = medical_dict.get_correction_targets() or None
    if not terms:
        return None
    # Callers only reach here when _DL_AVAILABLE is True (see the guard in
    # _replace), which is exactly when rapidfuzz set these to real values.
    assert process is not None and DamerauLevenshtein is not None
    matches = process.extract(
        word, terms, scorer=DamerauLevenshtein.distance,
        score_cutoff=max_distance, limit=25,
    )
    if not matches:
        return None
    best = min(
        matches,
        key=lambda m: (m[1], -_shared_prefix_len(word, m[0]), abs(len(m[0]) - len(word))),
    )
    return best[0]


# ---------------------------------------------------------------------------
# English-word guard
# ---------------------------------------------------------------------------
# A genuine typo is a *non-word*. We must never "correct" a word that is already
# valid English ("there", "around", "again") just because a real medical term
# ("marrow" vs "narrow") sits one edit away. pyspellchecker bundles an OFFLINE
# frequency dictionary (no network), so it's the guard. If it isn't installed,
# `_english_known` returns None and the stage drops back to the conservative
# ratio-only path — a missing dep can never *add* over-correction risk.
_ENGLISH = None  # None = not yet loaded; False = unavailable; else a SpellChecker


def _english_known(word: str) -> Optional[bool]:
    """Return True/False if *word* is/isn't standard English, or None if no checker."""
    global _ENGLISH
    if _ENGLISH is None:
        try:
            from spellchecker import SpellChecker
            _ENGLISH = SpellChecker()
        except Exception:  # not installed / failed to load — guard unavailable
            _ENGLISH = False
            # Loud, once: without this guard the stage drops to the conservative
            # ratio path and silently leaves the whole class of one-letter medical
            # misspellings uncorrected. A silent degradation here is exactly how
            # "the dictation keeps misspelling things" goes undiagnosed.
            logger.warning(
                "Spelling corrector degraded: pyspellchecker is not installed, so "
                "the English-word guard is off and one-letter medical misspellings "
                "(e.g. 'atelactasis'->'atelectasis', 'vertabra'->'vertebra') will "
                "NOT be corrected. Install it: pip install pyspellchecker"
            )
    return None if _ENGLISH is False else bool(_ENGLISH.known([word]))

# Inflectional suffixes the matcher must neither add nor strip. A fuzzy
# "correction" that only changes a word's grammatical number or tense is
# never a spelling fix — it silently rewrites what the radiologist said
# (e.g. "findings" → "finding", "margins" → "margines", "resolved" →
# "resolve"). The bundled wordlist stores singular/base forms, so without
# this guard every plural or past-tense word gets demoted to its base.
_INFLECTIONS = ("es", "ed", "ing", "s", "d")
_MIN_STEM = 4

# Acronyms preserved verbatim regardless of case.
_ACRONYMS: Set[str] = {
    "CT", "MRI", "US", "CXR", "PET", "SPECT", "ECG", "EEG", "DEXA",
    "ACL", "PCL", "MCL", "LCL", "UCL", "RCL", "CFL", "ATFL",
    "TFCC", "SLIL", "LTIL", "SLAP", "HAGL",
    "STIR", "PDFS", "FLAIR", "DWI", "ADC", "PD",
    "ARCO", "SPARCC", "FAI",
    "BMD", "BMI", "ROM", "OA", "RA", "AS", "SI", "APL", "EPB", "AVN",
}

# Terms already enforced by earlier stages — fuzzy matching must NOT
# rewrite these into US/alt forms.
PROTECTED_TERMS: Set[str] = {
    # UK spellings.
    "oedema", "anaemia", "haemorrhage", "orthopaedic",
    # MSK signal terms.
    "hyperintense", "hypointense", "isointense",
    "subchondral", "subarticular", "subacute",
    "infraspinatus", "supraspinatus", "subscapularis",
    "spondylolisthesis", "spondylolysis", "spondylosis", "spondyloarthropathy",
    "tendinopathy", "tendinosis", "tenosynovitis", "osteophytosis",
    "chondromalacia", "osteochondral", "transchondral",
    # Ultrasound echogenicity.
    "hyperechoic", "hypoechoic", "isoechoic", "anechoic",
    "echotexture", "echogenicity", "homogeneous", "heterogeneous",
    # MSK compound forms.
    "intrasubstance", "periarticular", "retrocalcaneal", "subperiosteal",
    "neuroforaminal", "paracentral",
    "hyperextension", "hyperflexion", "hypermobility",
    # Whisper mis-transcriptions already fixed.
    "tear", "tears", "rotator", "cruciate", "collateral",
    "posterior", "anterior", "labrum", "labral",
    "plantar", "fasciitis", "fascia",
    "enthesopathy", "enthesitis", "Achilles", "intact",
    # English words the fuzzy matcher tends to over-correct.
    "measures", "measured", "approximate", "approximately",
    "demonstrates", "demonstrated", "maintained",
    "identified", "visualised", "visualized",
    "appeared", "appears", "consistent",
    "suggesting", "suggested",
    "otherwise", "unremarkable", "normal", "within", "without",
    # Clinical words whose near-neighbour in the medical dict is a different concept.
    "decreased", "decreases", "increasing", "increased",
    "medial", "lateral", "superior", "inferior",
    "proximal", "distal", "dorsal", "volar", "palmar",
    # Common plural pairs.
    "erosions", "erosion", "calcifications", "calcification",
    "contusions", "contusion", "osteophytes", "osteophyte",
    "effusions", "effusion", "fractures", "fracture",
    # Enzymes already corrected.
    "amylases", "amylase", "lipases", "lipase",
    # UK spellings from terminology.
    "tumour", "tumours", "haematoma", "haematomas", "haematuria",
    "paediatric", "paediatrics",
    # Contrast terms.
    "gadolinium", "enhancement",
    # Hyphenated compounds.
    "post-contrast", "pre-contrast", "non-enhancing",
    "intra-articular", "extra-articular", "juxta-articular",
    "full-thickness", "partial-thickness", "high-grade", "low-grade",
    "non-displaced", "fat-saturated", "fat-suppressed",
    "t1-weighted", "t2-weighted",
    # General radiology terms.
    "consolidation", "consolidations", "pneumothorax", "atelectasis",
    "cardiomegaly", "hepatomegaly", "splenomegaly", "diaphragm",
    "interstitial", "cholelithiasis", "cholecystitis", "choledocholithiasis",
    "hydronephrosis", "nephrolithiasis", "urolithiasis", "pancreatitis",
    "diverticulitis", "diverticulosis", "appendicitis", "lymphadenopathy",
    "bronchiectasis", "emphysema", "pericardial", "mediastinal",
    # CT density terms.
    "hyperdense", "hypodense", "isodense",
    # Anatomical "colon" — must not be replaced with the punctuation rule.
    "colon",
}

# Lazy cache of the dictionary terms.
_MEDICAL_TERMS_CACHE: Set[str] = set()


def _ensure_medical_terms() -> Set[str]:
    global _MEDICAL_TERMS_CACHE
    if not _MEDICAL_TERMS_CACHE:
        _MEDICAL_TERMS_CACHE = medical_dict.get_medical_terms()
    return _MEDICAL_TERMS_CACHE


def _stems(word: str) -> Set[str]:
    """Return candidate base forms of *word* after undoing one inflection.

    Covers simple suffix stripping (-s, -ed, -ing, …) plus the English ``y→ies``
    plural ("opacities" → "opacity"). Without the ``ies→y`` case a correctly
    spelled plural whose singular is in the wordlist (but the plural is not)
    looks like a non-word and gets snapped to a near neighbour
    ("opacities" → "opacifies"); restoring the singular lets the inflection
    guard recognise it as already-correct and leave it alone.
    """
    stems = {
        word[: -len(suf)]
        for suf in _INFLECTIONS
        if word.endswith(suf) and len(word) - len(suf) >= _MIN_STEM
    }
    if word.endswith("ies") and len(word) >= _MIN_STEM + 2:
        stems.add(f"{word[:-3]}y")
    return stems


def _is_inflected_form(word: str, terms: Set[str]) -> bool:
    """True when *word* is an inflected form of a term that is in *terms*.

    e.g. "findings" → strip "s" → "finding" is in the medical dict.
    This means the word is already correct — just in a form the dict
    doesn't store — so the fuzzy stage must leave it alone.
    """
    return bool(_stems(word) & terms)


def _only_inflection_differs(a: str, b: str) -> bool:
    """True when *a* and *b* share a stem but differ only by an inflectional
    suffix with no other character changes.

    Uses stem overlap AND requires the shared stem to account for at least
    80% of each word's length, so genuine typos like "meniscuss→meniscus"
    (extra consonant, no suffix stripping involved) are not suppressed.
    """
    stems_a = _stems(a)
    stems_b = _stems(b)
    shared = stems_a & stems_b
    if not shared:
        return False
    # Accept the overlap only when the common stem is long relative to both words
    stem_len = max(len(s) for s in shared)
    return stem_len / len(a) >= 0.80 and stem_len / len(b) >= 0.80


def apply_medical_dictionary_suggestions(text: str) -> str:
    """Replace likely mis-transcribed words using high-similarity fuzzy match."""
    terms = _ensure_medical_terms()
    if not terms:
        return text

    def _cased(w: str, sug: str) -> str:
        """Re-apply *w*'s capitalisation to the lowercase suggestion."""
        if w.isupper():
            return sug.upper()
        return sug[0].upper() + sug[1:] if w[0].isupper() else sug

    def _replace(m: re.Match) -> str:
        w = m[0]
        if w.upper() in _ACRONYMS:
            return w
        wl = w.lower()
        if wl in terms or wl in PROTECTED_TERMS or len(wl) < _MIN_LEN:
            return w
        # An inflected form of a known term is already correct.
        if _is_inflected_form(wl, terms) or _is_inflected_form(wl, PROTECTED_TERMS):
            return w

        eng = _english_known(wl)
        if eng:
            # A valid English word is never a typo to fix — leave it untouched,
            # no matter how close a junk wordlist fragment sits.
            return w
        if eng is None or not _DL_AVAILABLE:
            # No English guard (or no distance metric) available — fall back to
            # the conservative ratio-only path so we never demote a real word.
            sug = medical_dict.suggest_correction(wl, cutoff=_LEGACY_CUTOFF)
            return w if not sug or _only_inflection_differs(wl, sug) else _cased(w, sug)
        # Confirmed non-word: snap to the nearest term within a typo's edit
        # distance — this is what catches the simple medical misspellings.
        sug = _nearest_term(wl, _max_edits(len(wl)))
        return w if not sug or _only_inflection_differs(wl, sug) else _cased(w, sug)

    return _WORD_RE.sub(_replace, text)
