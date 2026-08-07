"""Stage 7 — fuzzy medical-dictionary correction.

Replaces a misspelled word with the closest known medical term — any of the
~98k terms in :func:`medical_dict.get_medical_terms`, not just the 756-term
curated radiology lexicon — when it is within a small, length-scaled **edit
distance** of that term (via :func:`medical_dict.get_symspell`). Acronyms,
protected terms, and short words are never touched. Curated radiology-lexicon
entries are frequency-boosted in that index, so a tied edit distance still
prefers the clean radiology spelling ("efusion" -> "effusion", not "fusion")
rather than the alternative of restricting the candidate pool to the lexicon
alone — which left common terms like "esophageal" or "thyroid" uncorrectable
simply because they weren't in that smaller list.

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
from src.medical.medical_dict import is_english_word

logger = logging.getLogger(__name__)

try:  # symspellpy is a core dep, but mirror medical_dict's guard for stub envs
    from symspellpy import Verbosity  # type: ignore
    _SYMSPELL_AVAILABLE = True
except ImportError:
    Verbosity = None  # type: ignore
    _SYMSPELL_AVAILABLE = False

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


def _nearest_term(word: str, max_distance: int, sym: Optional[object] = None) -> Optional[str]:
    """Closest known medical term to *word* within *max_distance* edits, or None.

    Looks up the SymSpell index (:func:`medical_dict.get_symspell`), which
    covers every term in the ~98k-word membership set with curated
    radiology-lexicon entries frequency-boosted. ``Verbosity.CLOSEST`` returns
    every term at the smallest edit distance found, ranked by that boosted
    frequency — so a tie between "effusion" (lexicon) and "fusion" (generic)
    for "efusion" resolves to "effusion".

    *sym* lets the caller hoist :func:`medical_dict.get_symspell` out of a
    per-word loop and reuse the one cached index across a whole document.
    """
    if max_distance < 1:
        return None
    if sym is None:
        sym = medical_dict.get_symspell()
    if sym is None or Verbosity is None:
        return None
    suggestions = sym.lookup(
        word, Verbosity.CLOSEST, max_edit_distance=max_distance, transfer_casing=False,
    )
    return suggestions[0].term if suggestions else None


# ---------------------------------------------------------------------------
# English-word guard
# A genuine typo is a *non-word*, and this stage must never "correct" a word
# that is already valid English. The guard itself now lives beside the wordlists
# in :mod:`src.medical.medical_dict`, so this stage and the marking scan cannot
# disagree about what counts as a real word; the local name is kept because it
# reads better at the call sites here (and `warmup` imports it by this name).
_english_known = is_english_word

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


# British "ae"/"oe" digraphs that reduce to a single "e" in American spelling
# (e.g. "haematology" -> "hematology", "oestrogen" -> "estrogen",
# "leukaemia" -> "leukemia"). Correctly-spelled British medical terms are
# often absent from both the membership set and pyspellchecker's
# American-leaning dictionary, so without this check the fuzzy stage was
# silently "correcting" them to American spelling ("haemodynamic" ->
# "hemodynamic", "gynaecology" -> "gynecology") -- this only ever protects a
# word from correction, it never rewrites the output.
_BRITISH_DIGRAPHS = (("ae", "e"), ("oe", "e"))


def _is_british_spelling(word: str, terms: Set[str]) -> bool:
    """True if *word* is a correctly-spelled British variant of a known term."""
    for uk, us in _BRITISH_DIGRAPHS:
        if uk in word:
            candidate = word.replace(uk, us)
            if candidate in terms or _english_known(candidate):
                return True
    return False


# Per-word decision cache. The correction verdict for a given lowercased word is
# a pure function of that word and the load-once membership / protected / lexicon
# sets, so it is stable for the process's life. Caching it makes a repeated word
# — every live re-emit of a still-open sentence, every recurring anatomy term —
# an O(1) dict hit instead of re-paying the SpellChecker + SymSpell lookups that
# dominate this stage (measured 50ms mean / 257ms p95 without this cache). Value:
# the lowercase correction, or None to leave the word unchanged. A signature of
# the term-set sizes invalidates the cache transparently if the dictionary is
# reloaded mid-session (rare).
_DECISION_MEMO: dict = {}
_DECISION_MEMO_SIG: Optional[tuple] = None
# Sentinel distinguishing "cached: leave word unchanged (None)" from "not cached",
# so a memoised None is still a hit and never recomputed.
_MISS = object()


def _compute_decision(wl: str, terms: Set[str], sym: Optional[object]) -> Optional[str]:
    """Correction for already-lowercased non-word *wl*, or None to leave it alone.

    Pure function of *wl* and the load-once term sets (that is what makes the
    result safe to memoize). Casing is re-applied by the caller, never stored,
    so the cached value is independent of how the word was capitalised in the
    source. Mirrors the original per-word decision chain exactly — same guards,
    same order, same fallback conditions — only refactored out of the closure so
    it can be cached.
    """
    if wl in terms or wl in PROTECTED_TERMS:
        return None
    # An inflected form of a known term is already correct. Compute the stem set
    # once and test it against both membership and protected sets (was two
    # separate _stems() passes over the same word).
    stems = _stems(wl)
    if stems & terms or stems & PROTECTED_TERMS:
        return None
    # A correctly-spelled British variant ("haemodynamic", "oestrogen") of a
    # known term is already correct, even if neither the word itself nor
    # pyspellchecker recognises the British form directly.
    if _is_british_spelling(wl, terms):
        return None

    eng = _english_known(wl)
    if eng:
        # A valid English word is never a typo to fix — leave it untouched,
        # no matter how close a junk wordlist fragment sits.
        return None
    if eng is None or not _SYMSPELL_AVAILABLE:
        # No English guard (or no SymSpell index) available — fall back to the
        # conservative ratio-only path so we never demote a real word.
        sug = medical_dict.suggest_correction(wl, cutoff=_LEGACY_CUTOFF)
        return None if not sug or _only_inflection_differs(wl, sug) else sug
    # Confirmed non-word: snap to the nearest term within a typo's edit distance
    # — this is what catches the simple medical misspellings.
    sug = _nearest_term(wl, _max_edits(len(wl)), sym)
    return None if not sug or _only_inflection_differs(wl, sug) else sug


def apply_medical_dictionary_suggestions(text: str) -> str:
    """Replace likely mis-transcribed words using high-similarity fuzzy match."""
    global _DECISION_MEMO_SIG
    terms = _ensure_medical_terms()
    if not terms:
        return text

    # Hoist the SymSpell index once per document instead of re-fetching (and
    # re-validating its signature) inside every candidate word's lookup.
    sym = medical_dict.get_symspell()

    # Drop the per-word cache if the underlying term sets changed since it was
    # populated (e.g. a dictionary reload) — cheap size-signature check.
    sig = (len(terms), len(PROTECTED_TERMS))
    if sig != _DECISION_MEMO_SIG:
        _DECISION_MEMO.clear()
        _DECISION_MEMO_SIG = sig

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
        if len(wl) < _MIN_LEN:
            return w
        sug = _DECISION_MEMO.get(wl, _MISS)
        if sug is _MISS:
            sug = _compute_decision(wl, terms, sym)
            _DECISION_MEMO[wl] = sug
        return _cased(w, sug) if sug else w

    return _WORD_RE.sub(_replace, text)
