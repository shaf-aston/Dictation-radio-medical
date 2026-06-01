"""Stage 7 — fuzzy medical-dictionary correction.

Replaces uncertain words with the closest entry in the bundled medical
wordlist when similarity is at least :data:`_CUTOFF` (default 92%).
Acronyms, protected terms, and short words are never touched.
"""

from __future__ import annotations

import re
from typing import Set

from src.medical import medical_dict

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

_CUTOFF = 0.94                          # rapidfuzz similarity floor
_MIN_LEN = 5                            # ignore words shorter than this
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'\-]{2,}\b")

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
    """Return stems of *word* after stripping one inflectional suffix."""
    return {
        word[: -len(suf)]
        for suf in _INFLECTIONS
        if word.endswith(suf) and len(word) - len(suf) >= _MIN_STEM
    }


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
        sug = medical_dict.suggest_correction(wl, cutoff=_CUTOFF)
        # Reject suggestions that differ only in inflectional suffix (number/tense).
        if not sug or _only_inflection_differs(wl, sug):
            return w
        if w.isupper():
            return sug.upper()
        return sug[0].upper() + sug[1:] if w[0].isupper() else sug

    return _WORD_RE.sub(_replace, text)
