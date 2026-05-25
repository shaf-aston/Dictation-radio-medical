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

_CUTOFF = 0.92                          # rapidfuzz similarity floor
_MIN_LEN = 5                            # ignore words shorter than this
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'\-]{2,}\b")

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


def apply_medical_dictionary_suggestions(text: str) -> str:
    """Replace likely mis-transcribed words using high-similarity fuzzy match."""
    terms = _ensure_medical_terms()
    if not terms:
        return text

    def _replace(m: re.Match) -> str:
        w = m.group(0)
        if w.upper() in _ACRONYMS:
            return w
        wl = w.lower()
        if wl in terms or wl in PROTECTED_TERMS or len(wl) < _MIN_LEN:
            return w
        sug = medical_dict.suggest_correction(wl, cutoff=_CUTOFF)
        if not sug:
            return w
        if w.isupper():
            return sug.upper()
        if w[0].isupper():
            return sug[0].upper() + sug[1:]
        return sug

    return _WORD_RE.sub(_replace, text)
