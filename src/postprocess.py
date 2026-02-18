"""
Post-processing pipeline for radiology dictation transcripts.

Pipeline order (do NOT reorder without careful consideration):
  1. Correction commands  ("correct word X")
  2. Spoken punctuation   ("full stop", "comma", …)
  3. Space normalisation
  4. Measurement standardisation  (5 millimetres → 5 mm)
  5. MRI signal / sequence terminology  (T 1 → T1, hyper intense → hyperintense)
  6. MSK anatomy & pathology corrections
  7. General radiology / modality corrections
  8. Medical-dictionary fuzzy matching
  9. Smart capitalisation
"""

import re
from typing import Dict, List, Set, Tuple, Pattern

import medical_dict


# ---------------------------------------------------------------------------
# 1.  Voice correction commands
# ---------------------------------------------------------------------------

_CORRECTION_PATTERN1 = re.compile(
    r"([A-Za-z0-9'-]+)(\s*)(?:[,.!?;:]*)\s+(?:correct(?:\s+the)?\s+word)(?:\s+to)?\s+([A-Za-z0-9'-]+)\b",
    re.IGNORECASE,
)
_CORRECTION_PATTERN2 = re.compile(
    r"(?:^|[\s\n])(?:correct(?:\s+the)?\s+word)(?:\s+to)?\s+([A-Za-z0-9'-]+)\b",
    re.IGNORECASE,
)
_PREV_WORD_PATTERN = re.compile(r"\b([A-Za-z0-9'-]+)\b(?!.*\b[A-Za-z0-9'-]+\b)")
_WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z'\-]{2,}\b")


def apply_correction_commands(text: str) -> str:
    """Handle 'X correct word Y' and 'correct word Y' voice edits."""
    while True:
        text, n = _CORRECTION_PATTERN1.subn(
            lambda m: f"{m.group(3)}{m.group(2)}", text, count=1
        )
        if n == 0:
            break
    while True:
        m = _CORRECTION_PATTERN2.search(text)
        if not m:
            break
        new_word = m.group(1)
        cmd_start, cmd_end = m.start(), m.end()
        before = text[:cmd_start]
        m_prev = _PREV_WORD_PATTERN.search(before)
        if not m_prev:
            text = before.rstrip() + text[cmd_end:]
            continue
        ps, pe = m_prev.span(1)
        text = text[:ps] + new_word + text[pe:cmd_start] + text[cmd_end:]
    return text


# ---------------------------------------------------------------------------
# 2.  Spoken punctuation → symbols
# ---------------------------------------------------------------------------

_SPOKEN_PATTERNS: List[Tuple[Pattern, str]] = [
    (re.compile(r"\bfull[- ]?stop\b", re.IGNORECASE), "."),
    (re.compile(r"\bperiod\b", re.IGNORECASE), "."),
    (re.compile(r"\bcomma\b", re.IGNORECASE), ","),
    (re.compile(r"\bnew\s+line\b", re.IGNORECASE), "\n"),
    (re.compile(r"\bnewline\b", re.IGNORECASE), "\n"),
    (re.compile(r"\bnew\s+paragraph\b", re.IGNORECASE), "\n\n"),
    (re.compile(r"\bopen\s+bracket\b", re.IGNORECASE), "("),
    (re.compile(r"\bclose\s+bracket\b", re.IGNORECASE), ")"),
    (re.compile(r"\bquestion\s+mark\b", re.IGNORECASE), "?"),
    (re.compile(r"\bexclamation\s+mark\b", re.IGNORECASE), "!"),
    (re.compile(r"\bcolon\b", re.IGNORECASE), ":"),
    (re.compile(r"\bsemi[- ]?colon\b", re.IGNORECASE), ";"),
    (re.compile(r"\bhyphen\b", re.IGNORECASE), "-"),
    (re.compile(r"\bdash\b", re.IGNORECASE), " – "),
    (re.compile(r"\bnew\s+section\b", re.IGNORECASE), "\n\n"),
    (re.compile(r"\bnext\s+section\b", re.IGNORECASE), "\n\n"),
]


def apply_spoken_commands(text: str) -> str:
    for pattern, repl in _SPOKEN_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# 3.  Space normalisation
# ---------------------------------------------------------------------------

def normalize_spaces(text: str) -> str:
    text = re.sub(r"[^\S\n]+", " ", text)            # collapse horizontal whitespace
    text = re.sub(r"[ \t]+([.,;:!?)])", r"\1", text)  # remove space before punctuation
    return text.strip()


# ---------------------------------------------------------------------------
# 4.  Measurement standardisation
# ---------------------------------------------------------------------------

# Handles: "5 millimetre(s)", "5 millimeter(s)", "5 centimetre(s)", "5 centimeter(s)"
# and compound: "5 by 3 millimetres" → "5 x 3 mm"
_MEASURE_MM_COMPOUND = re.compile(
    r"(\d+(?:\.\d+)?)\s+(?:by|x|X)\s+(\d+(?:\.\d+)?)\s+millimet(?:re|er)s?\b",
    re.IGNORECASE,
)
_MEASURE_CM_COMPOUND = re.compile(
    r"(\d+(?:\.\d+)?)\s+(?:by|x|X)\s+(\d+(?:\.\d+)?)\s+(?:centimetre|centimeter)s?\b",
    re.IGNORECASE,
)
_MEASURE_MM = re.compile(
    r"(\d+(?:\.\d+)?)\s+millimet(?:re|er)s?\b", re.IGNORECASE
)
_MEASURE_CM = re.compile(
    r"(\d+(?:\.\d+)?)\s+(?:centimetre|centimeter)s?\b", re.IGNORECASE
)
_MEASURE_DEG = re.compile(
    r"(\d+(?:\.\d+)?)\s+degrees?\b", re.IGNORECASE
)


def apply_measurement_standardisation(text: str) -> str:
    text = _MEASURE_MM_COMPOUND.sub(r"\1 x \2 mm", text)
    text = _MEASURE_CM_COMPOUND.sub(r"\1 x \2 cm", text)
    text = _MEASURE_MM.sub(r"\1 mm", text)
    text = _MEASURE_CM.sub(r"\1 cm", text)
    text = _MEASURE_DEG.sub(r"\1°", text)
    return text


# ---------------------------------------------------------------------------
# 5.  MRI signal / sequence terminology
# ---------------------------------------------------------------------------

_SIGNAL_PATTERNS: List[Tuple[Pattern, str]] = [
    # Sequence names with spaces  (T 1 → T1, T 2 → T2, P D → PD)
    (re.compile(r"\bT\s+1\b"), "T1"),
    (re.compile(r"\bT\s+2\b"), "T2"),
    (re.compile(r"\bP\s+D\b", re.IGNORECASE), "PD"),
    (re.compile(r"\bS\s*T\s*I\s*R\b", re.IGNORECASE), "STIR"),
    (re.compile(r"\bP\s*D\s*F\s*S\b", re.IGNORECASE), "PDFS"),
    (re.compile(r"\bF\s*L\s*A\s*I\s*R\b", re.IGNORECASE), "FLAIR"),
    (re.compile(r"\bD\s*W\s*I\b", re.IGNORECASE), "DWI"),
    (re.compile(r"\bA\s*D\s*C\b", re.IGNORECASE), "ADC"),
    # Weighted sequences
    (re.compile(r"\bT1[- ]?weight(?:ed)?\b", re.IGNORECASE), "T1-weighted"),
    (re.compile(r"\bT2[- ]?weight(?:ed)?\b", re.IGNORECASE), "T2-weighted"),
    (re.compile(r"\bproton[- ]?density[- ]?weight(?:ed)?\b", re.IGNORECASE), "proton density-weighted"),
    # Hyper / hypo intensity (spoken with space)
    (re.compile(r"\bhyper[- ]?intens(?:e|ity)\b", re.IGNORECASE), "hyperintense"),
    (re.compile(r"\bhypo[- ]?intens(?:e|ity)\b", re.IGNORECASE), "hypointense"),
    (re.compile(r"\biso[- ]?intens(?:e|ity)\b", re.IGNORECASE), "isointense"),
    # Signal change compound terms
    (re.compile(r"\bbone\s+marrow\s+o[ea]dema\b", re.IGNORECASE), "bone marrow oedema"),
    (re.compile(r"\bmarrow\s+o[ea]dema\b", re.IGNORECASE), "bone marrow oedema"),
    (re.compile(r"\bBMO\b"), "bone marrow oedema"),
    # Sub-chondral
    (re.compile(r"\bsub[- ]?chondral\b", re.IGNORECASE), "subchondral"),
    (re.compile(r"\bsub[- ]?articular\b", re.IGNORECASE), "subarticular"),
    (re.compile(r"\bsub[- ]?acute\b", re.IGNORECASE), "subacute"),
    # Fat saturation
    (re.compile(r"\bfat[- ]?sat(?:urated|uration)?\b", re.IGNORECASE), "fat-saturated"),
    (re.compile(r"\bfat\s+suppressed\b", re.IGNORECASE), "fat-suppressed"),
]


def apply_signal_terminology(text: str) -> str:
    for pattern, repl in _SIGNAL_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# 6.  MSK anatomy & pathology corrections
# ---------------------------------------------------------------------------

_MSK_PATTERNS: List[Tuple[Pattern, str]] = [
    # Ligaments / tendons (ensure correct spelling)
    (re.compile(r"\bsupraspinatus\b", re.IGNORECASE), "supraspinatus"),
    (re.compile(r"\binfra[- ]?spinatus\b", re.IGNORECASE), "infraspinatus"),
    (re.compile(r"\bsubscapularis\b", re.IGNORECASE), "subscapularis"),
    (re.compile(r"\bteres\s+minor\b", re.IGNORECASE), "teres minor"),
    (re.compile(r"\bglenohumeral\b", re.IGNORECASE), "glenohumeral"),
    (re.compile(r"\bacromioclavicular\b", re.IGNORECASE), "acromioclavicular"),
    (re.compile(r"\btibiofemoral\b", re.IGNORECASE), "tibiofemoral"),
    (re.compile(r"\bpatellofemoral\b", re.IGNORECASE), "patellofemoral"),
    (re.compile(r"\bsacroiliac\b", re.IGNORECASE), "sacroiliac"),
    (re.compile(r"\biliotibial\b", re.IGNORECASE), "iliotibial"),
    # Menisci
    (re.compile(r"\bmenisci\b", re.IGNORECASE), "menisci"),
    (re.compile(r"\bmeniscus\b", re.IGNORECASE), "meniscus"),
    (re.compile(r"\bmeniscal\b", re.IGNORECASE), "meniscal"),
    # Cartilage
    (re.compile(r"\bchondral\b", re.IGNORECASE), "chondral"),
    (re.compile(r"\bchondromalacia\b", re.IGNORECASE), "chondromalacia"),
    (re.compile(r"\bosteochondral\b", re.IGNORECASE), "osteochondral"),
    (re.compile(r"\barticular\s+cartilage\b", re.IGNORECASE), "articular cartilage"),
    # Grading systems – ensure proper capitalisation and hyphenation
    (re.compile(r"\bkellgren[- ]?lawrence\b", re.IGNORECASE), "Kellgren-Lawrence"),
    (re.compile(r"\bouterbridge\b", re.IGNORECASE), "Outerbridge"),
    (re.compile(r"\bARCO\b", re.IGNORECASE), "ARCO"),
    (re.compile(r"\bSPARCC\b", re.IGNORECASE), "SPARCC"),
    (re.compile(r"\bmodic\b", re.IGNORECASE), "Modic"),
    # Pathologies
    (re.compile(r"\bosteophyte[s]?\b", re.IGNORECASE), "osteophyte"),
    (re.compile(r"\bosteophytosis\b", re.IGNORECASE), "osteophytosis"),
    (re.compile(r"\btendinopathy\b", re.IGNORECASE), "tendinopathy"),
    (re.compile(r"\btendinosis\b", re.IGNORECASE), "tendinosis"),
    (re.compile(r"\btenosynovitis\b", re.IGNORECASE), "tenosynovitis"),
    (re.compile(r"\bsynovitis\b", re.IGNORECASE), "synovitis"),
    (re.compile(r"\bbursitis\b", re.IGNORECASE), "bursitis"),
    (re.compile(r"\bspondylolisthesis\b", re.IGNORECASE), "spondylolisthesis"),
    (re.compile(r"\bspondylolysis\b", re.IGNORECASE), "spondylolysis"),
    (re.compile(r"\bspondylosis\b", re.IGNORECASE), "spondylosis"),
    (re.compile(r"\bspondyloarthrop(?:athy|athy)\b", re.IGNORECASE), "spondyloarthropathy"),
    (re.compile(r"\bavascular\s+necrosis\b", re.IGNORECASE), "avascular necrosis"),
    (re.compile(r"\bAVN\b"), "avascular necrosis"),
    (re.compile(r"\bimpingement\b", re.IGNORECASE), "impingement"),
    (re.compile(r"\btendinitis\b", re.IGNORECASE), "tendinopathy"),  # modern preferred term
    (re.compile(r"\bossification\b", re.IGNORECASE), "ossification"),
    (re.compile(r"\bcalcification[s]?\b", re.IGNORECASE), "calcification"),
    # Common fracture descriptors
    (re.compile(r"\bcommin(?:uted|ution)\b", re.IGNORECASE), "comminuted"),
    (re.compile(r"\bimpact(?:ed|ion)\b", re.IGNORECASE), "impacted"),
    (re.compile(r"\bavulsion\b", re.IGNORECASE), "avulsion"),
    (re.compile(r"\bcompression\s+fracture\b", re.IGNORECASE), "compression fracture"),
    # UK spelling preferences
    (re.compile(r"\b(?:oe|e)dema\b", re.IGNORECASE), "oedema"),
    (re.compile(r"\banaemia\b", re.IGNORECASE), "anaemia"),
    (re.compile(r"\bh[ae]emorrhage\b", re.IGNORECASE), "haemorrhage"),
    (re.compile(r"\borthop[ae]edic\b", re.IGNORECASE), "orthopaedic"),
]


def apply_msk_corrections(text: str) -> str:
    for pattern, repl in _MSK_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# 7.  General radiology / modality corrections
# ---------------------------------------------------------------------------

_RADIOLOGY_PATTERNS: List[Tuple[Pattern, str]] = [
    # Modalities
    (re.compile(r"\bcxr\b", re.IGNORECASE), "chest radiograph"),
    (re.compile(r"\bct\b", re.IGNORECASE), "CT"),
    (re.compile(r"\bmri\b", re.IGNORECASE), "MRI"),
    (re.compile(r"\busg\b", re.IGNORECASE), "ultrasound"),
    (re.compile(r"\bxray\b", re.IGNORECASE), "X-ray"),
    (re.compile(r"\bx-ray\b", re.IGNORECASE), "X-ray"),
    (re.compile(r"\bdexa\b", re.IGNORECASE), "DEXA"),
    (re.compile(r"\bpet[- ]?ct\b", re.IGNORECASE), "PET-CT"),
    # Common general terms
    (re.compile(r"\bpneumothorax\b", re.IGNORECASE), "pneumothorax"),
    (re.compile(r"\bconsolidation[s]?\b", re.IGNORECASE), "consolidation"),
    (re.compile(r"\bcardiomegaly\b", re.IGNORECASE), "cardiomegaly"),
    (re.compile(r"\batelectasis\b", re.IGNORECASE), "atelectasis"),
    (re.compile(r"\bpleural\s+effusion[s]?\b", re.IGNORECASE), "pleural effusion"),
    (re.compile(r"\bh[ae]patomegaly\b", re.IGNORECASE), "hepatomegaly"),
    (re.compile(r"\bsplenomegaly\b", re.IGNORECASE), "splenomegaly"),
    (re.compile(r"\bdiaphragm\b", re.IGNORECASE), "diaphragm"),
    (re.compile(r"\binterstitial\b", re.IGNORECASE), "interstitial"),
]


def apply_medical_corrections(text: str) -> str:
    for pattern, repl in _RADIOLOGY_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# 8.  Medical dictionary fuzzy matching
# ---------------------------------------------------------------------------

_ACRONYMS: Set[str] = {
    # Modalities
    "CT", "MRI", "US", "CXR", "PET", "SPECT", "ECG", "EEG", "DEXA",
    # MSK ligaments / structures
    "ACL", "PCL", "MCL", "LCL", "UCL", "RCL", "CFL", "ATFL",
    "TFCC", "SLIL", "LTIL", "SLAP", "HAGL",
    # Sequences
    "STIR", "PDFS", "FLAIR", "DWI", "ADC", "PD",
    # Grading / classification
    "ARCO", "SPARCC", "FAI",
    # Other medical
    "BMD", "BMI", "ROM", "OA", "RA", "AS", "SI", "APL", "EPB",
}

_MEDICAL_TERMS_CACHE: Set[str] = set()

# Terms already corrected by earlier pipeline steps (UK spellings, MSK terms).
# The fuzzy matcher must not override these with US alternatives.
_PROTECTED_TERMS: Set[str] = {
    # UK spellings enforced by step 6
    "oedema", "anaemia", "haemorrhage", "orthopaedic",
    # MSK terms enforced by steps 5-6 that fuzzy matching might mangle
    "hyperintense", "hypointense", "isointense", "subchondral", "subarticular",
    "subacute", "infraspinatus", "supraspinatus", "subscapularis",
    "spondylolisthesis", "spondylolysis", "spondylosis", "spondyloarthropathy",
    "tendinopathy", "tendinosis", "tenosynovitis", "osteophytosis",
    "chondromalacia", "osteochondral",
    # Common English words the fuzzy matcher incorrectly "corrects"
    "measures", "measured", "approximate", "approximately", "demonstrates",
    "demonstrated", "maintained", "identified", "visualised", "visualized",
    "appeared", "appears", "consistent", "suggesting", "suggested",
    "otherwise", "unremarkable", "normal", "within", "without",
}


def _ensure_medical_terms() -> Set[str]:
    global _MEDICAL_TERMS_CACHE
    if not _MEDICAL_TERMS_CACHE:
        _MEDICAL_TERMS_CACHE = medical_dict.get_medical_terms()
    return _MEDICAL_TERMS_CACHE


def apply_medical_dictionary_suggestions(text: str) -> str:
    """Replace likely mis-transcribed words using 90 % fuzzy similarity."""
    terms = _ensure_medical_terms()
    if not terms:
        return text

    def replace_word(m: re.Match) -> str:
        w = m.group(0)
        if w.upper() in _ACRONYMS:
            return w
        wl = w.lower()
        if wl in terms or wl in _PROTECTED_TERMS or len(wl) < 5:
            return w
        sug = medical_dict.suggest_correction(wl, cutoff=0.92)
        if not sug:
            return w
        if w.isupper():
            return sug.upper()
        if w[0].isupper():
            return sug[0].upper() + sug[1:]
        return sug

    return _WORD_PATTERN.sub(replace_word, text)


# ---------------------------------------------------------------------------
# 9.  Smart capitalisation
# ---------------------------------------------------------------------------

_PRESERVE_CAPS = re.compile(
    r"^(CT|MRI|US|CXR|STIR|PDFS|FLAIR|DWI|ADC|ACL|PCL|MCL|LCL|UCL|TFCC|SLAP|HAGL|"
    r"ARCO|SPARCC|DEXA|PET|BMD|FAI|SI|APL|EPB)\b"
)


def smart_capitalize(text: str) -> str:
    def _cap(s: str) -> str:
        s = s.strip()
        if not s:
            return s
        if _PRESERVE_CAPS.match(s):
            return s
        return s[0].upper() + s[1:]

    parts = re.split(r"([.!?]+\s+)", text)
    out: List[str] = []
    for i in range(0, len(parts), 2):
        out.append(_cap(parts[i]))
        if i + 1 < len(parts):
            out.append(parts[i + 1])
    return "".join(out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def postprocess_transcript(text: str) -> str:
    """Full post-processing pipeline for a raw dictation chunk."""
    text = apply_correction_commands(text)
    text = apply_spoken_commands(text)
    text = normalize_spaces(text)
    text = apply_measurement_standardisation(text)
    text = apply_signal_terminology(text)
    text = apply_msk_corrections(text)
    text = apply_medical_corrections(text)
    text = apply_medical_dictionary_suggestions(text)
    text = smart_capitalize(text)
    return text
