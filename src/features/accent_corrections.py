"""
Accent-specific Whisper mis-transcription correction patterns.

Whisper is trained on multi-accent data but still produces systematic errors
with certain accents.  These patterns correct the *Whisper output* – they
target the text that Whisper generates when it hears accented speech, not the
accent itself.

Patterns are eagerly compiled at module load to ensure they're ready for use.
A combined regex approach is used for faster matching where possible.

Supported accents:
  - south_asian   : Indian, Pakistani, Bangladeshi, Sri Lankan English
  - middle_eastern: Arabic, Farsi, Urdu-influenced English
  - east_asian    : Chinese, Japanese, Korean-influenced English
  - west_african  : Nigerian, Ghanaian English
  - neutral       : No accent-specific corrections (default)
"""

import re
from typing import Callable, Dict, List, Optional, Pattern, Tuple, Union

# Type alias – replacement can be a string or callable
_PatternList = List[Tuple[Pattern, Union[str, Callable[[re.Match], str]]]]

# ---------------------------------------------------------------------------
# South Asian English (India / Pakistan / Bangladesh / Sri Lanka)
#
# Common Whisper errors with South Asian accents:
#   - Aspirated stops: "rapture" for "rupture", "frachure" for "fracture"
#   - th→t/d: "toracic" for "thoracic"
#   - v↔w swap: "wertebra" for "vertebra", "wein" for "vein"
#   - Vowel shifts: "tandon" for "tendon", "normel" for "normal"
#   - Dropped/changed vowels: "legament" for "ligament"
# ---------------------------------------------------------------------------

_SOUTH_ASIAN: _PatternList = [
    # ── v ↔ w confusion ──
    # Whisper transcribes /v/ as "w" or vice versa in SA English
    (re.compile(r"\bwertebra\b", re.IGNORECASE), "vertebra"),
    (re.compile(r"\bwertebral\b", re.IGNORECASE), "vertebral"),
    (re.compile(r"\bwertebrae\b", re.IGNORECASE), "vertebrae"),
    (re.compile(r"\bwein\b", re.IGNORECASE), "vein"),
    (re.compile(r"\bweins\b", re.IGNORECASE), "veins"),
    (re.compile(r"\bwalve\b", re.IGNORECASE), "valve"),
    (re.compile(r"\bwalves\b", re.IGNORECASE), "valves"),
    (re.compile(r"\bwascular\b", re.IGNORECASE), "vascular"),
    (re.compile(r"\bwascularity\b", re.IGNORECASE), "vascularity"),
    (re.compile(r"\bwolar\b", re.IGNORECASE), "volar"),
    (re.compile(r"\bwisuali[sz]ed?\b", re.IGNORECASE), "visualised"),

    # ── th → t/d confusion ──
    (re.compile(r"\btoracic\b", re.IGNORECASE), "thoracic"),
    (re.compile(r"\btorassic\b", re.IGNORECASE), "thoracic"),
    (re.compile(r"\btorax\b", re.IGNORECASE), "thorax"),
    (re.compile(r"\btickness\b", re.IGNORECASE), "thickness"),
    (re.compile(r"\btic?kness\b", re.IGNORECASE), "thickness"),
    (re.compile(r"\bartritis\b", re.IGNORECASE), "arthritis"),
    (re.compile(r"\bartropathy\b", re.IGNORECASE), "arthropathy"),
    (re.compile(r"\bartroscopy\b", re.IGNORECASE), "arthroscopy"),
    (re.compile(r"\bartroplasty\b", re.IGNORECASE), "arthroplasty"),

    # ── Aspirated stops / vowel shifts ──
    # "rapture" (common SA Whisper error for "rupture")
    (re.compile(r"\brapture(?=\s+(?:of|in|at|the|is|was|has)\b)", re.IGNORECASE), "rupture"),
    (re.compile(r"\braptured\b", re.IGNORECASE), "ruptured"),
    # "tandon" → "tendon"
    (re.compile(r"\btandon\b", re.IGNORECASE), "tendon"),
    (re.compile(r"\btandons\b", re.IGNORECASE), "tendons"),
    (re.compile(r"\btandinopathy\b", re.IGNORECASE), "tendinopathy"),
    (re.compile(r"\btandinosis\b", re.IGNORECASE), "tendinosis"),
    # "legament" / "ligamant" → "ligament"
    (re.compile(r"\blegament\b", re.IGNORECASE), "ligament"),
    (re.compile(r"\blegaments\b", re.IGNORECASE), "ligaments"),
    (re.compile(r"\blegamentous\b", re.IGNORECASE), "ligamentous"),
    (re.compile(r"\bligamant\b", re.IGNORECASE), "ligament"),
    (re.compile(r"\bligamants\b", re.IGNORECASE), "ligaments"),
    # "normel" → "normal"
    (re.compile(r"\bnormel\b", re.IGNORECASE), "normal"),
    # "frachure" / "fraccher" → "fracture"
    (re.compile(r"\bfrachure\b", re.IGNORECASE), "fracture"),
    (re.compile(r"\bfrachures\b", re.IGNORECASE), "fractures"),
    (re.compile(r"\bfraccher\b", re.IGNORECASE), "fracture"),
    # "harniation" → "herniation"
    (re.compile(r"\bharniation\b", re.IGNORECASE), "herniation"),
    (re.compile(r"\bharniated\b", re.IGNORECASE), "herniated"),
    # "protusion" → "protrusion"
    (re.compile(r"\bprotusion\b", re.IGNORECASE), "protrusion"),
    (re.compile(r"\bprotusions\b", re.IGNORECASE), "protrusions"),
    # "efusion" / "afusion" → "effusion"
    (re.compile(r"\b[ae]fusion\b", re.IGNORECASE), "effusion"),
    (re.compile(r"\b[ae]fusions\b", re.IGNORECASE), "effusions"),
    # "arosion" → "erosion"
    (re.compile(r"\barosion\b", re.IGNORECASE), "erosion"),
    (re.compile(r"\barosions\b", re.IGNORECASE), "erosions"),
    # "pattella" → "patella"
    (re.compile(r"\bpattella\b", re.IGNORECASE), "patella"),
    (re.compile(r"\bpattellar\b", re.IGNORECASE), "patellar"),
    # "acitabulum" → "acetabulum"
    (re.compile(r"\bacitabulum\b", re.IGNORECASE), "acetabulum"),
    (re.compile(r"\bacitabular\b", re.IGNORECASE), "acetabular"),
    # "subchondrel" → "subchondral"
    (re.compile(r"\bsubchondrel\b", re.IGNORECASE), "subchondral"),
    # "osteofyte" → "osteophyte" (ph → f confusion)
    (re.compile(r"\bosteofyte\b", re.IGNORECASE), "osteophyte"),
    (re.compile(r"\bosteofytes\b", re.IGNORECASE), "osteophytes"),
    # "kalsification" → "calcification" (c → k)
    (re.compile(r"\bkalsification\b", re.IGNORECASE), "calcification"),
    (re.compile(r"\bkalsifications\b", re.IGNORECASE), "calcifications"),

    # ── Dropped syllables / slurred endings ──
    (re.compile(r"\btendonitis\b", re.IGNORECASE), "tendinopathy"),
    (re.compile(r"\bmenesci\b", re.IGNORECASE), "menisci"),
    (re.compile(r"\bmenescus\b", re.IGNORECASE), "meniscus"),
    (re.compile(r"\bmenescal\b", re.IGNORECASE), "meniscal"),
    (re.compile(r"\bsupraspinatous\b", re.IGNORECASE), "supraspinatus"),
    (re.compile(r"\binfraspinatous\b", re.IGNORECASE), "infraspinatus"),

    # ── Common SA medical English patterns ──
    # "disc" → "disk" and back (SA uses both; standardise to UK "disc")
    (re.compile(r"\bdisk\b", re.IGNORECASE), "disc"),
    (re.compile(r"\bdisks\b", re.IGNORECASE), "discs"),
]

# ---------------------------------------------------------------------------
# Middle Eastern / Arabic / Farsi-influenced English
#
# Primary confusion: /p/ ↔ /b/ (Arabic has no /p/)
# Also: /v/ → /f/, emphatic consonants
# ---------------------------------------------------------------------------

_MIDDLE_EASTERN: _PatternList = [
    # ── p ↔ b confusion ──
    (re.compile(r"\bbneumothorax\b", re.IGNORECASE), "pneumothorax"),
    (re.compile(r"\bbalmar\b", re.IGNORECASE), "palmar"),
    (re.compile(r"\bbosterior\b", re.IGNORECASE), "posterior"),
    (re.compile(r"\bbroximal\b", re.IGNORECASE), "proximal"),
    (re.compile(r"\bberoneal\b", re.IGNORECASE), "peroneal"),
    (re.compile(r"\bbatella\b", re.IGNORECASE), "patella"),
    (re.compile(r"\bbatellar\b", re.IGNORECASE), "patellar"),
    (re.compile(r"\bblanter\b", re.IGNORECASE), "plantar"),
    (re.compile(r"\bblantar\b", re.IGNORECASE), "plantar"),
    (re.compile(r"\bberiosteum\b", re.IGNORECASE), "periosteum"),
    (re.compile(r"\bberiosteal\b", re.IGNORECASE), "periosteal"),
    (re.compile(r"\bberiarticular\b", re.IGNORECASE), "periarticular"),
    (re.compile(r"\bbulmonary\b", re.IGNORECASE), "pulmonary"),
    (re.compile(r"\bbleural\b", re.IGNORECASE), "pleural"),
    (re.compile(r"\bbathological\b", re.IGNORECASE), "pathological"),
    (re.compile(r"\bbathology\b", re.IGNORECASE), "pathology"),

    # ── v → f confusion ──
    (re.compile(r"\bfertebra\b", re.IGNORECASE), "vertebra"),
    (re.compile(r"\bfertebral\b", re.IGNORECASE), "vertebral"),
    (re.compile(r"\bfein\b", re.IGNORECASE), "vein"),
    (re.compile(r"\bfeins\b", re.IGNORECASE), "veins"),
    (re.compile(r"\bfalfe\b", re.IGNORECASE), "valve"),
    (re.compile(r"\bfascular\b", re.IGNORECASE), "vascular"),

    # ── th → s/z confusion ──
    (re.compile(r"\bartritis\b", re.IGNORECASE), "arthritis"),
    (re.compile(r"\bsickness\b(?=\s+(?:of|tear|partial))", re.IGNORECASE), "thickness"),

    # Also include SA v↔w for Urdu speakers
    (re.compile(r"\bwertebra\b", re.IGNORECASE), "vertebra"),
    (re.compile(r"\bwertebral\b", re.IGNORECASE), "vertebral"),
    (re.compile(r"\bwalve\b", re.IGNORECASE), "valve"),
    (re.compile(r"\bwascular\b", re.IGNORECASE), "vascular"),
]

# ---------------------------------------------------------------------------
# East Asian (Chinese / Japanese / Korean-influenced English)
#
# Primary: /l/ ↔ /r/ confusion, final consonant cluster simplification
# ---------------------------------------------------------------------------

_EAST_ASIAN: _PatternList = [
    # ── l ↔ r confusion ──
    (re.compile(r"\brigament\b", re.IGNORECASE), "ligament"),
    (re.compile(r"\brigaments\b", re.IGNORECASE), "ligaments"),
    (re.compile(r"\brighament\b", re.IGNORECASE), "ligament"),
    (re.compile(r"\brateral\b", re.IGNORECASE), "lateral"),
    (re.compile(r"\brabrum\b", re.IGNORECASE), "labrum"),
    (re.compile(r"\brabral\b", re.IGNORECASE), "labral"),
    (re.compile(r"\bretinacurum\b", re.IGNORECASE), "retinaculum"),

    # ── Consonant cluster simplification ──
    (re.compile(r"\bfracchure\b", re.IGNORECASE), "fracture"),
    (re.compile(r"\bfraccture\b", re.IGNORECASE), "fracture"),
    (re.compile(r"\bstruccture\b", re.IGNORECASE), "structure"),
    (re.compile(r"\bruptchure\b", re.IGNORECASE), "rupture"),

    # ── Vowel insertions ──
    (re.compile(r"\bsupinatus\b", re.IGNORECASE), "supraspinatus"),
]

# ---------------------------------------------------------------------------
# West African (Nigerian / Ghanaian English)
#
# Generally well-handled by Whisper; a few specific patterns
# ---------------------------------------------------------------------------

_WEST_AFRICAN: _PatternList = [
    # ── th → t/d confusion ──
    (re.compile(r"\btickness\b", re.IGNORECASE), "thickness"),
    (re.compile(r"\btoracic\b", re.IGNORECASE), "thoracic"),
    (re.compile(r"\bartritis\b", re.IGNORECASE), "arthritis"),

    # ── Vowel shift ──
    (re.compile(r"\btandon\b", re.IGNORECASE), "tendon"),
    (re.compile(r"\btandons\b", re.IGNORECASE), "tendons"),
    (re.compile(r"\blegament\b", re.IGNORECASE), "ligament"),
]


# ---------------------------------------------------------------------------
# Registry – maps accent key → pattern list
# ---------------------------------------------------------------------------

ACCENT_PROFILES: Dict[str, _PatternList] = {
    "neutral": [],              # No extra corrections
    "south_asian": _SOUTH_ASIAN,
    "middle_eastern": _MIDDLE_EASTERN,
    "east_asian": _EAST_ASIAN,
    "west_african": _WEST_AFRICAN,
}

# Display labels for the UI combo box (order matters – shown in this order)
ACCENT_LABELS: Dict[str, str] = {
    "neutral": "Neutral",
    "south_asian": "South Asian",
    "middle_eastern": "Middle Eastern",
    "east_asian": "East Asian",
    "west_african": "West African",
}


# ---------------------------------------------------------------------------
# Quick-scan patterns for early exit optimization
# These are simple string fragments that indicate potential matches exist.
# If none are found, we can skip the more expensive regex processing.
# ---------------------------------------------------------------------------

_QUICK_SCAN_MARKERS: Dict[str, List[str]] = {
    "south_asian": [
        "wertebr", "wein", "walv", "wascular", "wolar", "wisuali",
        "torac", "torassic", "torax", "tickness", "artritis", "artrop",
        "artrosc", "artroplast", "raptur", "tandon", "tandino", "legament",
        "ligamant", "normel", "frachur", "fraccher", "harniat", "protusion",
        "fusion", "arosion", "pattell", "acitabul", "subchondrel", "osteofyte",
        "kalsific", "tendonitis", "menesci", "menescus", "menescal",
        "supraspinatous", "infraspinatous", "disk",
    ],
    "middle_eastern": [
        "bneumo", "balmar", "bosterior", "broximal", "beroneal", "batell",
        "blanter", "blantar", "beriost", "beriarticular", "bulmonary",
        "bleural", "batholog", "fertebr", "fein", "falfe", "fascular",
        "sickness", "wertebr", "walv", "wascular",
    ],
    "east_asian": [
        "rigament", "righament", "rateral", "rabrum", "rabral", "retinacur",
        "fracchure", "fraccture", "struccture", "ruptchure", "supinatus",
    ],
    "west_african": [
        "tickness", "toracic", "artritis", "tandon", "legament",
    ],
}


def _has_potential_matches(text_lower: str, accent: str) -> bool:
    """Quick check if text might contain any matchable patterns for this accent."""
    if markers := _QUICK_SCAN_MARKERS.get(accent, []):
        return any(marker in text_lower for marker in markers)
    else:
        return False


def apply_accent_corrections(text: str, accent: str) -> str:
    """Apply accent-specific Whisper error corrections.

    Parameters
    ----------
    text : str
        Transcript text (already through base MSK corrections).
    accent : str
        Key from ACCENT_PROFILES (e.g. "south_asian").  Falls back to
        "neutral" (no-op) for unknown keys.

    Returns
    -------
    str
        Corrected text.
    """
    if accent == "neutral" or accent not in ACCENT_PROFILES:
        return text

    # Quick scan optimization: skip regex processing if no potential matches
    text_lower = text.lower()
    if not _has_potential_matches(text_lower, accent):
        return text

    patterns = ACCENT_PROFILES.get(accent, [])
    for pattern, repl in patterns:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# Accent suggestion based on detected error patterns
# ---------------------------------------------------------------------------

# Characteristic error signatures for each accent (patterns → accent)
_ACCENT_SIGNATURES: Dict[str, List[str]] = {
    "south_asian": [
        r"\bwertebr", r"\bwein\b", r"\bwalve", r"\btandon",
        r"\blegament", r"\btoracic\b", r"\bfrachure",
    ],
    "middle_eastern": [
        r"\bbosterior\b", r"\bbroximal\b", r"\bberoneal\b",
        r"\bbalmar\b", r"\bblanter\b", r"\bbulmonary\b",
    ],
    "east_asian": [
        r"\brigament\b", r"\brateral\b", r"\brabrum\b",
        r"\bfracchure\b", r"\bruptchure\b",
    ],
    "west_african": [
        r"\btickness\b", r"\btandon\b",
    ],
}

# Cache for compiled signature patterns
_SIGNATURE_PATTERNS: Dict[str, List[Pattern]] = {}


def _get_signature_patterns(accent: str) -> List[Pattern]:
    """Get compiled signature patterns for an accent (cached)."""
    if accent not in _SIGNATURE_PATTERNS:
        raw = _ACCENT_SIGNATURES.get(accent, [])
        _SIGNATURE_PATTERNS[accent] = [re.compile(p, re.IGNORECASE) for p in raw]
    return _SIGNATURE_PATTERNS[accent]


def suggest_accent(text: str, min_matches: int = 2) -> Optional[str]:
    """Analyze text for accent-specific error patterns and suggest a profile.

    Parameters
    ----------
    text : str
        Sample text (ideally several sentences) to analyze.
    min_matches : int
        Minimum number of characteristic errors required to suggest an accent.

    Returns
    -------
    str or None
        Suggested accent key (e.g. "south_asian"), or None if no clear match.
    """
    if not text:
        return None

    text_lower = text.lower()
    best_accent: Optional[str] = None
    best_count = 0

    for accent in _ACCENT_SIGNATURES:
        patterns = _get_signature_patterns(accent)
        count = sum(bool(p.search(text_lower))
                for p in patterns)
        if count >= min_matches and count > best_count:
            best_count = count
            best_accent = accent

    return best_accent


def get_available_accents() -> List[str]:
    """Return list of available accent profile keys."""
    return list(ACCENT_PROFILES.keys())
