"""Stage 5 — radiology terminology corrections.

A single ordered list of regex substitutions runs in order, with longer
and more-specific patterns earlier so they win over shorter generic ones.
Sections in this file are presentational only — at runtime the table is
flat.

Three legacy stages (MRI signal, MSK anatomy, general radiology) have
been merged here.  The order within the table preserves their previous
relative precedence.
"""

from __future__ import annotations

import re
from typing import Callable, List, Pattern, Tuple, Union

# (compiled_pattern, replacement)
_Rule = Tuple[Pattern, Union[str, Callable[[re.Match], str]]]


def _ci(p: str) -> Pattern:
    """Shorthand: case-insensitive compiled regex."""
    return re.compile(p, re.IGNORECASE)


def _cs(p: str) -> Pattern:
    """Shorthand: case-sensitive compiled regex (for true acronyms)."""
    return re.compile(p)


_RULES: List[_Rule] = [
    # ------------------------------------------------------------------
    # MRI sequence names with stray spaces (T 1 → T1, P D → PD).
    # ------------------------------------------------------------------
    (_cs(r"\bT\s+1\b"), "T1"),
    (_cs(r"\bT\s+2\b"), "T2"),
    (_ci(r"\bP\s+D\b"), "PD"),
    (_ci(r"\bS\s*T\s*I\s*R\b"), "STIR"),
    (_ci(r"\bP\s*D\s*F\s*S\b"), "PDFS"),
    (_ci(r"\bF\s*L\s*A\s*I\s*R\b"), "FLAIR"),
    (_ci(r"\bD\s*W\s*I\b"), "DWI"),
    (_ci(r"\bA\s*D\s*C\b"), "ADC"),
    (_ci(r"\bT1[- ]?weight(?:ed)?\b"), "T1-weighted"),
    (_ci(r"\bT2[- ]?weight(?:ed)?\b"), "T2-weighted"),
    (_ci(r"\bproton[- ]?density[- ]?weight(?:ed)?\b"), "proton density-weighted"),

    # ------------------------------------------------------------------
    # Signal intensity / echogenicity compound terms.
    # ------------------------------------------------------------------
    (_ci(r"\bhyper[- ]?intens(?:e|ity)\b"), "hyperintense"),
    (_ci(r"\bhypo[- ]?intens(?:e|ity)\b"), "hypointense"),
    (_ci(r"\biso[- ]?intens(?:e|ity)\b"), "isointense"),
    (_ci(r"\bhyper[- ]?echo(?:ic|genicity)\b"), "hyperechoic"),
    (_ci(r"\bhypo[- ]?echo(?:ic|genicity)\b"), "hypoechoic"),
    (_ci(r"\biso[- ]?echo(?:ic|genicity)\b"), "isoechoic"),
    (_ci(r"\ban[- ]?echo(?:ic|genicity)\b"), "anechoic"),
    (_ci(r"\becho[- ]?texture\b"), "echotexture"),
    (_ci(r"\becho[- ]?genic(?:ity)?\b"), "echogenicity"),
    (_ci(r"\bhomo[- ]?geneous\b"), "homogeneous"),
    (_ci(r"\bhetero[- ]?geneous\b"), "heterogeneous"),
    (_ci(r"\bbone\s+marrow\s+o[ea]dema\b"), "bone marrow oedema"),
    (_ci(r"\bmarrow\s+o[ea]dema\b"), "bone marrow oedema"),
    (_cs(r"\bBMO\b"), "bone marrow oedema"),
    (_ci(r"\bsub[- ]?chondral\b"), "subchondral"),
    (_ci(r"\bsub[- ]?articular\b"), "subarticular"),
    (_ci(r"\bsub[- ]?acute\b"), "subacute"),
    (_ci(r"\bfat[- ]?sat(?:urated|uration)?\b"), "fat-saturated"),
    (_ci(r"\bfat\s+suppressed\b"), "fat-suppressed"),
    (_ci(r"\bpost[- ]?contrast\b"), "post-contrast"),
    (_ci(r"\bpre[- ]?contrast\b"), "pre-contrast"),
    (_ci(r"\bnon[- ]?enhancing\b"), "non-enhancing"),

    # ------------------------------------------------------------------
    # MSK Whisper-mishear corrections — compound forms BEFORE bare "tier".
    # ------------------------------------------------------------------
    (_ci(r"\broot[\s-]?tier\b"), "rotator"),
    (_ci(r"\broot[\s-]?a[\s-]?tier\b"), "rotator"),
    (_ci(r"\bpost[\s-]?tier\b"), "posterior"),
    (_ci(r"\bant[\s-]?tier\b"), "anterior"),
    (_ci(r"\btier(?=\s|[.,;:!?\n]|$)"), "tear"),
    (_ci(r"\btiers\b"), "tears"),

    # "median" → "medial" only in true anatomical contexts.
    (
        _ci(
            r"\bmedian\s+(?=meniscus|meniscal|collateral|compartment|condyle"
            r"|epicondyle|ligament|malleolus|tibial|femoral|retinaculum)"
        ),
        "medial ",
    ),
    # "crucial ligament" → "cruciate ligament" (context-aware).
    (_ci(r"\bcrucial\s+(?=ligament)"), "cruciate "),
    (_ci(r"\bcollar\s*all\b"), "collateral"),
    (_ci(r"\bin[- ]tact\b"), "intact"),
    (_ci(r"\bsupra[- ]?spinatus\b"), "supraspinatus"),
    (_ci(r"\binfra[- ]?spinatus\b"), "infraspinatus"),
    (_ci(r"\bsub[- ]?scapularis\b"), "subscapularis"),
    (_ci(r"\ba\s*(?:killes|killies|achilles)\b"), "Achilles"),
    (_ci(r"\bachilles\b"), "Achilles"),
    (_ci(r"\blab\s*rum\b"), "labrum"),
    (_ci(r"\blabram\b"), "labrum"),
    (_ci(r"\bin\s+the\s+substance\b"), "intrasubstance"),
    (_ci(r"\binter[- ]?substance\b"), "intrasubstance"),
    (_ci(r"\bintra[- ]?substance\b"), "intrasubstance"),
    (_ci(r"\ben(?:d\s+of|tho|theso)\s*(?:sopathy|pathy)\b"), "enthesopathy"),
    (_ci(r"\bplanter\b"), "plantar"),
    (_ci(r"\bfash[ie]a?\s*itis\b"), "fasciitis"),
    (_ci(r"\bbursar\b"), "bursa"),
    (_ci(r"\bo\s*steo\s*phyte\b"), "osteophyte"),
    (_ci(r"\bo\s*steo\s*phytes\b"), "osteophytes"),
    (_ci(r"\blig\s+ament\b"), "ligament"),
    (_ci(r"\btend\s+on\b"), "tendon"),
    (_ci(r"\bmen\s*iscus\b"), "meniscus"),
    (_ci(r"\bmen\s*isci\b"), "menisci"),
    (_ci(r"\bmen\s*iscal\b"), "meniscal"),
    (_ci(r"\bhyper[- ]?extension\b"), "hyperextension"),
    (_ci(r"\bhyper[- ]?flexion\b"), "hyperflexion"),
    (_ci(r"\bhyper[- ]?mobility\b"), "hypermobility"),
    (
        _ci(r"\blose\s+bod(?:y|ies)\b"),
        lambda m: "loose bodies" if "ies" in m.group().lower() else "loose body",
    ),
    (_ci(r"\bretro[- ]?calcaneal\b"), "retrocalcaneal"),
    (_ci(r"\bperi[- ]?articular\b"), "periarticular"),
    (_ci(r"\bintra[- ]?articular\b"), "intra-articular"),
    (_ci(r"\bextra[- ]?articular\b"), "extra-articular"),
    (_ci(r"\bjuxta[- ]?articular\b"), "juxta-articular"),
    (_ci(r"\bsub[- ]?periosteal\b"), "subperiosteal"),
    (_ci(r"\btrans[- ]?chondral\b"), "transchondral"),
    (_ci(r"\bosteo[- ]?chondral\b"), "osteochondral"),
    (_ci(r"\bosteo[- ]?arthritis\b"), "osteoarthritis"),
    (_ci(r"\bteno[- ]?synovitis\b"), "tenosynovitis"),
    (_ci(r"\bspondylo[- ]?listhesis\b"), "spondylolisthesis"),
    (_ci(r"\bspondylo[- ]?lysis\b"), "spondylolysis"),
    (_ci(r"\bspondylo[- ]?sis\b"), "spondylosis"),
    (_ci(r"\bspondylo[- ]?arthropathy\b"), "spondyloarthropathy"),
    (_ci(r"\bfull[- ]?thickness\b"), "full-thickness"),
    (_ci(r"\bpartial[- ]?thickness\b"), "partial-thickness"),
    (_ci(r"\bhigh[- ]?grade\b"), "high-grade"),
    (_ci(r"\blow[- ]?grade\b"), "low-grade"),

    # Grading systems (preserve canonical capitalisation).
    (_ci(r"\bkellgren[- ]?lawrence\b"), "Kellgren-Lawrence"),
    (_ci(r"\bouterbridge\b"), "Outerbridge"),
    (_ci(r"\bARCO\b"), "ARCO"),
    (_ci(r"\bSPARCC\b"), "SPARCC"),
    (_ci(r"\bmodic\b"), "Modic"),

    # Spine disc / joint compound terms.
    (_ci(r"\bneuro[- ]?foraminal\b"), "neuroforaminal"),
    (_ci(r"\bpara[- ]?central\b"), "paracentral"),

    # Fracture descriptors.
    (_ci(r"\bcommin(?:uted|ution)\b"), "comminuted"),
    (_ci(r"\bnon[- ]?displaced\b"), "non-displaced"),

    # Modern tendinopathy term.
    (_ci(r"\btendinitis\b"), "tendinopathy"),

    # UK spelling preferences (apply before fuzzy dictionary).
    (_ci(r"\badema\b"), "oedema"),
    (_ci(r"\b(?:oe|e)dema\b"), "oedema"),
    (_ci(r"\banaemia\b"), "anaemia"),
    (_ci(r"\bh[ae]emorrhage\b"), "haemorrhage"),
    (_ci(r"\borthop(?:ae|e)dic\b"), "orthopaedic"),

    # ------------------------------------------------------------------
    # General radiology / modality / lab terms.
    # ------------------------------------------------------------------
    (_ci(r"\bcxr\b"), "chest radiograph"),
    (_ci(r"\bct\b"), "CT"),
    (_ci(r"\bmri\b"), "MRI"),
    (_ci(r"\busg\b"), "ultrasound"),
    (_ci(r"\bxray\b"), "X-ray"),
    (_ci(r"\bx-ray\b"), "X-ray"),
    (_ci(r"\bdexa\b"), "DEXA"),
    (_ci(r"\bpet[- ]?ct\b"), "PET-CT"),
    (_ci(r"\batylactases\b"), "amylases"),
    (_ci(r"\batylactase\b"), "amylase"),

    # UK plurals before singulars (avoids partial overlap).
    (_ci(r"\btumou?rs\b"), "tumours"),
    (_ci(r"\btumou?r\b"), "tumour"),
    (_ci(r"\bh[ae]matomas\b"), "haematomas"),
    (_ci(r"\bh[ae]matoma\b"), "haematoma"),
    (_ci(r"\bh[ae]maturia\b"), "haematuria"),
    (_ci(r"\bp[ae]diatrics\b"), "paediatrics"),
    (_ci(r"\bp[ae]diatric\b"), "paediatric"),
]


def apply_terminology(text: str) -> str:
    """Apply MRI signal, MSK, and general radiology corrections in order."""
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)  # type: ignore[arg-type]
    return text
