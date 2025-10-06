import re
from typing import Dict


# Priority replacements applied case-insensitively with word boundaries
MEDICAL_REPLACEMENTS: Dict[str, str] = {
    r"\bpneumothorax\b": "pneumothorax",
    r"\bconsolidations?\b": "consolidation",
    r"\bcardiomegaly\b": "cardiomegaly",
    r"\batelectasis\b": "atelectasis",
    r"\bpleural effusions?\b": "pleural effusion",
    r"\binterstitial\b": "interstitial",
    r"\bheamorrhage\b": "haemorrhage",      # UK variant
    r"\bhemorrhage\b": "haemorrhage",        # normalize to UK spelling for NHS style
    r"\boesophagus\b": "oesophagus",
    r"\besophagus\b": "oesophagus",
    r"\bdiaphragm\b": "diaphragm",
    r"\bhepatomegaly\b": "hepatomegaly",
    r"\bsplenomegaly\b": "splenomegaly",
    r"\bcalcifications?\b": "calcification",
    r"\bosteophytes?\b": "osteophyte",
    r"\bcxr\b": "chest radiograph",
    r"\bct\b": "CT",
    r"\bmri\b": "MRI",
    r"\busg\b": "ultrasound",
}


def smart_capitalize(text: str) -> str:
    # Capitalize sentence starts while keeping acronyms (CT, MRI) intact
    def cap_sentence(s: str) -> str:
        s = s.strip()
        if not s:
            return s
        # If starts with acronym, keep it
        if re.match(r"^(CT|MRI|US|CXR)\b", s):
            return s[0:]
        return s[0:1].upper() + s[1:]

    parts = re.split(r"([.!?]+\s+)", text)
    out = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        out.append(cap_sentence(sentence))
        out.append(sep)
    return "".join(out)


def normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def apply_medical_corrections(text: str) -> str:
    out = text
    for pattern, repl in MEDICAL_REPLACEMENTS.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def postprocess_transcript(text: str) -> str:
    text = normalize_spaces(text)
    text = apply_medical_corrections(text)
    text = smart_capitalize(text)
    return text
