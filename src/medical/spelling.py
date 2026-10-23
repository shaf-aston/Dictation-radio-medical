"""British and US spelling: one place that knows how they differ.

Two callers:

* the pipeline stage ``apply_uk_spelling``, which writes British spelling;
* :func:`src.medical.medical_dict.is_english_word`, which must accept a British
  spelling as a real word. Its word list is American, so before this the fuzzy
  corrector rewrote "characterise" to "characters" and the word marks would
  have underlined every British spelling as suspect.

British spelling for general English words the recogniser writes the US way.

Parakeet was trained mostly on US English and writes "characterize",
"caliber", "favoring". Measured 2026-10-23 on the piper set: each of those
across all 5 voices. The medical words (oedema, haemorrhage) are already
handled by ``terminology.py``; this covers the ordinary English around them.

A setting, not a guess: ``spelling`` is "en-GB" (default, the convention of
this project's templates and lexicon) or "en-US", which leaves text alone.

Conservative by construction: a fixed list of whole words, plus the regular
-ize/-yze endings with the few words where "ize" is not a suffix excluded. A
word not listed is never touched.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional

#: US -> UK, whole words, lower case (case is restored on substitution).
_WORDS: Dict[str, str] = {
    "caliber": "calibre", "calibers": "calibres",
    "fiber": "fibre", "fibers": "fibres",
    "center": "centre", "centers": "centres", "centered": "centred",
    "tumor": "tumour", "tumors": "tumours",
    "color": "colour", "colors": "colours",
    "behavior": "behaviour",
    "favor": "favour", "favoring": "favouring", "favored": "favoured", "favors": "favours",
    "gray": "grey",
    "maneuver": "manoeuvre", "maneuvers": "manoeuvres",
    "ischemia": "ischaemia", "ischemic": "ischaemic",
    "anemia": "anaemia",
    "esophagus": "oesophagus", "esophageal": "oesophageal",
    "pediatric": "paediatric", "orthopedic": "orthopaedic",
    "leukemia": "leukaemia",
    "hematoma": "haematoma", "hematomas": "haematomas",
    "hemangioma": "haemangioma", "hemangiomas": "haemangiomas",
    "estrogen": "oestrogen",
    "diarrhea": "diarrhoea", "dyspnea": "dyspnoea", "apnea": "apnoea",
}

#: Words where "ize" is not the -ize suffix, so the rule must not touch them.
_NOT_A_SUFFIX = {"size", "sizes", "sized", "seize", "seized", "prize", "capsize", "baize"}

#: British -> US patterns, for recognising a British spelling as a real word.
_UK_TO_US = (
    (re.compile(r"is(e|es|ed|ing|ation|ations)$"), r"iz\1"),
    (re.compile(r"ys(e|es|ed|ing)$"), r"yz\1"),
    (re.compile(r"our(s|ed|ing|ite|ites)?$"), r"or\1"),
    (re.compile(r"([^aeiou])re(s)?$"), r"\1er\2"),
    (re.compile(r"ae"), "e"),
    (re.compile(r"oe"), "e"),
)

_WORD_RE = re.compile(r"\b(" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
_IZE_RE = re.compile(r"\b([a-z]{3,}i)z(e|es|ed|ing|ation|ations)\b", re.IGNORECASE)
_YZE_RE = re.compile(r"\b([a-z]{3,})yz(e|es|ed|ing)\b", re.IGNORECASE)


def _keep_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def us_variants(word: str) -> List[str]:
    """US spellings *word* could be the British form of (may be empty).

    Rule-based (-ise/-our/-re/ae/oe) plus the reverse of the word list; used
    only to *accept* a word as real, never to rewrite one.
    """
    w = word.lower()
    out = [us for us, uk in _WORDS.items() if uk == w]
    for pattern, repl in _UK_TO_US:
        candidate = pattern.sub(repl, w)
        if candidate != w:
            out.append(candidate)
    return out


@lru_cache(maxsize=1)
def _configured_locale() -> str:
    """The ``spelling`` setting, read once: a change takes effect on restart."""
    from src.core.settings import Settings  # noqa: PLC0415 - keep the stage importable alone
    return str(Settings().get("spelling") or "en-GB")


def apply_uk_spelling(text: str, locale: Optional[str] = None) -> str:
    """*text* with US spellings of the listed words made British; en-US: unchanged."""
    if (locale or _configured_locale()) != "en-GB":
        return text
    text = _WORD_RE.sub(lambda m: _keep_case(m.group(0), _WORDS[m.group(0).lower()]), text)

    def ize(m: re.Match) -> str:
        word = m.group(0)
        if word.lower() in _NOT_A_SUFFIX:
            return word
        return f"{m.group(1)}{'S' if word.isupper() else 's'}{m.group(2)}"

    text = _IZE_RE.sub(ize, text)
    return _YZE_RE.sub(lambda m: f"{m.group(1)}ys{m.group(2)}", text)
