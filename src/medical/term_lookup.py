"""The neighbourhood of a word: what looks like it, and what goes with it.

When the radiologist highlights a term in the report, both front-ends ask this
module for two short lists and show them side by side:

* **similar spelling** — curated radiology terms within a small edit distance of
  the highlighted word, found with the SymSpell index the fuzzy corrector
  already builds (:mod:`src.medical.medical_dict`). Only terms from
  ``src/resources/radiology_lexicon.txt`` are ever offered, for the same reason
  the corrector only snaps to that file: the broad 98k generic wordlist is full
  of chemistry and drug-name junk.
* **related** — terms that travel with it. Two layers, both offline:
  1. *stem families mined from the lexicon itself* — the shared prefix/suffix
     inventory is counted from the file at load time, never hardcoded, so
     ``pneumothorax`` finds ``haemothorax`` and ``pneumomediastinum`` without
     anyone writing that pair down;
  2. *curated relations* (``src/resources/related_terms.json``) for the pairs a
     stem cannot reach — ``pneumothorax`` -> ``chest drain``. That file is the
     place to extend this, and it holds the tuning knobs too.

**Related terms are associative hints for finding a word — not clinical
guidance.** They are not a differential, not advice, and nothing here is ever
applied to the report on its own; the front-ends only offer them, the
radiologist chooses.

Layer rules: pure — no Qt, no HTTP, no settings. It never raises into a caller
either; any internal failure is logged and returns empty lists, because a
lookup panel that cannot answer must not be able to interrupt dictation.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.features.file_manager import related_terms_path
from src.medical.medical_dict import (
    get_correction_targets,
    get_medical_terms,
    get_symspell,
    is_english_word,
)

logger = logging.getLogger(__name__)

#: A word as dictated: letters, plus the hyphen that real terms carry
#: ("ground-glass", "full-thickness").
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")

#: The same shape, but over the report as written — a scan needs the offsets of
#: "Pneumothorax" as much as of "pneumothorax". Lookups stay lowercase.
_WORD_IN_TEXT = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


@dataclass(frozen=True)
class Suggestion:
    """One offered term, with the short reason it was offered.

    ``note`` is what the UI prints under the term — the shared stem
    (``-thorax``) or the curated group's name (``chest trauma``). It exists so a
    suggestion never has to be taken on trust.
    """

    term: str
    note: str


@dataclass(frozen=True)
class Span:
    """One word worth a second look, at its place in the report text.

    ``start``/``end`` index the *original* string, so a front-end can mark the
    word without altering the text it is marking.
    """

    start: int
    end: int
    term: str


@dataclass(frozen=True)
class TermLookup:
    """The answer for one highlighted selection. Either list may be empty."""

    query: str
    #: The single word the lists were actually built from — for a multi-word
    #: selection this is the head word, so the UI can say what it looked up.
    key: str = ""
    similar_spelling: List[Suggestion] = field(default_factory=list)
    related: List[Suggestion] = field(default_factory=list)


_EMPTY = TermLookup(query="")


# ---------------------------------------------------------------------------
# The data behind the two tiers, built once and cached
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Index:
    tuning: Dict[str, int]
    lexicon: Set[str]
    #: (side, morpheme) -> the lexicon terms carrying it. side is "p" or "s".
    families: Dict[Tuple[str, str], Tuple[str, ...]]
    #: term -> the curated groups it belongs to, as (group name, members).
    groups: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]]


_INDEX: Optional[_Index] = None
_LOCK = threading.Lock()

#: Used only if the resource file is missing or unreadable — the module still
#: answers, it just answers from the lexicon alone.
_FALLBACK_TUNING: Dict[str, int] = {
    "max_query_chars": 60,
    "max_similar": 6,
    "max_related": 8,
    "max_curated_before_stems": 4,
    "spelling_edit_distance": 2,
    "min_morpheme_chars": 5,
    "min_remainder_chars": 2,
    "min_family_terms": 2,
    "max_family_terms": 12,
    # Marking (see suspect_terms): short words are noise, and a report with a
    # hundred marks has told the reader nothing.
    "min_suspect_chars": 4,
    "max_marks": 50,
}


def _load_relations() -> Tuple[Dict[str, int], list]:
    """Read the curated relations file. A bad file degrades, never crashes."""
    try:
        raw = json.loads(related_terms_path().read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read related_terms.json (%s) — related terms "
                       "fall back to lexicon stems only", exc)
        return dict(_FALLBACK_TUNING), []

    tuning = dict(_FALLBACK_TUNING)
    for key, value in (raw.get("tuning") or {}).items():
        if isinstance(value, int) and value > 0:
            tuning[key] = value
    return tuning, list(raw.get("groups") or [])


def _build_families(lexicon: Sequence[str], tuning: Dict[str, int]) -> Dict[Tuple[str, str], Tuple[str, ...]]:
    """Count the lexicon's own prefix/suffix inventory.

    Every prefix and suffix long enough to be a morpheme, and leaving enough of
    the word behind to be one, is counted across the whole file. A stem shared
    by two-to-a-dozen terms is a neighbourhood (``-thorax``, ``pneumo-``); one
    shared by fifty is grammar (``-ular``, ``-itis``) and is dropped by the
    family-size ceiling — which is why no list of morphemes is written down
    anywhere in this repo.
    """
    lo = tuning["min_morpheme_chars"]
    rest = tuning["min_remainder_chars"]
    counts: Dict[Tuple[str, str], List[str]] = {}
    for term in lexicon:
        for size in range(lo, len(term) - rest + 1):
            counts.setdefault(("p", term[:size]), []).append(term)
            counts.setdefault(("s", term[-size:]), []).append(term)
    return {
        key: tuple(terms)
        for key, terms in counts.items()
        if tuning["min_family_terms"] <= len(terms) <= tuning["max_family_terms"]
    }


def _build_groups(groups: list) -> Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]]:
    by_term: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {}
    for group in groups:
        name = str(group.get("name", "related")).strip()
        members = tuple(
            str(term).strip().lower()
            for term in (group.get("terms") or [])
            if str(term).strip()
        )
        for term in members:
            by_term.setdefault(term, []).append((name, members))
    return {term: tuple(entries) for term, entries in by_term.items()}


def _build_index() -> _Index:
    tuning, groups = _load_relations()
    lexicon = get_correction_targets()
    return _Index(
        tuning=tuning,
        lexicon=set(lexicon),
        families=_build_families(lexicon, tuning),
        groups=_build_groups(groups),
    )


def _index() -> _Index:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    with _LOCK:
        if _INDEX is None:
            _INDEX = _build_index()
        return _INDEX


def warm() -> None:
    """Build the indexes ahead of the first highlight (see dictation/warmup.py)."""
    _index()


def max_query_chars() -> int:
    """Longest selection worth looking up, so a caller can refuse a novel-sized one.

    The transport asks rather than keeping its own copy: the knob lives in
    ``related_terms.json``, and two copies of a limit is one limit too many.
    """
    return _index().tuning["max_query_chars"]


def reset() -> None:
    """Drop the cached indexes. For tests, and for reloading an edited file."""
    global _INDEX
    with _LOCK:
        _INDEX = None


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def _pick_key(tokens: Sequence[str], lexicon: Set[str]) -> str:
    """The one word a multi-word selection is looked up under.

    The longest word the lexicon knows, because that is the term carrying the
    meaning ("small right pneumothorax" -> "pneumothorax"); failing that, just
    the longest word, so a selection of unknown words still gets spelling help.
    """
    known = [t for t in tokens if t in lexicon]
    return max(known or tokens, key=len)


def _common_prefix(a: str, b: str) -> int:
    size = 0
    for x, y in zip(a, b):
        if x != y:
            break
        size += 1
    return size


def _is_inflection(a: str, b: str) -> bool:
    """True for a plural or an ending swap (bronchus/bronchi, effusion/effusions).

    Those share a long stem, so they would otherwise top the related list while
    telling the radiologist nothing they cannot already see.
    """
    return (
        abs(len(a) - len(b)) <= 3
        and _common_prefix(a, b) >= min(len(a), len(b)) - 1
    )


def _similar_spelling(key: str, index: _Index) -> List[Suggestion]:
    """Curated lexicon terms within the edit-distance ceiling of *key*.

    Runs against the shared SymSpell index (built over the full membership
    wordlist) and then keeps only lexicon terms — the same "correct toward
    radiology, never toward the generic junk" rule the fuzzy corrector follows.
    """
    sym = get_symspell()
    if sym is None:
        return []
    from symspellpy import Verbosity  # noqa: PLC0415 — optional dep, lazy

    hits = sym.lookup(
        key, Verbosity.ALL,
        max_edit_distance=index.tuning["spelling_edit_distance"],
    )
    ranked = sorted(
        (hit for hit in hits if hit.term != key and hit.term in index.lexicon),
        key=lambda hit: (hit.distance, len(hit.term), hit.term),
    )
    return [
        Suggestion(hit.term, f"{hit.distance} letter{'s' if hit.distance > 1 else ''} away")
        for hit in ranked[: index.tuning["max_similar"]]
    ]


def _stem_related(key: str, index: _Index) -> Dict[str, Tuple[Tuple[int, int], str]]:
    """Lexicon terms sharing a mined stem with *key*, scored by how specific it is.

    A longer stem is a stronger claim, and among equal stems a smaller family is
    a closer neighbourhood — so the score is (stem length, -family size).
    """
    lo = index.tuning["min_morpheme_chars"]
    rest = index.tuning["min_remainder_chars"]
    scored: Dict[str, Tuple[Tuple[int, int], str]] = {}
    for size in range(lo, len(key) - rest + 1):
        for side, stem, note in (
            ("p", key[:size], f"{key[:size]}-"),
            ("s", key[-size:], f"-{key[-size:]}"),
        ):
            family = index.families.get((side, stem))
            if not family:
                continue
            score = (size, -len(family))
            for term in family:
                if term == key or _is_inflection(term, key):
                    continue
                if term not in scored or score > scored[term][0]:
                    scored[term] = (score, note)
    return scored


def _curated_related(terms: Sequence[str], index: _Index) -> List[Suggestion]:
    """Members of every curated group any of *terms* belongs to, in file order."""
    out: List[Suggestion] = []
    seen: Set[str] = set(terms)
    for term in terms:
        for name, members in index.groups.get(term, ()):
            for member in members:
                if member not in seen:
                    seen.add(member)
                    out.append(Suggestion(member, name))
    return out


def lookup(text: str) -> TermLookup:
    """The two suggestion lists for a highlighted word or short phrase.

    Never raises: an empty, over-long, or unrecognisable selection — and any
    internal failure — comes back as empty lists.
    """
    try:
        return _lookup(text)
    except Exception as exc:
        logger.warning("Term lookup failed for %r: %s", text[:40], exc, exc_info=True)
        return _EMPTY


def _lookup(text: str) -> TermLookup:
    query = (text or "").strip()
    index = _index()
    if not query or len(query) > index.tuning["max_query_chars"]:
        return TermLookup(query=query[: index.tuning["max_query_chars"]])

    phrase = " ".join(query.lower().split())
    tokens = _WORD.findall(phrase)
    if not tokens:
        return TermLookup(query=query)

    key = _pick_key(tokens, index.lexicon)
    similar = _similar_spelling(key, index)

    # Curated first: a human wrote those down, so they outrank a mined stem.
    # The whole phrase is looked up as well as its words, because "chest drain"
    # is a curated term in its own right while "chest" and "drain" are not.
    asked = tokens if phrase in tokens else [phrase] + tokens
    curated = _curated_related(asked, index)
    stems = _stem_related(key, index)
    mined = [
        Suggestion(term, note)
        for term, (_, note) in sorted(
            stems.items(), key=lambda kv: (-kv[1][0][0], -kv[1][0][1], kv[0]),
        )
    ]
    # A large curated group would otherwise fill the whole list and hide the
    # stem family, which is often the better neighbourhood ("pneumothorax" ->
    # "pneumomediastinum"). Curated leads, stems follow, curated backfills.
    share = index.tuning["max_curated_before_stems"]
    related = curated[:share] + mined + curated[share:]

    # A term already offered as a spelling neighbour is never repeated below it.
    taken = {s.term for s in similar} | set(tokens) | {phrase}
    deduped: List[Suggestion] = []
    for suggestion in related:
        if suggestion.term in taken:
            continue
        taken.add(suggestion.term)
        deduped.append(suggestion)

    return TermLookup(
        query=query,
        key=key,
        similar_spelling=similar,
        related=deduped[: index.tuning["max_related"]],
    )


# ---------------------------------------------------------------------------
# Marking — which words are worth highlighting in the first place
# ---------------------------------------------------------------------------

def suspect_terms(text: str) -> List[Span]:
    """The words in *text* a reader should look at, with their offsets.

    Highlighting a word already answers it (:func:`lookup`); this answers the
    question before it — *which* word. Without it the feature is invisible: you
    have to suspect a word to select it, and the words worth suspecting are
    exactly the ones that read as plausible.

    A word is marked only when **all three** hold:

    * it is not standard English (:func:`~src.medical.medical_dict.is_english_word`)
      — the same guard the fuzzy corrector uses, and the reason "There" is not
      marked merely because "teres" is one edit away;
    * the membership wordlist does not know it either — the "is this already a
      real word? leave it alone" test; and
    * :func:`_similar_spelling` can name at least one curated lexicon term it
      might have been.

    The last is the point. A mark that opens onto an empty popup is a dead end,
    and a reader who hits two of those stops trusting the marks — so an
    unusual-looking word with nothing to offer is left unmarked.

    Never raises: any failure comes back as no marks, because the report must
    render whether or not this can answer.
    """
    try:
        return _suspect_terms(text)
    except Exception as exc:
        logger.warning("Suspect-term scan failed: %s", exc, exc_info=True)
        return []


def _suspect_terms(text: str) -> List[Span]:
    if not text:
        return []
    # Without the English guard every ordinary word ("there", "again") whose
    # neighbourhood happens to hold a medical term would be marked. A report
    # speckled with wrong marks is worse than no marks, so the feature switches
    # itself off rather than degrade — medical_dict warns loudly, once.
    if is_english_word("the") is None:
        return []

    index = _index()
    known = get_medical_terms()
    floor = index.tuning["min_suspect_chars"]
    ceiling = index.tuning["max_marks"]

    spans: List[Span] = []
    # One verdict per distinct word, not per occurrence: a report says
    # "pneumothorax" many times and the answer cannot change within one scan.
    verdicts: Dict[str, bool] = {}
    for match in _WORD_IN_TEXT.finditer(text):
        if len(spans) >= ceiling:
            break
        word = match.group()
        lowered = word.lower()
        if len(lowered) < floor or lowered in known:
            continue
        verdict = verdicts.get(lowered)
        if verdict is None:
            verdict = (
                not is_english_word(lowered)
                and bool(_similar_spelling(lowered, index))
            )
            verdicts[lowered] = verdict
        if verdict:
            spans.append(Span(match.start(), match.end(), word))
    return spans
