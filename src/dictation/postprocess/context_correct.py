"""Stage 7.5 — context-aware real-word correction.

Fixes the error class the fuzzy dictionary stage structurally cannot: a
correctly-spelled word that is nonetheless the *wrong* word, distinguishable
only from context. Whisper hears "spinal chord" for "spinal cord", "course"
for "coarse", "ileum" for "ilium" — nothing is misspelled, so no spell-checker
fires, but the surrounding words make the intended word obvious.

Two bounded, high-precision operations, both driven by data files (not code) and
both safe by construction — the worst case is a *visible* wrong pick within a
hand-curated pair, never a free rewrite:

* :func:`apply_context_correction` — for each word that belongs to a confusion
  set (``confusion_sets.yaml``), score every member of that set in the current
  context using the offline n-gram model (:mod:`context_model`) plus curated cue
  words and priors, and switch to the best-scoring member **only** when it beats
  the spoken word by a clear margin. It can only ever move *between members of
  the same set* — so an unlisted word is untouchable, and a listed word can only
  become one of its known confusables.

* :func:`rejoin_split_compounds` — rejoin a medical term Whisper split across two
  tokens ("hydro nephrosis" -> "hydronephrosis") when, and only when, the joined
  form is a known medical term and the two pieces are not both ordinary English
  words. Runs *before* the fuzzy stage so those fragments become a real term
  instead of being mangled by a per-fragment fuzzy match.

Every change made here surfaces in the corrections banner (the pipeline diffs
input vs output), so the radiologist always sees — and can reject — a real-word
substitution. Skipped entirely in "soft" cleanup mode.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from src.dictation.postprocess import context_model
from src.features.file_manager import confusion_sets_path

logger = logging.getLogger(__name__)

# A word token: a letter followed by letters / internal apostrophes / hyphens.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

# ── Calibration knobs (physical thresholds — keep here, not inlined) ──────────
# How many tokens on each side count as "nearby" for cue detection.
_CUE_WINDOW = 4
# Score added per matched cue word — deliberately large so an explicit curated
# cue ("spinal" near "cord") decisively outweighs weak n-gram noise.
_CUE_BONUS = 4.0
# Minimum score advantage the best candidate must have over the *spoken* word
# before we override it. High = precise (only correct when the evidence is
# strong). This is the master safety dial: raise it to correct less / more
# safely, lower it to correct more aggressively.
_SWITCH_MARGIN = 1.5
# A member this disfavoured (or worse) is treated as *virtually never correct*
# in radiology ("legion", "chord"): the prior alone may trigger a switch off it.
# Any member with a milder prior is a word that CAN be correct ("know",
# "discreet", "course"), so switching AWAY from it as spoken requires positive
# context evidence (a cue match or a corpus-observed word pair) — never the
# prior alone. This is what stops "the clinician should know" -> "...should no".
_NEVER_CORRECT_PRIOR = -8.0


class _ConfusionSets:
    """Parsed confusion-set data: which set each word belongs to, plus its
    per-member cues and priors. Built once from ``confusion_sets.yaml``."""

    def __init__(self, sets: List[dict]):
        self._word_to_set: Dict[str, dict] = {}
        for s in sets:
            raw = s.get("members", [])
            # Fail loud on the YAML boolean trap: an unquoted "no"/"yes"/"on"/
            # "off" parses as a Python bool, which would silently become the
            # member "false"/"true" and mis-correct real words. A member must be
            # a string — skip and warn rather than corrupt a clinical correction.
            if any(not isinstance(m, str) for m in raw):
                logger.warning(
                    "Confusion set %r has a non-string member (likely an "
                    "unquoted YAML boolean like `no`) — quote it in "
                    "confusion_sets.yaml; skipping this set.", raw,
                )
                continue
            members = [m.lower() for m in raw]
            if len(members) < 2:
                continue
            entry = {
                "members": members,
                "cues": {
                    str(k).lower(): [str(w).lower() for w in v]
                    for k, v in (s.get("cues") or {}).items()
                },
                "prior": {
                    str(k).lower(): float(v)
                    for k, v in (s.get("prior") or {}).items()
                },
            }
            for m in members:
                # First set wins if a word is (mis)listed twice — flag it loudly.
                if m in self._word_to_set:
                    logger.warning("Word %r appears in multiple confusion sets", m)
                    continue
                self._word_to_set[m] = entry

    def set_for(self, word: str) -> Optional[dict]:
        return self._word_to_set.get(word)


_SETS: Optional[_ConfusionSets] = None
_PREFIXES: Optional[frozenset] = None


def _load_config() -> Tuple[_ConfusionSets, frozenset]:
    """Parse ``confusion_sets.yaml`` once into (confusion sets, compound prefixes).

    Cached for the process. A missing or malformed file disables the feature
    (empty sets, empty prefixes) rather than raising — correction is an
    enhancement, never a hard dependency of the pipeline.
    """
    global _SETS, _PREFIXES
    if _SETS is not None and _PREFIXES is not None:
        return _SETS, _PREFIXES
    sets: List[dict] = []
    prefixes: List[str] = []
    try:
        import yaml  # noqa: PLC0415 — optional at import, needed only here
        with open(confusion_sets_path(), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        sets = data.get("sets") or []
        prefixes = data.get("compound_prefixes") or []
    except FileNotFoundError:
        logger.warning("confusion_sets.yaml missing — context correction disabled")
    except Exception as exc:
        logger.warning("Could not load confusion sets: %s — context correction off", exc)
    _SETS = _ConfusionSets(sets)
    _PREFIXES = frozenset(str(p).lower() for p in prefixes if isinstance(p, str))
    return _SETS, _PREFIXES


def _cased(original: str, replacement: str) -> str:
    """Re-apply *original*'s capitalisation to *replacement* (lowercase)."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_context_correction(text: str) -> str:
    """Swap confusable real words for the member that best fits the context.

    Only words listed in a confusion set are ever candidates, and each can only
    be replaced by another member of the *same* set, and only when the evidence
    clearly points away from the spoken word. Everything else passes through
    verbatim.

    Never raises: this stage runs on every live chunk, and a correction failure
    must degrade to "leave the text as-is", never break dictation. Any unexpected
    error returns the input unchanged.
    """
    try:
        return _apply_context_correction(text)
    except Exception as exc:  # correction is an enhancement, never load-bearing
        logger.warning("Context correction skipped (returning text as-is): %s", exc)
        return text


def _apply_context_correction(text: str) -> str:
    sets, _ = _load_config()
    if not sets._word_to_set:
        return text

    tokens = list(_WORD_RE.finditer(text))
    if not tokens:
        return text
    lowers = [m.group(0).lower() for m in tokens]
    ctx = context_model.get_context_model()

    # (start, end, replacement) edits, applied right-to-left so spans stay valid.
    edits: List[Tuple[int, int, str]] = []

    for i, tok in enumerate(tokens):
        spoken = lowers[i]
        entry = sets.set_for(spoken)
        if entry is None:
            continue
        left = lowers[i - 1] if i > 0 else None
        right = lowers[i + 1] if i + 1 < len(lowers) else None

        # Cue words within the ±window, and which members they support.
        window = set(lowers[max(0, i - _CUE_WINDOW): i]) | set(
            lowers[i + 1: i + 1 + _CUE_WINDOW]
        )
        cued = {
            m for m in entry["members"]
            if any(c in window for c in entry["cues"].get(m, ()))
        }

        # SAFETY RULE (precision >> recall): only replace the spoken word when
        # the evidence clearly points AWAY from it. Two trustworthy triggers:
        #
        #  (a) An exclusive cue for a *different* member is present AND the
        #      spoken word has NO cue of its own. "spinal chord" -> the cord cue
        #      "spinal" fires and "chord" has none. But "peroneal tendon" is left
        #      alone because "tendon" cues the spoken word itself, and "perineal
        #      region" is left alone because neither has an exclusive cue.
        #  (b) The spoken word is one we treat as virtually never correct in
        #      radiology (very negative prior — "legion", "chord"): replace it
        #      with the best-scoring alternative even without a cue.
        #
        # The sparse n-gram is deliberately NOT a standalone trigger — it only
        # ranks candidates once a trustworthy trigger has fired. This is what
        # prevents a thin corpus + a stray cue from flipping a correct word.
        if spoken in cued:
            continue  # the spoken word is itself supported — never touch it

        spoken_prior = entry["prior"].get(spoken, 0.0)
        candidates = cued - {spoken}
        if not candidates:
            if spoken_prior > _NEVER_CORRECT_PRIOR:
                continue  # no cue for any alternative and spoken can be correct
            candidates = {m for m in entry["members"] if m != spoken}

        # Score = context fit + prior + an exclusive-cue bonus. The cue bonus is
        # only ever added to a candidate here (the spoken word was excluded above
        # if it was cued), so it can only push a switch that the guards already
        # deemed trustworthy — it never flips a self-supported word.
        def _score(m: str) -> float:
            s = ctx.score(left, m, right) + entry["prior"].get(m, 0.0)
            return s + _CUE_BONUS if m in cued else s

        best_word = max(candidates, key=_score)
        if _score(best_word) - _score(spoken) < _SWITCH_MARGIN:
            continue
        edits.append((tok.start(), tok.end(), _cased(tok.group(0), best_word)))

    if not edits:
        return text
    out = text
    for start, end, repl in reversed(edits):
        out = out[:start] + repl + out[end:]
    return out


# ---------------------------------------------------------------------------
# Split-compound rejoin
# ---------------------------------------------------------------------------

def rejoin_split_compounds(text: str) -> str:
    """Rejoin a medical term Whisper split across two tokens.

    "hydro nephrosis" -> "hydronephrosis", "retro peritoneal" ->
    "retroperitoneal". Conservative: only joins when the concatenation is a known
    medical term AND (the first token is a medical combining form OR the two
    pieces are not both ordinary English words) — so "the rapist" is never joined
    into "therapist". Runs before the fuzzy stage so a real term is reconstructed
    rather than each fragment being fuzzy-matched (which previously corrupted
    "spleno" -> "seleno").

    Never raises — like :func:`apply_context_correction`, a failure degrades to
    leaving the text unchanged rather than breaking the live pipeline.
    """
    try:
        return _rejoin_split_compounds(text)
    except Exception as exc:
        logger.warning("Split-compound rejoin skipped (returning text as-is): %s", exc)
        return text


def _rejoin_split_compounds(text: str) -> str:
    from src.medical import medical_dict  # noqa: PLC0415 — avoid import cycle at module load
    from src.dictation.postprocess.medical_dict_match import _english_known  # noqa: PLC0415

    terms = medical_dict.get_medical_terms()
    if not terms:
        return text
    _, prefixes = _load_config()

    def _one_pass(s: str) -> str:
        tokens = list(_WORD_RE.finditer(s))
        edits: List[Tuple[int, int, str]] = []
        i = 0
        while i < len(tokens) - 1:
            a, b = tokens[i], tokens[i + 1]
            al, bl = a.group(0).lower(), b.group(0).lower()
            joined = al + bl
            # Only "word<space>word" is joinable — a hyphen means the author
            # already chose a compound form. The concatenation being a real
            # medical term is the strong guard; the extra check just refuses to
            # weld two ordinary English words ("the rapist" -> "therapist",
            # "no evidence" -> "noevidence" is already excluded by the term
            # check, but "to day" -> "today" would be, hence the guard).
            gap = s[a.end(): b.start()]
            if (
                gap == " "
                and len(al) >= 3
                and len(bl) >= 3
                and joined in terms
                # Justify the weld: either the first token is a medical combining
                # form ("retro", "hydro" — a compound was clearly split), or at
                # least one piece isn't an ordinary English word. This blocks
                # welding two plain words ("the rapist") whose concatenation only
                # coincidentally lands in the term list.
                and (al in prefixes or not (_english_known(al) and _english_known(bl)))
            ):
                edits.append((a.start(), b.end(), _cased(a.group(0), joined)))
                i += 2  # consumed both tokens
                continue
            i += 1
        if not edits:
            return s
        for start, end, repl in reversed(edits):
            s = s[:start] + repl + s[end:]
        return s

    # Iterate to a fixed point so a term split into three tokens
    # ("hepato spleno megaly") joins fully: pass 1 -> "hepato splenomegaly",
    # pass 2 -> "hepatosplenomegaly". Bounded to avoid any pathological loop.
    out = text
    for _ in range(4):
        joined = _one_pass(out)
        if joined == out:
            break
        out = joined
    return out
