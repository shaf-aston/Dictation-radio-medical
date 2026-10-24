"""
Adaptive Learning System for MSK Radiology Dictation.

Lightweight, passive learning that improves accuracy over time without
extra CPU load. All learning happens as a side-effect of normal use.

Design Principles:
1. Zero additional CPU - only runs during post-processing (already happening)
2. Passive collection - learns from user edits, not explicit training
3. Local storage - all data stays on user's machine
4. Incremental - no batch retraining needed
5. Reversible - user can reset all learned data

Learning Sources:
- Explicit corrections: "word correct word newword" voice command
- Manual edits: tracked when user modifies transcribed text
- Frequency: commonly used terms get priority

Data Structure:
corrections.json = {
    "word_corrections": {"wrong": "right", ...},
    "custom_terms": ["userterm1", "userterm2", ...],
    "term_frequency": {"term": count, ...},
    "accent_hints": {"pattern": "correction", ...}
}
"""

import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.core.json_store import read_json, update_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LEARNING_FILE = "learned_corrections.json"
_MAX_CORRECTIONS = 5000        # Limit stored corrections
_MAX_CUSTOM_TERMS = 2000       # Limit custom terms
_MAX_FREQUENCY_TERMS = 10000   # Limit frequency tracking
_MIN_WORD_LENGTH = 3           # Don't learn very short words
_SAVE_DEBOUNCE_SEC = 30        # Batch saves to reduce I/O
_PUNCTUATION = ".,;:!?()\"'"   # stripped before comparing two words


@dataclass
class _Learned:
    """Everything learned, in memory; the file holds the same four parts."""

    word_corrections: Dict[str, str] = field(default_factory=dict)
    custom_terms: Set[str] = field(default_factory=set)
    term_frequency: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    accent_hints: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data) -> "_Learned":
        data = data if isinstance(data, dict) else {}
        return cls(
            dict(data.get("word_corrections", {})),
            set(data.get("custom_terms", [])),
            defaultdict(int, data.get("term_frequency", {})),
            dict(data.get("accent_hints", {})),
        )

    def to_json(self) -> dict:
        return {
            "word_corrections": dict(self.word_corrections),
            "custom_terms": sorted(self.custom_terms),
            "term_frequency": dict(self.term_frequency),
            "accent_hints": dict(self.accent_hints),
        }

    def add_correction(self, wrong: str, correct: str) -> None:
        if wrong not in self.word_corrections and len(self.word_corrections) >= _MAX_CORRECTIONS:
            del self.word_corrections[next(iter(self.word_corrections))]  # the oldest
        self.word_corrections[wrong] = correct

    def add_term(self, term: str) -> None:
        if term not in self.custom_terms and len(self.custom_terms) >= _MAX_CUSTOM_TERMS:
            logger.warning("Custom terms full, evicted: %s", self.custom_terms.pop())
        self.custom_terms.add(term)

    def count_use(self, term: str) -> None:
        if term not in self.term_frequency and len(self.term_frequency) >= _MAX_FREQUENCY_TERMS:
            del self.term_frequency[min(self.term_frequency, key=self.term_frequency.__getitem__)]
        self.term_frequency[term] += 1

    def add_accent_hint(self, wrong: str, correct: str) -> None:
        self.accent_hints[wrong] = correct

    def clear(self) -> None:
        for part in (self.word_corrections, self.custom_terms,
                     self.term_frequency, self.accent_hints):
            part.clear()


class AdaptiveLearning:
    """
    Singleton class managing learned corrections and user vocabulary.

    Thread-safe for use from multiple threads (UI + worker).
    """

    _instance: Optional["AdaptiveLearning"] = None
    _lock = threading.Lock()

    def __new__(cls, data_dir: Optional[Path] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: Optional[Path] = None):
        if getattr(self, '_initialized', False):
            return

        self._data_dir = data_dir or self._default_data_dir()
        self._data_lock = threading.Lock()
        self._last_save = 0.0
        self._learned = _Learned()
        # Each change since the last save, replayed onto the file as it is
        # when saved: the desktop and web apps learn side by side, and writing
        # this process's whole state would undo whatever the other learned.
        self._unsaved: List[Callable[[_Learned], None]] = []

        # Compiled patterns (rebuilt when corrections change). Each entry is
        # ``(trigger_lower, pattern, replacement)``: trigger_lower is the cheap
        # substring gate so apply_learned_corrections can skip the regex when the
        # word can't be present (see apply_learned_corrections).
        self._correction_patterns: List[Tuple[str, re.Pattern, str]] = []

        # Load existing data
        self._load()
        self._initialized = True

        logger.info("AdaptiveLearning initialized with %d corrections, %d custom terms",
                    len(self._learned.word_corrections), len(self._learned.custom_terms))

    @staticmethod
    def _default_data_dir() -> Path:
        """Default storage directory (the data/ root) via the path authority."""
        from src.core.paths import learned_corrections_path
        return learned_corrections_path().parent

    # ------------------------------------------------------------------
    # Learning API
    # ------------------------------------------------------------------

    def learn_correction(self, wrong: str, correct: str) -> bool:
        """Learn a word correction from user input; False when it was refused."""
        if not wrong or not correct:
            return False
        wrong = wrong.lower().strip()
        correct = correct.lower().strip()

        if wrong == correct:
            return False
        if len(wrong) < _MIN_WORD_LENGTH or len(correct) < _MIN_WORD_LENGTH:
            return False
        if not _may_rewrite(wrong):
            # "medial" -> "lateral" in one report is a change of mind, not a
            # mishearing; learned, it would rewrite every later patient.
            logger.debug("Not learned: '%s' is a real word", wrong)
            return False

        self._change(lambda learned: learned.add_correction(wrong, correct), rebuild=True)
        logger.debug("Learned correction: '%s' → '%s'", wrong, correct)
        self._maybe_save()

        # Forward to the cloud training collector (no-op unless the user has
        # opted in). Kept as a lazy, fire-and-forget call so this module never
        # hard-depends on the training/cloud subsystem.
        try:
            from src.training.collector import get_correction_collector
            get_correction_collector().record_text_correction(wrong, correct)
        except Exception:
            pass
        return True

    def learn_term(self, term: str) -> None:
        """Add a term to the custom vocabulary (protects from 'correction')."""
        if not term or len(term) < _MIN_WORD_LENGTH:
            return
        term = term.lower().strip()

        self._change(lambda learned: learned.add_term(term))
        self._maybe_save()

    def record_term_usage(self, term: str) -> None:
        """Record that a term was used (for frequency weighting)."""
        if not term or len(term) < _MIN_WORD_LENGTH:
            return
        term = term.lower().strip()

        self._change(lambda learned: learned.count_use(term))
        self._maybe_save()  # debounced, so not a write per word

    def learn_accent_pattern(self, wrong: str, correct: str) -> None:
        """Learn an accent-specific correction pattern."""
        if not wrong or not correct:
            return

        wrong, correct = wrong.lower(), correct.lower()
        self._change(lambda learned: learned.add_accent_hint(wrong, correct), rebuild=True)
        self._maybe_save()

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_correction(self, word: str) -> Optional[str]:
        """Get learned correction for a word, or None if not found."""
        with self._data_lock:
            return self._learned.word_corrections.get(word.lower())

    def is_known_term(self, term: str) -> bool:
        """Check if term is in custom vocabulary."""
        with self._data_lock:
            return term.lower() in self._learned.custom_terms

    def custom_terms(self) -> Set[str]:
        """A copy of the words the radiologist has said to keep as spoken."""
        with self._data_lock:
            return set(self._learned.custom_terms)

    def get_term_frequency(self, term: str) -> int:
        """Get usage frequency of a term."""
        with self._data_lock:
            return self._learned.term_frequency.get(term.lower(), 0)

    def get_high_frequency_terms(self, min_count: int = 3) -> List[str]:
        """Get terms used at least min_count times."""
        with self._data_lock:
            return [t for t, c in self._learned.term_frequency.items() if c >= min_count]

    # ------------------------------------------------------------------
    # Post-processing integration
    # ------------------------------------------------------------------

    def apply_learned_corrections(self, text: str) -> str:
        """Apply all learned corrections to text. Run last in the pipeline.

        Quick-scan gate: the learned set can hold thousands of patterns, but a
        given transcript chunk contains only a handful of words. Lower-casing the
        text once and skipping any pattern whose trigger word isn't a substring
        turns "run every regex every chunk" into "run only the few that could
        match": the same optimisation accent corrections already use.
        """
        if not text:
            return text

        # _rebuild_patterns rebinds the list wholesale, so grabbing the reference
        # under the lock and iterating outside it is safe (and non-blocking).
        with self._data_lock:
            patterns = self._correction_patterns

        text_lower = text.lower()
        for trigger, pattern, replacement in patterns:
            if trigger in text_lower:
                text = pattern.sub(replacement, text)

        return text

    def get_custom_prompt_additions(self) -> str:
        """Get terms to add to Whisper's initial prompt.

        Includes:
        - All user-defined custom vocabulary (always; these are what the user
          explicitly told the app to recognise correctly).
        - High-frequency passively learned terms (used ≥5 times).

        Capped at 80 terms total to stay within Whisper's prompt budget.
        """
        with self._data_lock:
            # Custom vocabulary always injected
            custom = list(self._learned.custom_terms)

            # High-frequency passively learned terms
            freq_terms = [t for t, c in self._learned.term_frequency.items() if c >= 5]
            freq_terms_sorted = sorted(
                freq_terms,
                key=lambda t: self._learned.term_frequency.get(t, 0),
                reverse=True,
            )

        combined = list(dict.fromkeys(custom + freq_terms_sorted))[:80]
        return " ".join(combined) if combined else ""

    # ------------------------------------------------------------------
    # Text change tracking (for passive learning)
    # ------------------------------------------------------------------

    def track_edit(self, old_text: str, new_text: str) -> None:
        """Track user edits to learn corrections passively.

        Called when user manually edits the transcription. Detects
        single-word substitutions and learns them as corrections.
        """
        if not old_text or not new_text:
            return
        # Aligned by a real diff, not position: zipping the two word lists
        # paired "meniscus" with "medial" when one word was inserted ("the
        # meniscus" -> "the medial meniscus") and learned that as a fix.
        edits, fixes = _word_fixes(old_text, new_text)
        if len(edits) == 1 and len(fixes) == 1:
            wrong, right = fixes[0]
            if self._is_valid_correction(wrong, right):
                self.learn_correction(wrong, right)
        # Record frequency of terms in new text
        for word in new_text.lower().split():
            if len(word) >= _MIN_WORD_LENGTH and word.isalpha():
                self.record_term_usage(word)

    def _is_valid_correction(self, old: str, new: str) -> bool:
        """Check if a word change looks like a real correction."""
        # Both must be mostly alphabetic
        if not (old.replace("'", "").replace("-", "").isalpha()):
            return False
        if not (new.replace("'", "").replace("-", "").isalpha()):
            return False

        # Must be similar length (not wildly different words)
        len_ratio = len(old) / len(new) if new else 0
        if not (0.5 <= len_ratio <= 2.0):
            return False

        # Must share some characters (Levenshtein-ish heuristic)
        shared = len(set(old) & set(new))
        total = len(set(old) | set(new))
        return total <= 0 or shared / total >= 0.3

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_patterns(self) -> None:
        """Rebuild compiled regex patterns for corrections."""
        patterns = []

        for wrong, correct in self._learned.word_corrections.items():
            if not _may_rewrite(wrong):  # learned before this rule existed
                continue
            try:
                # Word boundary match, case insensitive
                pat = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
                patterns.append((wrong.lower(), pat, correct))
            except re.error:
                continue

        for wrong, correct in self._learned.accent_hints.items():
            if not _may_rewrite(wrong):
                continue
            try:
                pat = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
                patterns.append((wrong.lower(), pat, correct))
            except re.error:
                continue

        self._correction_patterns = patterns

    def _change(self, edit: Callable[[_Learned], None], rebuild: bool = False) -> None:
        """Apply *edit* now, and keep it to replay onto the file at the next save."""
        with self._data_lock:
            edit(self._learned)
            self._unsaved.append(edit)
            if rebuild:
                self._rebuild_patterns()

    def _maybe_save(self) -> None:
        """Save if anything changed and the debounce time has passed."""
        if self._unsaved and (time.time() - self._last_save) > _SAVE_DEBOUNCE_SEC:
            self._save()

    def _save(self) -> None:
        """Replay this process's changes onto the file as it is now, and adopt the result."""
        with self._data_lock:
            edits, self._unsaved = self._unsaved, []

        def replay(data: dict) -> None:
            learned = _Learned.from_json(data)
            for edit in edits:
                edit(learned)
            data.clear()
            data.update(learned.to_json())

        merged = update_json(self._data_dir / _LEARNING_FILE, {}, replay)
        with self._data_lock:
            if merged is None:  # not written: keep the edits for the next save
                self._unsaved = edits + self._unsaved
                return
            self._learned = _Learned.from_json(merged)
            for edit in self._unsaved:  # made while the file was being written
                edit(self._learned)
            self._rebuild_patterns()
            self._last_save = time.time()

    def _load(self) -> None:
        """Load learning data from disk."""
        learned = _Learned.from_json(read_json(self._data_dir / _LEARNING_FILE, {}))
        with self._data_lock:
            self._learned = learned
            self._rebuild_patterns()

        logger.info(
            "Loaded learning data: %d corrections, %d terms, %d frequencies",
            len(learned.word_corrections),
            len(learned.custom_terms),
            len(learned.term_frequency),
        )

    # ------------------------------------------------------------------
    # Management API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all learned data."""
        self._change(_Learned.clear, rebuild=True)
        self._save()
        logger.info("Reset all adaptive learning data")

    def export_corrections(self) -> Dict[str, str]:
        """Export learned corrections for review."""
        with self._data_lock:
            return dict(self._learned.word_corrections)

    def import_corrections(self, corrections: Dict[str, str]) -> None:
        """Import corrections (merge with existing)."""
        for wrong, correct in corrections.items():
            self.learn_correction(wrong, correct)

    def force_save(self) -> None:
        """Force immediate save of learning data."""
        self._save()

    def get_stats(self) -> Dict[str, int]:
        """Get learning statistics."""
        with self._data_lock:
            return {
                "corrections": len(self._learned.word_corrections),
                "custom_terms": len(self._learned.custom_terms),
                "tracked_terms": len(self._learned.term_frequency),
                "accent_hints": len(self._learned.accent_hints),
            }


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_adaptive_learning: Optional[AdaptiveLearning] = None

def get_adaptive_learning() -> AdaptiveLearning:
    """Get the singleton AdaptiveLearning instance."""
    global _adaptive_learning
    if _adaptive_learning is None:
        _adaptive_learning = AdaptiveLearning()
    return _adaptive_learning


def _may_rewrite(word: str) -> bool:
    """A learned rule applies everywhere, blind to context, so it may only fix
    a word that is not real: a mishearing. Real-word confusions need context
    (dictation/postprocess/confusion_sets.yaml)."""
    from src.medical.medical_dict import is_known_word  # noqa: PLC0415 - heavy wordlists, lazy
    return not is_known_word(word)


def apply_learned_corrections(text: str) -> str:
    """Shortcut to apply learned corrections to text."""
    return get_adaptive_learning().apply_learned_corrections(text)


def learn_from_edit(old_text: str, new_text: str) -> None:
    """Track an edit for passive learning.

    Also feeds the corrected text to the context model's on-device learned
    corpus (de-duplicated, so the debounced repeat calls are harmless) so
    real-word disambiguation improves for this radiologist's own reports over
    time. Best-effort: a failure here never blocks the primary edit tracking.
    """
    get_adaptive_learning().track_edit(old_text, new_text)
    try:
        from src.dictation.postprocess.context_model import append_learned_text
        append_learned_text(new_text)
    except Exception as exc:  # learning is an enhancement, never load-bearing
        logger.debug("Context-corpus learning skipped: %s", exc)


def learn_from_session(dictated: str, final: str) -> int:
    """Learn the one-word fixes the radiologist made to a finished dictation.

    *dictated* is what the app handed over; *final* is the text when the report
    was committed. Every one-word-for-one-word change that passes the same
    "looks like a real correction" test the live tracker uses is learned, so
    the next dictation corrects it on its own. Returns how many were learned.
    """
    learning = get_adaptive_learning()
    learned = 0
    for wrong, right in _word_fixes(dictated, final)[1]:
        if learning._is_valid_correction(wrong, right):
            learned += learning.learn_correction(wrong, right)
    return learned


def _word_fixes(before: str, after: str):
    """The edits from *before* to *after*, and the one-word-for-one-word ones.

    Returns ``(edits, fixes)``: every diff record, and ``(wrong, right)``
    lower-case pairs for the replacements of a single word by a single word,
    punctuation stripped. The one alignment both learning paths use.
    """
    from src.features.edit_tracking import diff_edits  # noqa: PLC0415
    edits = diff_edits(before, after)
    fixes = []
    for edit in edits:
        if edit["op"] != "replace" or " " in edit["before"] or " " in edit["after"]:
            continue
        wrong = edit["before"].strip(_PUNCTUATION).lower()
        right = edit["after"].strip(_PUNCTUATION).lower()
        if wrong and right and wrong != right:
            fixes.append((wrong, right))
    return edits, fixes


def get_custom_prompt_suffix() -> str:
    """Get high-frequency terms to append to Whisper's prompt."""
    return get_adaptive_learning().get_custom_prompt_additions()
