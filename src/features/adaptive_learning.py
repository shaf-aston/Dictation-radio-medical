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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from src.core.json_store import read_json, write_json

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
        self._dirty = False
        self._last_save = 0.0

        # Learning data structures
        self._word_corrections: Dict[str, str] = {}
        self._custom_terms: Set[str] = set()
        self._term_frequency: Dict[str, int] = defaultdict(int)
        self._accent_hints: Dict[str, str] = {}

        # Compiled patterns (rebuilt when corrections change). Each entry is
        # ``(trigger_lower, pattern, replacement)`` — trigger_lower is the cheap
        # substring gate so apply_learned_corrections can skip the regex when the
        # word can't be present (see apply_learned_corrections).
        self._correction_patterns: List[Tuple[str, re.Pattern, str]] = []

        # Load existing data
        self._load()
        self._initialized = True

        logger.info("AdaptiveLearning initialized with %d corrections, %d custom terms",
                    len(self._word_corrections), len(self._custom_terms))

    @staticmethod
    def _default_data_dir() -> Path:
        """Default storage directory (the data/ root) via the path authority."""
        from src.features.file_manager import learned_corrections_path
        return learned_corrections_path().parent

    # ------------------------------------------------------------------
    # Learning API
    # ------------------------------------------------------------------

    def learn_correction(self, wrong: str, correct: str) -> None:
        """Learn a word correction from user input."""
        if not wrong or not correct:
            return
        wrong = wrong.lower().strip()
        correct = correct.lower().strip()

        if wrong == correct:
            return
        if len(wrong) < _MIN_WORD_LENGTH or len(correct) < _MIN_WORD_LENGTH:
            return

        with self._data_lock:
            if len(self._word_corrections) >= _MAX_CORRECTIONS and self._word_corrections:
                oldest = next(iter(self._word_corrections))
                del self._word_corrections[oldest]

            self._word_corrections[wrong] = correct
            self._dirty = True
            self._rebuild_patterns()

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

    def learn_term(self, term: str) -> None:
        """Add a term to the custom vocabulary (protects from 'correction')."""
        if not term or len(term) < _MIN_WORD_LENGTH:
            return
        term = term.lower().strip()

        with self._data_lock:
            if len(self._custom_terms) >= _MAX_CUSTOM_TERMS and self._custom_terms:
                evicted = self._custom_terms.pop()
                logger.warning("Custom terms full, evicted: %s", evicted)

            self._custom_terms.add(term)
            self._dirty = True

        self._maybe_save()

    def record_term_usage(self, term: str) -> None:
        """Record that a term was used (for frequency weighting)."""
        if not term or len(term) < _MIN_WORD_LENGTH:
            return
        term = term.lower().strip()

        with self._data_lock:
            if len(self._term_frequency) >= _MAX_FREQUENCY_TERMS and self._term_frequency:
                min_term = min(self._term_frequency, key=lambda t: self._term_frequency[t])
                del self._term_frequency[min_term]

            self._term_frequency[term] += 1
            self._dirty = True

        # Don't save on every usage - too frequent

    def learn_accent_pattern(self, wrong: str, correct: str) -> None:
        """Learn an accent-specific correction pattern."""
        if not wrong or not correct:
            return

        with self._data_lock:
            self._accent_hints[wrong.lower()] = correct.lower()
            self._dirty = True
            self._rebuild_patterns()

        self._maybe_save()

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_correction(self, word: str) -> Optional[str]:
        """Get learned correction for a word, or None if not found."""
        with self._data_lock:
            return self._word_corrections.get(word.lower())

    def is_known_term(self, term: str) -> bool:
        """Check if term is in custom vocabulary."""
        with self._data_lock:
            return term.lower() in self._custom_terms

    def get_term_frequency(self, term: str) -> int:
        """Get usage frequency of a term."""
        with self._data_lock:
            return self._term_frequency.get(term.lower(), 0)

    def get_high_frequency_terms(self, min_count: int = 3) -> List[str]:
        """Get terms used at least min_count times."""
        with self._data_lock:
            return [t for t, c in self._term_frequency.items() if c >= min_count]

    # ------------------------------------------------------------------
    # Post-processing integration
    # ------------------------------------------------------------------

    def apply_learned_corrections(self, text: str) -> str:
        """Apply all learned corrections to text. Run last in the pipeline.

        Quick-scan gate: the learned set can hold thousands of patterns, but a
        given transcript chunk contains only a handful of words. Lower-casing the
        text once and skipping any pattern whose trigger word isn't a substring
        turns "run every regex every chunk" into "run only the few that could
        match" — the same optimisation accent corrections already use.
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
            custom = list(self._custom_terms)

            # High-frequency passively learned terms
            freq_terms = [t for t, c in self._term_frequency.items() if c >= 5]
            freq_terms_sorted = sorted(
                freq_terms,
                key=lambda t: self._term_frequency.get(t, 0),
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

        old_words = old_text.lower().split()
        new_words = new_text.lower().split()

        # Only learn from simple single-word edits
        if abs(len(old_words) - len(new_words)) > 1:
            return

        # Find differing words
        changes_found = 0
        for old, new in zip(old_words, new_words):
            if old != new:
                changes_found += 1
                if changes_found != 1:
                    # Multiple changes - too complex to learn
                    return

                # Learn this single-word correction
                if self._is_valid_correction(old, new):
                    self.learn_correction(old, new)
        # Record frequency of terms in new text
        for word in new_words:
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

        for wrong, correct in self._word_corrections.items():
            try:
                # Word boundary match, case insensitive
                pat = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
                patterns.append((wrong.lower(), pat, correct))
            except re.error:
                continue

        for wrong, correct in self._accent_hints.items():
            try:
                pat = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
                patterns.append((wrong.lower(), pat, correct))
            except re.error:
                continue

        self._correction_patterns = patterns

    def _maybe_save(self) -> None:
        """Save data if dirty and debounce time has passed."""
        now = time.time()
        if self._dirty and (now - self._last_save) > _SAVE_DEBOUNCE_SEC:
            self._save()

    def _save(self) -> None:
        """Persist learning data to disk (atomically, via the shared store)."""
        with self._data_lock:
            data = {
                "word_corrections": dict(self._word_corrections),
                "custom_terms": list(self._custom_terms),
                "term_frequency": dict(self._term_frequency),
                "accent_hints": dict(self._accent_hints),
            }

        write_json(self._data_dir / _LEARNING_FILE, data)
        with self._data_lock:
            self._dirty = False
            self._last_save = time.time()

    def _load(self) -> None:
        """Load learning data from disk."""
        data = read_json(self._data_dir / _LEARNING_FILE, None)
        if not data:
            return

        with self._data_lock:
            self._word_corrections = data.get("word_corrections", {})
            self._custom_terms = set(data.get("custom_terms", []))
            self._term_frequency = defaultdict(int, data.get("term_frequency", {}))
            self._accent_hints = data.get("accent_hints", {})
            self._rebuild_patterns()

        logger.info(
            "Loaded learning data: %d corrections, %d terms, %d frequencies",
            len(self._word_corrections),
            len(self._custom_terms),
            len(self._term_frequency),
        )

    # ------------------------------------------------------------------
    # Management API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all learned data."""
        with self._data_lock:
            self._word_corrections.clear()
            self._custom_terms.clear()
            self._term_frequency.clear()
            self._accent_hints.clear()
            self._correction_patterns.clear()
            self._dirty = True

        self._save()
        logger.info("Reset all adaptive learning data")

    def export_corrections(self) -> Dict[str, str]:
        """Export learned corrections for review."""
        with self._data_lock:
            return dict(self._word_corrections)

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
                "corrections": len(self._word_corrections),
                "custom_terms": len(self._custom_terms),
                "tracked_terms": len(self._term_frequency),
                "accent_hints": len(self._accent_hints),
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


def apply_learned_corrections(text: str) -> str:
    """Shortcut to apply learned corrections to text."""
    return get_adaptive_learning().apply_learned_corrections(text)


def learn_from_edit(old_text: str, new_text: str) -> None:
    """Track an edit for passive learning."""
    get_adaptive_learning().track_edit(old_text, new_text)


def get_custom_prompt_suffix() -> str:
    """Get high-frequency terms to append to Whisper's prompt."""
    return get_adaptive_learning().get_custom_prompt_additions()
