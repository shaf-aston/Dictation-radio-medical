"""The curated radiology lexicon is data, so it gets the checks data gets.

This file is the spelling authority: whatever is in it is what a mis-transcribed
word gets snapped to (``medical_dict_match.py``, stage 7). A bad line here does
not crash anything — it quietly rewrites a radiologist's word into a wrong one,
which is the failure mode this project measures as the false-correction rate.

Validating each entry on its own is not enough; the *set* has to be checked too,
which is what the duplicate test is for.
"""

from __future__ import annotations

from src.features.file_manager import radiology_lexicon_path


def _entries() -> list[str]:
    """Every real term in the file, in file order, comments and blanks dropped."""
    text = radiology_lexicon_path().read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class TestSetIntegrity:
    def test_no_term_appears_twice(self) -> None:
        # Case-insensitively, because the matcher folds case: "Hila" and "hila"
        # are one entry as far as snapping is concerned, and a second copy is a
        # silent contradiction rather than a harmless repeat.
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for term in _entries():
            key = term.lower()
            if key in seen:
                duplicates.append(f"{term!r} duplicates {seen[key]!r}")
            seen[key] = term
        assert not duplicates, f"duplicate lexicon entries: {duplicates}"

    def test_every_entry_is_lowercase(self) -> None:
        # One case convention, so a term cannot be added twice in two spellings
        # and pass the duplicate check above.
        odd = [t for t in _entries() if t != t.lower()]
        assert not odd, f"lexicon entries that are not lowercase: {odd}"


class TestNoExtractionNoise:
    """CLAUDE.md: never feed PDF/OCR-extracted text into the lexicon.

    Extraction artefacts are what make a snap target dangerous — a hyphenation
    fragment like "supra" is one short edit from a great many real words, so it
    pulls correct dictation toward nonsense.
    """

    def test_no_ligature_or_control_characters(self) -> None:
        bad = [t for t in _entries() if any(c in t for c in "ﬁﬂﬀ­​")]
        assert not bad, f"entries carrying ligature/soft-hyphen artefacts: {bad}"

    def test_no_dangling_hyphen_or_whitespace_inside(self) -> None:
        bad = [
            t for t in _entries()
            if t.startswith("-") or t.endswith("-") or any(c.isspace() for c in t)
        ]
        assert not bad, f"entries that look like split fragments: {bad}"

    def test_entries_are_long_enough_to_be_words(self) -> None:
        # Short entries are the expensive mistake: at 3 characters almost every
        # dictated word is within one edit, so the term becomes an attractor.
        # The known-good abbreviations the app really does dictate are listed
        # here explicitly rather than the rule being loosened for all of them.
        allowed = {"acl", "pcl", "mcl", "lcl", "atfl", "cfl", "slil", "ltil", "tfcc",
                   "ct", "mri", "pet", "iv", "ap", "pa", "rib", "hip", "arm", "leg",
                   "cbd", "ivc", "svc", "gb", "lv", "rv", "la", "ra", "si", "tmj"}
        bad = [t for t in _entries() if len(t) < 4 and t not in allowed]
        assert not bad, (
            f"entries too short to be safe snap targets: {bad} — add to the "
            f"allow-list above only if a radiologist genuinely dictates it"
        )
