"""Tests for the evaluation gold-set corpus layer.

No audio and no model: ``load_set`` only checks that a clip file exists, so an
empty placeholder is enough to exercise every path. The important cases here are
the *refusals* — an unreviewed machine draft and a vanished clip both have to be
loud, because either one silently accepted would make a later milestone look
better than it is.
"""

from __future__ import annotations

import pytest

from scripts.eval.build_sets import load_report_corpus
from scripts.eval.corpus import (
    UNREVIEWED_MARKER,
    Clip,
    load_set,
    write_manifest,
)
from src.features.file_manager import eval_set_dir


def _clip(name: str, reference: str, set_name: str = "bench") -> Clip:
    """A clip backed by an empty placeholder file in the isolated data dir."""
    path = eval_set_dir(set_name) / f"{name}.wav"
    path.touch()
    return Clip(set_name=set_name, audio_path=path, reference=reference)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_manifest_round_trip_preserves_reference_and_metadata():
    original = Clip(
        set_name="tts",
        audio_path=_clip("r0", "", set_name="tts").audio_path,
        reference="No acute cardiopulmonary process.",
        source="report_corpus.txt",
        licence="project-authored",
        synthetic=True,
        duration_sec=12.5,
    )
    write_manifest("tts", [original])

    loaded = load_set("tts")
    assert len(loaded) == 1
    assert loaded[0].reference == original.reference
    assert loaded[0].synthetic is True
    assert loaded[0].duration_sec == pytest.approx(12.5)
    assert loaded[0].clip_id == "tts/r0"


def test_manifest_rewrite_replaces_rather_than_appends():
    write_manifest("bench", [_clip("a", "first")])
    write_manifest("bench", [_clip("a", "second")])
    assert len(load_set("bench")) == 1


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_unreviewed_draft_is_refused_not_scored():
    write_manifest("bench", [_clip("draft", f"{UNREVIEWED_MARKER} machine guess")])
    with pytest.raises(ValueError, match=r"UNREVIEWED"):
        load_set("bench")


def test_one_unreviewed_draft_blocks_the_whole_set():
    # Partial scoring would quietly change the denominator between runs.
    write_manifest("bench", [
        _clip("good", "a corrected reference"),
        _clip("bad", f"{UNREVIEWED_MARKER} machine guess"),
    ])
    with pytest.raises(ValueError, match=r"bad\.reference\.txt"):
        load_set("bench")


def test_missing_manifest_names_the_build_command():
    with pytest.raises(FileNotFoundError, match=r"build_sets --set libri"):
        load_set("libri")


def test_set_with_every_clip_missing_raises():
    clip = _clip("gone", "reference text")
    write_manifest("bench", [clip])
    clip.audio_path.unlink()
    with pytest.raises(ValueError, match=r"zero usable clips"):
        load_set("bench")


def test_missing_clip_is_skipped_and_warned(caplog):
    present, absent = _clip("here", "kept"), _clip("vanished", "dropped")
    write_manifest("bench", [present, absent])
    absent.audio_path.unlink()

    with caplog.at_level("WARNING"):
        loaded = load_set("bench")

    assert [c.clip_id for c in loaded] == ["bench/here"]
    assert "no audio file" in caplog.text


# ---------------------------------------------------------------------------
# Hand-editable reference sidecar
# ---------------------------------------------------------------------------

def test_sidecar_reference_overrides_the_manifest():
    # Correcting a draft means editing the .txt next to the audio; that edit
    # must take effect without anyone rebuilding the manifest.
    clip = _clip("c0", f"{UNREVIEWED_MARKER} wrong words here")
    write_manifest("bench", [clip])
    sidecar = clip.audio_path.parent / "c0.reference.txt"
    sidecar.write_text("the corrected reference\n", encoding="utf-8")

    loaded = load_set("bench")
    assert loaded[0].reference == "the corrected reference"


def test_set_name_cannot_escape_the_eval_directory():
    # The name reaches this straight from a CLI argument. Traversal characters
    # are stripped rather than rejected, so assert the property that matters:
    # the resolved directory stays inside data/eval/.
    from src.features.file_manager import eval_dir

    escaped = eval_set_dir("../../etc")
    assert escaped.resolve().parent == eval_dir().resolve()


def test_set_name_that_sanitises_to_nothing_is_rejected():
    with pytest.raises(ValueError, match=r"Invalid evaluation set name"):
        eval_set_dir("../..")


# ---------------------------------------------------------------------------
# Report corpus
# ---------------------------------------------------------------------------

def test_report_corpus_parses_reports_and_drops_comments():
    reports = load_report_corpus()
    assert len(reports) >= 25
    assert not any(r.startswith("#") for r in reports)
    # Each is a single normalised line of dictated prose.
    assert all("\n" not in r for r in reports)
    assert all(len(r.split()) >= 25 for r in reports)


def test_report_corpus_exercises_the_radiology_lexicon():
    # The set exists to measure medical-term accuracy; if it barely touches the
    # lexicon, the term-error-rate metric would be measuring nothing. Read the
    # lexicon file directly — the medical_dict singleton is stubbed out here.
    from src.features.file_manager import radiology_lexicon_path

    lexicon = {
        ln.strip().lower()
        for ln in radiology_lexicon_path().read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }
    words = {w.strip(".,") for r in load_report_corpus() for w in r.lower().split()}
    assert len(words & lexicon) >= 50
