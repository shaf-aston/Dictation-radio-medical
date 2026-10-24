"""A de-identified training clip exists only while a staged record needs it.

Clips were written when a dictation ended and kept for ever: by a session with
no corrections, one whose corrections all failed the privacy check, one that
was replaced before it finished, and after consent was withdrawn.
"""

import pytest

from src.core import paths
from src.medical import deid
from src.training import staging_db
from src.training.collector import CorrectionCollector


@pytest.fixture
def collector(tmp_path, monkeypatch):
    paths._data_dir.cache_clear()
    monkeypatch.setattr(paths, "_data_dir", lambda: tmp_path)
    monkeypatch.setattr(staging_db, "_staging_db", staging_db.StagingDB(tmp_path / "s.db"))
    monkeypatch.setattr(CorrectionCollector, "_consent_active", staticmethod(lambda: True))

    def fake_deidentify(_patient, _wav, _segments, session_id):
        clip = paths.training_audio_dir() / f"{session_id}.wav"
        clip.write_bytes(b"RIFF")
        return str(clip)

    monkeypatch.setattr(CorrectionCollector, "_deidentify_audio", staticmethod(fake_deidentify))
    c = object.__new__(CorrectionCollector)
    c._init()
    return c


def _clips():
    return sorted(p.name for p in paths.training_audio_dir().iterdir())


def _record(c, session_id, patient=None, corrections=()):
    c.start_session(session_id, "rec.wav", patient_info=patient)
    for wrong, right in corrections:
        c.record_text_correction(wrong, right)
    c.prepare_audio()


def test_a_session_with_a_correction_keeps_its_clip(collector):
    _record(collector, "s1", corrections=[("menscus", "meniscus")])
    assert collector.finalize_session() == 1 and _clips() == ["s1.wav"]


def test_a_session_with_no_corrections_leaves_no_clip(collector):
    _record(collector, "s1")
    assert collector.finalize_session() == 0 and _clips() == []


def test_a_session_whose_corrections_all_fail_the_privacy_check_leaves_no_clip(collector, monkeypatch):
    def refuse(_self, _text):
        raise deid.PrivacyError("identifier left")
    monkeypatch.setattr(deid.DeIdentifier, "validate_clean", refuse)
    _record(collector, "s1", corrections=[("menscus", "meniscus")])
    assert collector.finalize_session() == 0 and _clips() == []


def test_a_session_replaced_before_it_finished_leaves_no_clip(collector):
    _record(collector, "s1", corrections=[("menscus", "meniscus")])
    _record(collector, "s2")
    assert _clips() == ["s2.wav"]


def test_consent_off_at_finalize_leaves_no_clip(collector, monkeypatch):
    _record(collector, "s1", corrections=[("menscus", "meniscus")])
    monkeypatch.setattr(CorrectionCollector, "_consent_active", staticmethod(lambda: False))
    assert collector.finalize_session() == 0 and _clips() == []


def test_withdrawing_consent_deletes_everything_not_yet_uploaded(collector):
    _record(collector, "s1", corrections=[("menscus", "meniscus")])
    collector.finalize_session()
    _record(collector, "s2", corrections=[("labrom", "labrum")])
    assert collector.withdraw_consent() == 1
    assert _clips() == [] and staging_db.get_staging_db().pending_count() == 0
    assert collector.finalize_session() == 0

