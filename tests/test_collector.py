"""Integration tests for the correction collector and its consent gate."""

from __future__ import annotations

import pytest

import src.training.staging_db as staging_db_module
from src.core.settings import Settings
from src.training.collector import CorrectionCollector


@pytest.fixture
def fresh_collector(tmp_path, monkeypatch):
    """Isolate the collector + staging DB singletons per test."""
    # Point the staging DB at a temp file and reset the module singleton.
    db = staging_db_module.StagingDB(db_path=tmp_path / "staging.db")
    staging_db_module._staging_db = db
    # Reset the collector singleton.
    CorrectionCollector._instance = None
    yield CorrectionCollector(), db
    staging_db_module._staging_db = None
    CorrectionCollector._instance = None


def _enable_cloud():
    Settings().batch_set({"cloud_enabled": True, "cloud_training_consent": True})


def test_no_capture_without_consent(fresh_collector):
    collector, db = fresh_collector
    # Default settings → consent off.
    collector.start_session("s1", wav_path=None, patient_info={})
    collector.record_text_correction("wertebra", "vertebra")
    assert collector.finalize_session() == 0
    assert db.pending_count() == 0


def test_capture_persists_with_consent(fresh_collector):
    collector, db = fresh_collector
    _enable_cloud()
    collector.start_session("s1", wav_path=None, patient_info={},
                            model_version="base", accent_profile="neutral")
    collector.update_segments([{"start": 1.0, "end": 2.0, "text": "wertebra noted"}])
    collector.record_text_correction("wertebra", "vertebra")
    saved = collector.finalize_session()
    assert saved == 1
    assert db.pending_count() == 1


def test_correction_text_is_deidentified(fresh_collector):
    collector, db = fresh_collector
    _enable_cloud()
    collector.start_session("s1", wav_path=None,
                            patient_info={"name": "John Smith"})
    # Correction text mentioning the patient name must be scrubbed.
    collector.record_text_correction("John Smith tendon", "John Smith tendon ok")
    collector.finalize_session()
    pending = db.get_pending()
    assert pending  # survived (name redacted, not dropped)
    assert all("John Smith" not in r.wrong_text for r in pending)
    assert all("John Smith" not in r.correct_text for r in pending)


def test_noop_correction_ignored(fresh_collector):
    collector, db = fresh_collector
    _enable_cloud()
    collector.start_session("s1", wav_path=None, patient_info={})
    collector.record_text_correction("same", "same")  # no real change
    assert collector.finalize_session() == 0


def test_review_time_corrections_captured(fresh_collector):
    """Spelling fixes typed *after* transcription (during review) must persist.

    This is the dominant workflow: record → audio prepared + WAV deleted →
    user reviews and corrects → session finalized on close/next recording.
    """
    collector, db = fresh_collector
    _enable_cloud()
    collector.start_session("s1", wav_path=None, patient_info={})
    collector.update_segments([{"start": 0.0, "end": 2.0, "text": "wertebra noted"}])

    # End-of-transcription phase: audio prepared, but session stays open.
    collector.prepare_audio()

    # User reviews the transcript and fixes a spelling mistake.
    collector.record_text_correction("wertebra", "vertebra")

    # Finalised later (new recording / app close).
    assert collector.finalize_session() == 1
    assert db.pending_count() == 1
    pending = db.get_pending()
    assert pending[0].wrong_text == "wertebra"
    assert pending[0].correct_text == "vertebra"


def test_new_session_finalizes_previous(fresh_collector):
    """Starting a new session must flush the previous (open) one."""
    collector, db = fresh_collector
    _enable_cloud()
    collector.start_session("s1", wav_path=None, patient_info={})
    collector.record_text_correction("wertebra", "vertebra")
    collector.prepare_audio()
    # No explicit finalize — instead a second recording begins.
    collector.finalize_session()  # mirrors _start_training_capture's flush
    collector.start_session("s2", wav_path=None, patient_info={})
    assert db.pending_count() == 1


def test_prepare_audio_without_consent_is_noop(fresh_collector):
    collector, db = fresh_collector
    # No consent enabled.
    collector.start_session("s1", wav_path=None, patient_info={})
    collector.prepare_audio()  # must not raise
    collector.record_text_correction("wertebra", "vertebra")
    assert collector.finalize_session() == 0
