"""Tests for the WhisperVoiceTask archive builder.

Guards the batch format that ``scripts/lightning/train_whisper.py`` consumes:
the manifest shape and the deduplication of shared audio clips (many correction
records reference one session clip; it must be bundled only once).
"""

from __future__ import annotations

import json
import tarfile
from datetime import datetime, timezone

from src.cloud.tasks.whisper_voice import WhisperVoiceTask
from src.training.schemas import CorrectionRecord


def _record(wrong, correct, audio_path=None) -> CorrectionRecord:
    return CorrectionRecord(
        session_id="sess1",
        created_at=datetime.now(timezone.utc).isoformat(),
        wrong_text=wrong,
        correct_text=correct,
        model_version="base",
        accent_profile="south_asian",
        audio_path=audio_path,
        deidentified=True,
    )


def test_build_archive_manifest_and_audio_dedup(tmp_path):
    clip = tmp_path / "sess1_clip.wav"
    clip.write_bytes(b"RIFFfake-wav-bytes")

    records = [
        _record("wertebra", "vertebra", audio_path=str(clip)),
        _record("tier", "tear", audio_path=str(clip)),    # same clip → deduped
        _record("akl", "ACL", audio_path=None),            # text-only record
    ]
    archive = WhisperVoiceTask().build_archive("batch_test", records, base_model="base")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
            manifest = json.loads(
                tar.extractfile("manifest.json").read().decode("utf-8")  # type: ignore[union-attr]
            )

        # Manifest shape.
        assert manifest["batch_id"] == "batch_test"
        assert manifest["base_model"] == "base"
        assert manifest["task_type"] == "whisper_voice"
        assert "created_at" in manifest
        assert len(manifest["records"]) == 3

        # The shared clip is bundled exactly once despite two references.
        audio_members = sorted(n for n in names if n.startswith("audio/"))
        assert audio_members == [f"audio/{clip.name}"]

        # Two records point at that member; the text-only record carries null.
        audio_files = [r["audio_file"] for r in manifest["records"]]
        assert audio_files.count(f"audio/{clip.name}") == 2
        assert audio_files.count(None) == 1

        # Each record carries the de-identified triple + timing fields.
        assert set(manifest["records"][0]) >= {
            "wrong", "correct", "accent", "ts_start", "ts_end", "audio_file"
        }
    finally:
        archive.unlink(missing_ok=True)


def test_build_archive_omits_missing_audio(tmp_path):
    # A record whose audio file is gone must not crash the build and must not
    # add a phantom member; its manifest entry falls back to text-only.
    records = [_record("akl", "ACL", audio_path=str(tmp_path / "nope.wav"))]
    archive = WhisperVoiceTask().build_archive("batch_missing", records, base_model="base")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
            manifest = json.loads(
                tar.extractfile("manifest.json").read().decode("utf-8")  # type: ignore[union-attr]
            )
        assert not [n for n in names if n.startswith("audio/")]
        assert manifest["records"][0]["audio_file"] is None
    finally:
        archive.unlink(missing_ok=True)
