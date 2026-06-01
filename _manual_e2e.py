"""Manual end-to-end test of the two-phase capture + audio de-identification.

Exercises the real modules (no mocks) against a temp data dir:
  consent on → start_session → segments (one with PHI) → prepare_audio
  → review-time correction → finalize → assert staged record + silenced audio.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Redirect all data writes to a throwaway dir BEFORE importing collectors.
tmp = Path(tempfile.mkdtemp(prefix="radio_e2e_"))
import src.features.file_manager as fm
fm._data_dir = lambda: tmp  # type: ignore

import src.core.settings as settings_module
settings_module.settings_file = lambda: tmp / "settings.json"  # type: ignore

from src.core.settings import Settings
from src.training.collector import CorrectionCollector
import src.training.staging_db as staging_db_module


def main() -> int:
    # Fresh singletons pointed at the temp DB.
    staging_db_module._staging_db = staging_db_module.StagingDB(db_path=tmp / "staging.db")
    CorrectionCollector._instance = None
    Settings().batch_set({"cloud_enabled": True, "cloud_training_consent": True})

    # --- Build a 4-second 16kHz WAV: PHI spoken 0.5–1.5s, finding 2.5–3.5s. ---
    sr = 16000
    audio = (0.3 * np.sin(2 * np.pi * 220 * np.arange(4 * sr) / sr)).astype("float32")
    wav = tmp / "session.wav"
    sf.write(str(wav), audio, sr)

    collector = CorrectionCollector()
    collector.start_session(
        session_id="sess-e2e",
        wav_path=str(wav),
        patient_info={"name": "John Smith"},
        model_version="base",
        accent_profile="south_asian",
    )
    # Segment 1 contains the patient name (PHI); segment 2 the dictation error.
    collector.update_segments([
        {"start": 0.5, "end": 1.5, "text": "patient John Smith referred"},
        {"start": 2.5, "end": 3.5, "text": "wertebra body intact"},
    ])

    # End of transcription: de-identify audio while the WAV exists.
    collector.prepare_audio()

    # Now the WAV would be deleted by cleanup_temp_audio in the app.
    wav.unlink()

    # User reviews and fixes the spelling mistake (post-transcription).
    collector.record_text_correction("wertebra", "vertebra")

    # Finalize (mirrors next-recording / app-close boundary).
    saved = collector.finalize_session()

    db = staging_db_module._staging_db
    pending = db.get_pending()

    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    print("Two-phase capture:")
    check("one record saved", saved == 1)
    check("one pending in DB", db.pending_count() == 1)
    check("correction wrong_text == 'wertebra'", pending and pending[0].wrong_text == "wertebra")
    check("correction correct_text == 'vertebra'", pending and pending[0].correct_text == "vertebra")
    check("record marked deidentified", pending and pending[0].deidentified)
    check("timestamps located from segments",
          pending and pending[0].ts_start == 2.5 and pending[0].ts_end == 3.5)

    clip_path = pending[0].audio_path if pending else None
    print("Audio de-identification:")
    check("audio clip path recorded", bool(clip_path))
    check("audio clip exists on disk", bool(clip_path) and Path(clip_path).exists())

    if clip_path and Path(clip_path).exists():
        clip, csr = sf.read(clip_path, dtype="float32")
        # PHI window 0.5–1.5s padded ±0.5s → silence 0.0–2.0s.
        phi_region = clip[int(0.6 * csr):int(1.4 * csr)]
        finding_region = clip[int(2.6 * csr):int(3.4 * csr)]
        check("PHI audio region silenced", float(np.abs(phi_region).max()) == 0.0)
        check("non-PHI (finding) audio preserved", float(np.abs(finding_region).max()) > 0.01)

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
