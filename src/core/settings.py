"""
Persistent application settings stored as JSON alongside the project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from src.core.json_store import read_json, write_json
from src.features.file_manager import settings_file

_DEFAULTS: dict = {
    "model_size": "base.en",   # English-only: faster AND more accurate than "base"
    "language": "en",
    "vad_filter": True,
    "accent": "neutral",
    "cleanup_level": "medium",  # post-dictation cleanup intensity: soft/medium/hard
    # The guessing post-processing stages may not rewrite a word the decoder
    # reported at or above this confidence (src/dictation/postprocess/
    # confidence_gate.py). None = off, which is what it measured its way to:
    # on the `tts` gold set at 0.90 the gate blocked two rewrites and both were
    # correct ones, so it could only subtract. Whisper is confidently wrong often
    # enough that its confidence does not separate a misheard word from a heard
    # one: at least not on synthetic audio, where every word scores high.
    # Set a float to switch it on; the honest test is the `own` set, real
    # acoustics, where confidence actually varies. See docs/dictation-accuracy.md.
    "correction_confidence_ceiling": None,
    "theme": "dark",
    "font_size": 13,
    "auto_save_interval": 60,   # seconds
    "pause_threshold": 2.5,     # seconds silence → new paragraph
    # --- Live transcription chunk policy (see src/dictation/stream/segmenter.py) ---
    # Each chunk is decoded exactly once, cut at a VAD silence boundary: never
    # shorter than chunk_min_sec, cut at the latest pause found by
    # chunk_soft_max_sec if one exists, otherwise force-cut at
    # chunk_force_cut_sec regardless of whether a pause was found (the only
    # case that can land mid-word: see ChunkPolicy's docstring).
    #
    # "auto" (the default) ignores the three numbers below and sizes chunks
    # from what the ASR engine says a call costs (src/dictation/stream/
    # policy.py): long chunks for local Whisper, which pays ~1-4s per call,
    # short ones for a ~0.2s cloud engine, where long chunks only make the kept
    # text trail the microphone. "manual" uses the numbers verbatim. Auto is
    # also what moves an existing install off the old 6 / 15 / 20: this file
    # persists every default it was written with, so those numbers are
    # literally in it.
    "chunk_policy": "auto",
    # When the live engine can stream (Deepgram's live socket), let it: audio
    # goes up as it is spoken, and settled text comes back at each pause
    # instead of waiting for a chunk to close. Any failure falls back to
    # decoding chunks for the rest of the dictation. False: always chunks.
    "asr_streaming": True,
    # 2 / 5 measured best on the replay harness (docs/dictation-accuracy.md,
    # 2026-09-06): commit lag p50 4.78s -> 2.93s and lower term error than 6 / 15.
    "chunk_min_sec": 2.0,
    "chunk_soft_max_sec": 5.0,
    "chunk_force_cut_sec": 20.0,
    # The last thing said before a pause used to wait for the speaker to start
    # talking again, because a cut point had to be a pause with more speech
    # after it to prove the silence was real. Now the clock proves it instead:
    # once this much audio has arrived with no speech in it, the pause counts.
    # Raise it if a mid-sentence breath is closing chunks; lower it to commit
    # sooner when someone stops to read the film.
    "chunk_trailing_silence_sec": 0.6,
    # CTranslate2 intra-op threads for CPU decode. Measured on a 10s tiny.en
    # decode, best of five: 8 threads = 1.007s, 14 threads (cpu_count-2 on this
    # 16-core box) = 1.405s. Eight is faster and uses less CPU. Clamped to the
    # host core count when the model loads, and takes effect on the next app
    # start: the loaded model is cached and is not rebuilt on a settings write.
    "asr_cpu_threads": 8,
    # --- Live transcription decode quality (see src/dictation/worker.py) ---
    "live_beam_size": 2,        # beam=1 caused repetition; beam=2 still real-time
    # Every decode that is not the live loop: the desktop's post-stop polish and
    # the web app's one-shot upload. Both are "transcribe this once, properly",
    # so they share one knob: the web app used to have its own copy of it.
    "final_beam_size": 5,
    # The live preview of the still-open tail is dropped once that tail is
    # longer than this AND the machine is measured to decode slower than speech
    # (src/dictation/worker.py::should_skip_preview). The preview is never
    # committed, so this trades early sight of a few words for the committed
    # chunks arriving on time. 0 keeps the preview no matter how far behind.
    "preview_max_lag_sec": 3.0,
    # A preview decode costs the same whatever it is handed, because Whisper
    # pads every clip to a 30-second window. So decoding a tail that is barely
    # started spends a full decode to show almost nothing, and delays the first
    # real preview by that much. Measured: the opening cycle decoded 0.048s of
    # audio for 1.15s of machine. Below this many seconds of open tail, wait.
    "preview_min_tail_sec": 1.0,
    # committed chunks below this mean word confidence get one re-decode after stop
    "polish_confidence_ceiling": 0.75,
    # A single word below this confidence gets a faint underline in the report.
    # Lower than polish_confidence_ceiling on purpose: that one decides whether
    # a whole chunk is worth re-decoding, this one decides whether one word is
    # worth a second look from the radiologist, and marking every third word
    # would make the marks worth nothing.
    "uncertain_word_confidence": 0.6,
    "autosave_retention_days": 30,  # days to keep autosave files
    # --- Web front-end (src/ui/web_app.py) ---
    "web_host": "127.0.0.1",    # loopback only: the app is offline by default
    "web_port": 8005,
    "max_upload_mb": 50,        # reject audio uploads larger than this
    # Two models, one dictation. This small one decodes the words that appear
    # while you are still speaking; `model_size` above re-decodes after Stop,
    # where being right matters more than being quick. Whisper's model cache
    # holds two (faster_whisper_engine._MODEL_CACHE_MAX), so this pair costs no reloads:
    # naming a third distinct model here would make them evict each other.
    "live_model_size": "tiny.en",
    "live_cycle_sec": 0.5,      # how often the live loop looks for new audio
    "recent_reports": [],
    "patient_info_visible": True,
    "macros_panel_visible": True,
    "panel_template_open": True,
    "panel_settings_open": False,
    # --- The marked-term lookup ---
    # How many taken suggestions before the marks hint stops explaining itself
    # and shows only the count, and the running total of takes. Shared by both
    # front-ends on purpose: learning the feature on the desktop should retire
    # the hint in the browser too.
    "term_lookup_hint_uses": 3,
    "term_lookup_uses": 0,
    # --- Run log (features/run_log.py, shown at /developer) ---
    # Local diagnostics: one record per dictation. Capped because each record
    # can hold a full report, and rolling beats growing without limit.
    "run_log_max": 200,
    # The report text alongside the numbers: that output is the point of the
    # page. Turn off to keep every timing and drop only the body.
    "run_log_store_text": True,
    # --- Live event diary (core/event_log.py, shown in the developer panel) ---
    # How many recent events the in-memory diary keeps. It is a ring buffer, so
    # this is the scroll-back length of the developer console, nothing more.
    "event_log_max": 1000,
    "last_template": "",
    "last_macro_region": "Knee",
    "window_width": 1200,
    "window_height": 760,
    "splitter_sizes": [220, 980],
    # --- Cloud training (Lightning AI) ---
    "cloud_enabled": False,            # master switch for all cloud features
    "cloud_training_consent": False,   # explicit user consent to upload de-identified data
    "lightning_project_id": "",        # Lightning AI project (API key lives in OS keychain)
    "min_corrections_before_upload": 20,  # don't train on tiny datasets
    "auto_download_models": True,      # auto-fetch fine-tuned models when jobs finish
    "active_model_version": None,      # active fine-tuned version, or None for base model
    "report_analysis_enabled": True,   # run local report pattern analysis on startup
    # --- Learning & UI state ---
    "learning_enabled": True,          # capture edits for adaptive learning
    "learning_consent_shown": False,   # has user seen the learning consent dialog
    "disclaimer_shown": False,         # has user seen the clinical disclaimer
}


def get_default(key: str) -> Any:
    """The shipped default for *key*: the single source of truth for defaults.

    Callers that mirror settings elsewhere (the web client's preferences
    payload) read defaults from here rather than re-listing them, which is how
    the desktop and web defaults drifted apart before.
    """
    return _DEFAULTS.get(key)


class Settings:
    def __init__(self) -> None:
        self._data: dict = {}
        self._path: Path = settings_file()
        self._mtime: float = -1.0
        self.load()

    @property
    def path(self) -> Path:
        """The settings file this instance is bound to."""
        return self._path

    def load(self) -> None:
        self._data = read_json(self._path, {})
        for key, val in _DEFAULTS.items():
            if key not in self._data:
                self._data[key] = val
        self._mtime = self._current_mtime()

    def refresh(self) -> None:
        """Re-read the file if it changed on disk since the last load.

        Lets a long-lived instance be cached (avoiding a re-parse per access)
        without going stale when the other front-end, or the user, edits
        ``dictation_settings.json``. A stat is cheap; a parse is not.
        """
        if self._current_mtime() != self._mtime:
            self.load()

    def _current_mtime(self) -> float:
        try:
            return self._path.stat().st_mtime
        except OSError:
            return -1.0

    def save(self) -> None:
        write_json(self._path, self._data)
        self._mtime = self._current_mtime()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, _DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def batch_set(self, updates: dict) -> None:
        """Update multiple settings and save once."""
        for key, value in updates.items():
            self._data[key] = value
        self.save()

    def add_recent_report(self, path: str) -> None:
        recent: List[str] = self._data.get("recent_reports", [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._data["recent_reports"] = recent[:20]
        self.save()

    def get_recent_reports(self) -> List[str]:
        recent: List[str] = self._data.get("recent_reports", [])
        return [p for p in recent if Path(p).is_file()][:20]
