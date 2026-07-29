"""
File and path management - centralized control of all project files.

Directory structure:
- data/temp/          Temporary audio files (cleaned on startup + after use)
- data/autosave/      Auto-saved reports (retention: 30 days)
- data/cache/         ALL rebuildable caches (SymSpell index, Whisper model
                      downloads, imaging embeddings). Safe to delete whole —
                      the app rebuilds/re-downloads on next use.
- data/macros.json    User-editable quick phrases
- templates/          Report templates (read-only)
"""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Project root — the parent of src/.

    This file lives at src/features/file_manager.py, so the root is three
    levels up: file → features/ → src/ → project root.
    """
    return Path(__file__).resolve().parents[2]

@functools.lru_cache(maxsize=1)
def _data_dir() -> Path:
    """data/ folder for all persistent/temp storage (cached after first call)"""
    path = _project_root() / "data"
    path.mkdir(exist_ok=True)
    return path

def temp_dir() -> Path:
    """Temp audio files (cleared on startup, after use)"""
    path = _data_dir() / "temp"
    path.mkdir(exist_ok=True)
    return path

def autosave_dir() -> Path:
    """Auto-saved dictation reports"""
    path = _data_dir() / "autosave"
    path.mkdir(exist_ok=True)
    return path

def cache_dir() -> Path:
    """Single home for every rebuildable on-disk cache.

    Everything in here is derived data — the SymSpell index, downloaded
    Whisper models, imaging embeddings. Deleting the whole folder is always
    safe; each cache is rebuilt or re-downloaded on next use.
    """
    path = _data_dir() / "cache"
    path.mkdir(exist_ok=True)
    return path

def whisper_cache_dir() -> Path:
    """Download root for stock Whisper models (faster-whisper).

    Keeps the multi-hundred-MB model downloads inside data/cache/ instead of
    the hidden per-user HuggingFace cache, so they are visible and deletable
    with the rest of the caches.
    """
    path = cache_dir() / "whisper"
    path.mkdir(exist_ok=True)
    return path

def onnx_asr_cache_dir(model_name: str) -> Path:
    """Download directory for one onnx-asr model (Parakeet), under data/cache/.

    Same reason as :func:`whisper_cache_dir` — keep the several-hundred-MB
    download visible and deletable with the other caches instead of hidden in
    the per-user HuggingFace cache.

    The returned directory is deliberately NOT created. onnx-asr treats an
    existing directory as "already downloaded, work offline", so pre-making it
    would turn the very first run into a missing-file error instead of a
    download. It creates the directory on the one download; every later load
    then resolves locally, which is what keeps the offline invariant.
    """
    root = cache_dir() / "onnx_asr"
    root.mkdir(exist_ok=True)
    return root / model_name.replace("/", "_")

def imaging_embeddings_dir() -> Path:
    """Cache of imaging embeddings + similarity indices (rebuildable)."""
    path = cache_dir() / "imaging_embeddings"
    path.mkdir(exist_ok=True)
    return path

def clear_cache() -> None:
    """Delete every cached artefact under data/cache/ (all rebuildable)."""
    import shutil
    try:
        shutil.rmtree(cache_dir(), ignore_errors=True)
        logger.info("Cleared cache directory")
    except Exception as exc:
        logger.warning("Cache clear failed: %s", exc)

def templates_dir() -> Path:
    """Report templates"""
    return _project_root() / "src" / "templates"

def medical_wordlist_path() -> Path:
    """Broad generic medical wordlist — the membership net (read-only static)."""
    return _project_root() / "src" / "resources" / "medical_terms.txt"

def radiology_lexicon_path() -> Path:
    """Curated radiology lexicon — the clean spelling-correction snap targets.

    Distinct from :func:`medical_wordlist_path`: that broad list answers "is this
    already a real word?", while this curated, radiology-only list is what a
    mis-transcribed word is *corrected to*, so typos snap to genuine radiology
    terms rather than to generic-wordlist junk. Read-only static resource.
    """
    return _project_root() / "src" / "resources" / "radiology_lexicon.txt"

def related_terms_path() -> Path:
    """Curated term relations + tuning for the highlight-a-word lookup.

    The half of "the neighbourhood of a word" a stem cannot find
    (pneumothorax -> chest drain); see :mod:`src.medical.term_lookup`.
    Read-only static resource.
    """
    return _project_root() / "src" / "resources" / "related_terms.json"

def medical_dict_cache_path() -> Path:
    """Cached SymSpell index built from the medical wordlist + radiology lexicon.

    Rebuilding the index from ~98k terms takes a few seconds; this on-disk
    cache (keyed by a term-count signature, see
    :func:`src.medical.medical_dict.get_symspell`) avoids paying that cost on
    every launch.
    """
    return cache_dir() / "medical_symspell.pkl"

def radiology_prompt_path() -> Path:
    """Whisper initial-prompt text fed to the model before transcription."""
    return _project_root() / "src" / "dictation" / "resources" / "radiology_prompt.txt"

def confusion_sets_path() -> Path:
    """Curated confusable-word sets (YAML) for context-aware correction.

    Bundled, version-controlled resource. Each entry groups real words a
    radiologist never means to swap ("cord"/"chord", "coarse"/"course") plus
    optional context cues; see
    :mod:`src.dictation.postprocess.context_correct`.
    """
    return (
        _project_root() / "src" / "dictation" / "postprocess"
        / "resources" / "confusion_sets.yaml"
    )

def context_seed_corpus_path() -> Path:
    """Bundled radiology sentences that seed the context n-gram model.

    Cold-start training text so the context corrector works on a fresh install
    before the user's own reports have accumulated. Version-controlled.
    """
    return (
        _project_root() / "src" / "dictation" / "postprocess"
        / "resources" / "context_seed_corpus.txt"
    )

def context_model_cache_path() -> Path:
    """Cached n-gram counts for the context corrector (rebuildable).

    Derived from the seed corpus + any learned report text; keyed by a source
    signature so it rebuilds when either changes. Safe to delete.
    """
    return cache_dir() / "context_ngram.pkl"

def learned_context_corpus_path() -> Path:
    """Append-only radiology text learned from the user's finalized reports.

    Feeds the context n-gram model so disambiguation improves with use. Stays
    on-device (never uploaded) — the reports are PHI. Written finalize-time by
    the context corrector's learning hook; separate from the shipped seed
    corpus so an app update never clobbers it.
    """
    return _data_dir() / "learned_context_corpus.txt"

def macros_file() -> Path:
    """Macros JSON file"""
    return _data_dir() / "macros.json"

def user_corrections_path() -> Path:
    """Site-added spelling-correction rules (YAML), editable via the app.

    Kept in the data dir, separate from the shipped, version-controlled
    ``corrections.yaml``, so an app update never clobbers a site's own rules.
    """
    return _data_dir() / "user_corrections.yaml"

def learned_corrections_path() -> Path:
    """Passively-learned word corrections / custom vocabulary (JSON).

    Written by ``features/adaptive_learning.py``; the canonical location lives
    here so consumers never hand-roll the path from the data dir.
    """
    return _data_dir() / "learned_corrections.json"

def audit_log_path() -> Path:
    """Append-only audit log of report actions (JSON lines).

    Written by ``features/audit_log.py``. Never auto-deleted (institutional
    retention); see that module for the retention policy.
    """
    return _data_dir() / "audit.log"

# ---------------------------------------------------------------------------
# Cloud training storage (Lightning AI integration)
# ---------------------------------------------------------------------------

def training_dir() -> Path:
    """Root for locally-staged training data (SQLite + audio clips)."""
    path = _data_dir() / "training"
    path.mkdir(exist_ok=True)
    return path

def training_audio_dir() -> Path:
    """De-identified audio clips awaiting upload to Lightning AI."""
    path = training_dir() / "audio_clips"
    path.mkdir(exist_ok=True)
    return path

def staging_db_path() -> Path:
    """SQLite database staging correction triples for cloud training."""
    return training_dir() / "staging.db"

def models_dir() -> Path:
    """Root for fine-tuned model artefacts downloaded from Lightning AI."""
    path = _data_dir() / "models"
    path.mkdir(exist_ok=True)
    return path

def fine_tuned_dir() -> Path:
    """Directory holding versioned CTranslate2 fine-tuned model folders."""
    path = models_dir() / "fine_tuned"
    path.mkdir(exist_ok=True)
    return path

def model_registry_path() -> Path:
    """JSON index of downloaded fine-tuned model versions."""
    return models_dir() / "registry.json"

def analysis_dir() -> Path:
    """Directory for local report-analysis output."""
    path = _data_dir() / "analysis"
    path.mkdir(exist_ok=True)
    return path

# ---------------------------------------------------------------------------
# Dictation evaluation corpus (accuracy/speed measurement — local only)
# ---------------------------------------------------------------------------

def eval_dir() -> Path:
    """Root for the dictation evaluation gold sets and their reports.

    Not under :func:`cache_dir` on purpose: the ``own`` set is the user's own
    voice recordings and the ``bench`` set is hand-corrected — neither is
    rebuildable, so a ``clear_cache()`` must never take them. The ``libri`` and
    ``tts`` subdirectories *are* regenerable via ``scripts/eval/build_sets.py``
    and are safe to delete individually.

    Lives under ``data/`` (gitignored), which is also what keeps the user's own
    dictated audio out of version control.
    """
    path = _data_dir() / "eval"
    path.mkdir(exist_ok=True)
    return path

def eval_set_dir(name: str) -> Path:
    """Directory for one evaluation gold set (audio + ``manifest.jsonl``).

    The name is sanitised because it reaches this function straight from a CLI
    argument, and a set name must never be able to escape the eval directory.
    """
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_")
    if not safe:
        raise ValueError(f"Invalid evaluation set name: {name!r}")
    path = eval_dir() / safe
    path.mkdir(exist_ok=True)
    return path

def eval_manifest_path(name: str) -> Path:
    """JSONL manifest listing one gold set's clips and reference transcripts."""
    return eval_set_dir(name) / "manifest.jsonl"

def eval_reports_dir() -> Path:
    """Evaluation report JSONs, one per run — the milestone-to-milestone record."""
    path = eval_dir() / "reports"
    path.mkdir(exist_ok=True)
    return path

def bench_audio_dir() -> Path:
    """Pre-existing benchmark clips shipped in the working tree."""
    return _data_dir() / "bench_audio"

def dictation_edits_path() -> Path:
    """JSONL log of post-dictation edits (the 'was dictation wrong?' signal).

    Local-only, append-only; one JSON record per line. Mined by
    ``scripts/mine_corrections.py`` and written by ``features/edit_tracking.py``.
    """
    return analysis_dir() / "dictation_edits.jsonl"

# ---------------------------------------------------------------------------
# Imaging assistant storage (scan classification, embeddings, datasets)
# ---------------------------------------------------------------------------

def imaging_dir() -> Path:
    """Root for scan assistant storage (datasets, overlays, thresholds).

    Rebuildable embedding caches live under :func:`imaging_embeddings_dir`
    (data/cache/), not here — this dir holds only non-derived data.
    """
    path = _data_dir() / "imaging"
    path.mkdir(exist_ok=True)
    return path

def imaging_training_dir() -> Path:
    """De-identified images staged for scan fine-tuning."""
    path = imaging_dir() / "training"
    path.mkdir(exist_ok=True)
    return path

def imaging_overlay_dir() -> Path:
    """Rendered overlays from Grad-CAM localization."""
    path = imaging_dir() / "overlays"
    path.mkdir(exist_ok=True)
    return path

def imaging_thresholds_path() -> Path:
    """Per-pathology confidence thresholds for abstention gate."""
    return imaging_dir() / "thresholds.json"

def settings_file() -> Path:
    """Settings JSON file"""
    return _project_root() / "dictation_settings.json"


# ---------------------------------------------------------------------------
# Report file naming
# ---------------------------------------------------------------------------

def report_filename(
    patient_info: dict, ext: str, prefix: str = "", fallback_id: str = "report"
) -> str:
    """Build a ``<prefix><patient-id>_<timestamp><ext>`` report filename.

    The desktop save dialog, the web download header, and autosave each built
    this name themselves and drifted apart. One builder, one naming scheme.

    The patient ID is sanitised: it lands in a filename (and, on the web, in a
    ``Content-Disposition`` header), so path separators, quotes, and control
    characters must not survive it.
    """
    raw = str(patient_info.get("id") or "").strip()
    pid = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in raw).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{pid or fallback_id}_{timestamp}{ext}"


# ---------------------------------------------------------------------------
# Temp WAV file management
# ---------------------------------------------------------------------------

def create_temp_wav() -> str:
    """Create a new temp WAV file. Returns absolute path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = temp_dir() / f"recording_{ts}.wav"
    return str(path)

def cleanup_temp_files() -> None:
    """Delete all files in temp/ directory."""
    try:
        count = 0
        for file in temp_dir().iterdir():
            if file.is_file():
                file.unlink()
                count += 1
        if count > 0:
            logger.info("Cleaned up %d temp audio file(s)", count)
    except Exception as exc:
        logger.warning("Temp cleanup failed: %s", exc)

# ---------------------------------------------------------------------------
# Autosave management
# ---------------------------------------------------------------------------

def cleanup_old_autosaves(retention_days: int = 30) -> None:
    """Delete autosave files older than retention_days."""
    try:
        cutoff = datetime.now() - timedelta(days=retention_days)
        count = 0
        for file in autosave_dir().iterdir():
            if file.is_file() and file.suffix == ".txt":
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < cutoff:
                    file.unlink()
                    count += 1
        if count > 0:
            logger.info(
                "Deleted %d old autosave file(s) (>%d days)",
                count,
                retention_days,
            )
    except Exception as exc:
        logger.warning("Autosave cleanup failed: %s", exc)

# ---------------------------------------------------------------------------
# Startup cleanup
# ---------------------------------------------------------------------------

def startup_cleanup(retention_days: int = 30) -> None:
    """Run all cleanup tasks on app startup."""
    cleanup_temp_files()
    cleanup_old_autosaves(retention_days=retention_days)
    _remove_legacy_temp_files()
    _remove_legacy_cache_locations()


def _remove_legacy_cache_locations() -> None:
    """Delete caches from their pre-data/cache/ homes (all rebuildable).

    Earlier versions kept the SymSpell pickle in data/resources/ and imaging
    embeddings in data/imaging/embeddings/. Both now live under
    :func:`cache_dir`; the old copies are stale derived data, safe to drop.
    """
    import shutil
    try:
        count = 0
        for legacy in (
            _data_dir() / "resources",
            _data_dir() / "imaging" / "embeddings",
        ):
            if legacy.exists():
                shutil.rmtree(legacy, ignore_errors=True)
                count += 1
        if count > 0:
            logger.info("Removed %d legacy cache location(s)", count)
    except Exception as exc:
        logger.warning("Legacy cache cleanup failed: %s", exc)

def _remove_legacy_temp_files() -> None:
    """Remove leftover temp files and directories (tmpclaude-*, etc)."""
    import shutil
    try:
        root = _project_root()
        patterns = ["tmpclaude-*", "tmp*-cwd"]
        count = 0
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    path.unlink()
                    count += 1
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    count += 1
        if count > 0:
            logger.info("Removed %d legacy temp item(s)", count)
    except Exception as exc:
        logger.warning("Legacy cleanup failed: %s", exc)
