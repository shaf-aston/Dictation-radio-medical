"""
File and path management - centralized control of all project files.

Directory structure:
- data/temp/          Temporary audio files (cleaned on startup + after use)
- data/autosave/      Auto-saved reports (retention: 30 days)
- data/resources/     Resource files (medical wordlist, etc)
- data/macros.json    User-editable quick phrases
- templates/          Report templates (read-only)
"""
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

def resources_dir() -> Path:
    """Resource files (medical wordlists, etc)"""
    path = _data_dir() / "resources"
    path.mkdir(exist_ok=True)
    return path

def templates_dir() -> Path:
    """Report templates"""
    return _project_root() / "src" / "templates"

def medical_wordlist_path() -> Path:
    """Bundled medical terminology wordlist (read-only static resource)."""
    return _project_root() / "src" / "resources" / "medical_terms.txt"

def radiology_prompt_path() -> Path:
    """Whisper initial-prompt text fed to the model before transcription."""
    return _project_root() / "src" / "dictation" / "resources" / "radiology_prompt.txt"

def macros_file() -> Path:
    """Macros JSON file"""
    return _data_dir() / "macros.json"

def settings_file() -> Path:
    """Settings JSON file"""
    return _project_root() / "dictation_settings.json"

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
