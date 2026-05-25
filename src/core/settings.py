"""
Persistent application settings stored as JSON alongside the project root.
"""
import json
import logging
from pathlib import Path
from typing import Any, List

from src.features.file_manager import settings_file

logger = logging.getLogger(__name__)

_DEFAULTS: dict = {
    "model_size": "base",
    "language": "en",
    "vad_filter": True,
    "accent": "neutral",
    "theme": "dark",
    "font_size": 13,
    "auto_save_interval": 60,   # seconds
    "pause_threshold": 2.5,     # seconds silence → new paragraph
    "autosave_retention_days": 30,  # days to keep autosave files
    "recent_reports": [],
    "patient_info_visible": True,
    "macros_panel_visible": True,
    "last_template": "",
    "last_macro_region": "Knee",
    "window_width": 1200,
    "window_height": 760,
    "splitter_sizes": [220, 980],
}


class Settings:
    def __init__(self) -> None:
        self._data: dict = {}
        self._path: Path = settings_file()
        self.load()

    def load(self) -> None:
        if self._path.is_file():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as exc:
                logger.warning("Could not load settings: %s", exc)
                self._data = {}
        for key, val in _DEFAULTS.items():
            if key not in self._data:
                self._data[key] = val

    def save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save settings: %s", exc)

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
