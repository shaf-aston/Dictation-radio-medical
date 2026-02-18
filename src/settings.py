"""
Persistent application settings stored as JSON alongside the project root.
"""
import os
import json
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

_DEFAULTS: dict = {
    "model_size": "base",
    "language": "en",
    "vad_filter": True,
    "theme": "dark",
    "font_size": 13,
    "auto_save_interval": 60,   # seconds
    "pause_threshold": 2.5,     # seconds silence → new paragraph
    "recent_reports": [],
    "patient_info_visible": True,
    "macros_panel_visible": True,
    "last_template": "",
    "last_macro_region": "Knee",
    "window_width": 1200,
    "window_height": 760,
    "splitter_sizes": [220, 980],
}


def _settings_path() -> str:
    # Store beside the project root (one level up from src/)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "dictation_settings.json")


class Settings:
    def __init__(self) -> None:
        self._data: dict = {}
        self.load()

    def load(self) -> None:
        path = _settings_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as exc:
                logger.warning("Could not load settings: %s", exc)
                self._data = {}
        # Fill missing keys with defaults (non-destructive)
        for key, val in _DEFAULTS.items():
            if key not in self._data:
                self._data[key] = val

    def save(self) -> None:
        path = _settings_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save settings: %s", exc)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, _DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def add_recent_report(self, path: str) -> None:
        recent: List[str] = self._data.get("recent_reports", [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._data["recent_reports"] = recent[:20]

    def get_recent_reports(self) -> List[str]:
        recent: List[str] = self._data.get("recent_reports", [])
        return [p for p in recent if os.path.isfile(p)][:20]
