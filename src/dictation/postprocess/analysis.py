"""Logging and analysis of post-processing pipeline."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

from src.features.file_manager import analysis_dir

logger = logging.getLogger(__name__)

_ANALYSIS_KEEP = 200  # retain only the N most recent analysis files


def _prune_analysis_dir() -> None:
    files = sorted(analysis_dir().glob("analysis_*.json"), key=lambda p: p.stat().st_mtime)
    for f in files[:-_ANALYSIS_KEEP]:
        f.unlink(missing_ok=True)


class PipelineAnalyzer:
    """Track pipeline execution for debugging and analysis."""

    def __init__(self):
        self.start_time = time.time()
        self.stages: List[Dict[str, Any]] = []
        self.input_text = ""
        self.output_text = ""

    def record_input(self, text: str) -> None:
        """Record the input transcript."""
        self.input_text = text

    def record_stage(self, stage_name: str, output: str, duration: float) -> None:
        """Record the output of a pipeline stage."""
        self.stages.append({
            "stage": stage_name,
            "output": output,
            "duration_ms": round(duration * 1000, 2),
        })
        self.output_text = output

    def save_analysis(self, accent: str = "neutral") -> None:
        """Save the analysis to JSON."""
        elapsed = time.time() - self.start_time
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "accent": accent,
            "input": self.input_text,
            "output": self.output_text,
            "total_duration_ms": round(elapsed * 1000, 2),
            "stages": self.stages,
        }

        file_path = analysis_dir() / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            logger.debug("Analysis saved to %s", file_path)
            _prune_analysis_dir()
        except Exception as e:
            logger.error("Failed to save analysis: %s", e)
