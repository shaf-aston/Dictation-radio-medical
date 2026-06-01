"""Logging and analysis of post-processing pipeline."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_ANALYSIS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "analysis"


def _analysis_dir() -> Path:
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    return _ANALYSIS_DIR


class PipelineAnalyzer:
    """Track pipeline execution for debugging and analysis."""

    def __init__(self):
        self.start_time = time.time()
        self.stages: List[Dict[str, Any]] = []
        self.input_text = ""
        self.output_text = ""

    def record_input(self, text: str):
        """Record the input transcript."""
        self.input_text = text

    def record_stage(self, stage_name: str, output: str, duration: float):
        """Record the output of a pipeline stage."""
        self.stages.append({
            "stage": stage_name,
            "output": output,
            "duration_ms": round(duration * 1000, 2),
        })
        self.output_text = output

    def save_analysis(self, accent: str = "neutral"):
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

        file_path = _analysis_dir() / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        try:
            with open(file_path, "w") as f:
                json.dump(analysis, f, indent=2)
            logger.debug("Analysis saved to %s", file_path)
        except Exception as e:
            logger.error("Failed to save analysis: %s", e)
