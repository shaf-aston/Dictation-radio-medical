"""Measure dictation accuracy and speed against the gold sets.

    python -m scripts.eval.run_eval --set tts --set bench
    python -m scripts.eval.run_eval --set tts --baseline data/eval/reports/m0.json
    python -m scripts.eval.run_eval --set tts --label m1-asr-port

Transcribes every clip in the chosen sets, runs the post-processing pipeline on
the result, and scores three things the project could not previously see:

* **WER** — the general yardstick.
* **medical-term error rate** — WER restricted to curated radiology-lexicon
  terms. A report can post a decent WER while mangling every anatomical word
  in it, and this is the number that catches that.
* **false-correction rate** — of the edits the post-processing pipeline made,
  the share that took a *correct* word and made it wrong. The pipeline's own
  scoreboard, and the reason raw and processed text are both kept per clip.

Plus a real-time factor per clip (decode seconds / audio seconds), which is the
speed number every later milestone is judged on.

Everything runs locally. Reports land in ``data/eval/reports/``; pass
``--baseline`` to print a signed delta against an earlier run so a regression is
loud rather than something you have to eyeball across two files.
"""

from __future__ import annotations

import argparse
import logging
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scripts.eval.corpus import Clip, KNOWN_SETS, load_set
from scripts.eval.metrics import correction_effect, term_error_rate, word_error_rate
from src.core.json_store import write_json
from src.core.logging_setup import setup_logging
from src.core.settings import Settings
from src.features.file_manager import eval_reports_dir

logger = logging.getLogger(__name__)

#: Metrics where a *lower* number is better — drives the direction of the
#: baseline delta arrows so a regression can never read as an improvement.
_LOWER_IS_BETTER = {"wer", "term_error_rate", "false_correction_rate", "rtf",
                    "rtf_median", "rtf_p95", "decode_sec_total"}


# ---------------------------------------------------------------------------
# Per-clip result
# ---------------------------------------------------------------------------

@dataclass
class ClipResult:
    """One clip's scores, plus the texts needed to explain them."""

    clip_id: str
    audio_sec: float
    decode_sec: float
    postprocess_sec: float
    rtf: float
    wer: Dict[str, Any]
    terms: Dict[str, Any]
    correction: Dict[str, Any]
    reference: str = ""
    raw: str = ""
    processed: str = ""

    def as_dict(self, include_text: bool) -> dict:
        data = asdict(self)
        if not include_text:
            for key in ("reference", "raw", "processed"):
                data.pop(key)
        return data


@dataclass
class SetResult:
    """Aggregate over one gold set."""

    set_name: str
    clips: int
    audio_sec: float
    summary: Dict[str, float] = field(default_factory=dict)
    clip_results: List[ClipResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Transcription seam
# ---------------------------------------------------------------------------

class WhisperRunner:
    """Transcribes a clip with the current faster-whisper engine.

    Deliberately narrow: ``transcribe(path) -> (text, seconds)``. M1 replaces
    the body with the ``AsrEngine`` port without the evaluator changing, which
    is what lets M1 prove itself by producing numbers identical to M0.
    """

    def __init__(self, model_size: str, beam_size: int) -> None:
        from src.dictation.transcriber import Transcriber

        self.model_size = model_size
        self.beam_size = beam_size
        self._transcriber = Transcriber(model_size=model_size)

    def describe(self) -> Dict[str, Any]:
        return {
            "engine": "faster-whisper",
            "model_size": self.model_size,
            "beam_size": self.beam_size,
            "compute_type": getattr(self._transcriber, "compute_type", None),
        }

    def warmup(self) -> None:
        """Load the model before timing anything.

        Model load is seconds; folding it into the first clip would make that
        clip's real-time factor a fiction.
        """
        self._transcriber.preload()

    def transcribe(self, path: Path) -> tuple[str, float]:
        start = time.perf_counter()
        text, _ = self._transcriber.transcribe(str(path), beam_size=self.beam_size)
        return text, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_clip(
    clip: Clip, runner: WhisperRunner, lexicon: Sequence[str], accent: str,
    cleanup_level: str,
) -> ClipResult:
    """Transcribe, post-process, and score one clip."""
    from src.dictation.postprocess.pipeline import postprocess_transcript

    raw, decode_sec = runner.transcribe(clip.audio_path)

    pp_start = time.perf_counter()
    processed = postprocess_transcript(raw, accent=accent, cleanup_level=cleanup_level)
    pp_sec = time.perf_counter() - pp_start

    audio_sec = clip.duration_sec or 0.0
    return ClipResult(
        clip_id=clip.clip_id,
        audio_sec=round(audio_sec, 3),
        decode_sec=round(decode_sec, 3),
        postprocess_sec=round(pp_sec, 4),
        rtf=round(decode_sec / audio_sec, 4) if audio_sec else 0.0,
        wer=word_error_rate(clip.reference, processed).as_dict(),
        terms=term_error_rate(clip.reference, processed, lexicon).as_dict(),
        correction=correction_effect(clip.reference, raw, processed).as_dict(),
        reference=clip.reference,
        raw=raw,
        processed=processed,
    )


def summarise(results: List[ClipResult]) -> Dict[str, float]:
    """Aggregate clip results into the headline numbers.

    WER and term error rate are pooled over total word counts rather than
    averaged per clip — a one-sentence clip must not carry the same weight as a
    ninety-word report. The real-time factor *is* a per-clip mean (plus p95),
    because that is how latency is actually experienced.
    """
    if not results:
        return {}

    def total(section: str, key: str) -> int:
        return sum(int(r.__dict__[section].get(key, 0)) for r in results)

    ref_words = total("wer", "ref_words")
    errors = sum(
        total("wer", k) for k in ("substitutions", "deletions", "insertions")
    )
    term_tokens = total("terms", "term_tokens")
    term_errors = total("terms", "term_errors")
    true_fixes = total("correction", "true_fixes")
    false_corrections = total("correction", "false_corrections")
    meaningful = true_fixes + false_corrections

    rtfs = sorted(r.rtf for r in results if r.rtf > 0)
    return {
        "wer": round(errors / ref_words, 5) if ref_words else 0.0,
        "term_error_rate": round(term_errors / term_tokens, 5) if term_tokens else 0.0,
        "false_correction_rate": (
            round(false_corrections / meaningful, 5) if meaningful else 0.0
        ),
        "true_fixes": true_fixes,
        "false_corrections": false_corrections,
        "net_gain": true_fixes - false_corrections,
        # Wall-clock speed on this machine is genuinely noisy — the same 30
        # clips have measured a 0.74 and a 1.00 mean RTF on consecutive runs,
        # with single clips ranging 0.5-2.4. The median is the number to compare
        # across milestones; the mean and p95 are kept because a long tail is
        # exactly what a radiologist feels as a stall. For a *reproducible*
        # speed gate use the redundancy ratio (decoded seconds / audio seconds),
        # which is a count rather than a timing.
        "rtf": round(statistics.mean(rtfs), 4) if rtfs else 0.0,
        "rtf_median": round(statistics.median(rtfs), 4) if rtfs else 0.0,
        "decode_sec_total": round(sum(r.decode_sec for r in results), 2),
        "rtf_p95": round(rtfs[min(len(rtfs) - 1, int(len(rtfs) * 0.95))], 4) if rtfs else 0.0,
        "ref_words": ref_words,
        "term_tokens": term_tokens,
        "postprocess_ms_mean": round(
            statistics.mean(r.postprocess_sec for r in results) * 1000, 2
        ),
    }


def evaluate_sets(
    set_names: List[str], runner: WhisperRunner, accent: str, cleanup_level: str,
) -> List[SetResult]:
    from src.medical.medical_dict import get_correction_targets

    lexicon = list(get_correction_targets())
    logger.info("Scoring medical terms against %d curated lexicon entries", len(lexicon))

    out: List[SetResult] = []
    for name in set_names:
        clips = load_set(name)
        logger.info("Set '%s': %d clip(s) - %s", name, len(clips), KNOWN_SETS.get(name, ""))
        results = [
            evaluate_clip(c, runner, lexicon, accent, cleanup_level) for c in clips
        ]
        for i, r in enumerate(results, 1):
            logger.info(
                "  [%d/%d] %s  wer=%.3f  term_er=%.3f  rtf=%.2f",
                i, len(results), r.clip_id, r.wer["wer"], r.terms["term_error_rate"], r.rtf,
            )
        out.append(SetResult(
            set_name=name,
            clips=len(results),
            audio_sec=round(sum(r.audio_sec for r in results), 2),
            summary=summarise(results),
            clip_results=results,
        ))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(
    sets: List[SetResult], runner: WhisperRunner, accent: str, cleanup_level: str,
    label: str, include_text: bool,
) -> dict:
    return {
        "label": label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "config": {
            **runner.describe(),
            "accent": accent,
            "cleanup_level": cleanup_level,
        },
        "sets": {
            s.set_name: {
                "clips": s.clips,
                "audio_sec": s.audio_sec,
                "summary": s.summary,
                "clip_results": [r.as_dict(include_text) for r in s.clip_results],
            }
            for s in sets
        },
    }


def print_summary(report: dict, baseline: dict | None) -> None:
    """Print the headline table, with a signed delta when a baseline is given.

    ASCII only: the Windows console defaults to cp1252 and dies on box-drawing
    characters, which would take the whole run down after the expensive part had
    already finished.
    """
    print()
    for set_name, data in report["sets"].items():
        summary = data["summary"]
        base = (baseline or {}).get("sets", {}).get(set_name, {}).get("summary", {})
        note = " (synthetic audio - vocabulary only)" if set_name == "tts" else ""
        print(f"== {set_name}{note}: "
              f"{data['clips']} clips, {data['audio_sec'] / 60:.1f} min audio")
        for key in ("wer", "term_error_rate", "false_correction_rate",
                    "rtf_median", "rtf", "rtf_p95"):
            value = summary.get(key, 0.0)
            line = f"   {key:<22} {value:>8.4f}"
            if key in base:
                delta = value - base[key]
                better = (delta < 0) if key in _LOWER_IS_BETTER else (delta > 0)
                mark = "same" if abs(delta) < 1e-6 else ("BETTER" if better else "WORSE")
                line += f"   {delta:+.4f} {mark}"
            print(line)
        print(f"   {'true fixes / false':<22} "
              f"{summary.get('true_fixes', 0):>4} / {summary.get('false_corrections', 0)}"
              f"   net {summary.get('net_gain', 0):+d}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate dictation accuracy and speed")
    parser.add_argument("--set", dest="sets", action="append", required=True,
                        choices=list(KNOWN_SETS), help="gold set to score (repeatable)")
    parser.add_argument("--label", default="", help="name for this run, e.g. 'm0-baseline'")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="an earlier report JSON to diff against")
    parser.add_argument("--model", default="", help="override the model size from settings")
    parser.add_argument("--beam-size", type=int, default=0, help="override beam width")
    parser.add_argument("--cleanup-level", default="", choices=["", "soft", "medium", "hard"],
                        help="override the post-processing level from settings")
    parser.add_argument("--no-text", action="store_true",
                        help="omit per-clip transcripts from the report (smaller file)")
    args = parser.parse_args(argv)

    setup_logging()
    settings = Settings()
    model = args.model or settings.get("model_size", "base")
    beam = args.beam_size or int(settings.get("beam_size", 5))
    accent = settings.get("accent", "neutral")
    cleanup_level = args.cleanup_level or settings.get("cleanup_level", "medium")

    baseline = None
    if args.baseline:
        if not args.baseline.exists():
            logger.error("Baseline report not found: %s", args.baseline)
            return 1
        import json

        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    runner = WhisperRunner(model_size=model, beam_size=beam)
    logger.info("Loading %s (beam=%d, cleanup=%s)...", model, beam, cleanup_level)
    runner.warmup()

    try:
        sets = evaluate_sets(args.sets, runner, accent, cleanup_level)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    label = args.label or f"{model}-beam{beam}"
    report = build_report(sets, runner, accent, cleanup_level, label, not args.no_text)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = eval_reports_dir() / f"{label.replace('/', '_')}_{stamp}.json"
    write_json(out_path, report)

    print_summary(report, baseline)
    print(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
