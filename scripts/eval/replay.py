"""Replay a recording through the real live loop and time when words stick.

The rest of this harness decodes a whole file in one ``engine.transcribe()``
call (``run_eval.EngineRunner``), which never touches ``segmenter.cut_chunks``,
``ChunkLedger`` or ``LiveSession``. That measures accuracy honestly and says
nothing at all about the number the radiologist complains about: how long a
word waits between being spoken and becoming permanent.

Worse, the two disagree. Handed a whole file, ``cut_chunks`` takes the LATEST
pause up to ``soft_max_sec``; the live loop only ever sees the audio that has
arrived, so the same function takes the FIRST pause past ``min_sec``. An
offline harness therefore measures a chunk policy the live app does not run.

This drives the real ``LiveSession`` on the same schedule the WebSocket uses
and reports commit latency next to the text, so a chunk-policy change can be
scored on speed and accuracy at once instead of one at a time.

Wall-clock honesty: by default the replay runs flat out, so latency is
reported in AUDIO seconds (how far behind the microphone a word was when it
became permanent), which is a property of the policy and not of the machine.
``--realtime`` instead feeds audio at true speaking pace and also reports wall
seconds, which is what a person actually experiences and which a loaded
machine will make worse.

Run:
    python -m scripts.eval.replay --set tts --limit 5
    python -m scripts.eval.replay --audio data/bench_audio/chest_long.wav --realtime
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from scripts.eval import corpus
from scripts.eval.metrics import term_error_rate, word_error_rate
from src.core.settings import Settings
from src.dictation.asr.factory import DEFAULT_ENGINE, create_engine, model_kwargs
from src.dictation.stream.live_session import LiveSession
from src.dictation.stream.rules import build_context_prompt
from src.dictation.stream.segmenter import ChunkPolicy
from src.dictation.stream.vad import SAMPLE_RATE


@dataclass
class ReplayResult:
    """One recording's worth of live-loop behaviour."""

    name: str
    audio_sec: float
    chunks: int
    final_text: str
    # Audio seconds between a word being spoken and it becoming permanent.
    commit_lag: List[float] = field(default_factory=list)
    wall_sec: float = 0.0
    polish_sec: float = 0.0
    polished: int = 0
    first_word_sec: Optional[float] = None
    wer: Optional[float] = None
    term_error: Optional[float] = None

    def percentile(self, p: float) -> float:
        if not self.commit_lag:
            return float("nan")
        ordered = sorted(self.commit_lag)
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    def summary(self) -> Dict[str, float]:
        return {
            "audio_sec": round(self.audio_sec, 2),
            "wall_sec": round(self.wall_sec, 2),
            "polish_sec": round(self.polish_sec, 2),
            "polished": self.polished,
            "chunks": self.chunks,
            "words": len(self.commit_lag),
            "first_word_sec": (
                round(self.first_word_sec, 2) if self.first_word_sec is not None else None
            ),
            "lag_p50": round(self.percentile(0.50), 2),
            "lag_p90": round(self.percentile(0.90), 2),
            "lag_max": round(max(self.commit_lag), 2) if self.commit_lag else None,
        }


def load_audio(path: Path) -> np.ndarray:
    """Load *path* as mono float32 at 16 kHz.

    Always through ``decode_audio``. The eval clips are 22050 Hz and the live
    path is 16 kHz throughout; resampling by picking nearest samples aliases
    them badly enough to send WER from ~10% to ~62% and make every clip look
    truncated. That trap cost an hour once already, see
    docs/dictation-speed-review.md.
    """
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def _policy_from(settings: Settings, overrides: Optional[Dict[str, float]] = None) -> ChunkPolicy:
    """The shipped policy, with any --chunk-* override applied on top.

    Overriding here rather than by editing dictation_settings.json keeps an
    A/B run from mutating the user's own configuration, and keeps the two arms
    of a comparison in one command each.
    """
    values = {
        "min_sec": float(settings.get("chunk_min_sec")),
        "soft_max_sec": float(settings.get("chunk_soft_max_sec")),
        "force_cut_sec": float(settings.get("chunk_force_cut_sec")),
        "trailing_silence_sec": float(settings.get("chunk_trailing_silence_sec")),
    }
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return ChunkPolicy(**values)


def replay(
    audio: np.ndarray,
    name: str,
    settings: Settings,
    *,
    engine_name: str = DEFAULT_ENGINE,
    live_model: str = "",
    final_model: str = "",
    final_engine_name: str = "",
    polish_ceiling: Optional[float] = None,
    realtime: bool = False,
    policy: Optional[ChunkPolicy] = None,
) -> ReplayResult:
    """Drive one recording through the live loop exactly as the socket does."""
    sr = SAMPLE_RATE
    cycle_sec = float(settings.get("live_cycle_sec"))
    block = max(1, int(cycle_sec * sr))

    session = LiveSession(
        create_engine(engine_name, **model_kwargs(engine_name, live_model)),
        create_engine(final_engine_name or engine_name,
                      **model_kwargs(final_engine_name or engine_name, final_model)),
        language=str(settings.get("language")),
        accent=str(settings.get("accent")),
        cleanup_level=str(settings.get("cleanup_level")),
        policy=policy or _policy_from(settings),
        pause_threshold=float(settings.get("pause_threshold")),
        live_beam_size=int(settings.get("live_beam_size")),
        final_beam_size=int(settings.get("final_beam_size")),
        preview_max_lag_sec=float(settings.get("preview_max_lag_sec")),
        preview_min_tail_sec=float(settings.get("preview_min_tail_sec")),
        polish_confidence_ceiling=(
            polish_ceiling if polish_ceiling is not None
            else float(settings.get("polish_confidence_ceiling"))
        ),
        uncertain_word_confidence=float(settings.get("uncertain_word_confidence")),
        initial_prompt=build_context_prompt(),
        sr=sr,
    )

    result = ReplayResult(name=name, audio_sec=len(audio) / sr, chunks=0, final_text="")
    committed_words = 0
    frontier = 0.0
    fed = 0
    wall_start = time.perf_counter()

    while fed < len(audio):
        if realtime:
            # Feed every block that is DUE by the wall clock, not one per
            # cycle. The socket keeps draining PCM while a decode runs
            # (web_app.py runs session.cycle in a background task), so feeding
            # one block per cycle would model a slower app than the real one.
            due_samples = min(len(audio), int((time.perf_counter() - wall_start) * sr))
            if due_samples <= fed:
                time.sleep(0.01)
                continue
            session.feed(audio[fed:due_samples])
            fed = due_samples
        else:
            session.feed(audio[fed : fed + block])
            fed = min(len(audio), fed + block)

        update = session.cycle()
        if update is None:
            continue

        # A word's lag is how much audio had arrived when it became permanent,
        # minus roughly when it was spoken. The ledger freezes whole chunks, so
        # the spoken time is approximated by spreading the new words evenly
        # across the audio the chunk covered. Exact per-word timings would mean
        # threading the decoder's word offsets through the ledger: a change to
        # the app, not to its instrument.
        fresh = len(update.committed.split()) - committed_words
        if fresh > 0:
            now_audio = session.audio_sec
            new_frontier = session.committed_sec
            span = max(0.0, new_frontier - frontier)
            for i in range(fresh):
                spoken_at = frontier + span * (i + 0.5) / fresh
                result.commit_lag.append(max(0.0, now_audio - spoken_at))
            committed_words += fresh
            frontier = new_frontier
            if result.first_word_sec is None:
                result.first_word_sec = (
                    time.perf_counter() - wall_start if realtime else now_audio
                )

    session.close_open_tail_fast()
    polish_start = time.perf_counter()
    result.final_text = session.finalize()
    result.polish_sec = time.perf_counter() - polish_start
    result.polished = session.chunks_polished
    result.chunks = session.chunks_decoded
    result.wall_sec = time.perf_counter() - wall_start
    return result


def _clips(args: argparse.Namespace) -> List[Path]:
    if args.audio:
        return [Path(args.audio)]
    root = Path("data") / "eval" / args.set
    clips = sorted(root.glob("*.wav"))
    if not clips:
        raise SystemExit(
            f"no audio in {root}. For the 'own' set, record it first:\n"
            f"    python -m scripts.eval.build_sets --set own --record"
        )
    return clips[: args.limit] if args.limit else clips


def _references(set_name: str) -> Dict[str, str]:
    """Ground truth by clip stem, empty when the set has none to offer.

    A latency number on its own is worth nothing here: a chunk policy can
    always commit sooner by cutting mid-word, and the whole reason this
    project measures at all is that a report can post a good WER while
    mangling every anatomical word. So the harness refuses to report speed
    without reporting what it cost, whenever a reference exists.
    """
    try:
        return {c.audio_path.stem: c.reference for c in corpus.load_set(set_name)}
    except Exception as exc:  # a missing or unreviewed manifest is not fatal
        print(f"(no references for set {set_name!r}: {exc})")
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="tts", help="gold set under data/eval/")
    ap.add_argument("--audio", help="one WAV instead of a set")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only")
    ap.add_argument("--realtime", action="store_true",
                    help="feed at true speaking pace and report wall latency")
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--live-model", default="", help="override live_model_size")
    ap.add_argument("--final-model", default="", help="override model_size")
    ap.add_argument("--final-engine", default="", help="engine for the polish (default: --engine)")
    ap.add_argument("--polish-ceiling", type=float, help="override polish_confidence_ceiling")
    ap.add_argument("--out-dir", default=str(Path("data") / "eval" / "reports"))
    ap.add_argument("--label", default="", help="name this run in the JSON output")
    ap.add_argument("--chunk-min", type=float, help="override chunk_min_sec")
    ap.add_argument("--chunk-soft-max", type=float, help="override chunk_soft_max_sec")
    ap.add_argument("--chunk-force-cut", type=float, help="override chunk_force_cut_sec")
    ap.add_argument("--trailing-silence", type=float,
                    help="override chunk_trailing_silence_sec; a huge value "
                         "effectively turns the trailing-pause cut off")
    args = ap.parse_args()

    settings = Settings()
    final_engine = args.final_engine or args.engine
    # The model settings name Whisper sizes; Parakeet has no such model, so it
    # keeps its own default unless one is passed on the command line.
    live_model = args.live_model or (
        "" if args.engine == "parakeet" else str(settings.get("live_model_size")))
    final_model = args.final_model or (
        "" if final_engine == "parakeet" else str(settings.get("model_size")))
    policy = _policy_from(settings, {
        "min_sec": args.chunk_min,
        "soft_max_sec": args.chunk_soft_max,
        "force_cut_sec": args.chunk_force_cut,
        "trailing_silence_sec": args.trailing_silence,
    })

    ceiling = (args.polish_ceiling if args.polish_ceiling is not None
               else float(settings.get("polish_confidence_ceiling")))
    print(f"engine={args.engine} live={live_model} final_engine={final_engine} "
          f"final={final_model} polish_ceiling={ceiling}")
    print(f"policy min={policy.min_sec} soft={policy.soft_max_sec} "
          f"force={policy.force_cut_sec} trailing={policy.trailing_silence_sec}")
    print(f"mode={'realtime' if args.realtime else 'flat out (audio-seconds lag)'}\n")

    references = {} if args.audio else _references(args.set)
    lexicon = []
    if references:
        from src.medical.medical_dict import get_correction_targets
        lexicon = list(get_correction_targets())

    results = []
    for clip in _clips(args):
        res = replay(
            load_audio(clip), clip.stem, settings,
            engine_name=args.engine, live_model=live_model,
            final_model=final_model, final_engine_name=final_engine,
            polish_ceiling=ceiling, realtime=args.realtime, policy=policy,
        )
        results.append(res)
        s = res.summary()
        ref = references.get(res.name, "")
        if ref:
            res.wer = word_error_rate(ref, res.final_text).wer
            res.term_error = term_error_rate(ref, res.final_text, lexicon).error_rate
        scored = (f"  wer {res.wer:.1%}  term {res.term_error:.1%}"
                  if res.wer is not None else "")
        print(f"{res.name:<16} audio {s['audio_sec']:>6.2f}s  chunks {s['chunks']:>2}  "
              f"first {s['first_word_sec']}  p50 {s['lag_p50']}  p90 {s['lag_p90']}  "
              f"max {s['lag_max']}{scored}")

    if not results:
        return

    every_lag = [x for r in results for x in r.commit_lag]
    print(f"\n{len(results)} clips, {len(every_lag)} committed words")
    if every_lag:
        ordered = sorted(every_lag)
        print(f"commit lag  p50 {statistics.median(ordered):.2f}s  "
              f"p90 {ordered[int(0.9 * len(ordered))]:.2f}s  max {max(ordered):.2f}s")
    print(f"mean chunk  {sum(r.audio_sec for r in results) / max(1, sum(r.chunks for r in results)):.2f}s")
    scored_runs = [r for r in results if r.term_error is not None]
    if scored_runs:
        # The veto metric. A latency win that moves this up is not a win.
        print(f"accuracy    wer {statistics.mean(r.wer for r in scored_runs):.2%}  "
              f"medical-term error {statistics.mean(r.term_error for r in scored_runs):.2%}"
              f"   ({len(scored_runs)} scored)")
    else:
        print("accuracy    NOT SCORED: no reference for this audio")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or "replay"
    dest = out / f"{label}.replay.json"
    dest.write_text(json.dumps({
        "label": label,
        "engine": args.engine,
        "final_engine": final_engine,
        "polish_ceiling": ceiling,
        "live_model": live_model,
        "final_model": final_model,
        "realtime": args.realtime,
        "policy": {
            "min_sec": policy.min_sec, "soft_max_sec": policy.soft_max_sec,
            "force_cut_sec": policy.force_cut_sec,
            "trailing_silence_sec": policy.trailing_silence_sec,
        },
        "clips": [{"name": r.name, **r.summary(), "wer": r.wer,
                   "term_error": r.term_error, "text": r.final_text} for r in results],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
