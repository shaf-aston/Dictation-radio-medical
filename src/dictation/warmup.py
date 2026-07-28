"""Background pre-warming of the heavy dictation singletons.

Two things dominate first-use latency, and both are built lazily on the *first*
spoken chunk — so the app feels frozen at exactly the moment the radiologist
starts talking:

* the SymSpell medical-spelling index — ~1.3 s to build (or unpickle) on every
  launch, paid inside the fuzzy-match post-process stage;
* the Whisper model — a few seconds on the first ``transcribe()``.

This module moves that cost to app startup, on a background thread, so first use
is instant. It only calls the *existing* lazy loaders — it owns no data and does
no I/O of its own, so it adds no new failure mode: any warmer that raises is
logged and skipped, and the ordinary lazy path still runs on first use.

Design notes (loose coupling / SRP):

* A **warmer** is just ``(name, callable)``. The set is data, not hard-coded
  control flow — front-ends compose the list they need (see
  :func:`postprocess_warmers` and :func:`transcriber_warmer`).
* No Qt, no ``settings`` import here. The caller resolves the model size and
  passes it in, so this module stays a pure, testable utility usable from the
  desktop GUI, the web app, or the CLI benchmark alike.

Front-ends call :func:`warm_up_async` once at startup. Tests and the CLI
benchmark call :func:`warm_up` to block until warm.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Iterable, List, Optional, Tuple

from src.core import perf

logger = logging.getLogger(__name__)

# A unit of pre-warming: a label (for perf timings/logs) and a zero-arg thunk
# that triggers one lazy build. Kept as plain data so callers can compose them.
Warmer = Tuple[str, Callable[[], object]]

_start_lock = threading.Lock()
_started = False


def _warm_symspell() -> None:
    from src.medical import medical_dict

    medical_dict.get_symspell()


def _warm_terminology() -> None:
    from src.dictation.postprocess.terminology import apply_terminology

    apply_terminology("warmup")


def _warm_context_model() -> None:
    # Builds (or unpickles) the n-gram context model used by the real-word
    # confusion corrector. Tiny (~4 ms), but warming it keeps the first spoken
    # chunk off the build path along with the other singletons.
    from src.dictation.postprocess import context_model

    context_model.get_context_model()


def _warm_english_guard() -> None:
    # Loads pyspellchecker's offline English dictionary (~200 ms on first use),
    # which the fuzzy stage consults to tell a real word from a typo. Left lazy
    # it lands on the first real chunk even after the SymSpell index is warm.
    from src.dictation.postprocess.medical_dict_match import _english_known

    _english_known("warmup")


def _warm_term_lookup() -> None:
    # Mines the lexicon's stem families and reads the curated relations file
    # (~150 ms) for the highlight-a-word panel. Warmed here, with the rest,
    # because both front-ends already call this list — a warmer added only to
    # one of them is how the two drift apart.
    from src.medical import term_lookup

    term_lookup.warm()


def postprocess_warmers() -> List[Warmer]:
    """Warmers for the pure-Python singletons — no model, no settings needed.

    Always safe to run: these are device-agnostic index builds, wanted by every
    front-end before the radiologist's first word (or first highlight).
    """
    return [
        ("medical_symspell", _warm_symspell),
        ("english_guard", _warm_english_guard),
        ("terminology", _warm_terminology),
        ("context_model", _warm_context_model),
        ("term_lookup", _warm_term_lookup),
    ]


def transcriber_warmer(model_size: str, model_path: Optional[str] = None) -> Warmer:
    """A warmer that loads the Whisper model for *model_size* ahead of first use.

    Front-ends resolve the active model from settings and pass it here, so this
    module never reaches up into settings itself.
    """

    def _warm() -> None:
        from src.dictation.asr import create_engine

        create_engine(model_size=model_size, model_path=model_path).preload()

    label = f"whisper_model[{model_path or model_size}]"
    return (label, _warm)


def _run_warmer(name: str, fn: Callable[[], object]) -> None:
    """Run one warmer, timed under ``warmup.<name>`` and guarded."""
    try:
        with perf.stage(f"warmup.{name}"):
            fn()
        logger.debug("Pre-warmed %s", name)
    except Exception as exc:  # never let a warmer break startup
        logger.debug("Warmup skipped for %s: %s", name, exc)


def warm_up(warmers: Optional[Iterable[Warmer]] = None) -> None:
    """Run every warmer once — concurrently — timed under ``warmup.<name>``.

    Each warmer runs on its own daemon thread so the multi-second Whisper model
    load overlaps the pure-Python index builds instead of queuing behind them:
    total warm time becomes the slowest single warmer, not their sum (≈11s vs
    ≈14.5s serial when the model load and the SymSpell build are both present).
    The warmers touch independent singletons, each guarded by its own lock
    (`_MODEL_CACHE_LOCK`, `medical_dict._LOCK`) or idempotent, and `perf.record`
    is lock-guarded — so concurrent execution is safe.

    Blocks until every warmer has finished. A warmer that raises is logged and
    skipped — pre-warming is a latency optimisation, never a correctness
    dependency, so it must never break startup.
    """
    resolved = postprocess_warmers() if warmers is None else list(warmers)
    threads = [
        threading.Thread(
            target=_run_warmer, args=(name, fn),
            name=f"warmup-{name}", daemon=True,
        )
        for name, fn in resolved
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def warm_up_async(
    warmers: Optional[Iterable[Warmer]] = None,
) -> Optional[threading.Thread]:
    """Warm up on a daemon thread. Idempotent — only the first call starts it.

    Returns the started thread, or ``None`` if warming has already begun this
    process (so a second front-end call, or a test, is a harmless no-op).
    """
    global _started
    with _start_lock:
        if _started:
            return None
        _started = True
    resolved = postprocess_warmers() if warmers is None else list(warmers)
    thread = threading.Thread(
        target=warm_up, args=(resolved,), name="dictation-warmup", daemon=True,
    )
    thread.start()
    return thread


def reset() -> None:
    """Clear the once-only guard so :func:`warm_up_async` can start again.

    For tests only — production warms exactly once per process.
    """
    global _started
    with _start_lock:
        _started = False
