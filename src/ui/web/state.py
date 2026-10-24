"""What the web server holds between requests, in one place.

This is a loopback, single-radiologist workstation with one report open at a
time: the same shape as the one desktop window. So the state is a handful of
process-wide objects, and every router reads them from here rather than
keeping its own copy. The day a second radiologist shares one server, this is
the file that turns into per-user state; nothing else has to learn about it.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Optional

from src.core.paths import settings_file
from src.core.settings import Settings
from src.dictation.asr import AsrEngine, create_engine
from src.dictation.asr.models import resolve_model
from src.features.report_release import OutstandingFindings
from src.medical import macros

#: What the browser records at and what the decoder expects. One number,
#: handed to the page in the bootstrap payload so the two cannot drift.
LIVE_SAMPLE_RATE = 16000

#: Settings keys the browser client mirrors. The *values* come from
#: ``settings._DEFAULTS``: re-listing them here is how the web and desktop
#: defaults drifted apart.
PREFERENCE_KEYS = ("model_size", "language", "vad_filter", "accent", "cleanup_level")

_settings_instance: Optional[Settings] = None


def settings() -> Settings:
    """The process-wide settings object.

    Previously a fresh ``Settings()`` per call, which re-read and re-parsed the
    JSON file from disk on every request *and* several times per page render.
    Cached and refreshed only when the file actually changes, so an edit made in
    the desktop app is still picked up. The cache is rebuilt if the settings
    *path* itself changes underneath us.
    """
    global _settings_instance
    path = settings_file()
    if _settings_instance is None or _settings_instance.path != path:
        _settings_instance = Settings()
    else:
        _settings_instance.refresh()
    return _settings_instance


def current_preferences() -> dict:
    current = settings()
    prefs = {key: current.get(key) for key in PREFERENCE_KEYS}
    macro_region = current.get("last_macro_region")
    if macro_region not in macros.REGION_ORDER and macros.REGION_ORDER:
        macro_region = macros.REGION_ORDER[0]
    prefs["macro_region"] = macro_region
    return prefs


def macros_payload() -> dict:
    """Return the current macro regions and phrases in JSON-friendly form."""
    regions = [
        {
            "name": region,
            "phrases": [
                {"label": label, "text": text}
                for label, text in macros.MACROS.get(region, [])
            ],
        }
        for region in macros.REGION_ORDER
    ]
    return {"regions": regions, "selected_region": current_preferences()["macro_region"]}


# -- speech engines ----------------------------------------------------------

_engine_lock = Lock()
#: One engine per model name: the live model and the final model coexist.
_engines: dict[str, AsrEngine] = {}


def get_engine(model_size: Optional[str] = None) -> AsrEngine:
    """Return an ASR engine for *model_size* (default: the saved setting).

    Engines are cached per model name, so the live dictation loop's fast model
    and the accurate one that re-decodes after Stop are each built once and
    then reused for the life of the process. Whisper's own weights cache holds
    two (``faster_whisper_engine._MODEL_CACHE_MAX``), which is what makes holding exactly
    this pair free.
    """
    name = resolve_model(model_size or settings().get("model_size"))
    with _engine_lock:
        if name not in _engines:
            _engines[name] = create_engine(model_size=name)
        return _engines[name]


def drop_engine(model_size: Optional[str] = None) -> None:
    """Forget one cached engine (or all), so the next request builds it afresh."""
    with _engine_lock:
        if model_size is None:
            _engines.clear()
        else:
            _engines.pop(resolve_model(model_size), None)


# -- the open report ---------------------------------------------------------

#: How many open reports' answers are remembered (tabs, reloads, past reports).
_REPORTS_KEPT = 32
_findings_lock = Lock()
_findings: "OrderedDict[str, OutstandingFindings]" = OrderedDict()


def findings_for(report_id: str) -> OutstandingFindings:
    """Which findings one report has, and which have been answered for.

    Keyed by the id the page gives each report (on load, New report, a
    template): one process-wide object carried an acknowledgement across a
    reload, a second tab or a change of patient, so the next patient's
    identical finding read as already communicated and was never audited. A
    report id the server has not seen starts with nothing answered.

    The *same* class the desktop reads (``features/report_release
    .OutstandingFindings``), so the count, the marks and the meaning of
    "acknowledged" cannot drift between the two front-ends.
    """
    if not report_id:  # a client that names no report gets no memory of one
        return OutstandingFindings()
    with _findings_lock:
        found = _findings.pop(report_id, None) or OutstandingFindings()
        _findings[report_id] = found
        while len(_findings) > _REPORTS_KEPT:
            _findings.popitem(last=False)
        return found


def forget_report(report_id: str) -> None:
    with _findings_lock:
        _findings.pop(report_id, None)

#: Bumped by every new dictation. An older one still polishing sees it change
#: and stops between sections, so its polish stops taking the CPU from the new
#: recording's live text. The desktop does the same in
#: recording_session._abandon_unfinished_session.
dictation_generation = 0


def next_dictation() -> int:
    """Start a new dictation: every older one's polish should now stop."""
    global dictation_generation
    dictation_generation += 1
    return dictation_generation
