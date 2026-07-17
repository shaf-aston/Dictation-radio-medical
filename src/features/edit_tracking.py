"""Dictation edit tracking — the "was the dictation itself wrong?" signal.

When a radiologist edits the text *after* dictation has finished, that edit is
the highest-quality evidence the dictation + correction pipeline got something
wrong: the clinician literally fixed it. Nothing captured this in an analysable
form. The existing single-word passive learner
(:meth:`adaptive_learning.track_edit`) only kept same-length one-word swaps and
fed them straight into the live correction map — it dropped multi-word fixes,
insertions and deletions, and left no record to mine.

This module snapshots the dictation OUTPUT and, when the report is committed
(export, clear, close, or the next recording), diffs it against the DELIVERED
text and logs each change as a structured record to
``data/analysis/dictation_edits.jsonl``. The miner
(:mod:`scripts.mine_corrections`) reads it as a ranked source of pipeline gaps.

Privacy & scope:

* **Local only.** Records never leave the device through this path (same model
  as ``learned_corrections.json``). The cloud uploader does not read this file.
* **Consent-aligned.** Gated on the ``learning_enabled`` setting (default on);
  turning it off stops all capture.
* **Minimal.** Only the changed token spans plus a few words of context are
  stored — never the whole report — and the log can be reset.
* **Lossless / never fatal.** Any failure is swallowed; tracking must never
  interrupt dictation or block a save.
"""

from __future__ import annotations

import difflib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.core.json_store import append_jsonl, read_jsonl

logger = logging.getLogger(__name__)

# A single edit whose changed side is longer than this many tokens is almost
# certainly a wholesale rewrite or a pasted template block, not a dictation fix
# worth mining — so we skip it to keep the signal clean.
_MAX_SPAN_TOKENS = 8
# Words of surrounding context kept on each side, so a reviewer (and the miner)
# can see how a change was used without storing the whole sentence.
_CONTEXT_TOKENS = 3


def _enabled() -> bool:
    """True unless the user has turned passive learning off."""
    try:
        from src.core.settings import Settings
        return bool(Settings().get("learning_enabled", True))
    except Exception:
        return True


def diff_edits(
    before: str,
    after: str,
    *,
    max_span: int = _MAX_SPAN_TOKENS,
    context: int = _CONTEXT_TOKENS,
) -> List[Dict[str, str]]:
    """Return structured edit records describing how *before* became *after*.

    Word-level :class:`difflib.SequenceMatcher` opcodes are turned into one
    record per changed span. ``op`` is ``replace`` (dictation said X, became Y),
    ``delete`` (X removed) or ``insert`` (Y added). Unchanged regions — e.g. a
    surrounding template — produce nothing. Spans longer than *max_span* tokens
    on the changed side are dropped as likely rewrites rather than fixes.
    """
    a = before.split()
    b = after.split()
    if a == b:
        return []

    records: List[Dict[str, str]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        before_span = a[i1:i2]
        after_span = b[j1:j2]
        if max(len(before_span), len(after_span)) > max_span:
            continue
        records.append({
            "op": tag,
            "before": " ".join(before_span),
            "after": " ".join(after_span),
            "before_context": " ".join(a[max(0, i1 - context):i2 + context]),
            "after_context": " ".join(b[max(0, j1 - context):j2 + context]),
        })
    return records


def record_session_edits(dictated: str, final: str) -> int:
    """Diff a dictation's output against the delivered text and log the edits.

    *dictated* is the pipeline's output captured when recording finished;
    *final* is the editor's text at the moment the report is committed. Returns
    the number of edit records written (0 if disabled, unchanged, or on error).
    """
    if not _enabled() or not dictated or dictated == final:
        return 0
    try:
        records = diff_edits(dictated, final)
    except Exception as exc:  # never let tracking break a save or close
        logger.warning("Could not diff dictation edits: %s", exc)
        return 0
    if not records:
        return 0

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for rec in records:
        rec["ts"] = ts
    from src.features.file_manager import dictation_edits_path
    written = append_jsonl(dictation_edits_path(), records)
    if written:
        logger.info("Logged %d post-dictation edit(s)", written)
    return written


def load_edits(limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Return logged edit records, newest last; missing log -> ``[]``."""
    from src.features.file_manager import dictation_edits_path
    records = read_jsonl(dictation_edits_path())
    return records[-limit:] if limit else records


def edit_stats() -> Dict[str, int]:
    """Summary counts for the learning-statistics view."""
    records = load_edits()
    by_op: Dict[str, int] = {}
    for r in records:
        by_op[r.get("op", "?")] = by_op.get(r.get("op", "?"), 0) + 1
    return {"total": len(records), **by_op}


def reset_edits() -> None:
    """Delete the edit log."""
    try:
        from src.features.file_manager import dictation_edits_path
        path = dictation_edits_path()
        if path.exists():
            path.unlink()
            logger.info("Cleared dictation edit log")
    except Exception as exc:
        logger.warning("Could not reset dictation edits: %s", exc)
