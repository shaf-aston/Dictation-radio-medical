"""The one gate every way a report leaves the app has to pass.

Holds two rules, asked in this order. Both front-ends call both: the desktop
shows a dialog, the web app answers 409. Neither owns the rules, so neither can
drift from the other; this module has no Qt, no HTTP and no dialog vocabulary
in it.

1. **Unfilled template fields** (``unfilled_fields``) — report quality. The
   answer *can* cancel: "No" means the report does not leave. Not audited.
2. **Critical or urgent findings** (``check_release`` / ``record_release``) —
   clinical safety. The report is **never withheld**: a radiologist must always
   be able to get their report out, so both answers proceed and only the audit
   entry differs.

The cancellable rule is asked first, so cancelling can never leave an
acknowledgement in the audit log for a report that then did not leave.

Autosave is deliberately *not* gated. It writes into the app's own autosave
directory, so the report is not leaving the device, and a prompt firing on a
timer is exactly what trains people to click the real one away.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Tuple

from src.features import audit_log
from src.medical.critical_findings import (
    CriticalFinding,
    format_findings_for_dialog,
    scan_for_critical_findings,
)

logger = logging.getLogger(__name__)

#: What a template leaves behind for the radiologist to fill in: a SHOUTED
#: bracketed name (``[FINDINGS]``) or a moustache placeholder (``{{name}}``).
#: Deliberately narrow — ``[5 mm]`` and ``[Findings]`` are ordinary text.
_UNFILLED_FIELD_RE = re.compile(r"\[([A-Z][A-Z0-9 _/-]{1,40})\]|\{\{([^}]{1,40})\}\}")


def unfilled_fields(text: str) -> Tuple[str, ...]:
    """Return the placeholder names still in *text*, in first-appearance order.

    Each name appears once however often it was left in the report — the
    question asked is "which fields are unfilled?", not "how many brackets are
    there?". Empty when the report is clean.
    """
    names = [m[0] or m[1] for m in _UNFILLED_FIELD_RE.findall(text)]
    return tuple(dict.fromkeys(names))


@dataclass(frozen=True)
class ReleaseCheck:
    """What one report was found to contain, in the form both front-ends need."""

    findings: Tuple[CriticalFinding, ...] = ()
    #: Human-readable list of the findings; "" when there is nothing to show.
    summary: str = ""
    #: Most severe level present — 1 = life-threatening, 2 = urgent, 0 = clear.
    worst_level: int = 0

    @property
    def needs_acknowledgement(self) -> bool:
        return bool(self.findings)


_CLEAR = ReleaseCheck()


def check_release(text: str) -> ReleaseCheck:
    """Scan *text* for findings the radiologist must acknowledge before release.

    Fails open on purpose: a broken scanner must never stop a radiologist
    sending a report, so any fault is logged and the report is treated as clear.
    """
    if not text.strip():
        return _CLEAR
    try:
        findings = scan_for_critical_findings(text)
        if not findings:
            return _CLEAR
        return ReleaseCheck(
            findings=tuple(findings),
            summary=format_findings_for_dialog(findings),
            worst_level=min(f.level for f in findings),
        )
    except Exception as exc:  # a scanner fault must not block a report
        logger.warning("Critical findings scan failed: %s", exc)
        return _CLEAR


def record_release(check: ReleaseCheck, patient_id: str, acknowledged: bool) -> None:
    """Write the radiologist's answer to the audit log.

    ``acknowledged`` True means verbal communication was confirmed — logged per
    finding. False means they proceeded anyway — logged once, as an override.
    """
    if not check.needs_acknowledgement:
        return
    if acknowledged:
        for f in check.findings:
            audit_log.log_critical_finding_acknowledged(f.term, patient_id, f.level)
    else:
        audit_log.log_critical_finding_overridden(
            "; ".join(f.term for f in check.findings), patient_id
        )
