"""The one gate every way a report leaves the app has to pass.

Asks a single question — does this report name a critical or urgent finding the
radiologist must have seen? — and records the answer. Both front-ends call it:
the desktop shows a dialog, the web app answers 409. Neither owns the rule, so
neither can drift from the other; this module has no Qt, no HTTP and no dialog
vocabulary in it.

The report is **never withheld**. A radiologist must always be able to get their
report out; both answers proceed and only the audit entry differs.

Autosave is deliberately *not* gated. It writes into the app's own autosave
directory, so the report is not leaving the device, and a prompt firing on a
timer is exactly what trains people to click the real one away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

from src.features import audit_log
from src.medical.critical_findings import (
    CriticalFinding,
    format_findings_for_dialog,
    scan_for_critical_findings,
)

logger = logging.getLogger(__name__)


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
