"""The one gate every way a report leaves the app has to pass.

Holds two rules, asked in this order. Both front-ends call both: the desktop
shows a dialog, the web app answers 409. Neither owns the rules, so neither can
drift from the other; this module has no Qt, no HTTP and no dialog vocabulary
in it.

1. **Report checks** (``report_warnings``): unfilled template fields, plus
   the consistency checks in :mod:`src.features.report_checks` (side, sex,
   impression). Asked together, once. The answer *can* cancel: "No" means the
   report does not leave. Not audited.
2. **Critical or urgent findings** (``check_release`` / ``record_release``):
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
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence, Tuple

from src.features import audit_log, report_checks, template_fields
from src.medical.critical_findings import (
    CriticalFinding,
    format_findings_for_dialog,
    scan_for_critical_findings,
)

logger = logging.getLogger(__name__)

def unfilled_fields(text: str) -> Tuple[str, ...]:
    """The template fields still in *text*, once each, in first-appearance order.

    What counts as a field is decided in :mod:`src.features.template_fields`,
    the same definition field navigation uses. Empty when the report is clean.
    """
    return template_fields.unfilled_fields(text)


#: How much of the report-checks question fits in a dialog.
_LINE_MAX = 90
_ISSUES_SHOWN = 8
_FIELDS_SHOWN = 10


def _short(line: str) -> str:
    return line if len(line) <= _LINE_MAX else line[: _LINE_MAX - 1] + "…"


@dataclass(frozen=True)
class ReportWarnings:
    """What the radiologist is asked about before release; may cancel."""

    fields: Tuple[str, ...] = ()
    issues: Tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.fields or self.issues)

    def message(self) -> str:
        """The question itself, worded once for both front-ends."""
        parts = []
        if self.issues:
            parts.append("Worth a second look:\n" + "\n".join(
                f"  • {_short(i)}" for i in self.issues[:_ISSUES_SHOWN]))
        if self.fields:
            parts.append(
                f"{len(self.fields)} unfilled template field(s):\n"
                + "\n".join(f"  • {_short(f)}" for f in self.fields[:_FIELDS_SHOWN])
                + ("\n  …" if len(self.fields) > _FIELDS_SHOWN else "")
            )
        return "\n\n".join(parts) + "\n\nRelease the report anyway?"


def report_warnings(text: str, sex: str = "") -> ReportWarnings:
    """Unfilled fields and consistency issues in *text*, for one question.

    Fails open, like the findings scan: a broken check must never stop a
    report leaving, so a fault is logged and that check is skipped.
    """
    try:
        issues = tuple(i.message for i in report_checks.check_report(text, sex))
    except Exception as exc:  # a checker fault must not block a report
        logger.warning("Report consistency check failed: %s", exc)
        issues = ()
    return ReportWarnings(unfilled_fields(text), issues)


@dataclass(frozen=True)
class ReleaseCheck:
    """What one report was found to contain, in the form both front-ends need."""

    findings: Tuple[CriticalFinding, ...] = ()
    #: Human-readable list of the findings; "" when there is nothing to show.
    summary: str = ""
    #: Most severe level present: 1 = life-threatening, 2 = urgent, 0 = clear.
    worst_level: int = 0

    @property
    def needs_acknowledgement(self) -> bool:
        return bool(self.findings)

    @property
    def title(self) -> str:
        """What the communication question is headed with."""
        return "Life-threatening finding" if self.worst_level == 1 else "Urgent finding"


_CLEAR = ReleaseCheck()


def check_release(text: str) -> ReleaseCheck:
    """Scan *text* for findings the radiologist must acknowledge before release.

    Fails open on purpose: a broken scanner must never stop a radiologist
    sending a report, so any fault is logged and the report is treated as clear.
    """
    if not text.strip():
        return _CLEAR
    try:
        return _release_check(scan_for_critical_findings(text))
    except Exception as exc:  # a scanner fault must not block a report
        logger.warning("Critical findings scan failed: %s", exc)
        return _CLEAR


def _release_check(findings: Sequence[CriticalFinding]) -> ReleaseCheck:
    if not findings:
        return _CLEAR
    return ReleaseCheck(
        findings=tuple(findings),
        summary=format_findings_for_dialog(list(findings)),
        worst_level=min(f.level for f in findings),
    )


def _finding_key(finding: CriticalFinding) -> str:
    """What "the same finding" means when deciding if an answer still holds."""
    return finding.term.lower()


@dataclass
class OutstandingFindings:
    """Which findings this report has, and which have been answered for.

    Acknowledgement is a property of the report, not a dialog that fired once:
    the question "is anything still outstanding?" has to be answerable at export
    time, however long ago the radiologist was asked.

    Answers are kept per finding, so an edit that introduces a *new* finding
    leaves that one outstanding while the ones already communicated stay
    answered: and deleting a finding never un-answers the rest.

    Answers belong to **one report**. Whoever owns an instance must call
    :meth:`reset` when the editor moves to a different report, or an
    acknowledgement given for one patient silently answers the identical
    finding for the next one.

    Read by both front-ends: the desktop gutter (:mod:`src.ui.finding_marks`)
    and the web app's ``/api/report/findings``, which draws the same strip.
    """

    #: What the last :meth:`update` found. Empty until the first scan.
    check: ReleaseCheck = _CLEAR
    #: Keys of the findings the radiologist has answered for.
    answered: frozenset = frozenset()

    def update(self, text: str) -> None:
        """Re-scan *text*. Answers already given survive; new findings do not."""
        self.check = check_release(text)

    def reset(self) -> None:
        """Forget this report entirely: a different report is being loaded."""
        self.check = _CLEAR
        self.answered = frozenset()

    def acknowledge(self, check: Optional[ReleaseCheck] = None) -> None:
        """Record that the findings in *check* were communicated.

        *check* defaults to the current scan, but the caller should pass the
        exact scan it showed the radiologist: a dialog runs a nested event
        loop, during which a debounced re-scan can replace :attr:`check`, and
        answering for findings nobody was shown is how an acknowledgement goes
        missing.
        """
        answered_now = check if check is not None else self.check
        self.answered = self.answered | {
            _finding_key(f) for f in answered_now.findings
        }

    @property
    def outstanding(self) -> Tuple[CriticalFinding, ...]:
        """The findings still waiting for an answer."""
        return tuple(
            f for f in self.check.findings if _finding_key(f) not in self.answered
        )

    @property
    def pending(self) -> ReleaseCheck:
        """The outstanding findings, as the question the radiologist is asked.

        Only these are asked about, written into the report and audited at an
        exit: one already communicated needs no second call, and asking again
        appended a second "communicated" sentence whose answer was never
        audited. A finding added since is outstanding, so it is still asked.
        """
        return _release_check(self.outstanding)

    @property
    def count(self) -> int:
        """How many findings the report contains, answered or not."""
        return len(self.check.findings)

    @property
    def label(self) -> str:
        """The count display's words: quiet when clear, never a red zero.

        A warning shown on every clean report is one nobody reads by the time
        a report has a finding, so a clear report says "No findings".
        """
        if not self.check.findings:
            return "No findings"
        outstanding = len(self.outstanding)
        shown = outstanding or self.count
        noun = "finding" if shown == 1 else "findings"
        return f"{shown} {noun} {'to communicate' if outstanding else 'acknowledged'}"

    @property
    def state(self) -> str:
        """One word for the count display: clear, outstanding, or acknowledged."""
        if not self.check.findings:
            return "clear"
        return "outstanding" if self.outstanding else "acknowledged"


#: How a finding can be communicated; the first is the default offered.
COMMUNICATION_METHODS: Tuple[str, ...] = (
    "telephone", "in person", "secure message", "bleep",
)


@dataclass(frozen=True)
class Communication:
    """Who a finding was communicated to, how, when, and by whom.

    The ACR practice parameter on communicating imaging findings asks for the
    date, time, method and recipient to be documented, and for the record to
    say it happened, not that it will ("will be notified" does not count).
    """

    recipient: str
    method: str
    at: datetime
    radiologist: str = ""


def _list_words(words: Sequence[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def communication_statement(
    findings: Sequence[CriticalFinding], comm: Communication
) -> str:
    """The sentence that goes into the report, in the past tense.

    ``"Findings of tension pneumothorax were communicated to Dr Patel by
    telephone at 14:20 on 23/10/2026 by Dr Shah."``
    """
    terms = list(dict.fromkeys(f.term.lower() for f in findings))
    noun = "Finding" if len(terms) == 1 else "Findings"
    verb = "was" if len(terms) == 1 else "were"
    by = f" by {comm.radiologist}" if comm.radiologist else ""
    return (
        f"{noun} of {_list_words(terms)} {verb} communicated to {comm.recipient} "
        f"by {comm.method} at {comm.at:%H:%M} on {comm.at:%d/%m/%Y}{by}."
    )


def add_communication_statement(text: str, statement: str) -> str:
    """*text* with *statement* as its last paragraph; unchanged if already there."""
    if statement in text:
        return text
    return f"{text.rstrip()}\n\n{statement}\n"


def record_release(
    findings: Tuple[CriticalFinding, ...], patient_id: str, acknowledged: bool,
    communication: Optional[Communication] = None,
) -> None:
    """Write the radiologist's answer to the audit log.

    ``acknowledged`` True means verbal communication was confirmed: logged per
    finding. False means they proceeded anyway: logged once, as an override.

    Takes the findings actually being answered for, not the whole scan: an audit
    entry claims a phone call happened, so exporting the same report twice must
    not write the same call twice. A caller holding acknowledgement state passes
    :attr:`OutstandingFindings.outstanding`; a stateless one passes
    ``check.findings``.
    """
    if not findings:
        return
    if acknowledged:
        comm = communication
        for f in findings:
            audit_log.log_critical_finding_acknowledged(
                f.term, patient_id, f.level,
                recipient=comm.recipient if comm else "",
                method=comm.method if comm else "",
                communicated_at=comm.at.isoformat(timespec="minutes") if comm else "",
                radiologist=comm.radiologist if comm else "",
            )
    else:
        audit_log.log_critical_finding_overridden(
            "; ".join(f.term for f in findings), patient_id
        )
