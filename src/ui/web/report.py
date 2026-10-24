"""The report leaving the app: findings, the release gate, and the exports.

Every rule here lives in ``features/report_release.py``, alongside the desktop
front-end's dialogs: this is only its HTTP shape. Enforced on the server
rather than in the browser so a client that forgets to ask cannot silently
skip a warning.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from src.core.paths import report_filename
from src.features import audit_log
from src.features.report_manager import DOCX_AVAILABLE, export_to_word_bytes, format_plain_text_report
from src.features.report_release import (
    add_communication_statement,
    communication_statement,
    record_release,
    report_warnings,
)
from src.ui.web import state
from src.ui.web.schemas import ReportRef, ReportRequest, StatementRequest, TextRequest

logger = logging.getLogger(__name__)
router = APIRouter()

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@router.post("/api/report/findings")
async def report_findings_endpoint(payload: TextRequest):
    """Re-scan the open report and return what the gutter strip should show.

    Positions come back with the findings so the browser can place a mark
    without knowing any of the scanning rules: exactly what the desktop gutter
    reads off the same object.
    """
    findings = state.findings_for(payload.report_id)
    findings.update(payload.text)
    outstanding = {id(f) for f in findings.outstanding}
    return {
        "count": findings.count,
        "state": findings.state,
        "findings": [
            {
                "term": f.term,
                "level": f.level,
                "start": f.start,
                "end": f.end,
                "outstanding": id(f) in outstanding,
            }
            for f in findings.check.findings
        ],
    }


@router.post("/api/report/findings/reset")
async def reset_findings_endpoint(payload: ReportRef):
    """The report was cleared: forget its findings and its acknowledgements.

    The page moves to a fresh report id at the same time, so nothing answered
    here can reach the next report either way.
    """
    state.forget_report(payload.report_id)
    audit_log.log_report_cleared("")
    return {"count": 0, "state": "clear", "findings": []}


def confirm_release(payload: ReportRequest) -> None:
    """Refuse to emit a report the radiologist has not answered for yet.

    Both rules refuse with 409 and a ``reason``, one at a time. Unfilled fields go
    first because that is the only answer that can cancel: asking about findings
    first would audit an acknowledgement for a report that then did not leave.

    Raises:
        HTTPException: 409 with the unfilled fields and consistency issues
            while proceed_with_warnings is False, then 409 with the findings
            while acknowledged is still None.
    """
    found = report_warnings(payload.text, payload.patient.sex)
    if found and not payload.proceed_with_warnings:
        raise HTTPException(
            status_code=409,
            detail={
                "reason": "report_checks",
                "fields": list(found.fields),
                "issues": list(found.issues),
            },
        )

    # Scanned through this report's state, so this answer settles the same
    # count the strip in the browser is showing: and a finding typed in after
    # an acknowledgement puts it back to outstanding, rather than riding out on
    # an answer given before it existed.
    findings = state.findings_for(payload.report_id)
    findings.update(payload.text)
    # Only what is still outstanding is asked about and audited: an audit
    # entry claims a phone call happened, so a second export must not record
    # (or ask for) a second one.
    pending = findings.pending
    if not pending.needs_acknowledgement:
        return
    if payload.acknowledged is None:
        raise HTTPException(
            status_code=409,
            detail={
                "reason": "critical_findings",
                "summary": pending.summary,
                "worst_level": pending.worst_level,
            },
        )
    record_release(
        pending.findings,
        payload.patient.id,
        payload.acknowledged,
        payload.communication.to_record() if payload.communication else None,
    )
    if payload.acknowledged:
        findings.acknowledge(pending)


@router.post("/api/report/communication")
async def communication_statement_endpoint(payload: StatementRequest):
    """The report with its communication sentence added, built by the shared rule.

    The browser shows the result in the editor before releasing the report, so
    the sentence the radiologist sees is the one that is exported. Also
    remembers the radiologist's name for next time, as the desktop does.
    """
    comm = payload.communication.to_record()
    if comm.radiologist:
        state.settings().set("radiologist_name", comm.radiologist)
    report = state.findings_for(payload.report_id)
    report.update(payload.text)
    findings = report.pending.findings
    if not findings:
        return {"text": payload.text, "statement": ""}
    statement = communication_statement(findings, comm)
    return {"text": add_communication_statement(payload.text, statement), "statement": statement}


@router.post("/api/report/check")
async def check_report_endpoint(payload: ReportRequest):
    """Run the release gate without emitting a file.

    Copy puts the report on the clipboard, which leaves the app just as surely as a
    download does. It has no file to fetch, so it asks the gate this way instead.
    It is only called by Copy, so passing it is what audits the copy.
    """
    confirm_release(payload)
    audit_log.log_report_copied(payload.patient.id, len(payload.text.split()))
    return {"ok": True}


def _attachment(content: bytes | str, filename: str, media_type: str) -> Response:
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=content, media_type=media_type, headers=headers)


def _filename(payload: ReportRequest, ext: str) -> str:
    return report_filename(
        payload.patient.model_dump(), f".{ext}", prefix="dictation_", fallback_id="unknown"
    )


@router.post("/api/report/save-txt")
async def save_report_txt_endpoint(payload: ReportRequest):
    confirm_release(payload)
    content = format_plain_text_report(payload.text, payload.patient.model_dump())
    filename = _filename(payload, "txt")
    audit_log.log_report_saved(filename, payload.patient.id, len(payload.text.split()))
    return _attachment(content, filename, "text/plain; charset=utf-8")


@router.post("/api/report/export-word")
async def export_word_endpoint(payload: ReportRequest):
    if not DOCX_AVAILABLE:
        raise HTTPException(status_code=503, detail="Word export is unavailable")
    confirm_release(payload)
    try:
        docx_bytes = export_to_word_bytes(payload.text, payload.patient.model_dump())
    except Exception as exc:
        logger.error("Word export failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to export Word document") from exc
    filename = _filename(payload, "docx")
    audit_log.log_report_exported(filename, payload.patient.id, "docx")
    return _attachment(docx_bytes, filename, DOCX_MIME)
