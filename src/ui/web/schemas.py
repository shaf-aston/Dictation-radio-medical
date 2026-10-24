"""The shapes of what the browser sends. Validation only, no behaviour."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.core.patient_schema import PATIENT_KEYS
from src.features.report_release import Communication


class ThemeUpdate(BaseModel):
    theme: Literal["dark", "light"]


class PreferencesUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_size: str
    language: str = "en"
    vad_filter: bool = True
    accent: str = "neutral"
    cleanup_level: str = "medium"
    macro_region: str = ""


class PatientInfo(BaseModel):
    name: str = ""
    id: str = ""
    dob: str = ""
    study_date: str = ""
    referring: str = ""
    accession: str = ""
    sex: str = ""


# The API contract must not drift from the fields the de-identifier scrubs and
# the report header prints. Adding a field to the shared schema without adding
# it here fails at import, loudly, rather than silently shipping an
# un-scrubbed identifier to the transport layer.
if tuple(PatientInfo.model_fields) != PATIENT_KEYS:
    raise RuntimeError(
        f"PatientInfo fields {tuple(PatientInfo.model_fields)} do not match the "
        f"shared patient schema {PATIENT_KEYS}"
    )


#: Which open report a request is about (state.findings_for); the page makes
#: one per report. Blank for a client that does not send one.
ReportId = Field(default="", max_length=64)


class TextRequest(BaseModel):
    """Just the report text, for the endpoints that only read it."""

    text: str = ""
    report_id: str = ReportId


class ReportRef(BaseModel):
    report_id: str = ReportId


class SessionEditsRequest(BaseModel):
    """What a dictation handed over, and what the report said when committed."""

    dictated: str = Field(default="", max_length=200_000)
    final: str = Field(default="", max_length=200_000)


class TermChoiceRequest(BaseModel):
    """One answer to a marked word: take *replacement*, or keep *word* as is."""

    word: str = Field(min_length=1, max_length=64)
    replacement: Optional[str] = Field(default=None, min_length=1, max_length=64)


class ConsentRequest(BaseModel):
    enabled: bool


class CommunicationIn(BaseModel):
    """Who a critical finding was communicated to, as the browser's dialog asks."""

    recipient: str = Field(min_length=1, max_length=120)
    method: str = Field(min_length=1, max_length=40)
    #: Local time as the radiologist typed it, ISO ``YYYY-MM-DDTHH:MM``.
    at: datetime
    radiologist: str = Field(default="", max_length=120)

    def to_record(self) -> Communication:
        return Communication(
            self.recipient.strip(), self.method.strip(), self.at, self.radiologist.strip()
        )


class StatementRequest(BaseModel):
    text: str = ""
    report_id: str = ReportId
    communication: CommunicationIn


class ReportRequest(BaseModel):
    text: str = ""
    report_id: str = ReportId
    patient: PatientInfo = Field(default_factory=PatientInfo)
    #: None = the client has not yet been shown the critical-findings warning.
    #: True = radiologist confirmed verbal communication; False = proceeded anyway.
    #: Both outcomes are audited; only None is refused (see report.confirm_release).
    acknowledged: Optional[bool] = None
    #: Who was told, how and when; sent with acknowledged=True. Optional so
    #: that an answer without it is still audited, with the fields blank.
    communication: Optional[CommunicationIn] = None
    #: The answer to the report-checks question (unfilled fields, side, sex,
    #: impression), which is a plain permission rather than an audited one:
    #: only True releases the report. Not asked yet and "no, take me back" are
    #: the same thing here, so both are False.
    proceed_with_warnings: bool = False
