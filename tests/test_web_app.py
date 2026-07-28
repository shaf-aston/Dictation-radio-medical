from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from src.core.settings import Settings
from src.dictation.asr import AsrResult
import src.features.report_release as report_release
import src.ui.web_app as web_app


class DummyEngine:
    def transcribe(self, *args, **kwargs) -> AsrResult:
        return AsrResult(text="", segments=())

    def preload(self) -> None:
        pass

    def capabilities(self):
        from src.dictation.asr import EngineCaps
        return EngineCaps(word_confidence=False, hotwords=False)


class DummySettings:
    def __init__(self, initial: dict | None = None) -> None:
        self._data = dict(initial or {})

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value) -> None:
        self._data[key] = value

    def batch_set(self, updates: dict) -> None:
        self._data.update(updates)


def _client(monkeypatch):
    monkeypatch.setattr(web_app, "create_engine", lambda **kwargs: DummyEngine())
    return TestClient(web_app.app)


def _make_template_dir(files: dict[str, str]) -> Path:
    base = Path(__file__).resolve().parents[1] / ".pytest_temp" / f"radio-dictate-web-{uuid4().hex}"
    base.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (base / name).write_text(content, encoding="utf-8")
    return base


def _cleanup_template_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def test_root_uses_saved_theme(monkeypatch) -> None:
    Settings().set("theme", "light")

    with _client(monkeypatch) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'data-theme="light"' in response.text
    assert "radio-dictate-theme" in response.text


def test_get_theme_reflects_persisted_setting(monkeypatch) -> None:
    Settings().set("theme", "dark")

    with _client(monkeypatch) as client:
        response = client.get("/api/theme")

    assert response.status_code == 200
    assert response.json() == {"theme": "dark"}


def test_put_theme_persists_setting(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.put("/api/theme", json={"theme": "light"})

    assert response.status_code == 200
    assert response.json() == {"theme": "light"}
    assert Settings().get("theme") == "light"


def test_put_theme_rejects_invalid_value(monkeypatch) -> None:
    Settings().set("theme", "dark")

    with _client(monkeypatch) as client:
        response = client.put("/api/theme", json={"theme": "midnight"})

    assert response.status_code == 422
    assert Settings().get("theme") == "dark"


def test_get_templates_lists_files_and_selection(monkeypatch) -> None:
    template_dir = _make_template_dir({"alpha.txt": "alpha", "beta.txt": "beta"})
    settings = DummySettings({"theme": "dark", "last_template": "beta.txt"})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)
    monkeypatch.setattr(web_app, "templates_dir", lambda: template_dir)

    try:
        with _client(monkeypatch) as client:
            response = client.get("/api/templates")

        assert response.status_code == 200
        assert response.json() == {
            "templates": ["alpha.txt", "beta.txt"],
            "selected": "beta.txt",
        }
    finally:
        _cleanup_template_dir(template_dir)


def test_root_renders_template_dropdown(monkeypatch) -> None:
    template_dir = _make_template_dir({"alpha.txt": "alpha", "beta.txt": "beta"})
    settings = DummySettings({"theme": "dark", "last_template": "beta.txt"})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)
    monkeypatch.setattr(web_app, "templates_dir", lambda: template_dir)

    try:
        with _client(monkeypatch) as client:
            response = client.get("/")

        assert response.status_code == 200
        # The select is populated by the page from the bootstrap payload, so
        # what the server must get right is the data, not the markup.
        assert '<select id="templateSelect"' in response.text
        assert '"templates":["alpha.txt","beta.txt"]' in response.text
        assert '"selected_template":"beta.txt"' in response.text
    finally:
        _cleanup_template_dir(template_dir)


def test_root_renders_workstation_controls(monkeypatch) -> None:
    settings = DummySettings({"theme": "dark"})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)

    with _client(monkeypatch) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'id="patientName"' in response.text
    assert 'id="modelSelect"' in response.text
    assert 'id="macroRegionSelect"' in response.text
    assert 'id="saveTxtBtn"' in response.text
    assert 'id="exportWordBtn"' in response.text
    assert 'id="editor"' in response.text
    assert 'id="dictateBtn"' in response.text
    # The shell links the front-end rather than carrying it inline.
    assert '/static/app.css' in response.text
    assert '/static/app.js' in response.text


def test_script_regex_literals_are_not_double_escaped(monkeypatch) -> None:
    # These were mangled once by living inside a Python r-string. The script is
    # a real file now so the hazard is gone by construction, but the assertion
    # is what proves it stayed gone.
    with _client(monkeypatch) as client:
        script = client.get("/static/app.js")

    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert r"/[\s\n]$/" in script.text
    assert r"/filename\*=UTF-8''([^;]+)|filename=" in script.text
    assert r"/[\\s\\n]$/" not in script.text
    # The placeholder regex used to be here too, a second copy of a rule the
    # server enforces. It now exists once, in features/report_release.py, and
    # the browser is told the field names — so it must not reappear here.
    assert "[A-Z0-9 _/-]" not in script.text


def test_static_route_serves_only_the_bundled_front_end(monkeypatch) -> None:
    # The name comes straight off the URL, so it is an allow-list, not a path.
    with _client(monkeypatch) as client:
        assert client.get("/static/app.css").status_code == 200
        assert client.get("/static/favicon.svg").status_code == 200
        # app.html is rendered at "/" with state injected; serving the raw
        # shell with its placeholders unfilled would be a broken page.
        assert client.get("/static/app.html").status_code == 404
        assert client.get("/static/settings.json").status_code == 404
        assert client.get("/static/..%2Fweb_app.py").status_code == 404


def test_post_template_loads_content_and_persists_selection(monkeypatch) -> None:
    template_dir = _make_template_dir({"msk_mri_knee.txt": "Knee template content"})
    settings = DummySettings({"theme": "dark"})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)
    monkeypatch.setattr(web_app, "templates_dir", lambda: template_dir)

    try:
        with _client(monkeypatch) as client:
            response = client.post("/api/templates/msk_mri_knee.txt/load")

        assert response.status_code == 200
        assert response.json() == {
            "name": "msk_mri_knee.txt",
            "content": "Knee template content",
        }
        assert settings.get("last_template") == "msk_mri_knee.txt"
    finally:
        _cleanup_template_dir(template_dir)


def test_get_preferences_reflects_persisted_settings(monkeypatch) -> None:
    settings = DummySettings(
        {
            "theme": "dark",
            "model_size": "small",
            "language": "en-GB",
            "vad_filter": False,
            "accent": "east_asian",
            "last_macro_region": "Knee",
        }
    )
    monkeypatch.setattr(web_app, "_settings", lambda: settings)

    with _client(monkeypatch) as client:
        response = client.get("/api/preferences")

    assert response.status_code == 200
    assert response.json() == {
        "model_size": "small",
        "language": "en-GB",
        "vad_filter": False,
        "accent": "east_asian",
        "cleanup_level": "medium",
        "macro_region": "Knee",
    }


def test_put_preferences_persists_settings(monkeypatch) -> None:
    settings = DummySettings({"theme": "dark"})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)

    with _client(monkeypatch) as client:
        response = client.put(
            "/api/preferences",
            json={
                "model_size": "base",
                "language": "en",
                "vad_filter": True,
                "accent": "neutral",
                "macro_region": "Spine",
            },
        )

    assert response.status_code == 200
    assert settings.get("model_size") == "base"
    assert settings.get("last_macro_region") in {"Spine", web_app.macros.REGION_ORDER[0]}


def test_put_preferences_rejects_bogus_cleanup_level(monkeypatch) -> None:
    """An out-of-range cleanup_level from a client is coerced to 'medium'.

    This is a trust boundary: the value reaches the pipeline and gates AI
    cleanup, so an arbitrary string must not be persisted verbatim.
    """
    settings = DummySettings({"theme": "dark"})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)

    with _client(monkeypatch) as client:
        response = client.put(
            "/api/preferences",
            json={
                "model_size": "base",
                "language": "en",
                "vad_filter": False,
                "accent": "neutral",
                "cleanup_level": "bogus",
                "macro_region": "Spine",
            },
        )

    assert response.status_code == 200
    assert settings.get("cleanup_level") == "medium"


def test_get_macros_returns_regions(monkeypatch) -> None:
    settings = DummySettings({"theme": "dark", "last_macro_region": "Knee"})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)

    with _client(monkeypatch) as client:
        response = client.get("/api/macros")

    assert response.status_code == 200
    payload = response.json()
    assert "regions" in payload
    assert isinstance(payload["regions"], list)
    assert payload["selected_region"]


def test_report_save_txt_downloads_formatted_report(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.post(
            "/api/report/save-txt",
            json={
                "text": "FINDINGS:\nNo acute abnormality.",
                "patient": {
                    "name": "Jane Doe",
                    "id": "12345",
                    "dob": "01/01/1980",
                    "study_date": "04/05/2026",
                    "referring": "Dr Smith",
                    "accession": "ACC-001",
                },
            },
        )

    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment;")
    assert "Jane Doe" in response.text
    assert "No acute abnormality." in response.text


def test_report_export_word_downloads_docx(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.post(
            "/api/report/export-word",
            json={
                "text": "FINDINGS:\nNo acute abnormality.",
                "patient": {
                    "name": "Jane Doe",
                    "id": "12345",
                    "dob": "01/01/1980",
                    "study_date": "04/05/2026",
                    "referring": "Dr Smith",
                    "accession": "ACC-001",
                },
            },
        )

    if web_app.DOCX_AVAILABLE:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(web_app.DOCX_MIME)
        assert response.content[:2] == b"PK"
    else:
        assert response.status_code == 503


def test_transcribe_returns_text(monkeypatch) -> None:
    """The /transcribe endpoint runs the pipeline and returns JSON with a text key."""
    with _client(monkeypatch) as client:
        audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", audio_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    assert "text" in response.json()


def test_transcribe_rejects_oversized_file(monkeypatch) -> None:
    """Files over 50 MB must be rejected with 413."""
    with _client(monkeypatch) as client:
        big = b"\x00" * (50 * 1024 * 1024 + 2)
        response = client.post(
            "/transcribe",
            files={"file": ("big.wav", big, "audio/wav")},
        )

    assert response.status_code == 413


# ---------------------------------------------------------------------------
# Critical-findings gate
#
# The clinical safety rule: a report naming an urgent finding is never withheld,
# but the radiologist must have been shown it, and the outcome is always audited.
# Enforced server-side so a browser that forgets to ask cannot skip the warning.
# ---------------------------------------------------------------------------

_URGENT_TEXT = "Findings: Large right pneumothorax with mediastinal shift."


def _save_txt(client, text: str, acknowledged=None):
    body: dict = {"text": text, "patient": {"id": "P1"}}
    if acknowledged is not None:
        body["acknowledged"] = acknowledged
    return client.post("/api/report/save-txt", json=body)


def test_urgent_finding_is_refused_until_the_radiologist_has_seen_it(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = _save_txt(client, _URGENT_TEXT)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "critical_findings"
    assert "pneumothorax" in detail["summary"].lower()


def test_acknowledged_urgent_finding_saves_and_is_audited(monkeypatch) -> None:
    logged: list[tuple] = []
    monkeypatch.setattr(
        report_release.audit_log, "log_critical_finding_acknowledged",
        lambda term, patient_id, level: logged.append((term, patient_id, level)),
    )
    with _client(monkeypatch) as client:
        response = _save_txt(client, _URGENT_TEXT, acknowledged=True)

    assert response.status_code == 200
    assert "pneumothorax" in response.text.lower()
    assert [t[0] for t in logged] == ["pneumothorax"]
    assert logged[0][1] == "P1"


def test_overridden_urgent_finding_saves_and_is_audited_separately(monkeypatch) -> None:
    # Proceeding anyway is allowed — the report is never withheld — but it is
    # recorded as an override, not as an acknowledgement.
    overrides: list[tuple] = []
    acknowledged: list[tuple] = []
    monkeypatch.setattr(
        report_release.audit_log, "log_critical_finding_overridden",
        lambda terms, patient_id: overrides.append((terms, patient_id)),
    )
    monkeypatch.setattr(
        report_release.audit_log, "log_critical_finding_acknowledged",
        lambda term, patient_id, level: acknowledged.append((term, patient_id, level)),
    )
    with _client(monkeypatch) as client:
        response = _save_txt(client, _URGENT_TEXT, acknowledged=False)

    assert response.status_code == 200
    assert len(overrides) == 1 and "pneumothorax" in overrides[0][0]
    assert acknowledged == []


def test_report_without_an_urgent_finding_is_not_gated(monkeypatch) -> None:
    # The gate must not block ordinary reports, or every export would stall.
    with _client(monkeypatch) as client:
        response = _save_txt(client, "Findings: No acute cardiopulmonary process.")

    assert response.status_code == 200


def test_copy_is_gated_and_audited_like_every_other_way_out(monkeypatch) -> None:
    # Copy is the primary bar action and puts the report on the clipboard, which
    # leaves the app exactly as a download does. It gets the same gate and the same
    # audit entry — otherwise the fastest button is the one with no warning.
    overrides: list[tuple] = []
    monkeypatch.setattr(
        report_release.audit_log, "log_critical_finding_overridden",
        lambda terms, patient_id: overrides.append((terms, patient_id)),
    )
    with _client(monkeypatch) as client:
        body: dict = {"text": _URGENT_TEXT, "patient": {"id": "P1"}}
        refused = client.post("/api/report/check", json=body)
        answered = client.post("/api/report/check", json={**body, "acknowledged": False})

    assert refused.status_code == 409
    assert refused.json()["detail"]["reason"] == "critical_findings"
    assert answered.status_code == 200
    assert len(overrides) == 1 and "pneumothorax" in overrides[0][0]


def test_every_way_a_report_leaves_the_app_runs_the_gate() -> None:
    # A new export route that forgets _confirm_release is the exact defect
    # Copy had, so the set of exits is asserted rather than left to review.
    exits = {"/api/report/check", "/api/report/save-txt", "/api/report/export-word"}
    routed = {
        r.path for r in web_app.app.routes  # type: ignore[attr-defined]
        if getattr(r, "path", "").startswith("/api/report/")
    }
    assert routed == exits


# ---------------------------------------------------------------------------
# Unfilled-fields gate
#
# Report quality, not clinical safety: unlike the findings gate this one *can*
# cancel, and it is not audited. Server-side for the same reason — the browser
# used to hold its own copy of the rule, and Copy skipped it entirely.
# ---------------------------------------------------------------------------

_UNFILLED_TEXT = "Findings: [FINDINGS]\nImpression: normal study."


def test_an_unfilled_field_is_refused_until_answered_for(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = _save_txt(client, _UNFILLED_TEXT)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "unfilled_fields"
    assert detail["fields"] == ["FINDINGS"]


def test_proceeding_past_an_unfilled_field_exports_the_report(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.post(
            "/api/report/save-txt",
            json={"text": _UNFILLED_TEXT, "patient": {"id": "P1"},
                  "proceed_unfilled": True},
        )

    assert response.status_code == 200
    assert "[FINDINGS]" in response.text


def test_declining_an_unfilled_field_exports_nothing(monkeypatch) -> None:
    # "No, take me back" is the client simply not re-sending; a request that
    # still says False must never produce a file, or the answer meant nothing.
    with _client(monkeypatch) as client:
        response = client.post(
            "/api/report/save-txt",
            json={"text": _UNFILLED_TEXT, "patient": {"id": "P1"},
                  "proceed_unfilled": False},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "unfilled_fields"


def test_copy_is_gated_on_unfilled_fields_like_every_other_way_out(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        body: dict = {"text": _UNFILLED_TEXT, "patient": {"id": "P1"}}
        refused = client.post("/api/report/check", json=body)
        answered = client.post("/api/report/check", json={**body, "proceed_unfilled": True})

    assert refused.status_code == 409
    assert refused.json()["detail"]["reason"] == "unfilled_fields"
    assert answered.status_code == 200


def test_the_cancellable_question_is_asked_before_the_audited_one(monkeypatch) -> None:
    # Order matters: acknowledging a critical finding is written to the audit
    # log, so it must not be asked for a report the radiologist then cancels.
    logged: list = []
    monkeypatch.setattr(
        report_release.audit_log, "log_critical_finding_overridden",
        lambda terms, patient_id: logged.append((terms, patient_id)),
    )
    both = _URGENT_TEXT + "\nImpression: [IMPRESSION]"
    with _client(monkeypatch) as client:
        body: dict = {"text": both, "patient": {"id": "P1"}}
        first = client.post("/api/report/save-txt", json=body)
        second = client.post("/api/report/save-txt", json={**body, "proceed_unfilled": True})
        third = client.post(
            "/api/report/save-txt",
            json={**body, "proceed_unfilled": True, "acknowledged": False},
        )

    assert first.json()["detail"]["reason"] == "unfilled_fields"
    assert second.json()["detail"]["reason"] == "critical_findings"
    assert third.status_code == 200
    # Nothing was audited until the report actually left.
    assert len(logged) == 1


def test_a_scanner_fault_never_blocks_a_report(monkeypatch) -> None:
    # Fail open here on purpose: a crashing scanner must not stop a radiologist
    # sending a report. The failure is logged, not swallowed silently.
    def _boom(_text):
        raise RuntimeError("scanner exploded")

    monkeypatch.setattr(report_release, "scan_for_critical_findings", _boom)
    with _client(monkeypatch) as client:
        response = _save_txt(client, _URGENT_TEXT)

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# First-launch clinical disclaimer
#
# The desktop app has shown it since day one and the web app showed nothing,
# while both read the same setting — so a web-only radiologist never saw it.
# ---------------------------------------------------------------------------

def _bootstrap_from(html: str) -> dict:
    marker = "window.__BOOTSTRAP__="
    start = html.index(marker) + len(marker)
    return json.loads(html[start:html.index(";</script>", start)])


def test_first_load_ships_the_disclaimer_and_the_modal_to_show_it_in(monkeypatch) -> None:
    settings = DummySettings({"theme": "dark"})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)

    with _client(monkeypatch) as client:
        response = client.get("/")

    disclaimer = _bootstrap_from(response.text)["disclaimer"]
    assert disclaimer["needed"] is True
    assert "FDA" in disclaimer["text"]
    assert disclaimer["title"]
    # The page has somewhere to put it, and the wording is not duplicated there.
    assert 'id="disclaimerModal"' in response.text
    assert 'id="disclaimerAckBtn"' in response.text


def test_an_acknowledged_disclaimer_is_not_shown_again(monkeypatch) -> None:
    settings = DummySettings({"theme": "dark", "disclaimer_shown": True})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)

    with _client(monkeypatch) as client:
        response = client.get("/")

    assert _bootstrap_from(response.text)["disclaimer"]["needed"] is False


def test_the_ack_endpoint_records_it_in_the_setting_both_front_ends_read(monkeypatch) -> None:
    settings = DummySettings({"theme": "dark"})
    monkeypatch.setattr(web_app, "_settings", lambda: settings)

    with _client(monkeypatch) as client:
        response = client.post("/api/disclaimer/ack")

    assert response.status_code == 200
    assert response.json() == {"acknowledged": True}
    assert settings.get("disclaimer_shown") is True


def test_term_lookup_returns_both_tiers(monkeypatch) -> None:
    """The endpoint is a pass-through: the service decides, this only shapes it."""
    monkeypatch.setattr(
        web_app.term_lookup, "lookup",
        lambda q: web_app.term_lookup.TermLookup(
            query=q, key=q,
            similar_spelling=[web_app.term_lookup.Suggestion("pneumothorax", "1 letter away")],
            related=[web_app.term_lookup.Suggestion("chest drain", "pleural space")],
        ),
    )
    with _client(monkeypatch) as client:
        response = client.get("/api/terms/lookup", params={"q": "pnemothorax"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "pnemothorax",
        "key": "pnemothorax",
        "similar_spelling": [{"term": "pneumothorax", "note": "1 letter away"}],
        "related": [{"term": "chest drain", "note": "pleural space"}],
    }


def test_term_lookup_refuses_an_oversized_selection(monkeypatch) -> None:
    """A whole report dragged into the endpoint is refused at the boundary."""
    limit = web_app.term_lookup.max_query_chars()
    with _client(monkeypatch) as client:
        response = client.get("/api/terms/lookup", params={"q": "x" * (limit + 1)})

    assert response.status_code == 422
