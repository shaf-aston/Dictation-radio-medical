const THEME_STORAGE_KEY = window.__THEME_STORAGE_KEY__;
const BOOTSTRAP = window.__BOOTSTRAP__;
const PATIENT_STORAGE_KEY = "radio-dictate-web-patient";
let isRecording = false;
// A dictation owns the report from Record until its text is handed back:
// idle → starting → recording → finishing → idle. Only idle may start one, and
// nothing else may write the report, or let it leave, until it is idle again.
// Record pressed while the last one was finishing threw its hand-back away
// (usually the impression), and any other write was overwritten by the next
// live update.
let sessionPhase = 'idle';
// Which report the server's findings state belongs to (state.findings_for):
// a new one on load, New report, Clear and a template, so an acknowledgement
// given for one patient can never answer for the next.
let reportId = newReportId();
let undoStack = [''];
let undoIndex = 0;
let themeRequestInFlight = false;
let templateLoadInFlight = false;
let preferenceSaveInFlight = false;
let preferenceSaveQueued = false;
let preferenceSaveTimer = null;
let macroReloadInFlight = false;
let reportRequestInFlight = false;

const dictateBtn = document.getElementById('dictateBtn');
const micIcon = document.getElementById('micIcon');
const cancelBtn = document.getElementById('cancelBtn');
const recMeter = document.getElementById('recMeter');
const recTimer = document.getElementById('recTimer');
const sessionTimeline = document.getElementById('sessionTimeline');
const tlRibbon = document.getElementById('tlRibbon');
const tlPulse = document.getElementById('tlPulse');
const tlNow = document.getElementById('tlNow');
const tlLiveMsg = document.getElementById('tlLiveMsg');
const tlLiveSub = document.getElementById('tlLiveSub');
const tlTrail = document.getElementById('tlTrail');
const editor = document.getElementById('editor');
const livePreview = document.getElementById('livePreview');
// What a template field is: the server's own pattern (template_fields.py).
const FIELD_RE = new RegExp(BOOTSTRAP.field_pattern, 'g');
const NOT_A_FIELD_RE = new RegExp(BOOTSTRAP.not_a_field_pattern, 'i');
const newReportBtn = document.getElementById('newReportBtn');
const saveTxtBtn = document.getElementById('saveTxtBtn');
const exportWordBtn = document.getElementById('exportWordBtn');
const templateSelect = document.getElementById('templateSelect');
const loadTemplateBtn = document.getElementById('loadTemplateBtn');
const loadTemplateText = document.getElementById('loadTemplateText');
// One input per field of core/patient_schema.py, each naming its key, so a new
// field is markup only (a test checks the keys match the schema).
const patientInputs = [...document.querySelectorAll('[data-patient]')];
const modelSelect = document.getElementById('modelSelect');
const languageInput = document.getElementById('languageInput');
const accentSelect = document.getElementById('accentSelect');
const cleanupSelect = document.getElementById('cleanupSelect');
const vadCheckbox = document.getElementById('vadCheckbox');
const reloadMacrosBtn = document.getElementById('reloadMacrosBtn');
const macroRegionSelect = document.getElementById('macroRegionSelect');
const macroButtons = document.getElementById('macroButtons');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const copyText = document.getElementById('copyText');
const copyIcon = document.getElementById('copyIcon');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const themeBtn = document.getElementById('themeBtn');
const themeText = document.getElementById('themeText');
const themeIcon = document.getElementById('themeIcon');

function normalizeTheme(value) {
    return value === 'light' ? 'light' : 'dark';
}

function currentTheme() {
    return normalizeTheme(document.documentElement.dataset.theme || 'dark');
}

function renderTheme(theme, persist = true) {
    const next = normalizeTheme(theme);
    document.documentElement.dataset.theme = next;
    document.documentElement.style.colorScheme = next;
    themeBtn.setAttribute('aria-pressed', next === 'dark' ? 'true' : 'false');
    themeBtn.title = next === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    themeText.textContent = next === 'dark' ? 'Dark' : 'Light';
    themeIcon.className = next === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    if (persist) {
        localStorage.setItem(THEME_STORAGE_KEY, next);
    }
}

function setThemeSavingState(isSaving) {
    themeBtn.disabled = isSaving;
    if (isSaving) {
        themeText.textContent = 'Saving...';
    } else {
        renderTheme(currentTheme(), true);
    }
}

// The status pill's states, matching the desktop window's: recording, working,
// done, failed. The colour is a class app.css owns (.is-rec / .is-busy /
// .is-ok / .is-error) so it comes from tokens.json like every other colour:
// this used to set colours inline here, which both hardcoded them and named
// classes app.css no longer has, so the dot never changed at all.
const STATUS_STATES = ['is-rec', 'is-busy', 'is-ok', 'is-warn', 'is-error'];
let statusGeneration = 0;

function showStatus(message, state = '', timeout = 0) {
    statusGeneration += 1;
    const generation = statusGeneration;
    statusIndicator.classList.remove(...STATUS_STATES);
    if (state) {
        statusIndicator.classList.add(state);
    }
    statusText.textContent = message;
    if (timeout) {
        // Only the newest message may clear itself: otherwise a short one
        // scheduled earlier wipes the state of whatever is running now.
        setTimeout(() => {
            if (generation === statusGeneration) hideStatus();
        }, timeout);
    }
}

function hideStatus() {
    statusGeneration += 1;
    statusIndicator.classList.remove(...STATUS_STATES);
    statusText.textContent = 'Ready';
}

function showError(message) {
    showStatus(message, 'is-error', 5000);
}

async function persistTheme(theme) {
    if (themeRequestInFlight) {
        return;
    }
    themeRequestInFlight = true;
    const previousTheme = currentTheme();
    setThemeSavingState(true);
    renderTheme(theme, false);

    try {
        const response = await fetch('/api/theme', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ theme })
        });

        const result = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(result.detail || 'Theme update failed');
        }

        renderTheme(normalizeTheme(result.theme || theme), true);
    } catch (err) {
        renderTheme(previousTheme, true);
        showError('Theme preference could not be saved: ' + err.message);
    } finally {
        setThemeSavingState(false);
        themeRequestInFlight = false;
    }
}

function toggleTheme() {
    persistTheme(currentTheme() === 'dark' ? 'light' : 'dark');
}

function syncCachedTheme() {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    const serverTheme = currentTheme();
    if (savedTheme && normalizeTheme(savedTheme) !== serverTheme) {
        localStorage.setItem(THEME_STORAGE_KEY, serverTheme);
    } else if (!savedTheme) {
        localStorage.setItem(THEME_STORAGE_KEY, serverTheme);
    }
}

function safeParseJson(raw, fallback) {
    try {
        return raw ? JSON.parse(raw) : fallback;
    } catch (err) {
        return fallback;
    }
}

function loadPatientDraft() {
    return safeParseJson(localStorage.getItem(PATIENT_STORAGE_KEY), BOOTSTRAP.patient_defaults || {});
}

function savePatientDraft() {
    localStorage.setItem(PATIENT_STORAGE_KEY, JSON.stringify(collectPatientInfo()));
}

function collectPatientInfo() {
    return Object.fromEntries(patientInputs.map((el) => [el.dataset.patient, el.value.trim()]));
}

function applyPatientInfo(info) {
    const data = Object.assign({}, BOOTSTRAP.patient_defaults || {}, info || {});
    patientInputs.forEach((el) => { el.value = data[el.dataset.patient] || ''; });
}

function populateModelOptions() {
    modelSelect.innerHTML = '';
    (BOOTSTRAP.supported_models || []).forEach((model) => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    });
}

function populateTemplateOptions() {
    // Built here rather than injected into the page server-side: the names are
    // already in the bootstrap payload, so building them twice was the only
    // reason three placeholder strings existed in the Python template.
    const names = BOOTSTRAP.templates || [];
    templateSelect.innerHTML = '';
    if (!names.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No templates found';
        option.disabled = true;
        option.selected = true;
        templateSelect.appendChild(option);
    } else {
        names.forEach((name) => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            templateSelect.appendChild(option);
        });
    }
    templateSelect.disabled = !names.length;
}

function populateAccentOptions() {
    accentSelect.innerHTML = '';
    (BOOTSTRAP.accent_options || []).forEach((accent) => {
        const option = document.createElement('option');
        option.value = accent.key;
        option.textContent = accent.label;
        accentSelect.appendChild(option);
    });
}

function populateCleanupOptions() {
    cleanupSelect.innerHTML = '';
    (BOOTSTRAP.cleanup_options || []).forEach((level) => {
        const option = document.createElement('option');
        option.value = level.key;
        option.textContent = level.label;
        cleanupSelect.appendChild(option);
    });
}

function populateMacroRegions() {
    macroRegionSelect.innerHTML = '';
    (BOOTSTRAP.macros?.regions || []).forEach((region) => {
        const option = document.createElement('option');
        option.value = region.name;
        option.textContent = region.name;
        macroRegionSelect.appendChild(option);
    });
}

function currentMacroRegion() {
    return macroRegionSelect.value || (BOOTSTRAP.macros?.selected_region || '');
}

function renderMacros(regionName) {
    const region = (BOOTSTRAP.macros?.regions || []).find((item) => item.name === regionName) || BOOTSTRAP.macros?.regions?.[0];
    macroButtons.innerHTML = '';

    if (!region || !region.phrases || region.phrases.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'card-subtitle';
        empty.textContent = 'No quick phrases are available in this region.';
        macroButtons.appendChild(empty);
        return;
    }

    region.phrases.forEach((phrase) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'macro-chip';

        const title = document.createElement('strong');
        title.textContent = phrase.label;
        btn.appendChild(title);

        const preview = document.createElement('span');
        preview.textContent = phrase.text.length > 78 ? `${phrase.text.slice(0, 75)}...` : phrase.text;
        btn.appendChild(preview);

        btn.addEventListener('click', () => insertTextAtCursor(phrase.text));
        macroButtons.appendChild(btn);
    });
}

function pushUndoState() {
    if (undoIndex < undoStack.length - 1) {
        undoStack.length = undoIndex + 1;
    }
    undoStack.push(editor.value);
    undoIndex++;
    if (undoStack.length > 50) {
        undoStack.shift();
        undoIndex = undoStack.length - 1;
    }
}

/* Say that the report changed, for the overlays that read it.
 *
 * Setting `editor.value` from script fires no `input` event, so anything
 * listening for typing never sees a dictated report, a loaded template or an
 * undo: which is exactly the text the findings strip exists to check. Every
 * programmatic write calls this; the `input` listeners cover the typing. */
// The count is on the report's own head strip, so it has to be refreshed
// everywhere the text can change -- typing, a template load, a live update, a
// suggestion applied. announceReportChanged() already runs on all of those.
function updateWordCount() {
    const el = document.getElementById('wordCount');
    if (!el) return;
    // What is on screen, which while recording includes the dimmed tail. A
    // count that ignores the words you can plainly read says "0 words" at the
    // very moment the app is proving it heard you.
    const shown = [editor.value, isRecording ? previewText : ''].join(' ').trim();
    const words = shown ? shown.split(/\s+/).length : 0;
    el.textContent = words === 1 ? '1 word' : `${words} words`;
}

function announceReportChanged() {
    updateWordCount();
    scheduleMarks();
    scheduleFindings();
}

function newReportId() {
    return window.crypto && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function sessionOpen() {
    return sessionPhase !== 'idle';
}

function setSessionPhase(phase) {
    sessionPhase = phase;
    editor.readOnly = phase !== 'idle';
    // The button is Record when idle and Stop while recording; in between it
    // is neither.
    dictateBtn.disabled = phase === 'starting' || phase === 'finishing';
    syncReportButtons();
    syncTemplateButton();
}

function refuseWhileDictating() {
    if (!sessionOpen()) return false;
    showStatus('Finish dictating first', 'is-busy', 2000);
    return true;
}

// The one way a whole new report replaces the editor: New report, Clear and a
// template. Never while a dictation owns the editor; asks first when *ask* is
// given; stays one undo step away; and starts a new report, so the findings
// answered for the old one stay with it.
function replaceReport(text, { ask = '', caret = 'end' } = {}) {
    if (refuseWhileDictating()) return false;
    if (ask && !confirm(ask)) return false;
    flushLearning();
    forgetReport(reportId);
    reportId = newReportId();
    editor.value = text;
    const pos = caret === 'start' ? 0 : text.length;
    editor.setSelectionRange(pos, pos);
    pushUndoState();
    announceReportChanged();
    return true;
}

function insertTextAtCursor(text) {
    if (refuseWhileDictating()) return;
    const current = editor.value;
    const start = typeof editor.selectionStart === 'number' ? editor.selectionStart : current.length;
    const end = typeof editor.selectionEnd === 'number' ? editor.selectionEnd : current.length;
    const before = current.slice(0, start);
    const after = current.slice(end);
    const needsSpacer = before.length > 0 && !/[\s\n]$/.test(before) && !/^[\s\n]/.test(text);
    const insertion = `${needsSpacer ? ' ' : ''}${text}`;
    editor.value = before + insertion + after;
    announceReportChanged();
    if (typeof editor.setSelectionRange === 'function') {
        const pos = before.length + insertion.length;
        editor.setSelectionRange(pos, pos);
    }
    editor.focus();
    pushUndoState();
}

function loadReportSettingsFromServer() {
    const prefs = BOOTSTRAP.preferences || {};
    modelSelect.value = prefs.model_size || (BOOTSTRAP.supported_models || [])[0] || 'base';
    languageInput.value = prefs.language || 'en';
    accentSelect.value = prefs.accent || 'neutral';
    cleanupSelect.value = prefs.cleanup_level || 'medium';
    vadCheckbox.checked = Boolean(prefs.vad_filter);
    macroRegionSelect.value = prefs.macro_region || currentMacroRegion();
    if (!macroRegionSelect.value && macroRegionSelect.options.length > 0) {
        macroRegionSelect.selectedIndex = 0;
    }
    templateSelect.value = BOOTSTRAP.selected_template || templateSelect.value;
    BOOTSTRAP.macros = BOOTSTRAP.macros || { regions: [], selected_region: '' };
    renderMacros(currentMacroRegion());
    syncTemplateButton();
    syncReportButtons();
    updateSettingsBadge();
}

function updateSettingsBadge() {
    const badge = document.getElementById('settingsBadge');
    if (!badge) return;
    // The engine that answers, when it is not the Whisper model picked here.
    const m = BOOTSTRAP.engine_label && !BOOTSTRAP.engine_label.startsWith('Whisper')
        ? BOOTSTRAP.engine_label : (modelSelect.value || 'base');
    const l = languageInput.value.trim() || 'en';
    const a = accentSelect.options[accentSelect.selectedIndex]?.text || 'Neutral';
    const v = vadCheckbox.checked ? 'VAD' : '';
    badge.textContent = [m, l, a, v].filter(Boolean).join(' · ');
}

function applyBootstrapState() {
    populateModelOptions();
    populateTemplateOptions();
    populateAccentOptions();
    populateCleanupOptions();
    populateMacroRegions();
    loadPatientFromStorage();
    loadReportSettingsFromServer();
    BOOTSTRAP.docx_available = Boolean(BOOTSTRAP.docx_available);
    exportWordBtn.disabled = !BOOTSTRAP.docx_available;
}

function syncReportButtons() {
    // A half-dictated report must not leave the app, nor be replaced under the
    // dictation writing it.
    const busy = reportRequestInFlight || sessionOpen();
    exportWordBtn.disabled = !BOOTSTRAP.docx_available || busy;
    exportWordBtn.title = BOOTSTRAP.docx_available
        ? 'Export a Word document'
        : 'Install python-docx to enable Word export';
    saveTxtBtn.disabled = busy;
    newReportBtn.disabled = busy;
    clearBtn.disabled = busy;
    copyBtn.disabled = busy;
}

function schedulePreferenceSave() {
    if (preferenceSaveTimer) {
        clearTimeout(preferenceSaveTimer);
    }
    preferenceSaveTimer = setTimeout(() => {
        persistPreferences();
    }, 250);
}

async function persistPreferences() {
    if (preferenceSaveInFlight) {
        preferenceSaveQueued = true;
        return;
    }
    preferenceSaveInFlight = true;
    const payload = {
        model_size: modelSelect.value || 'base',
        language: languageInput.value.trim() || 'en',
        vad_filter: vadCheckbox.checked,
        accent: accentSelect.value || 'neutral',
        cleanup_level: cleanupSelect.value || 'medium',
        macro_region: macroRegionSelect.value || '',
    };

    try {
        const response = await fetch('/api/preferences', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(result.detail || 'Preference update failed');
        }

        BOOTSTRAP.preferences = Object.assign({}, payload, result.preferences || {});
        if (result.macros) {
            BOOTSTRAP.macros = result.macros;
        }
        modelSelect.value = BOOTSTRAP.preferences.model_size || payload.model_size;
        languageInput.value = BOOTSTRAP.preferences.language || payload.language;
        accentSelect.value = BOOTSTRAP.preferences.accent || payload.accent;
        cleanupSelect.value = BOOTSTRAP.preferences.cleanup_level || payload.cleanup_level;
        vadCheckbox.checked = Boolean(BOOTSTRAP.preferences.vad_filter);
        macroRegionSelect.value = BOOTSTRAP.preferences.macro_region || payload.macro_region;
        renderMacros(currentMacroRegion());
        showStatus('Settings saved', 'is-ok', 1200);
    } catch (err) {
        showError('Settings could not be saved: ' + err.message);
    } finally {
        preferenceSaveInFlight = false;
        syncReportButtons();
        if (preferenceSaveQueued) {
            preferenceSaveQueued = false;
            persistPreferences();
        }
    }
}

async function reloadMacros() {
    if (macroReloadInFlight) {
        return;
    }
    macroReloadInFlight = true;
    reloadMacrosBtn.disabled = true;
    reloadMacrosBtn.innerHTML = '<i class="fas fa-spinner fa-spin" aria-hidden="true"></i><span>Reloading...</span>';
    try {
        const response = await fetch('/api/macros/reload', { method: 'POST' });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(result.detail || 'Macro reload failed');
        }
        // Same shape as the preferences answer: {macros: {regions, selected_region}}.
        BOOTSTRAP.macros = result.macros || BOOTSTRAP.macros;
        populateMacroRegions();
        macroRegionSelect.value = BOOTSTRAP.macros.selected_region || macroRegionSelect.value;
        renderMacros(currentMacroRegion());
        showStatus('Phrases reloaded', 'is-ok', 1500);
    } catch (err) {
        showError('Macros could not be reloaded: ' + err.message);
    } finally {
        macroReloadInFlight = false;
        reloadMacrosBtn.disabled = false;
        reloadMacrosBtn.innerHTML = '<i class="fas fa-rotate-right" aria-hidden="true"></i><span>Reload macros</span>';
        syncReportButtons();
    }
}

function loadPatientFromStorage() {
    applyPatientInfo(loadPatientDraft());
}

function resetPatientInfo() {
    applyPatientInfo(BOOTSTRAP.patient_defaults || {});
    savePatientDraft();
}

function parseDownloadFilename(contentDisposition, fallback) {
    const match = /filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i.exec(contentDisposition || '');
    const candidate = match && (match[1] || match[2]);
    return candidate ? decodeURIComponent(candidate) : fallback;
}

function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function postReport(url, answers) {
    return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: editor.value,
            report_id: reportId,
            patient: collectPatientInfo(),
            acknowledged: answers.acknowledged,
            proceed_with_warnings: answers.proceed_with_warnings,
            communication: answers.communication || null,
        }),
    });
}

// Returns the refusal the server wants answered, or null if this response was
// not one: an ordinary error still has to reach the caller as an error.
async function readReleaseGate(response) {
    if (response.status !== 409) {
        return null;
    }
    try {
        const body = await response.clone().json();
        const detail = body && body.detail;
        const known = detail && (detail.reason === 'report_checks' || detail.reason === 'critical_findings');
        return known ? detail : null;
    } catch (err) {
        return null;
    }
}

// Cancellable: No means the report does not leave. The question, wording
// included, comes from the server (report_release.ReportWarnings.message), so
// the browser holds no copy of the rules or of how they are put.
function confirmReportChecks(gate) {
    return window.confirm(gate.message);
}

// Never cancellable: the report is not withheld, only the audit entry differs.
// "Communicated" records who was told, how and when, and can add that sentence
// to the report first, so the text that leaves is the text on screen.
const commModal = document.getElementById('commModal');
const commForm = document.getElementById('commForm');
const commMethod = document.getElementById('commMethod');
const commRadiologist = document.getElementById('commRadiologist');
(BOOTSTRAP.communication?.methods || ['telephone']).forEach((method) => {
    const option = document.createElement('option');
    option.value = method;
    option.textContent = method[0].toUpperCase() + method.slice(1);
    commMethod.append(option);
});
commRadiologist.value = BOOTSTRAP.communication?.radiologist || '';

function localMinute(date) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`
        + `T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function askCommunication(gate) {
    document.getElementById('commTitle').textContent = gate.title;
    document.getElementById('commSummary').textContent = gate.summary
        + '\n\nConfirm it was communicated to the referring clinician.';
    document.getElementById('commAt').value = localMinute(new Date());
    commModal.hidden = false;
    document.getElementById('commRecipient').focus();
    return new Promise((resolve) => {
        const done = (answer) => {
            commModal.hidden = true;
            commForm.onsubmit = null;
            document.getElementById('commSkip').onclick = null;
            resolve(answer);
        };
        document.getElementById('commSkip').onclick = () => done({ acknowledged: false });
        commForm.onsubmit = async (event) => {
            event.preventDefault();
            const communication = {
                recipient: document.getElementById('commRecipient').value.trim(),
                method: commMethod.value,
                at: document.getElementById('commAt').value,
                radiologist: commRadiologist.value.trim(),
            };
            try {
                if (document.getElementById('commAdd').checked) {
                    const response = await fetch('/api/report/communication', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: editor.value, report_id: reportId, communication }),
                    });
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const { text } = await response.json();
                    if (text !== editor.value) {
                        editor.value = text;
                        pushUndoState();
                        announceReportChanged();
                    }
                }
            } catch (err) {
                // The acknowledgement still stands and is audited; only the
                // sentence is missing, and the radiologist is told so.
                showError('Could not add the communication sentence: ' + err.message);
            }
            done({ acknowledged: true, communication });
        };
    });
}

// Sends the report and clears the release gate on the way. Both answers start
// unset; the server refuses with 409 one rule at a time, unfilled fields first,
// then critical findings, we ask, and re-send with the radiologist's answer.
// Same questions the desktop app asks. Every path that lets the report leave the
// app goes through here, clipboard included: a gate one button can skip is not a
// gate. Returns null when the radiologist cancelled: not an error, just a stop.
async function postGatedReport(endpoint) {
    const answers = { acknowledged: null, proceed_with_warnings: false, communication: null };
    let response = await postReport(endpoint, answers);

    for (let rule = 0; rule < 2; rule++) {
        const gate = await readReleaseGate(response);
        if (!gate) {
            break;
        }
        if (gate.reason === 'report_checks') {
            if (!confirmReportChecks(gate)) {
                return null;
            }
            answers.proceed_with_warnings = true;
        } else {
            const answer = await askCommunication(gate);
            answers.acknowledged = answer.acknowledged;
            answers.communication = answer.communication || null;
        }
        response = await postReport(endpoint, answers);
    }
    return response;
}

async function downloadReport(kind) {
    if (reportRequestInFlight) {
        return;
    }

    reportRequestInFlight = true;
    syncReportButtons();
    const endpoint = kind === 'word' ? '/api/report/export-word' : '/api/report/save-txt';
    const fallbackName = kind === 'word' ? 'radiology_report.docx' : 'radiology_report.txt';

    try {
        const response = await postGatedReport(endpoint);
        if (response === null) {
            // The radiologist chose to go back and fill the report in.
            showStatus('Export cancelled', '', 1800);
            return;
        }

        if (!response.ok) {
            const errorText = await response.text();
            let detail = 'Report download failed';
            try {
                const parsed = JSON.parse(errorText);
                detail = parsed.detail || detail;
            } catch (err) {
                if (errorText) {
                    detail = errorText;
                }
            }
            throw new Error(detail);
        }

        const blob = await response.blob();
        const filename = parseDownloadFilename(response.headers.get('content-disposition'), fallbackName);
        triggerDownload(blob, filename);
        showStatus(kind === 'word' ? 'Word file ready' : 'Text file ready', 'is-ok', 1800);
    } catch (err) {
        showError('Report export failed: ' + err.message);
    } finally {
        reportRequestInFlight = false;
        syncReportButtons();
    }
}

function syncTemplateButton() {
    loadTemplateBtn.disabled = templateLoadInFlight || !templateSelect.value || sessionOpen();
    loadTemplateBtn.title = templateSelect.value
        ? `Load ${templateSelect.value}`
        : 'Select a template first';
}

function setTemplateLoadingState(isLoading) {
    templateLoadInFlight = isLoading;
    loadTemplateText.textContent = isLoading ? 'Loading...' : 'Load template';
    syncTemplateButton();
}

async function loadSelectedTemplate() {
    const name = templateSelect.value;
    if (!name || templateLoadInFlight || refuseWhileDictating()) {
        return;
    }
    // Asked before the fetch, so a slow server cannot put a question in front
    // of a report that has meanwhile changed.
    if (editor.value.trim() && !confirm(`Replace the current report with ${name}? Undo brings it back.`)) {
        return;
    }

    try {
        setTemplateLoadingState(true);
        showStatus('Loading template', 'is-busy');

        const response = await fetch(`/api/templates/${encodeURIComponent(name)}/load`, {
            method: 'POST',
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(result.detail || 'Template load failed');
        }

        // The top of the template, not the bottom. A freshly loaded report is
        // something you read down from the first heading; dropping the caret at
        // the end scrolled the first line half out of view and left the
        // radiologist looking at "IMPRESSION: 1." before they had read anything.
        if (!replaceReport(result.content || '', { caret: 'start' })) return;
        editor.scrollTop = 0;
        editor.focus();
        showStatus('Template loaded', 'is-ok', 2000);
    } catch (err) {
        showError('Template load failed: ' + err.message);
    } finally {
        setTemplateLoadingState(false);
    }
}

// Track text changes for undo and persist patient fields locally.
editor.addEventListener('input', pushUndoState);
editor.addEventListener('input', updateWordCount);
patientInputs.forEach((field) => {
    field.addEventListener('input', () => {
        savePatientDraft();
    });
});

newReportBtn.addEventListener('click', () => {
    const hasContent = editor.value.trim() !== '' || Object.values(collectPatientInfo()).some(Boolean);
    const ask = hasContent ? 'Clear the current report and patient information?' : '';
    if (!replaceReport('', { ask })) return;
    editor.focus();
    resetPatientInfo();
    hideStatus();
});

clearBtn.addEventListener('click', () => {
    if (editor.value.trim() === '') return;
    if (replaceReport('', { ask: 'Clear all text? Undo brings it back.' })) hideStatus();
});

saveTxtBtn.addEventListener('click', () => { flushLearning(); downloadReport('txt'); });
exportWordBtn.addEventListener('click', () => { flushLearning(); downloadReport('word'); });
copyBtn.addEventListener('click', () => flushLearning());

// Copy with visual feedback. The clipboard is an exit from the app, so it clears the
// same release gate the downloads do before the text goes anywhere.
copyBtn.addEventListener('click', async () => {
    if (editor.value.trim() === '' || reportRequestInFlight) {
        if (!reportRequestInFlight) {
            copyText.textContent = 'Nothing to copy';
            setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
        }
        return;
    }

    reportRequestInFlight = true;
    syncReportButtons();
    try {
        const response = await postGatedReport('/api/report/check');
        if (response === null) {
            // The radiologist chose to go back and fill the report in.
            showStatus('Copy cancelled', '', 1800);
            return;
        }
        if (!response.ok) {
            throw new Error('Report check failed');
        }
        await navigator.clipboard.writeText(editor.value);
        copyText.textContent = 'Copied!';
        copyIcon.className = 'fas fa-check';
        setTimeout(() => {
            copyText.textContent = 'Copy';
            copyIcon.className = 'fas fa-copy';
        }, 2000);
    } catch (err) {
        console.error('Failed to copy', err);
        copyText.textContent = 'Copy failed';
        setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
    } finally {
        reportRequestInFlight = false;
        syncReportButtons();
    }
});

modelSelect.addEventListener('change', () => { schedulePreferenceSave(); updateSettingsBadge(); });
languageInput.addEventListener('input', () => { schedulePreferenceSave(); updateSettingsBadge(); });
accentSelect.addEventListener('change', () => { schedulePreferenceSave(); updateSettingsBadge(); });
cleanupSelect.addEventListener('change', () => { schedulePreferenceSave(); updateSettingsBadge(); });
vadCheckbox.addEventListener('change', () => { schedulePreferenceSave(); updateSettingsBadge(); });
macroRegionSelect.addEventListener('change', () => {
    renderMacros(currentMacroRegion());
    schedulePreferenceSave();
});
reloadMacrosBtn.addEventListener('click', reloadMacros);
templateSelect.addEventListener('change', syncTemplateButton);
loadTemplateBtn.addEventListener('click', loadSelectedTemplate);
themeBtn.addEventListener('click', toggleTheme);

// -- template fields and bound keys -------------------------------------------
// What a field is comes from the server (template_fields.py); the keys are the
// same settings the desktop binds, so a dictation mic's buttons work in both.

const KEYS = BOOTSTRAP.keys || {};

// "Shift+Tab", "Ctrl+Alt+K", "F2": the spelling the settings use (Qt's).
function canonicalKeys(spec) {
    const parts = String(spec || '').split('+').map((p) => p.trim()).filter(Boolean);
    const key = parts.pop() || '';
    const mods = ['Ctrl', 'Alt', 'Shift', 'Meta'].filter((m) => parts.includes(m));
    return [...mods, key.length === 1 ? key.toUpperCase() : key].join('+');
}

function pressedKeys(e) {
    const mods = [];
    if (e.ctrlKey) mods.push('Ctrl');
    if (e.altKey) mods.push('Alt');
    if (e.shiftKey) mods.push('Shift');
    if (e.metaKey) mods.push('Meta');
    const key = e.key === ' ' ? 'Space' : e.key;
    return [...mods, key.length === 1 ? key.toUpperCase() : key].join('+');
}

function keyMatches(e, name) {
    return Boolean(KEYS[name]) && pressedKeys(e) === canonicalKeys(KEYS[name]);
}

function templateFields(text) {
    return [...text.matchAll(FIELD_RE)]
        .filter((m) => !NOT_A_FIELD_RE.test(m[0]))
        .map((m) => ({ start: m.index, end: m.index + m[0].length }));
}

// Select the next (or previous) field. False when there is none that way, so
// the key keeps its ordinary meaning in a report with no template.
function selectField(forward) {
    const fields = templateFields(editor.value);
    const field = forward
        ? fields.find((f) => f.start >= editor.selectionEnd)
        : fields.filter((f) => f.end <= editor.selectionStart).pop();
    if (!field) return false;
    editor.setSelectionRange(field.start, field.end);
    // Re-focusing is what makes a textarea scroll its selection into view.
    editor.blur();
    editor.focus();
    return true;
}

editor.addEventListener('keydown', (e) => {
    if (sessionOpen() || dialogOpen()) return;
    const forward = keyMatches(e, 'next_field');
    if ((forward || keyMatches(e, 'previous_field')) && selectField(forward)) {
        e.preventDefault();
    }
});

(function showKeyHint() {
    const hint = document.getElementById('keyHint');
    if (!hint || !KEYS.record_toggle) return;
    const parts = [];
    const add = (spec, label) => {
        canonicalKeys(spec).split('+').forEach((key, i) => {
            if (i) parts.push('+');
            const el = document.createElement('kbd');
            el.textContent = key;
            parts.push(el);
        });
        parts.push(label);
    };
    add(KEYS.record_toggle, ' to record · ');
    add(KEYS.next_field, ' next field · ');
    add('Ctrl+Z', ' to undo');
    hint.replaceChildren(...parts);
})();

function dialogOpen() {
    return Boolean(document.querySelector('.scrim:not([hidden])'));
}

// Every app shortcut, in one place, under three rules. Nothing fires behind an
// open dialog (F2 used to start recording under the critical-findings
// question). A key held down repeats, and a toggle must not: a held F2 started
// and stopped a recording within 120 ms. And the report's own editing keys act
// only in the report: Ctrl+Z in the patient's name field undid the report.
// The buttons it presses refuse for themselves when disabled.
document.addEventListener('keydown', (e) => {
    if (dialogOpen()) return;
    const inReport = e.target === editor;
    const toggle = keyMatches(e, 'record_toggle')
        || (e.code === 'Space' && e.target === document.body && !sessionOpen());
    if (toggle) {
        e.preventDefault();
        if (!e.repeat) dictateBtn.click();
        return;
    }
    if (!(e.ctrlKey || e.metaKey)) return;
    const key = e.key.toLowerCase();
    const press = (button) => { e.preventDefault(); if (!e.repeat) button.click(); };
    if (key === 'z' && inReport) { e.preventDefault(); undo(); }
    else if (key === 'y' && inReport) { e.preventDefault(); redo(); }
    else if (key === 'n') press(newReportBtn);
    else if (key === 's') press(saveTxtBtn);
    else if (key === 'w' && e.shiftKey) press(exportWordBtn);
    else if (key === 't') press(loadTemplateBtn);
    else if (key === 'r') press(reloadMacrosBtn);
});

function undo() {
    if (sessionOpen()) return;
    if (undoIndex > 0) {
        undoIndex--;
        editor.value = undoStack[undoIndex];
        announceReportChanged();
        editor.focus();
    }
}

function redo() {
    if (sessionOpen()) return;
    if (undoIndex < undoStack.length - 1) {
        undoIndex++;
        editor.value = undoStack[undoIndex];
        announceReportChanged();
        editor.focus();
    }
}

renderTheme(document.documentElement.dataset.theme || 'dark', false);
syncCachedTheme();
applyBootstrapState();
syncReportButtons();

// ---------------------------------------------------------------------------
// Option A: the panels below the editor remember whether you left them open.
// This is the whole of "configurable to needs": you shape the screen by using
// it. Kept in localStorage, not settings, because it is per-browser chrome
// state and has no business going through the server.
// ---------------------------------------------------------------------------

// Bumped when the defaults below changed: someone who had already visited was
// otherwise pinned to the old folded-everything layout forever, and would have
// had to find the fix by hand.
const PANEL_STORAGE_KEY = "radio-dictate-web-panels-v2";

function loadOpenPanels() {
    const saved = safeParseJson(localStorage.getItem(PANEL_STORAGE_KEY), null);
    // First visit: the two you reach for while dictating are open. Patient
    // details and Settings are things you set once, so they start folded.
    return Array.isArray(saved) ? saved : ["template", "phrases"];
}

function saveOpenPanels() {
    const open = [...document.querySelectorAll("details[data-panel][open]")]
        .map((el) => el.dataset.panel);
    try {
        localStorage.setItem(PANEL_STORAGE_KEY, JSON.stringify(open));
    } catch (err) {
        // A full or blocked storage must never stop someone dictating.
        console.warn("Could not save panel state:", err);
    }
}

function initPanels() {
    const open = new Set(loadOpenPanels());
    document.querySelectorAll("details[data-panel]").forEach((el) => {
        el.open = open.has(el.dataset.panel);
        el.addEventListener("toggle", saveOpenPanels);
    });
}

// ---------------------------------------------------------------------------
// Overflow menu: one primary action stays on the bar, the rest live in here.
// ---------------------------------------------------------------------------

function initOverflowMenu() {
    const btn = document.getElementById("moreBtn");
    const menu = document.getElementById("moreMenu");
    if (!btn || !menu) return;

    const close = () => {
        menu.hidden = true;
        btn.setAttribute("aria-expanded", "false");
    };

    btn.addEventListener("click", (event) => {
        event.stopPropagation();
        const willOpen = menu.hidden;
        menu.hidden = !willOpen;
        btn.setAttribute("aria-expanded", String(willOpen));
    });

    // Clicking any action inside closes the menu; the action's own handler
    // is already bound elsewhere and still runs.
    menu.addEventListener("click", (event) => {
        if (event.target.closest("button")) close();
    });

    document.addEventListener("click", (event) => {
        if (!menu.hidden && !menu.contains(event.target) && event.target !== btn) close();
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !menu.hidden) {
            close();
            btn.focus();
        }
    });
}

// ---------------------------------------------------------------------------
// First-launch clinical disclaimer. The desktop app has always shown this and
// the web app never did, while both read the same "already seen" setting. The
// wording and that decision live in src/features/clinical_disclaimer.py; this
// only renders what the bootstrap hands over.
// ---------------------------------------------------------------------------

function initDisclaimer() {
    const info = BOOTSTRAP.disclaimer;
    const scrim = document.getElementById('disclaimerModal');
    const ackBtn = document.getElementById('disclaimerAckBtn');
    if (!scrim || !ackBtn || !info || !info.needed) return;

    document.getElementById('disclaimerTitle').textContent = info.title || '';
    document.getElementById('disclaimerText').textContent = info.text || '';
    scrim.hidden = false;
    ackBtn.focus();

    // It is shown once in the life of the install, so Escape must not skip it:
    // that would silently spend the only showing. Tab is pinned to the single
    // button, and nothing behind the modal hears a keystroke.
    const guard = (event) => {
        if (event.key === 'Escape' || event.key === 'Tab') {
            event.preventDefault();
            ackBtn.focus();
        }
        event.stopPropagation();
    };
    document.addEventListener('keydown', guard, true);

    ackBtn.addEventListener('click', async () => {
        ackBtn.disabled = true;
        try {
            const response = await fetch('/api/disclaimer/ack', { method: 'POST' });
            if (!response.ok) {
                throw new Error('Acknowledgement was not recorded');
            }
        } catch (err) {
            // Deliberate trade-off: a failed ack still dismisses the modal, but
            // the flag stays unset so it appears again next load. Better to show
            // it twice than to lock a radiologist out of the app over a POST.
            showError('The disclaimer will be shown again next time: ' + err.message);
        } finally {
            document.removeEventListener('keydown', guard, true);
            scrim.hidden = true;
            ackBtn.disabled = false;
            editor.focus();
        }
    });
}

// ---------------------------------------------------------------------------
// Learning from the radiologist. Two signals, both gated on the same consent
// the desktop asks for: an explicit answer to a marked word (take a fix, or
// keep it as spoken), and the one-word fixes made to a finished dictation,
// sent when the report is committed. The rules for what counts live in
// src/features/adaptive_learning.py; the browser only reports what happened.
// ---------------------------------------------------------------------------

let learningEnabled = !!(BOOTSTRAP.learning && BOOTSTRAP.learning.enabled);
// What the last dictation handed over; null once its edits have been sent.
let learnSnapshot = null;

async function learnChoice(word, replacement) {
    if (!learningEnabled || !word.trim()) return;
    try {
        await fetch('/api/learn/choice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word.trim(), replacement }),
        });
    } catch (err) {
        console.warn('Could not record the choice:', err);
    }
}

function flushLearning() {
    const dictated = learnSnapshot;
    learnSnapshot = null;
    if (!learningEnabled || !dictated || dictated === editor.value) return;
    const body = JSON.stringify({ dictated, final: editor.value });
    // sendBeacon survives the page closing, which is one of the commit points.
    if (!navigator.sendBeacon('/api/learn/session', new Blob([body], { type: 'application/json' }))) {
        console.warn('Could not send dictation edits for learning');
    }
}

function initLearningConsent() {
    window.addEventListener('pagehide', flushLearning);
    const info = BOOTSTRAP.learning;
    // Never stacked on the disclaimer: that one comes first, this one next load.
    if (!info || !info.consent_needed || (BOOTSTRAP.disclaimer && BOOTSTRAP.disclaimer.needed)) return;
    const scrim = document.getElementById('learningModal');
    const answer = async (enabled) => {
        scrim.hidden = true;
        learningEnabled = enabled;
        try {
            await fetch('/api/learn/consent', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled }),
            });
        } catch (err) {
            showError('Your answer was not saved and will be asked again: ' + err.message);
        }
        editor.focus();
    };
    document.getElementById('learningYesBtn').addEventListener('click', () => answer(true));
    document.getElementById('learningNoBtn').addEventListener('click', () => answer(false));
    scrim.hidden = false;
    document.getElementById('learningYesBtn').focus();
}

// ---------------------------------------------------------------------------
// The neighbourhood of a highlighted word. Highlight a term and a small panel
// offers what it might have been (spelling) and what goes with it (related).
// Both lists, and their order, come from src/medical/term_lookup.py, the
// same service the desktop window asks, so the two front-ends cannot suggest
// different things for the same word. Nothing is ever applied on its own.
// ---------------------------------------------------------------------------

const TERM_POP_DELAY_MS = 300;   // wait for the highlight to settle
const TERM_POP_OFFSET = 12;      // gap between the pointer and the panel

const termPop = document.getElementById('termPop');
const termPopKey = document.getElementById('tpKey');
const termPopSpelling = document.getElementById('tpSpelling');
const termPopRelated = document.getElementById('tpRelated');

let termPopTimer = null;
let termPopRequest = 0;
let termPopSelection = null;
let lastPointer = null;

function hideTermPop() {
    if (termPopTimer) {
        clearTimeout(termPopTimer);
        termPopTimer = null;
    }
    termPopRequest += 1;
    termPop.hidden = true;
    termPopSelection = null;
}

function placeTermPop() {
    // Anchored to where the highlight was made. A textarea gives no caret
    // rectangle, so the pointer is the honest answer; a keyboard selection
    // falls back to the editor's own top-left.
    const rect = editor.getBoundingClientRect();
    const from = lastPointer || { x: rect.left + 16, y: rect.top + 16 };
    const size = termPop.getBoundingClientRect();
    const left = Math.max(8, Math.min(from.x, window.innerWidth - size.width - 8));
    const below = from.y + TERM_POP_OFFSET;
    const top = below + size.height > window.innerHeight - 8
        ? Math.max(8, from.y - TERM_POP_OFFSET - size.height)
        : below;
    termPop.style.left = `${Math.round(left)}px`;
    termPop.style.top = `${Math.round(top)}px`;
}

// Is the word a suggestion was looked up for still where it was? Script writes
// (the polish, undo, the communication sentence) move text without an input
// event, and replacing by stale offsets rewrites the wrong characters.
function stillThere(range) {
    return Boolean(range) && !sessionOpen()
        && editor.value.slice(range.start, range.end) === range.word;
}

function applyTermSuggestion(term) {
    const range = termPopSelection;
    hideTermPop();
    if (!stillThere(range)) return;
    learnChoice(range.word, term);
    editor.value = editor.value.slice(0, range.start) + term + editor.value.slice(range.end);
    const caret = range.start + term.length;
    editor.setSelectionRange(caret, caret);
    editor.focus();
    pushUndoState();
    noteLookupUsed();
    announceReportChanged();
}

// ---------------------------------------------------------------------------
// Which words are worth highlighting in the first place. The lookup above only
// helps a radiologist who already suspects a word, and the words worth
// suspecting are exactly the ones that read as plausible: so the app points.
//
// Marks are drawn on a transparent copy of the text sitting *behind* the
// textarea. Nothing is inserted into the report itself, which is why Copy,
// Save TXT and Export Word all emit exactly what was typed.
// ---------------------------------------------------------------------------

const MARKS_DELAY_MS = 400;   // longer than the popup's: this reads the whole report

const editorMarks = document.getElementById('editorMarks');
const marksHint = document.getElementById('marksHint');

let marksTimer = null;
let marksRequest = 0;
// Words the decoder itself was unsure of, sent by the server with each update.
// They are underlined more faintly than the suspect-term marks: one says "the
// machine guessed at this", the other says "this word has alternatives worth
// seeing", and collapsing them into one mark would lose that difference.
let uncertainWords = new Set();
// The spans painted last, so a click on an underlined word can open its fixes.
let markedSpans = [];
// Whether the marks still explain themselves (term_lookup.lookup_hint_active).
let lookupHintActive = Boolean(BOOTSTRAP.lookup_hint_active);

function clearMarks() {
    if (marksTimer) {
        clearTimeout(marksTimer);
        marksTimer = null;
    }
    marksRequest += 1;
    editorMarks.innerHTML = '';
    marksHint.hidden = true;
    markedSpans = [];
}

// Every word the decoder was unsure of, wherever it appears in the report.
// Matched by word rather than by position because the correction pipeline has
// rewritten the text since the decode, so the original offsets no longer point
// anywhere real -- and marking one word too many is the safe direction.
function uncertainSpans(text, taken) {
    if (!uncertainWords.size) return [];
    const out = [];
    const word = /[A-Za-z][A-Za-z'-]*/g;
    let hit;
    while ((hit = word.exec(text)) !== null) {
        if (!uncertainWords.has(hit[0].toLowerCase())) continue;
        const start = hit.index;
        const end = start + hit[0].length;
        // A suspect-term mark already covers this word and says more.
        if (taken.some((s) => start < s.end && end > s.start)) continue;
        out.push({ start, end, kind: 'unsure' });
    }
    return out;
}

function paintMarks(spans) {
    const text = editor.value;
    const all = spans.concat(uncertainSpans(text, spans))
        .sort((a, b) => a.start - b.start);
    markedSpans = all;
    const out = document.createDocumentFragment();
    let at = 0;
    all.forEach((span) => {
        if (span.start < at) return;          // overlapping spans cannot happen, but never trust
        out.append(text.slice(at, span.start));
        const mark = document.createElement('mark');
        if (span.kind === 'unsure') mark.className = 'unsure';
        mark.textContent = text.slice(span.start, span.end);
        out.append(mark);
        at = span.end;
    });
    // The trailing newline keeps the last line's wrapping identical to the
    // textarea's, which otherwise reserves a line the underlay does not.
    out.append(text.slice(at) + '\n');
    editorMarks.replaceChildren(out);
    syncMarksScroll();

    // Both kinds of underline are counted, and named apart. An underline the
    // radiologist cannot account for is worse than no underline: they either
    // learn to ignore all of them or stop trusting the text around them.
    const unsure = all.length - spans.length;
    if (!all.length) {
        marksHint.hidden = true;
        return;
    }
    const parts = [];
    if (spans.length) {
        parts.push(`${spans.length} ${spans.length === 1 ? 'word' : 'words'} to check`
            + (lookupHintActive ? ': highlight one to see alternatives' : ''));
    }
    if (unsure) {
        // A whole phrase, not a tail. When there are no suspect terms this is
        // the only part there is, and "4 the machine wasn't sure of" on its
        // own is not a sentence anyone can read.
        parts.push(`${unsure} ${unsure === 1 ? 'word' : 'words'} the machine wasn't sure of`);
    }
    marksHint.textContent = parts.join(' · ');
    marksHint.hidden = false;
}

function syncMarksScroll() {
    editorMarks.scrollTop = editor.scrollTop;
    editorMarks.scrollLeft = editor.scrollLeft;
}

async function rescanMarks() {
    // Recording rewrites the report every second; marks under moving text are
    // noise, and the spans would be stale before they were painted.
    if (sessionOpen()) {
        clearMarks();
        return;
    }
    marksRequest += 1;
    const request = marksRequest;
    try {
        const response = await fetch('/api/terms/suspect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: editor.value }),
        });
        if (request !== marksRequest) return;   // a newer edit won
        if (!response.ok) {
            clearMarks();
            return;
        }
        paintMarks((await response.json()).spans || []);
    } catch (err) {
        // Marking that cannot answer must never interrupt the report.
        console.warn('Term marking failed:', err);
        clearMarks();
    }
}

function scheduleMarks() {
    if (marksTimer) clearTimeout(marksTimer);
    marksTimer = setTimeout(rescanMarks, MARKS_DELAY_MS);
}

async function noteLookupUsed() {
    if (!lookupHintActive) return;
    try {
        const response = await fetch('/api/terms/used', { method: 'POST' });
        if (response.ok) lookupHintActive = Boolean((await response.json()).hint_active);
    } catch (err) {
        console.warn('Could not record lookup use:', err);
    }
}

function initTermMarks() {
    editor.addEventListener('input', scheduleMarks);
    editor.addEventListener('scroll', syncMarksScroll);
    window.addEventListener('resize', scheduleMarks);
    scheduleMarks();
}

// ---------------------------------------------------------------------------
// Critical findings: a tick in the strip beside the report for each one, and a
// count that is on show whether or not there is anything to show.
//
// Every rule lives on the server, in the same OutstandingFindings object the
// desktop window reads (features/report_release.py): what counts as a finding,
// whether a negation clears it, and whether an acknowledgement still holds. The
// browser only draws the answer, which is why the two front-ends cannot come to
// different conclusions about the same report.
// ---------------------------------------------------------------------------

const FINDINGS_DELAY_MS = 400;   // matches the term marks: one scan per settle

const findingGutter = document.getElementById('findingGutter');
const findingsPill = document.getElementById('findingsPill');

let findingsTimer = null;
let findingsRequest = 0;

function paintFindings(data) {
    const findings = data.findings || [];
    findingGutter.replaceChildren();

    // Placed by where the finding sits in the document rather than by where the
    // text is scrolled to, so a finding further down the report still has a
    // tick: and reaching one you cannot already see is the whole interaction.
    const last = Math.max(1, editor.value.length);
    findings.forEach((f) => {
        const mark = document.createElement('button');
        mark.type = 'button';
        mark.className = 'finding-mark';
        mark.dataset.outstanding = String(Boolean(f.outstanding));
        mark.style.top = `${(Math.min(f.start, last) / last) * 100}%`;
        mark.title = `${f.level === 1 ? 'Critical' : 'Urgent'}: ${f.term}`;
        mark.setAttribute('aria-label', `Jump to ${f.term}`);
        mark.addEventListener('click', () => {
            // Selecting scrolls the textarea to it; focus last so the caret lands.
            editor.focus();
            editor.setSelectionRange(f.start, f.end);
        });
        findingGutter.append(mark);
    });

    findingsPill.dataset.findings = data.state || 'clear';
    // The words are the server's (OutstandingFindings.label), the same the
    // desktop pill shows.
    findingsPill.textContent = data.label || '';
    const count = data.count || 0;
    const outstanding = findings.filter((f) => f.outstanding).length;
    // A phone bar has no room for the sentence, so the pill carries a count the
    // stylesheet can show instead; the full wording stays as the tooltip.
    findingsPill.dataset.short = String(outstanding || count);
    findingsPill.title = count ? findingsPill.textContent : 'No critical or urgent findings in this report';
}

async function rescanFindings() {
    // Recording rewrites the report every second, so offsets taken now would be
    // stale before they were drawn. Stand down: and leave the last answer on
    // screen rather than replacing it with "No findings", which would claim a
    // report is clear at the exact moment nothing is re-checking it.
    if (sessionOpen()) return;
    findingsRequest += 1;
    const request = findingsRequest;
    try {
        const response = await fetch('/api/report/findings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: editor.value, report_id: reportId }),
        });
        if (request !== findingsRequest) return;   // a newer edit won
        if (!response.ok) return;
        paintFindings(await response.json());
    } catch (err) {
        // A strip that cannot answer must never interrupt the report, and must
        // never quietly claim the report is clear either: leave what is shown.
        console.warn('Findings scan failed:', err);
    }
}

function scheduleFindings() {
    if (findingsTimer) clearTimeout(findingsTimer);
    findingsTimer = setTimeout(rescanFindings, FINDINGS_DELAY_MS);
}

// The report was replaced: the server can forget its findings and answers.
// The page has already moved to a new report id, so this is housekeeping, not
// what keeps one patient's answers from the next.
async function forgetReport(id) {
    findingsRequest += 1;
    try {
        const response = await fetch('/api/report/findings/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ report_id: id }),
        });
        if (response.ok) paintFindings(await response.json());
    } catch (err) {
        console.warn('Could not reset findings:', err);
    }
}

function initFindingMarks() {
    editor.addEventListener('input', scheduleFindings);
    scheduleFindings();
}

function renderTermTier(host, items) {
    host.innerHTML = '';
    items.forEach((item) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.title = `Replace with "${item.term}"`;
        const term = document.createElement('span');
        term.textContent = item.term;
        btn.appendChild(term);
        if (item.note) {
            const why = document.createElement('span');
            why.className = 'tp-why';
            why.textContent = item.note;
            btn.appendChild(why);
        }
        btn.addEventListener('click', () => applyTermSuggestion(item.term));
        host.appendChild(btn);
    });
}

function showTermPop(result, range) {
    const spelling = result.similar_spelling || [];
    const related = result.related || [];
    // A marked word always opens, even with nothing to offer: "Keep as is"
    // is still an answer. A plain highlight with nothing to show stays shut.
    if (!spelling.length && !related.length && !range.marked) {
        hideTermPop();
        return;
    }
    termPopSelection = range;
    termPopKey.textContent = result.key || '';
    renderTermTier(termPopSpelling, spelling);
    renderTermTier(termPopRelated, related);
    termPop.hidden = false;
    placeTermPop();
}

async function lookupSelectedTerm() {
    // Recording rewrites the report every second; a panel pinned to text that
    // is moving underneath it is noise. Lookups are for reviewing.
    if (sessionOpen() || document.activeElement !== editor) {
        hideTermPop();
        return;
    }
    // A double-click word-select in a textarea can include the trailing
    // space (Chrome does this at some word boundaries). Trim the range
    // itself, not just the text sent to the lookup, so applying a
    // suggestion replaces only the word and never eats the space next to it.
    const raw = editor.value;
    let start = editor.selectionStart;
    let end = editor.selectionEnd;
    while (start < end && /\s/.test(raw[start])) start += 1;
    while (end > start && /\s/.test(raw[end - 1])) end -= 1;
    const selected = raw.slice(start, end);
    const range = { start, end, word: selected, marked: markedSpans.some((m) => m.start === start && m.end === end) };
    if (!selected) {
        hideTermPop();
        return;
    }

    termPopRequest += 1;
    const request = termPopRequest;
    try {
        const response = await fetch(`/api/terms/lookup?q=${encodeURIComponent(selected)}`);
        if (request !== termPopRequest) return;   // a newer highlight won
        if (!response.ok) {
            hideTermPop();
            return;
        }
        showTermPop(await response.json(), range);
    } catch (err) {
        // A lookup that cannot answer must never interrupt the report.
        console.warn('Term lookup failed:', err);
        hideTermPop();
    }
}

function scheduleTermLookup() {
    if (termPopTimer) clearTimeout(termPopTimer);
    termPopTimer = setTimeout(lookupSelectedTerm, TERM_POP_DELAY_MS);
}

function initTermPop() {
    // One click on an underlined word selects it, which opens its fixes. The
    // marks sit behind the textarea, so the click is matched by caret offset.
    editor.addEventListener('click', () => {
        if (sessionOpen() || editor.selectionStart !== editor.selectionEnd) return;
        const at = editor.selectionStart;
        const hit = markedSpans.find((m) => at >= m.start && at <= m.end);
        if (hit) editor.setSelectionRange(hit.start, hit.end);
    });
    document.getElementById('tpKeep').addEventListener('click', () => {
        const range = termPopSelection;
        hideTermPop();
        if (!stillThere(range)) return;
        const word = range.word;
        uncertainWords.delete(word.toLowerCase());
        learnChoice(word, null).then(scheduleMarks);
        editor.setSelectionRange(range.end, range.end);
        editor.focus();
    });
    editor.addEventListener('mousemove', (event) => {
        lastPointer = { x: event.clientX, y: event.clientY };
    });
    document.addEventListener('selectionchange', () => {
        if (document.activeElement === editor) scheduleTermLookup();
    });
    editor.addEventListener('input', hideTermPop);
    editor.addEventListener('blur', () => {
        // Clicking a suggestion blurs the editor; let that click land first.
        setTimeout(() => {
            if (!termPop.contains(document.activeElement)) hideTermPop();
        }, 0);
    });
    window.addEventListener('resize', hideTermPop);
    document.addEventListener('scroll', hideTermPop, true);
    document.addEventListener('click', (event) => {
        if (!termPop.hidden && !termPop.contains(event.target) && event.target !== editor) {
            hideTermPop();
        }
    });

    // Keyboard: Down from the editor steps into the list, arrows move along
    // it, Escape closes and hands the report back.
    document.addEventListener('keydown', (event) => {
        if (termPop.hidden) return;
        const items = [...termPop.querySelectorAll('button')];
        if (event.key === 'Escape') {
            event.preventDefault();
            hideTermPop();
            editor.focus();
        } else if (event.key === 'ArrowDown' && document.activeElement === editor) {
            event.preventDefault();
            items[0]?.focus();
        } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            const at = items.indexOf(document.activeElement);
            if (at === -1) return;
            event.preventDefault();
            const step = event.key === 'ArrowDown' ? 1 : -1;
            items[(at + step + items.length) % items.length].focus();
        }
    });
}

initPanels();
initOverflowMenu();
initDisclaimer();
initLearningConsent();
initTermPop();
initTermMarks();
initFindingMarks();

// ---------------------------------------------------------------------------
// Developer console
//
// Two clocks, one stream. The server's diary (src/core/event_log.py) says what
// the machine did and how long each decode took; the browser's own marks say
// when the text actually reached the screen. Both are needed, because the
// complaint "it feels slow" is about the second one and the cause is nearly
// always in the first.
//
// It polls rather than opening a second socket: the dictation socket must
// never share a connection with diagnostics, and a poll that only asks for
// events newer than the last one it printed costs almost nothing.
//
// Everything here is local. The endpoints read in-process buffers and are
// served on loopback; nothing is written to disk and nothing leaves the device.
// ---------------------------------------------------------------------------

const DEV_STORAGE_KEY = 'radio-dictate-web-dev';
const DEV_POLL_MS = 700;          // how often the server diary is drained
const DEV_PERF_MS = 2500;         // the rolling averages move slowly
const DEV_MAX_LINES = BOOTSTRAP.event_log_max || 1000;  // scroll-back: the server's ring
const DEV_SLOW_MS = 1500;         // a decode over this is worth the eye landing on
const DEV_LONG_CHARS = 220;       // past this a line is folded until clicked

const devDrawer = document.getElementById('devDrawer');
const devConsole = document.getElementById('devConsole');
const devFilterInput = document.getElementById('devFilter');

const dev = {
    open: false,
    size: 'normal',   // or 'min' (bar only) / 'max' (fills the window)
    paused: false,
    lastSeq: 0,
    pollTimer: null,
    nowTimer: null,
    perfTimer: null,
    entries: [],
    filter: '',
    showAll: false,   // off: only lines that took time or went wrong
    dirty: false,
    running: [],      // the server's open timed blocks, from the last poll
    // What this browser measured about the recording in progress.
    recordStart: 0,
    firstWordsMs: null,
    lastUpdateAt: 0,
    updates: 0,
    bytesSent: 0,
};

function devSetStat(id, text, over = false) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('over', Boolean(over));
}

// One entry. `source` is a short subsystem name; `fields` is whatever numbers
// make the line readable. Recorded whether or not the drawer is open, so
// opening it after a slow dictation still shows that dictation.
function devMark(source, message, fields = {}, { level = 'info', ms = null } = {}) {
    devPush({ t: Date.now() / 1000, source, message, ms, fields, level, browser: true });
}

function devPush(entry) {
    const last = dev.entries[dev.entries.length - 1];
    // The server's diary arrives in batches, so one of its lines can reach the
    // page after a browser line that happened later. A console whose clock runs
    // backwards reads as a broken console, so the order is repaired and the
    // stream redrawn once the batch has landed.
    const outOfOrder = Boolean(last) && entry.t < last.t;
    dev.entries.push(entry);
    if (outOfOrder) {
        dev.entries.sort((a, b) => a.t - b.t);
        dev.dirty = true;
    }
    if (dev.entries.length > DEV_MAX_LINES) {
        dev.entries.splice(0, dev.entries.length - DEV_MAX_LINES);
        dev.dirty = true;
    }
    if (!dev.dirty && dev.open && !dev.paused) devAppend(entry);
}

// `k=v k=v`: the one spelling of an entry's fields, drawn, filtered and copied.
function devPairs(entry) {
    return Object.entries(entry.fields || {}).map(([k, v]) => `${k}=${v}`).join(' ');
}

function devMatches(entry) {
    // Chatter mirrored from the ordinary log (model loaded, rules read) is
    // hidden unless asked for: what you came to see is what took time.
    if (!dev.showAll && entry.fields && entry.fields.mirrored && entry.level === 'info') return false;
    if (!dev.filter) return true;
    const hay = `${entry.source} ${entry.message} ${devPairs(entry)}`.toLowerCase();
    return hay.includes(dev.filter);
}

function devClock(t) {
    const d = new Date(t * 1000);
    const pad = (n, w = 2) => String(n).padStart(w, '0');
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

function devAppend(entry) {
    if (!devMatches(entry)) return;
    const atBottom = devConsole.scrollTop + devConsole.clientHeight >= devConsole.scrollHeight - 24;

    const line = document.createElement('span');
    line.className = 'dev-line';
    if (entry.level === 'error' || entry.level === 'critical') line.classList.add('is-error');
    if (entry.browser) line.classList.add('is-browser');

    const add = (cls, text) => {
        const el = document.createElement('span');
        el.className = cls;
        el.textContent = text;
        line.appendChild(el);
        line.appendChild(document.createTextNode(' '));
    };

    add('t', devClock(entry.t));
    add('src', entry.browser ? 'browser' : entry.source);
    add('msg', entry.message);

    const pairs = devPairs(entry);
    if (pairs) add('kv', pairs);

    if (entry.ms !== null && entry.ms !== undefined) {
        const el = document.createElement('span');
        el.className = entry.ms >= DEV_SLOW_MS ? 'ms over' : 'ms';
        el.textContent = `${Math.round(entry.ms)}ms`;
        line.appendChild(el);
    }

    if (line.textContent.length > DEV_LONG_CHARS) {
        line.classList.add('is-long');
        line.setAttribute('role', 'button');
        line.setAttribute('tabindex', '0');
        line.setAttribute('aria-expanded', 'false');
        line.title = 'Click to show the whole line';
    }

    devConsole.appendChild(line);
    while (devConsole.childElementCount > DEV_MAX_LINES) {
        devConsole.removeChild(devConsole.firstChild);
    }
    // Follow the tail only if you were already at the tail. Scrolling up to
    // read a line and being yanked back down is how a console becomes useless.
    if (atBottom) devConsole.scrollTop = devConsole.scrollHeight;
}

function devRedraw() {
    dev.dirty = false;
    devConsole.innerHTML = '';
    dev.entries.forEach(devAppend);
    devConsole.scrollTop = devConsole.scrollHeight;
}

async function devPoll() {
    if (!dev.open || dev.paused) return;
    try {
        const response = await fetch(`/api/debug/events?after=${dev.lastSeq}`);
        if (!response.ok) return;
        const data = await response.json();
        dev.running = data.running || [];
        devDrawNow();
        (data.events || []).forEach((event) => {
            dev.lastSeq = Math.max(dev.lastSeq, event.seq);
            devPush(Object.assign({}, event, { browser: false }));
        });
        if (dev.dirty && dev.open && !dev.paused) devRedraw();
    } catch (err) {
        // The console failing must never be louder than what it reports on.
        console.warn('Developer console poll failed:', err);
    }
}

// "now": the oldest open timed block on the server and how long it has been
// open, ticking between polls so a decode that is taking too long is visible
// while it is still taking too long.
function devDrawNow() {
    const el = document.getElementById('devNow');
    const first = dev.running[0];
    if (!first) {
        el.textContent = 'idle';
        el.className = 'idle';
        return;
    }
    const sec = Math.max(0, Date.now() / 1000 - first.started);
    el.textContent = `${first.message} ${sec.toFixed(1)}s`;
    el.className = sec * 1000 >= DEV_SLOW_MS ? 'over' : '';
}

async function devPollPerf() {
    if (!dev.open || dev.paused) return;
    try {
        const response = await fetch('/api/debug/perf');
        if (!response.ok) return;
        const data = await response.json();
        const stages = data.stages || {};
        const names = Object.keys(stages);
        const rows = document.getElementById('devPerfRows');
        const empty = document.getElementById('devPerfEmpty');
        rows.innerHTML = '';
        empty.hidden = names.length > 0;
        names.slice(0, 14).forEach((name) => {
            const stat = stages[name];
            const tr = document.createElement('tr');
            [name, stat.count, `${Math.round(stat.mean_ms)}ms`, `${Math.round(stat.p95_ms)}ms`]
                .forEach((value, index) => {
                    const td = document.createElement('td');
                    if (index > 0) td.className = 'num';
                    td.textContent = value;
                    tr.appendChild(td);
                });
            rows.appendChild(tr);
        });
    } catch (err) {
        console.warn('Developer perf read failed:', err);
    }
}

// Minimise and Maximise each toggle: pressing the active one restores the
// normal height, so there is no third "restore" button to explain.
function devSetSize(size) {
    dev.size = dev.size === size ? 'normal' : size;
    devDrawer.classList.toggle('is-min', dev.size === 'min');
    devDrawer.classList.toggle('is-max', dev.size === 'max');
    document.getElementById('devMinBtn').setAttribute('aria-pressed', String(dev.size === 'min'));
    document.getElementById('devMaxBtn').setAttribute('aria-pressed', String(dev.size === 'max'));
}

function devSetOpen(open) {
    dev.open = open;
    devDrawer.hidden = !open;
    const btn = document.getElementById('devBtn');
    if (btn) btn.setAttribute('aria-pressed', String(open));
    try {
        localStorage.setItem(DEV_STORAGE_KEY, open ? '1' : '0');
    } catch (err) {
        console.warn('Could not remember the console state:', err);
    }

    clearInterval(dev.pollTimer);
    clearInterval(dev.perfTimer);
    clearInterval(dev.nowTimer);
    if (!open) return;

    devRedraw();
    devPoll();
    devPollPerf();
    dev.pollTimer = setInterval(devPoll, DEV_POLL_MS);
    dev.nowTimer = setInterval(devDrawNow, 200);
    dev.perfTimer = setInterval(devPollPerf, DEV_PERF_MS);
}

// -- what the browser itself measures ---------------------------------------

function devRecordingStarted() {
    dev.recordStart = performance.now();
    dev.firstWordsMs = null;
    dev.lastUpdateAt = 0;
    dev.updates = 0;
    dev.bytesSent = 0;
    devSetStat('devFirstWords', '–');
    devSetStat('devLastUpdate', '–');
    devSetStat('devBehind', '–');
    devSetStat('devUpdates', '0');
    devSetStat('devAudioSent', '0.0s');
}

function devAudioSent(byteLength) {
    dev.bytesSent += byteLength;
    // 16-bit samples at the live rate: two bytes is one sample.
    const seconds = dev.bytesSent / 2 / LIVE_SAMPLE_RATE;
    if (dev.open) devSetStat('devAudioSent', `${seconds.toFixed(1)}s`);
}

function devWordsIn(text) {
    const trimmed = (text || '').trim();
    return trimmed ? trimmed.split(/\s+/).length : 0;
}

// Called for every 'partial' the socket delivers -- the moment the radiologist
// actually sees new words, which is the only latency they feel.
function devUpdateArrived(msg) {
    const now = performance.now();
    const sinceStart = now - dev.recordStart;
    const gap = dev.lastUpdateAt ? now - dev.lastUpdateAt : null;
    dev.lastUpdateAt = now;
    dev.updates += 1;

    const words = devWordsIn(msg.committed);
    const previewWords = devWordsIn(msg.preview);
    if (dev.firstWordsMs === null && (words || previewWords)) {
        dev.firstWordsMs = sinceStart;
        devMark('browser', 'first words on screen', {}, { ms: sinceStart });
        // The wait everyone feels. Budget is the measured 3s; past that it is
        // worth the radiologist knowing the start was slow, not just that it
        // eventually arrived.
        tlStep('First words', sinceStart / 1000, sinceStart > 3000 ? 'slow' : 'good');
    }

    // How far the text on screen is behind the microphone: wall time since the
    // button was pressed, minus how much audio the server says it has read.
    const behind = Math.max(0, sinceStart / 1000 - (msg.audioSec || 0));

    devSetStat(
        'devFirstWords',
        dev.firstWordsMs === null ? '–' : `${(dev.firstWordsMs / 1000).toFixed(1)}s`,
        dev.firstWordsMs !== null && dev.firstWordsMs > 6000,
    );
    devSetStat('devLastUpdate', gap === null ? '–' : `${(gap / 1000).toFixed(1)}s`, gap !== null && gap > 4000);
    devSetStat('devBehind', `${behind.toFixed(1)}s`, behind > 5);
    devSetStat('devUpdates', String(dev.updates));

    devMark('browser', 'text on screen', {
        words,
        preview_words: previewWords,
        audio_sec: msg.audioSec,
        behind_sec: behind.toFixed(1),
        state: msg.state,
    }, { ms: gap });

    // The same three numbers the console just printed, handed to the timeline
    // rather than computed a second time and drifting.
    return { sinceStart, behind, gap, words };
}

function initDevConsole() {
    const btn = document.getElementById('devBtn');
    if (btn) btn.addEventListener('click', () => devSetOpen(!dev.open));

    document.getElementById('devCloseBtn').addEventListener('click', () => devSetOpen(false));
    document.getElementById('devMinBtn').addEventListener('click', () => devSetSize('min'));
    document.getElementById('devMaxBtn').addEventListener('click', () => devSetSize('max'));

    // One listener for every folded line, present or future.
    const toggleLine = (event) => {
        const line = event.target.closest('.dev-line.is-long');
        if (!line) return;
        if (event.type === 'keydown') {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
        }
        const open = line.classList.toggle('is-open');
        line.setAttribute('aria-expanded', String(open));
    };
    devConsole.addEventListener('click', toggleLine);
    devConsole.addEventListener('keydown', toggleLine);

    const allBtn = document.getElementById('devAllBtn');
    allBtn.addEventListener('click', () => {
        dev.showAll = !dev.showAll;
        allBtn.setAttribute('aria-pressed', String(dev.showAll));
        devRedraw();
    });

    const pauseBtn = document.getElementById('devPauseBtn');
    pauseBtn.addEventListener('click', () => {
        dev.paused = !dev.paused;
        pauseBtn.textContent = dev.paused ? 'Resume' : 'Pause';
        pauseBtn.setAttribute('aria-pressed', String(dev.paused));
        // Resuming prints what was missed rather than silently skipping it --
        // a gap you cannot see is worse than no console at all.
        if (!dev.paused) { devRedraw(); devPoll(); }
    });

    document.getElementById('devClearBtn').addEventListener('click', () => {
        dev.entries = [];
        devConsole.innerHTML = '';
    });

    document.getElementById('devCopyBtn').addEventListener('click', async () => {
        const text = dev.entries.filter(devMatches).map((entry) => {
            const pairs = devPairs(entry);
            const ms = (entry.ms === null || entry.ms === undefined) ? '' : ` ${Math.round(entry.ms)}ms`;
            const who = entry.browser ? 'browser' : entry.source;
            return `${devClock(entry.t)} ${who} ${entry.message} ${pairs}${ms}`.trim();
        }).join('\n');
        try {
            await navigator.clipboard.writeText(text);
            showStatus('Console copied', 'is-ok', 1800);
        } catch (err) {
            showError('Could not copy the console.');
        }
    });

    devFilterInput.addEventListener('input', () => {
        dev.filter = devFilterInput.value.trim().toLowerCase();
        devRedraw();
    });

    document.addEventListener('keydown', (event) => {
        if (event.ctrlKey && event.shiftKey && (event.key === 'D' || event.key === 'd')) {
            event.preventDefault();
            devSetOpen(!dev.open);
        }
    });

    let wasOpen = false;
    try {
        wasOpen = localStorage.getItem(DEV_STORAGE_KEY) === '1';
    } catch (err) {
        console.warn('Could not read the console state:', err);
    }
    if (wasOpen) devSetOpen(true);
}

initDevConsole();
updateWordCount();
