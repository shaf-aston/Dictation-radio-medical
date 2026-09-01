const THEME_STORAGE_KEY = window.__THEME_STORAGE_KEY__;
const BOOTSTRAP = window.__BOOTSTRAP__;
const PATIENT_STORAGE_KEY = "radio-dictate-web-patient";
let isRecording = false;
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
const editor = document.getElementById('editor');
const newReportBtn = document.getElementById('newReportBtn');
const saveTxtBtn = document.getElementById('saveTxtBtn');
const exportWordBtn = document.getElementById('exportWordBtn');
const templateSelect = document.getElementById('templateSelect');
const loadTemplateBtn = document.getElementById('loadTemplateBtn');
const loadTemplateText = document.getElementById('loadTemplateText');
const patientName = document.getElementById('patientName');
const patientId = document.getElementById('patientId');
const patientDob = document.getElementById('patientDob');
const patientStudyDate = document.getElementById('patientStudyDate');
const patientReferrer = document.getElementById('patientReferrer');
const patientAccession = document.getElementById('patientAccession');
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
const STATUS_STATES = ['is-rec', 'is-busy', 'is-ok', 'is-error'];
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
    return {
        name: patientName.value.trim(),
        id: patientId.value.trim(),
        dob: patientDob.value.trim(),
        study_date: patientStudyDate.value.trim(),
        referring: patientReferrer.value.trim(),
        accession: patientAccession.value.trim(),
    };
}

function applyPatientInfo(info) {
    const data = Object.assign({}, BOOTSTRAP.patient_defaults || {}, info || {});
    patientName.value = data.name || '';
    patientId.value = data.id || '';
    patientDob.value = data.dob || '';
    patientStudyDate.value = data.study_date || '';
    patientReferrer.value = data.referring || '';
    patientAccession.value = data.accession || '';
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

function setEditorValue(value, { moveCaretToEnd = true } = {}) {
    editor.value = value;
    if (moveCaretToEnd && typeof editor.setSelectionRange === 'function') {
        const end = editor.value.length;
        editor.setSelectionRange(end, end);
    }
    pushUndoState();
    announceReportChanged();
}

function insertTextAtCursor(text) {
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
    const m = modelSelect.value || 'base';
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
    exportWordBtn.disabled = !BOOTSTRAP.docx_available || reportRequestInFlight;
    exportWordBtn.title = BOOTSTRAP.docx_available
        ? 'Export a Word document'
        : 'Install python-docx to enable Word export';
    saveTxtBtn.disabled = reportRequestInFlight;
    newReportBtn.disabled = reportRequestInFlight;
    clearBtn.disabled = reportRequestInFlight;
    copyBtn.disabled = reportRequestInFlight;
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
        BOOTSTRAP.macros = result.macros || BOOTSTRAP.macros;
        populateMacroRegions();
        macroRegionSelect.value = result.selected_region || macroRegionSelect.value;
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
            patient: collectPatientInfo(),
            acknowledged: answers.acknowledged,
            proceed_unfilled: answers.proceed_unfilled,
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
        const known = detail && (detail.reason === 'unfilled_fields' || detail.reason === 'critical_findings');
        return known ? detail : null;
    } catch (err) {
        return null;
    }
}

// Cancellable: No means the report does not leave. Field names come from the
// server, so the browser holds no copy of what counts as unfilled.
function confirmUnfilledFields(fields) {
    const names = fields || [];
    return window.confirm(
        `The report contains ${names.length} unfilled field(s):\n\n`
        + names.slice(0, 10).map((name) => `  • ${name}`).join('\n')
        + (names.length > 10 ? '\n  …' : '')
        + '\n\nDo you want to proceed anyway?'
    );
}

// Never cancellable: the report is not withheld, only the audit entry differs.
function confirmCriticalFindings(info) {
    return window.confirm(
        (info.worst_level === 1
            ? 'LIFE-THREATENING FINDING DETECTED\n\n'
            : 'URGENT FINDING DETECTED\n\n')
        + info.summary
        + '\nConfirm verbal communication with the referring clinician '
        + 'before releasing this report.\n\n'
        + 'OK = I have communicated this finding\n'
        + 'Cancel = proceed without acknowledging (recorded in the audit log)'
    );
}

// Sends the report and clears the release gate on the way. Both answers start
// unset; the server refuses with 409 one rule at a time, unfilled fields first,
// then critical findings, we ask, and re-send with the radiologist's answer.
// Same questions the desktop app asks. Every path that lets the report leave the
// app goes through here, clipboard included: a gate one button can skip is not a
// gate. Returns null when the radiologist cancelled: not an error, just a stop.
async function postGatedReport(endpoint) {
    const answers = { acknowledged: null, proceed_unfilled: false };
    let response = await postReport(endpoint, answers);

    for (let rule = 0; rule < 2; rule++) {
        const gate = await readReleaseGate(response);
        if (!gate) {
            break;
        }
        if (gate.reason === 'unfilled_fields') {
            if (!confirmUnfilledFields(gate.fields)) {
                return null;
            }
            answers.proceed_unfilled = true;
        } else {
            answers.acknowledged = confirmCriticalFindings(gate);
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
    loadTemplateBtn.disabled = templateLoadInFlight || !templateSelect.value;
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
    if (!name || templateLoadInFlight) {
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

        editor.value = result.content || '';
        announceReportChanged();
        // The top of the template, not the bottom. A freshly loaded report is
        // something you read down from the first heading; dropping the caret at
        // the end scrolled the first line half out of view and left the
        // radiologist looking at "IMPRESSION: 1." before they had read anything.
        if (typeof editor.setSelectionRange === 'function') {
            editor.setSelectionRange(0, 0);
        }
        editor.scrollTop = 0;
        editor.focus();
        undoStack = [editor.value];
        undoIndex = 0;
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
[patientName, patientId, patientDob, patientStudyDate, patientReferrer, patientAccession].forEach((field) => {
    field.addEventListener('input', () => {
        savePatientDraft();
    });
});

newReportBtn.addEventListener('click', () => {
    const hasContent = editor.value.trim() !== '' || Object.values(collectPatientInfo()).some(Boolean);
    if (hasContent && !confirm('Clear the current report and patient information?')) {
        return;
    }
    editor.value = '';
    undoStack = [''];
    undoIndex = 0;
    editor.focus();
    resetPatientInfo();
    hideStatus();
    announceReportChanged();
    resetFindings();
});

// Clear with confirmation
clearBtn.addEventListener('click', () => {
    if (editor.value.trim() === '') return;
    if (confirm('Are you sure you want to clear all text? This cannot be undone.')) {
        setEditorValue('');
        hideStatus();
        resetFindings();
    }
});

saveTxtBtn.addEventListener('click', () => downloadReport('txt'));
exportWordBtn.addEventListener('click', () => downloadReport('word'));

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

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z' && !isRecording) { e.preventDefault(); undo(); }
        if (e.key === 'y' && !isRecording) { e.preventDefault(); redo(); }
        if (e.key === 'n') { e.preventDefault(); newReportBtn.click(); }
        if (e.key === 's') { e.preventDefault(); saveTxtBtn.click(); }
        if (e.shiftKey && e.key.toLowerCase() === 'w') { e.preventDefault(); exportWordBtn.click(); }
        if (e.key === 't') { e.preventDefault(); loadTemplateBtn.click(); }
        if (e.key === 'r') { e.preventDefault(); reloadMacrosBtn.click(); }
    }
    if (e.code === 'Space' && e.target === document.body && !isRecording) {
        e.preventDefault();
        dictateBtn.click();
    }
});

function undo() {
    if (undoIndex > 0) {
        undoIndex--;
        editor.value = undoStack[undoIndex];
        announceReportChanged();
        editor.focus();
    }
}

function redo() {
    if (undoIndex < undoStack.length - 1) {
        undoIndex++;
        editor.value = undoStack[undoIndex];
        announceReportChanged();
        editor.focus();
    }
}

// ---------------------------------------------------------------------------
// Live dictation.
//
// The microphone streams raw 16-bit PCM at 16 kHz straight to /ws/dictate, and
// text comes back while you are still speaking. It is raw PCM rather than the
// browser's own MediaRecorder output because a webm/opus blob cannot be decoded
// a piece at a time: waiting for the container to close is exactly the pause
// this replaces.
//
// Two kinds of text arrive. `committed` is decoded once, corrected, and final.
// `preview` is a guess about the words still being spoken; it is shown dimmed
// and never enters the undo history, because presenting a guess as settled text
// is the one thing a report editor must not do.
// ---------------------------------------------------------------------------

// From the server, so the capture rate and the decoder's rate are one fact.
const LIVE_SAMPLE_RATE = BOOTSTRAP.live_sample_rate;

let liveSocket = null;
let audioContext = null;
let micStream = null;
let micNode = null;
let committedText = '';       // what the server has frozen this session
let previewText = '';         // the provisional tail, dimmed on screen
let baseText = '';            // whatever was in the editor before recording
let recordStartedAt = 0;
let timerHandle = null;
let handedOverText = null;  // what Stop handed back, to detect edits since
// When Stop was pressed, so the developer console can price the two things
// that follow it separately: the hand-back (should be instant) and the
// accuracy pass behind it (allowed to take as long as it needs).
let stopPressedAt = 0;

// The worklet only forwards frames. Every decision stays on the main thread, so
// UI work can never block the audio thread.
const PCM_WORKLET = [
    'class PcmTap extends AudioWorkletProcessor {',
    '    process(inputs) {',
    '        const ch = inputs[0] && inputs[0][0];',
    '        if (ch) this.port.postMessage(ch.slice(0));',
    '        return true;',
    '    }',
    '}',
    'registerProcessor("pcm-tap", PcmTap);',
].join('\n');

function floatToPcm16(frame) {
    const out = new Int16Array(frame.length);
    for (let i = 0; i < frame.length; i++) {
        const clamped = Math.max(-1, Math.min(1, frame[i]));
        out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
    return out;
}

function peakLevel(frame) {
    let peak = 0;
    for (let i = 0; i < frame.length; i++) {
        const v = Math.abs(frame[i]);
        if (v > peak) peak = v;
    }
    return peak;
}

// -- what the radiologist sees ---------------------------------------------

function renderLiveText() {
    const gap = baseText && !baseText.endsWith('\n') ? ' ' : '';
    const settled = baseText + (committedText ? gap + committedText : '');
    editor.value = settled;
    editor.scrollTop = editor.scrollHeight;
    paintPreview(settled);
    // The count is the only thing on screen saying the report is growing while
    // you speak. It is one split of the text, unlike the marks and findings
    // scans, which is why it runs every cycle and they do not.
    updateWordCount();
}

// The dimmed tail is painted by the existing marks underlay rather than a
// second overlay mechanism, and nothing is inserted into the textarea, so an
// export can never contain a provisional word.
function paintPreview(settled) {
    if (!editorMarks) return;
    if (!previewText) { editorMarks.replaceChildren(); return; }
    // The settled half is copied in transparently only to push the preview to
    // the right place on the line; the textarea above paints those same
    // characters for real.
    const out = document.createDocumentFragment();
    out.append(settled + (settled ? ' ' : ''));
    const tail = document.createElement('span');
    tail.className = 'preview';
    tail.textContent = previewText;
    out.append(tail);
    editorMarks.replaceChildren(out);
    syncMarksScroll();
}

//: Kept from the markup so the prompt is written down once, in the HTML.
const EDITOR_PLACEHOLDER = editor.getAttribute('placeholder') || '';

function setRecordingUi(on) {
    dictateBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    dictateBtn.classList.toggle('is-recording', on);
    micIcon.className = on ? 'fas fa-stop' : 'fas fa-microphone';
    dictateBtn.setAttribute('aria-label', on ? 'Stop recording' : 'Start recording');
    recMeter.hidden = !on;
    cancelBtn.hidden = !on;
    // The live preview is painted in the underlay BEHIND the textarea, and an
    // empty textarea still paints its placeholder on top of it. The two drew
    // over each other as "Predsictherpmicrophone and start speaking." for the
    // first seconds of every dictation, which is the first thing you see.
    editor.placeholder = on ? '' : EDITOR_PLACEHOLDER;
    if (!on) setLevel(0);
}

// The real peak of the last frame: a meter that only ever shows "something"
// is a meter that cannot tell you the microphone is dead.
function setLevel(peak) {
    recMeter.style.setProperty('--level', Math.min(1, peak * 2.2).toFixed(3));
}

function startTimer() {
    recordStartedAt = Date.now();
    const tick = () => {
        const secs = Math.floor((Date.now() - recordStartedAt) / 1000);
        const mm = String(Math.floor(secs / 60)).padStart(2, '0');
        const ss = String(secs % 60).padStart(2, '0');
        recTimer.textContent = `${mm}:${ss}`;
    };
    tick();
    timerHandle = setInterval(tick, 500);
}

function stopTimer() {
    if (timerHandle) clearInterval(timerHandle);
    timerHandle = null;
}

// -- the socket -------------------------------------------------------------

function openSocket() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${scheme}://${location.host}/ws/dictate`);
    socket.binaryType = 'arraybuffer';

    socket.onmessage = (event) => {
        const msg = safeParseJson(event.data, null);
        if (!msg) return;
        if (msg.type === 'partial') {
            devUpdateArrived(msg);
            committedText = msg.committed || '';
            previewText = msg.preview || '';
            uncertainWords = new Set(msg.uncertain || []);
            renderLiveText();
            showStatus(
                msg.state === 'catching_up'
                    ? 'Catching up'
                    : 'Listening',
                'is-rec',
            );
        } else if (msg.type === 'stopped') {
            devMark('browser', 'report handed back', { words: devWordsIn(msg.text) },
                { ms: stopPressedAt ? performance.now() - stopPressedAt : null });
            // The report is yours now. The accurate re-decode is still running,
            // but you can read and edit while it does.
            previewText = '';
            committedText = msg.text || '';
            uncertainWords = new Set(msg.uncertain || []);
            renderLiveText();
            handedOverText = editor.value;
            finishSession(null);
            showStatus('Yours to edit · polishing', 'is-busy');
        } else if (msg.type === 'final') {
            devMark('browser', 'accuracy pass arrived', { words: devWordsIn(msg.text) },
                { ms: stopPressedAt ? performance.now() - stopPressedAt : null });
            stopPressedAt = 0;
            const improved = msg.text || '';
            if (editor.value !== handedOverText) {
                // You edited while it was working. Your words win: silently
                // replacing them with the machine's would be the worst possible
                // outcome for a clinical report.
                showStatus('Kept your edits', 'is-ok', 4000);
            } else {
                committedText = improved;
                uncertainWords = new Set(msg.uncertain || []);
                renderLiveText();
                pushUndoState();
                announceReportChanged();
                showStatus('Polished', 'is-ok', 2000);
            }
            handedOverText = null;
        } else if (msg.type === 'error') {
            devMark('browser', msg.message || 'dictation failed', {}, { level: 'error' });
            showError(msg.message || 'Dictation failed.');
            finishSession(null);
        }
    };

    socket.onerror = () => showError('Lost the connection to the dictation service.');
    socket.onclose = () => { if (isRecording) teardownMic(); };
    return socket;
}

// One undo entry per dictation and one scan of the finished report: not one of
// each per partial, which would both blow the 50-entry history in seconds and
// spend the whole machine re-scanning half-sentences.
function finishSession(okMessage) {
    stopTimer();
    setRecordingUi(false);
    previewText = '';
    paintPreview('');
    pushUndoState();
    announceReportChanged();
    syncReportButtons();
    if (okMessage) showStatus(okMessage, 'is-ok', 2500);
}

async function teardownMic() {
    isRecording = false;
    if (micNode) { micNode.disconnect(); micNode = null; }
    if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
    if (audioContext) {
        await audioContext.close().catch(() => {});
        audioContext = null;
    }
}

// -- start / stop / cancel --------------------------------------------------

async function startRecording() {
    devRecordingStarted();
    const micAskedAt = performance.now();
    try {
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
    } catch (err) {
        devMark('browser', 'microphone refused', {}, { level: 'error' });
        console.error('Microphone access denied:', err);
        showError('Microphone access denied. Allow microphone permissions in your browser settings, then try again.');
        return;
    }

    devMark('browser', 'microphone granted', {}, { ms: performance.now() - micAskedAt });

    try {
        // Asking the context for 16 kHz makes the browser resample for us, so
        // there is no hand-written downsampler to get wrong.
        audioContext = new AudioContext({ sampleRate: LIVE_SAMPLE_RATE });
        const workletUrl = URL.createObjectURL(new Blob([PCM_WORKLET], { type: 'application/javascript' }));
        await audioContext.audioWorklet.addModule(workletUrl);
        URL.revokeObjectURL(workletUrl);

        const socketAskedAt = performance.now();
        liveSocket = openSocket();
        await new Promise((resolve, reject) => {
            liveSocket.addEventListener('open', resolve, { once: true });
            liveSocket.addEventListener('error', reject, { once: true });
        });
        devMark('browser', 'dictation socket open', {}, { ms: performance.now() - socketAskedAt });

        micNode = new AudioWorkletNode(audioContext, 'pcm-tap');
        micNode.port.onmessage = (event) => {
            const frame = event.data;
            setLevel(peakLevel(frame));
            if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
                const pcm = floatToPcm16(frame).buffer;
                devAudioSent(pcm.byteLength);
                liveSocket.send(pcm);
            }
        };
        audioContext.createMediaStreamSource(micStream).connect(micNode);
    } catch (err) {
        console.error('Could not start live dictation:', err);
        await teardownMic();
        showError('Could not start dictation: ' + err.message);
        return;
    }

    baseText = editor.value.trim();
    committedText = '';
    previewText = '';
    // The doubts belong to the dictation that raised them. A new session will
    // report its own, and text from the last one has already been read.
    uncertainWords = new Set();
    isRecording = true;
    setRecordingUi(true);
    startTimer();
    devMark('browser', 'recording started', {}, { ms: performance.now() - dev.recordStart });
    showStatus('Listening', 'is-rec');
}

async function stopRecording() {
    if (!isRecording) return;
    stopPressedAt = performance.now();
    devMark('browser', 'stop pressed', { updates: dev.updates });
    await teardownMic();
    setRecordingUi(false);
    stopTimer();
    showStatus('Finishing', 'is-busy');
    if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
        liveSocket.send(JSON.stringify({ command: 'stop' }));
    } else {
        finishSession(null);
    }
}

// Cancel means cancel: the report goes back to exactly what it was.
async function cancelRecording() {
    devMark('browser', 'recording discarded', { updates: dev.updates });
    if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
        liveSocket.send(JSON.stringify({ command: 'cancel' }));
    }
    await teardownMic();
    committedText = '';
    previewText = '';
    editor.value = baseText;
    paintPreview('');
    stopTimer();
    setRecordingUi(false);
    hideStatus();
}

dictateBtn.addEventListener('click', () => {
    if (isRecording) stopRecording(); else startRecording();
});
cancelBtn.addEventListener('click', cancelRecording);

dictateBtn.setAttribute('aria-pressed', 'false');
dictateBtn.setAttribute('role', 'button');
dictateBtn.setAttribute('aria-label', 'Start or stop recording');
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

function applyTermSuggestion(term) {
    const range = termPopSelection;
    hideTermPop();
    if (!range) return;
    editor.value = editor.value.slice(0, range.start) + term + editor.value.slice(range.end);
    const caret = range.start + term.length;
    editor.setSelectionRange(caret, caret);
    editor.focus();
    pushUndoState();
    noteLookupUsed();
    scheduleMarks();
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
let lookupUses = BOOTSTRAP.term_lookup_uses || 0;
const lookupHintUses = BOOTSTRAP.term_lookup_hint_uses || 3;

function clearMarks() {
    if (marksTimer) {
        clearTimeout(marksTimer);
        marksTimer = null;
    }
    marksRequest += 1;
    editorMarks.innerHTML = '';
    marksHint.hidden = true;
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
            + (lookupUses < lookupHintUses ? ': highlight one to see alternatives' : ''));
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
    if (isRecording) {
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
    if (lookupUses >= lookupHintUses) return;
    lookupUses += 1;
    try {
        await fetch('/api/terms/used', { method: 'POST' });
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
    const count = data.count || 0;
    if (!count) {
        // "No findings", never a red zero: a warning shown on every clear
        // report is one that stops being read on the report that has one.
        findingsPill.textContent = 'No findings';
        // A phone bar has no room for the sentence, so the pill carries a
        // count the stylesheet can show instead. The full wording stays as the
        // tooltip -- a number nobody can expand is not a warning.
        findingsPill.dataset.short = '0';
        findingsPill.title = 'No critical or urgent findings in this report';
        return;
    }
    const outstanding = findings.filter((f) => f.outstanding).length;
    const shown = outstanding || count;
    const noun = shown === 1 ? 'finding' : 'findings';
    findingsPill.textContent = outstanding
        ? `${shown} ${noun} to communicate`
        : `${shown} ${noun} acknowledged`;
    findingsPill.dataset.short = String(shown);
    findingsPill.title = findingsPill.textContent;
}

async function rescanFindings() {
    // Recording rewrites the report every second, so offsets taken now would be
    // stale before they were drawn. Stand down: and leave the last answer on
    // screen rather than replacing it with "No findings", which would claim a
    // report is clear at the exact moment nothing is re-checking it.
    if (isRecording) return;
    findingsRequest += 1;
    const request = findingsRequest;
    try {
        const response = await fetch('/api/report/findings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: editor.value }),
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

async function resetFindings() {
    // A new report: the previous patient's acknowledgements must not answer for
    // this one's identical finding.
    findingsRequest += 1;
    try {
        const response = await fetch('/api/report/findings/reset', { method: 'POST' });
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
    if (!spelling.length && !related.length) {
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
    if (isRecording || document.activeElement !== editor) {
        hideTermPop();
        return;
    }
    const range = { start: editor.selectionStart, end: editor.selectionEnd };
    const selected = editor.value.slice(range.start, range.end).trim();
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
const DEV_MAX_LINES = 800;        // scroll-back, matched to the server's ring
const DEV_SLOW_MS = 1500;         // a decode over this is worth the eye landing on

const devDrawer = document.getElementById('devDrawer');
const devConsole = document.getElementById('devConsole');
const devFilterInput = document.getElementById('devFilter');

const dev = {
    open: false,
    paused: false,
    lastSeq: 0,
    pollTimer: null,
    perfTimer: null,
    entries: [],
    filter: '',
    dirty: false,
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

function devMatches(entry) {
    if (!dev.filter) return true;
    const hay = `${entry.source} ${entry.message} ${JSON.stringify(entry.fields || {})}`.toLowerCase();
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

    const fields = entry.fields || {};
    const pairs = Object.keys(fields).map((k) => `${k}=${fields[k]}`).join(' ');
    if (pairs) add('kv', pairs);

    if (entry.ms !== null && entry.ms !== undefined) {
        const el = document.createElement('span');
        el.className = entry.ms >= DEV_SLOW_MS ? 'ms over' : 'ms';
        el.textContent = `${Math.round(entry.ms)}ms`;
        line.appendChild(el);
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
    if (!open) return;

    devRedraw();
    devPoll();
    devPollPerf();
    dev.pollTimer = setInterval(devPoll, DEV_POLL_MS);
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
}

function initDevConsole() {
    const btn = document.getElementById('devBtn');
    if (btn) btn.addEventListener('click', () => devSetOpen(!dev.open));

    document.getElementById('devCloseBtn').addEventListener('click', () => devSetOpen(false));

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
            const pairs = Object.entries(entry.fields || {}).map(([k, v]) => `${k}=${v}`).join(' ');
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
