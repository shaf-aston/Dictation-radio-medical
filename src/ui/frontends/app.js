const THEME_STORAGE_KEY = window.__THEME_STORAGE_KEY__;
const BOOTSTRAP = window.__BOOTSTRAP__;
const PATIENT_STORAGE_KEY = "radio-dictate-web-patient";
let mediaRecorder;
let audioChunks = [];
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
const statusDot = document.getElementById('statusDot');
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

function showStatus(message, kind = 'info', timeout = 0) {
    statusIndicator.classList.add('is-visible');
    statusIndicator.classList.toggle('is-error', kind === 'error');
    statusText.textContent = message;
    statusDot.classList.toggle('is-processing', kind === 'processing');
    if (kind === 'processing') {
        statusDot.style.setProperty('--dot-color', 'var(--brand)');
        statusDot.style.setProperty('--dot-glow', 'rgba(96, 165, 250, 0.22)');
    } else if (kind === 'success') {
        statusDot.style.setProperty('--dot-color', 'var(--success)');
        statusDot.style.setProperty('--dot-glow', 'rgba(74, 222, 128, 0.22)');
    } else if (kind === 'error') {
        statusDot.style.setProperty('--dot-color', 'var(--danger)');
        statusDot.style.setProperty('--dot-glow', 'rgba(248, 113, 113, 0.22)');
    } else {
        statusDot.style.setProperty('--dot-color', 'var(--brand)');
        statusDot.style.setProperty('--dot-glow', 'rgba(96, 165, 250, 0.22)');
    }
    if (timeout) {
        setTimeout(() => {
            statusIndicator.classList.remove('is-visible', 'is-error');
        }, timeout);
    }
}

function hideStatus() {
    statusIndicator.classList.remove('is-visible', 'is-error');
    statusDot.style.removeProperty('--dot-color');
    statusDot.style.removeProperty('--dot-glow');
}

function showError(message) {
    showStatus(message, 'error', 5000);
    statusText.style.color = '';
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

function setEditorValue(value, { moveCaretToEnd = true } = {}) {
    editor.value = value;
    if (moveCaretToEnd && typeof editor.setSelectionRange === 'function') {
        const end = editor.value.length;
        editor.setSelectionRange(end, end);
    }
    pushUndoState();
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
        showStatus('Settings saved', 'success', 1200);
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
        showStatus('Macros reloaded', 'success', 1500);
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

function confirmTemplateFieldCompletion() {
    const matches = editor.value.match(/\[([A-Z][A-Z0-9 _/-]{1,40})\]|\{\{([^}]{1,40})\}\}/g);
    if (!matches || matches.length === 0) {
        return true;
    }
    const unique = Array.from(new Set(matches));
    return confirm(
        `The report contains ${unique.length} unfilled field(s):\n\n` +
        unique.slice(0, 10).map((item) => `• ${item}`).join('\n') +
        (unique.length > 10 ? '\n…' : '') +
        '\n\nContinue anyway?'
    );
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

async function downloadReport(kind) {
    if (reportRequestInFlight) {
        return;
    }
    if (!confirmTemplateFieldCompletion()) {
        return;
    }

    reportRequestInFlight = true;
    syncReportButtons();
    const endpoint = kind === 'word' ? '/api/report/export-word' : '/api/report/save-txt';
    const fallbackName = kind === 'word' ? 'radiology_report.docx' : 'radiology_report.txt';

    const postReport = (url, acknowledged) => fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: editor.value,
            patient: collectPatientInfo(),
            acknowledged: acknowledged,
        }),
    });

    // Returns the findings payload, or null if this 409 was something else.
    const readCriticalFindings = async (response) => {
        try {
            const body = await response.clone().json();
            const detail = body && body.detail;
            return (detail && detail.reason === 'critical_findings') ? detail : null;
        } catch (err) {
            return null;
        }
    };

    try {
        // acknowledged starts unset. The server refuses with 409 when the report
        // carries a critical finding, we show it, and only then re-send with the
        // radiologist's answer. Same warning the desktop app gives.
        let acknowledged = null;
        let response = await postReport(endpoint, acknowledged);

        if (response.status === 409) {
            const info = await readCriticalFindings(response);
            if (info) {
                acknowledged = window.confirm(
                    (info.worst_level === 1
                        ? 'LIFE-THREATENING FINDING DETECTED\n\n'
                        : 'URGENT FINDING DETECTED\n\n')
                    + info.summary
                    + '\nConfirm verbal communication with the referring clinician '
                    + 'before releasing this report.\n\n'
                    + 'OK = I have communicated this finding\n'
                    + 'Cancel = proceed without acknowledging (recorded in the audit log)'
                );
                response = await postReport(endpoint, acknowledged);
            }
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
        showStatus(kind === 'word' ? 'Word export ready' : 'TXT download ready', 'success', 1800);
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
        showStatus(`Loading template: ${name}`, 'processing');

        const response = await fetch(`/api/templates/${encodeURIComponent(name)}/load`, {
            method: 'POST',
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(result.detail || 'Template load failed');
        }

        editor.value = result.content || '';
        if (typeof editor.setSelectionRange === 'function') {
            const end = editor.value.length;
            editor.setSelectionRange(end, end);
        }
        editor.focus();
        undoStack = [editor.value];
        undoIndex = 0;
        showStatus(`Template loaded: ${result.name}`, 'success', 2000);
    } catch (err) {
        showError('Template load failed: ' + err.message);
    } finally {
        setTemplateLoadingState(false);
    }
}

// Track text changes for undo and persist patient fields locally.
editor.addEventListener('input', pushUndoState);
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
});

// Clear with confirmation
clearBtn.addEventListener('click', () => {
    if (editor.value.trim() === '') return;
    if (confirm('Are you sure you want to clear all text? This cannot be undone.')) {
        setEditorValue('');
        hideStatus();
    }
});

saveTxtBtn.addEventListener('click', () => downloadReport('txt'));
exportWordBtn.addEventListener('click', () => downloadReport('word'));

// Copy with visual feedback
copyBtn.addEventListener('click', () => {
    if (editor.value.trim() === '') {
        copyText.textContent = 'Nothing to copy';
        setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
        return;
    }
    navigator.clipboard.writeText(editor.value)
        .then(() => {
            copyText.textContent = 'Copied!';
            copyIcon.className = 'fas fa-check';
            setTimeout(() => {
                copyText.textContent = 'Copy';
                copyIcon.className = 'fas fa-copy';
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy', err);
            copyText.textContent = 'Copy failed';
            setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
        });
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
        editor.focus();
    }
}

function redo() {
    if (undoIndex < undoStack.length - 1) {
        undoIndex++;
        editor.value = undoStack[undoIndex];
        editor.focus();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = sendAudio;

        audioChunks = [];
        mediaRecorder.start();
        isRecording = true;
        dictateBtn.setAttribute('aria-pressed', 'true');
        dictateBtn.classList.add('is-recording');
        micIcon.className = 'fas fa-stop';

        showStatus('Recording...', 'processing');
    } catch (err) {
        console.error('Microphone access denied:', err);
        showError('Microphone access denied. Please allow microphone permissions in your browser settings and try again.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        isRecording = false;
        dictateBtn.setAttribute('aria-pressed', 'false');
        dictateBtn.classList.remove('is-recording');
        micIcon.className = 'fas fa-microphone';

        showStatus('Transcribing...', 'processing');
    }
}

dictateBtn.addEventListener('click', () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

async function sendAudio() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append("file", audioBlob, "dictation.webm");

    try {
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Transcription failed');
        }

        const result = await response.json();

        if (result.text) {
            const nextValue = editor.value.trim() !== '' ? `${editor.value} ${result.text}` : result.text;
            setEditorValue(nextValue);
            editor.scrollTop = editor.scrollHeight;
            showStatus('Transcription complete', 'success', 2000);
        } else {
            hideStatus();
        }
    } catch (err) {
        console.error('Transcription error:', err);
        showError('Transcription failed: ' + err.message + '. Try again.');
    } finally {
        statusDot.classList.remove('is-processing');
    }
}

dictateBtn.setAttribute('aria-pressed', 'false');
dictateBtn.setAttribute('role', 'button');
dictateBtn.setAttribute('aria-label', 'Start or stop recording');
renderTheme(document.documentElement.dataset.theme || 'dark', false);
syncCachedTheme();
applyBootstrapState();
syncReportButtons();

// ---------------------------------------------------------------------------
// Option A: the panels below the editor remember whether you left them open.
// This is the whole of "configurable to needs" — you shape the screen by using
// it. Kept in localStorage, not settings, because it is per-browser chrome
// state and has no business going through the server.
// ---------------------------------------------------------------------------

const PANEL_STORAGE_KEY = "radio-dictate-web-panels";

function loadOpenPanels() {
    const saved = safeParseJson(localStorage.getItem(PANEL_STORAGE_KEY), null);
    // First visit: patient details open, the rest folded. A new user sees the
    // fields they need for an export without having to discover them.
    return Array.isArray(saved) ? saved : ["patient"];
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
// Overflow menu — one primary action stays on the bar, the rest live in here.
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

initPanels();
initOverflowMenu();
