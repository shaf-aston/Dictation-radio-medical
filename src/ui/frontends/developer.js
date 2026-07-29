// The /developer page: read the run log, render it, and answer the long-vs-short
// question. Read-only — nothing here writes to the report or the settings.

const runRows = document.getElementById('runRows');
const runsEmpty = document.getElementById('runsEmpty');
const lengthRows = document.getElementById('lengthRows');
const devSummary = document.getElementById('devSummary');
const outputScrim = document.getElementById('outputScrim');
const outputBody = document.getElementById('outputBody');

// Audio-length bands for the long-vs-short comparison. Coarse on purpose: the
// question is "does cost climb with length", not "what is the exact curve".
const BANDS = [
    { label: 'under 1 min', max: 60 },
    { label: '1–5 min', max: 300 },
    { label: '5–15 min', max: 900 },
    { label: 'over 15 min', max: Infinity },
];

function secs(value) {
    if (!value && value !== 0) return '—';
    return value >= 60 ? `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`
                       : `${value.toFixed(1)}s`;
}

function when(iso) {
    const at = new Date(iso);
    return Number.isNaN(at.getTime()) ? '—' : at.toLocaleString();
}

function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function cell(row, text, className) {
    const td = document.createElement('td');
    td.textContent = text;
    if (className) td.className = className;
    row.appendChild(td);
    return td;
}

function showOutput(run) {
    outputBody.textContent = run.text;
    outputScrim.hidden = false;
    document.getElementById('outputClose').focus();
}

function hideOutput() {
    outputScrim.hidden = true;
}

function renderRuns(runs) {
    runRows.replaceChildren();
    runsEmpty.hidden = runs.length > 0;

    runs.forEach((run) => {
        const row = document.createElement('tr');
        cell(row, when(run.started));
        cell(row, run.front_end || '—');
        cell(row, run.model || '—');
        cell(row, secs(run.audio_sec), 'num');
        cell(row, secs(run.duration_sec), 'num');
        cell(row, run.finalise_sec ? secs(run.finalise_sec) : '—', 'num');

        const ratio = run.decode_ratio;
        const td = cell(row, ratio == null ? '—' : ratio.toFixed(2), 'num');
        if (ratio != null && ratio > 1) td.classList.add('over');

        cell(row, String(run.word_count ?? 0), 'num');
        cell(row, String(run.chunk_count ?? 0), 'num');

        // The output is behind a click, never inline: a long report would
        // otherwise blow the row height out and make the table unreadable.
        const last = document.createElement('td');
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'dev-link';
        if (run.text) {
            button.textContent = 'View';
            button.addEventListener('click', () => showOutput(run));
        } else {
            button.textContent = 'not stored';
            button.disabled = true;
        }
        last.appendChild(button);
        row.appendChild(last);
        runRows.appendChild(row);
    });
}

function renderBands(runs) {
    lengthRows.replaceChildren();
    BANDS.forEach((band, index) => {
        const floor = index ? BANDS[index - 1].max : 0;
        const inBand = runs.filter((run) => run.audio_sec > floor && run.audio_sec <= band.max);
        const ratios = inBand.map((run) => run.decode_ratio).filter((value) => value != null);
        const tails = inBand.map((run) => run.finalise_sec).filter(Boolean);

        const row = document.createElement('tr');
        cell(row, band.label);
        cell(row, String(inBand.length), 'num');
        const ratio = median(ratios);
        const td = cell(row, ratio == null ? '—' : ratio.toFixed(2), 'num');
        if (ratio != null && ratio > 1) td.classList.add('over');
        const tail = median(tails);
        cell(row, tail == null ? '—' : secs(tail), 'num');
        lengthRows.appendChild(row);
    });
}

async function load() {
    try {
        const response = await fetch('/api/runs');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const runs = (await response.json()).runs || [];
        renderRuns(runs);
        renderBands(runs);
        const audio = runs.reduce((total, run) => total + (run.audio_sec || 0), 0);
        devSummary.textContent = runs.length
            ? `${runs.length} runs · ${secs(audio)} of audio`
            : 'nothing recorded yet';
    } catch (err) {
        devSummary.textContent = `Could not read the run log: ${err.message}`;
    }
}

document.getElementById('outputClose').addEventListener('click', hideOutput);
outputScrim.addEventListener('click', (event) => {
    if (event.target === outputScrim) hideOutput();
});
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !outputScrim.hidden) hideOutput();
});

load();
