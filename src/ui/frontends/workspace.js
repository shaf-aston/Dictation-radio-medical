/* ===========================================================================
   Workspace: the arrangement round the report.

   - Sections are *read* from the report text, never drawn into it: a heading
     is a line such as "FINDINGS:" or "Impression:". The editor stays one plain
     textarea, so typing, undo and dictation behave exactly as before.
   - The outline lists them; clicking one puts the caret at the end of that
     section, which is where the next dictation goes.
   - Panels (Dictation, Sections, Templates & settings) dock left, right or at
     the bottom, float, or hide: by dragging the grip or from the dock menu.
   - Pinned sections are mirrored, read-only, under the report.
   - Report text size is the radiologist's own.

   Loaded after app.js and dictate.js; reads only the `editor` element and the
   record button's aria-pressed, so nothing in those files depends on this one.
   =========================================================================== */
'use strict';

/* ---------------------------------------------------------------- parsing */

/** Top-level report headings, matched in any case ("Findings:" or "FINDINGS:"). */
const WS_MAIN_HEADINGS = new Set([
    'EXAMINATION', 'EXAM', 'CLINICAL HISTORY', 'CLINICAL INDICATION', 'CLINICAL DETAILS',
    'CLINICAL INFORMATION', 'HISTORY', 'INDICATION', 'TECHNIQUE', 'PROTOCOL', 'SEQUENCES',
    'COMPARISON', 'FINDINGS', 'RESULTS', 'IMPRESSION', 'CONCLUSION', 'OPINION', 'SUMMARY',
    'RECOMMENDATION', 'RECOMMENDATIONS', 'COMMENT', 'COMMENTS', 'REPORT',
]);

// A heading is a label of up to six words at the start of a line, then a colon.
// Anything may follow on the same line ("EXAMINATION: CT Head").
const WS_HEADING = /^[ \t]*([A-Za-z][A-Za-z0-9 &/()'\-]{0,48}?)[ \t]*:(?![0-9])[ \t]*([^\r]*)\r?$/;

/**
 * Sections of *text*, in order. Level 1: a known report heading in any case.
 * Level 2: any other ALL-CAPS label ("AIRWAYS:", "HAEMORRHAGE:"). Title-case
 * labels that are not known headings ("Trachea:") are body text, not sections:
 * a template is full of them and an outline of every one is no outline.
 *
 * Each section: { title, level, start, headEnd, end, body }
 *   start   index of the heading line
 *   headEnd index just past the heading line's text
 *   end     index where the section's own text ends (trailing blank lines excluded)
 *   body    the section's text, heading removed, for summaries and pins
 */
function parseSections(text) {
    const sections = [];
    let pos = 0;
    const lines = text.split('\n');
    for (const line of lines) {
        const m = WS_HEADING.exec(line);
        if (m) {
            const label = m[1].trim();
            const upper = label.toUpperCase();
            const words = label.split(/\s+/).length;
            let level = 0;
            if (WS_MAIN_HEADINGS.has(upper)) level = 1;
            else if (label === upper && /[A-Z]{2}/.test(label) && words <= 6) level = 2;
            if (level) {
                const inline = m[2] + (line.endsWith('\r') ? '\r' : '');
                sections.push({
                    title: label, key: upper, level, start: pos,
                    headEnd: pos + line.length,
                    bodyStart: pos + line.length - inline.length,
                });
            }
        }
        pos += line.length + 1;
    }
    for (let i = 0; i < sections.length; i++) {
        const s = sections[i];
        const nextAny = i + 1 < sections.length ? sections[i + 1].start : text.length;
        // A level-1 section owns its level-2 children, for pins and summaries.
        let nextSame = text.length;
        for (let j = i + 1; j < sections.length; j++) {
            if (sections[j].level <= s.level) { nextSame = sections[j].start; break; }
        }
        s.end = trimEndIndex(text, s.bodyStart, nextAny);
        s.ownEnd = trimEndIndex(text, s.bodyStart, nextSame);
        s.body = text.slice(s.bodyStart, s.ownEnd).trim();
        s.own = text.slice(s.bodyStart, s.end).trim();   // without sub-sections: for the outline's one-line summary
    }
    return sections;
}

function trimEndIndex(text, from, to) {
    let e = to;
    while (e > from && /\s/.test(text[e - 1])) e--;
    return e;
}

/** The section the index *pos* sits in, and its level-1 parent. */
function sectionAt(sections, pos) {
    let here = null, parent = null;
    for (const s of sections) {
        if (s.start > pos) break;
        here = s;
        if (s.level === 1) parent = s;
    }
    if (here && here.level === 1) parent = null;
    return { here, parent };
}

/** One line of summary: the first words of a section, without template fields. */
function summarise(body, max = 64) {
    const flat = body.replace(/\s+/g, ' ').trim();
    return flat.length > max ? flat.slice(0, max - 1).trimEnd() + '…' : flat;
}

if (typeof module !== 'undefined') module.exports = { parseSections, sectionAt, summarise };

/* -------------------------------------------------------------------- UI */

if (typeof document !== 'undefined') (() => {
    const KEY = 'radio-dictate-web-workspace-v1';
    const ORDER = ['outline', 'mic', 'library'];
    const DEFAULT = {
        dock: { outline: 'left', mic: 'right', library: 'right' },
        float: {},
        pins: ['IMPRESSION', 'CONCLUSION', 'OPINION'],
        fs: null,
        focus: false,
        lastLibrary: 'right',
    };
    const NAMES = { outline: 'Sections', mic: 'Dictation', library: 'Templates & settings' };
    const ZONES = { left: 'Dock left', right: 'Dock right', bottom: 'Dock at the bottom', float: 'Float', hidden: 'Hide' };
    const ZICON = { left: 'fa-dock-left', right: 'fa-dock-right', bottom: 'fa-dock-bottom', float: 'fa-dock-float', hidden: 'fa-eye-slash' };

    const $ = (id) => document.getElementById(id);
    const ed = $('editor');
    const main = document.querySelector('.main.ws');
    if (!ed || !main) return;

    const W = {};
    document.querySelectorAll('.ws-w[data-w]').forEach((w) => { W[w.dataset.w] = w; });

    let S;
    try {
        const saved = JSON.parse(localStorage.getItem(KEY) || '{}');
        S = { ...DEFAULT, ...saved, dock: { ...DEFAULT.dock, ...(saved.dock || {}) }, float: { ...(saved.float || {}) } };
    } catch (_) {
        S = { ...DEFAULT, dock: { ...DEFAULT.dock }, float: {} };
    }
    if (S.dock.mic === 'hidden') S.dock.mic = 'right';   // the record button is never lost
    const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (_) { /* private mode */ } };
    const wide = window.matchMedia('(min-width: 68rem)');

    /* ---------------------------------------------------------- docking */

    function effectiveZone(k) {
        if (S.focus) return k === 'mic' ? 'float' : 'hidden';
        return S.dock[k];
    }

    function layout() {
        for (const k of ORDER) {
            const w = W[k];
            if (!w) continue;
            const z = effectiveZone(k);
            w.style.left = w.style.top = w.style.width = '';
            if (z === 'hidden') $('wsParking').append(w);
            else if (z === 'float') { $('wsFloat').append(w); placeFloat(k); }
            else $('wsZone' + z[0].toUpperCase() + z.slice(1)).append(w);
        }
        document.body.classList.toggle('ws-focus', !!S.focus);
        $('wsFocusBtn').setAttribute('aria-pressed', String(!!S.focus));
        $('wsLibBtn').setAttribute('aria-pressed', String(effectiveZone('library') !== 'hidden'));
        save();
        // The editor's width may have changed: let app.js re-wrap its marks.
        window.dispatchEvent(new Event('resize'));
    }

    function placeFloat(k) {
        const w = W[k];
        const box = main.getBoundingClientRect();
        const pos = (S.focus && k === 'mic') ? {} : (S.float[k] || {});
        const bw = w.offsetWidth || 288, bh = w.offsetHeight || 260;
        let x = pos.x ?? box.width - bw - 18;
        let y = pos.y ?? box.height - bh - 18;
        x = Math.max(6, Math.min(box.width - bw - 6, x));
        y = Math.max(6, Math.min(box.height - bh - 6, y));
        w.style.left = x + 'px';
        w.style.top = y + 'px';
    }

    function dock(k, z) {
        if (k === 'mic' && z === 'hidden') z = 'float';
        if (k === 'library' && z !== 'hidden') S.lastLibrary = z;
        S.focus = false;
        S.dock[k] = z;
        layout();
    }

    // Drag a panel by its grip; drop on a zone to dock, anywhere else to float.
    main.addEventListener('pointerdown', (e) => {
        const grip = e.target.closest('.ws-grip');
        if (!grip || !wide.matches || e.button !== 0) return;
        e.preventDefault();
        const w = grip.closest('.ws-w'), k = w.dataset.w;
        const box = main.getBoundingClientRect(), r = w.getBoundingClientRect();
        const dx = e.clientX - r.left, dy = e.clientY - r.top;
        document.body.classList.add('ws-dragging');
        w.classList.add('is-dragging');
        w.style.width = r.width + 'px';
        $('wsFloat').append(w);
        let target = null;
        const move = (ev) => {
            w.style.left = (ev.clientX - box.left - dx) + 'px';
            w.style.top = (ev.clientY - box.top - dy) + 'px';
            w.style.visibility = 'hidden';
            const under = document.elementFromPoint(ev.clientX, ev.clientY);
            w.style.visibility = '';
            const zone = under && under.closest('.ws-zone');
            document.querySelectorAll('.ws-zone').forEach((z) => z.classList.toggle('is-drop', z === zone));
            target = zone ? zone.dataset.zone : null;
        };
        const up = () => {
            window.removeEventListener('pointermove', move);
            window.removeEventListener('pointerup', up);
            document.body.classList.remove('ws-dragging');
            w.classList.remove('is-dragging');
            document.querySelectorAll('.ws-zone').forEach((z) => z.classList.remove('is-drop'));
            if (target) dock(k, target);
            else {
                S.float[k] = { x: parseFloat(w.style.left), y: parseFloat(w.style.top) };
                dock(k, 'float');
            }
        };
        window.addEventListener('pointermove', move);
        window.addEventListener('pointerup', up);
    });

    // The same moves by click: a dock menu per panel, and one for all panels.
    const pop = $('wsPop');
    function openPop(anchor, keys) {
        pop.innerHTML = keys.map((k) => `<h6>${NAMES[k]}</h6>` + Object.keys(ZONES)
            .filter((z) => !(k === 'mic' && z === 'hidden'))
            .map((z) => `<button type="button" role="menuitemradio" data-k="${k}" data-z="${z}" aria-checked="${!S.focus && S.dock[k] === z}"><i class="fas ${ZICON[z]}" aria-hidden="true"></i>${ZONES[z]}</button>`)
            .join('')).join('<hr>')
            + (keys.length > 1 ? '<hr><button type="button" data-reset><i class="fas fa-rotate-right" aria-hidden="true"></i>Reset layout</button>' : '');
        pop.hidden = false;
        const r = anchor.getBoundingClientRect();
        const pw = pop.offsetWidth, ph = pop.offsetHeight;
        pop.style.left = Math.max(8, Math.min(innerWidth - pw - 8, r.right - pw)) + 'px';
        pop.style.top = (r.bottom + ph + 6 > innerHeight ? Math.max(8, r.top - ph - 6) : r.bottom + 6) + 'px';
        const first = pop.querySelector('button');
        if (first) first.focus();
    }
    document.addEventListener('click', (e) => {
        const menuBtn = e.target.closest('[data-dockmenu]');
        if (menuBtn) { e.stopPropagation(); openPop(menuBtn, [menuBtn.closest('.ws-w').dataset.w]); return; }
        const item = e.target.closest('#wsPop button');
        if (item) {
            if (item.hasAttribute('data-reset')) {
                S.dock = { ...DEFAULT.dock }; S.float = {}; S.focus = false; layout();
            } else dock(item.dataset.k, item.dataset.z);
            pop.hidden = true;
            return;
        }
        if (!e.target.closest('#wsPop') && !e.target.closest('#wsLayoutBtn')) pop.hidden = true;
    });
    $('wsLayoutBtn').addEventListener('click', (e) => {
        e.stopPropagation();
        if (!pop.hidden) { pop.hidden = true; return; }
        openPop(e.currentTarget, ORDER);
    });
    $('wsLibBtn').addEventListener('click', toggleLibrary);
    $('wsFocusBtn').addEventListener('click', toggleFocus);

    function toggleLibrary() {
        if (S.focus) { S.focus = false; S.dock.library = S.lastLibrary || 'right'; layout(); return; }
        dock('library', S.dock.library === 'hidden' ? (S.lastLibrary || 'right') : 'hidden');
    }
    function toggleFocus() { S.focus = !S.focus; layout(); }

    window.addEventListener('resize', () => {
        for (const k of ORDER) if (effectiveZone(k) === 'float') placeFloat(k);
    });

    /* --------------------------------------------------------- sections */

    let sections = [];
    let lastText = null, lastCaret = -1;
    const outline = $('wsOutline');

    function refresh(force) {
        const text = ed.value, caret = ed.selectionStart;
        const textChanged = text !== lastText;
        if (!force && !textChanged && caret === lastCaret) return;
        lastCaret = caret;
        if (textChanged || force) {
            lastText = text;
            sections = parseSections(text);
            renderOutline();
            renderPins();
        }
        markHere();
    }

    function renderOutline() {
        $('wsOutlineEmpty').hidden = sections.length > 0;
        outline.innerHTML = '';
        sections.forEach((s, i) => {
            const li = document.createElement('li');
            li.dataset.i = i;
            li.className = (s.level === 2 ? 'is-sub' : '') + (s.own || s.body ? '' : ' is-empty')
                + (S.pins.includes(s.key) ? ' is-pinned' : '');
            const go = document.createElement('button');
            go.type = 'button';
            go.className = 'ws-go';
            go.title = 'Go to ' + s.title + ': the next dictation goes at the end of it';
            const h = document.createElement('span');
            h.className = 'ws-h';
            h.textContent = s.title;
            const sm = document.createElement('span');
            sm.className = 'ws-s';
            sm.textContent = s.own ? summarise(s.own) : (s.body ? '' : 'empty');
            go.append(h, sm);
            go.addEventListener('click', () => goTo(s));
            li.append(go);
            if (s.level === 1) {
                const pin = document.createElement('button');
                pin.type = 'button';
                pin.className = 'ws-ibtn ws-pin';
                pin.title = S.pins.includes(s.key) ? 'Unpin from under the report' : 'Pin under the report, so it stays in view';
                pin.setAttribute('aria-pressed', String(S.pins.includes(s.key)));
                pin.setAttribute('aria-label', 'Pin ' + s.title);
                pin.innerHTML = '<i class="fas fa-thumbtack" aria-hidden="true"></i>';
                pin.addEventListener('click', () => togglePin(s.key));
                li.append(pin);
            }
            outline.append(li);
        });
    }

    function markHere() {
        const { here, parent } = sectionAt(sections, ed.selectionStart);
        outline.querySelectorAll('li').forEach((li) => {
            const s = sections[+li.dataset.i];
            li.classList.toggle('is-here', s === here);
        });
        const where = $('wsWhere');
        if (!here) { where.textContent = ''; return; }
        const path = (parent ? parent.title + ' › ' : '') + here.title;
        const rec = document.body.classList.contains('ws-rec');
        where.textContent = (rec ? 'Dictating into ' : 'Dictation goes to ') + path;
    }

    /** Put the caret at the end of section *s* and bring it into view. */
    function goTo(s) {
        const at = s.end > s.bodyStart ? s.end : s.headEnd;
        if (!ed.readOnly) {
            ed.focus({ preventScroll: true });
            ed.setSelectionRange(at, at);
        }
        scrollToIndex(s.start);
        refresh(true);
    }

    // Where index *i* sits in the textarea, measured on a hidden copy laid out
    // with the editor's own wrapping (same trick as the marks underlay).
    let mirror = null;
    function scrollToIndex(i) {
        const cs = getComputedStyle(ed);
        if (!mirror) {
            mirror = document.createElement('div');
            mirror.setAttribute('aria-hidden', 'true');
            mirror.style.cssText = 'position:absolute;visibility:hidden;top:0;left:-9999px;white-space:pre-wrap;overflow-wrap:break-word;';
            document.body.append(mirror);
        }
        for (const p of ['font', 'letterSpacing', 'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft', 'boxSizing', 'borderTopWidth', 'borderLeftWidth', 'borderRightWidth', 'tabSize']) {
            mirror.style[p] = cs[p];
        }
        mirror.style.width = ed.clientWidth + 'px';
        mirror.textContent = ed.value.slice(0, i);
        const mark = document.createElement('span');
        mark.textContent = '\u200b';
        mirror.append(mark);
        const top = mark.offsetTop - parseFloat(cs.paddingTop);
        ed.scrollTo({ top: Math.max(0, top - ed.clientHeight * 0.15), behavior: 'smooth' });
    }

    $('wsWhere').addEventListener('click', () => {
        const { here } = sectionAt(sections, ed.selectionStart);
        if (here) scrollToIndex(here.start);
    });

    ['input', 'click', 'keyup', 'select', 'focus'].forEach((ev) => ed.addEventListener(ev, () => refresh(false)));
    // Templates and dictation set the text from script, which fires no event;
    // a cheap comparison catches both without touching app.js or dictate.js.
    setInterval(() => refresh(false), 250);

    // Recording state, from the record button itself.
    const recBtn = $('dictateBtn');
    if (recBtn) {
        new MutationObserver(() => {
            document.body.classList.toggle('ws-rec', recBtn.getAttribute('aria-pressed') === 'true');
            markHere();
        }).observe(recBtn, { attributes: true, attributeFilter: ['aria-pressed'] });
    }

    /* ------------------------------------------------------------- pins */

    function togglePin(key) {
        S.pins = S.pins.includes(key) ? S.pins.filter((p) => p !== key) : [...S.pins, key];
        save();
        renderOutline();
        renderPins();
        markHere();
    }

    function renderPins() {
        const tray = $('wsPins');
        const pinned = sections.filter((s) => s.level === 1 && S.pins.includes(s.key));
        tray.hidden = pinned.length === 0;
        const open = new Set([...tray.querySelectorAll('.ws-pin-card.is-open')].map((c) => c.dataset.key));
        tray.innerHTML = '';
        for (const s of pinned) {
            const card = document.createElement('div');
            card.className = 'ws-pin-card' + (open.has(s.key) ? ' is-open' : '');
            card.dataset.key = s.key;
            const h = document.createElement('span');
            h.className = 'ws-pin-h';
            h.innerHTML = '<i class="fas fa-thumbtack" aria-hidden="true"></i>';
            h.append(document.createTextNode(s.title));
            const acts = document.createElement('span');
            const goBtn = document.createElement('button');
            goBtn.type = 'button';
            goBtn.className = 'ws-ibtn';
            goBtn.title = 'Go to ' + s.title;
            goBtn.setAttribute('aria-label', 'Go to ' + s.title);
            goBtn.innerHTML = '<i class="fas fa-chevron-down" aria-hidden="true"></i>';
            goBtn.addEventListener('click', () => goTo(s));
            const un = document.createElement('button');
            un.type = 'button';
            un.className = 'ws-ibtn';
            un.title = 'Unpin';
            un.setAttribute('aria-label', 'Unpin ' + s.title);
            un.innerHTML = '<i class="fas fa-eye-slash" aria-hidden="true"></i>';
            un.addEventListener('click', () => togglePin(s.key));
            acts.append(goBtn, un);
            const t = document.createElement('p');
            t.className = 'ws-pin-t' + (s.body ? '' : ' is-empty');
            t.textContent = s.body || 'Nothing dictated here yet.';
            t.title = 'Click to show all of it';
            t.addEventListener('click', () => card.classList.toggle('is-open'));
            card.append(h, acts, t);
            tray.append(card);
        }
    }

    /* -------------------------------------------------------- text size */

    const base = parseFloat(getComputedStyle(ed).fontSize) || 16;
    function setFs(px) {
        S.fs = px == null ? null : Math.max(12, Math.min(30, Math.round(px)));
        if (S.fs == null) document.documentElement.style.removeProperty('--ws-report-fs');
        else document.documentElement.style.setProperty('--ws-report-fs', S.fs + 'px');
        $('wsZoomReset').textContent = Math.round((S.fs || base) / base * 100) + '%';
        save();
        window.dispatchEvent(new Event('resize'));
    }
    $('wsZoomIn').addEventListener('click', () => setFs((S.fs || base) + 1));
    $('wsZoomOut').addEventListener('click', () => setFs((S.fs || base) - 1));
    $('wsZoomReset').addEventListener('click', () => setFs(null));

    /* -------------------------------------------------------- shortcuts */

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !pop.hidden) { pop.hidden = true; return; }
        if (!e.altKey || e.ctrlKey || e.metaKey) return;
        const k = e.key.toLowerCase();
        if (k === 't') { e.preventDefault(); toggleLibrary(); }
        else if (k === 'f') { e.preventDefault(); toggleFocus(); }
        else if (k === '=' || k === '+') { e.preventDefault(); setFs((S.fs || base) + 1); }
        else if (k === '-') { e.preventDefault(); setFs((S.fs || base) - 1); }
        else if (/^[1-9]$/.test(k)) {
            const top = sections.filter((s) => s.level === 1)[+k - 1];
            if (top) { e.preventDefault(); goTo(top); }
        }
    });

    /* ------------------------------------------------------------- boot */

    setFs(S.fs);
    layout();
    refresh(true);
})();
