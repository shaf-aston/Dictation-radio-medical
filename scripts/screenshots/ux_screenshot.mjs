import { chromium } from 'playwright';
import { mkdirSync, existsSync, readFileSync } from 'fs';

const OUT = './ux_screenshots';
mkdirSync(OUT, { recursive: true });

// ── 1. Resolve BASE URL ───────────────────────────────────────────────────
// Priority: APP_URL env → package.json port hint → config file hint → port probe
async function resolveBase(browser) {
  if (process.env.APP_URL) return process.env.APP_URL;

  // Parse package.json dev/start scripts for explicit port
  try {
    const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
    const scripts = Object.values(pkg.scripts || {}).join(' ');
    const m = scripts.match(/(?:--port[= ]|PORT=)(\d{4,5})/);
    if (m) return `http://localhost:${m[1]}`;
  } catch {}

  // Check common framework config files
  for (const cfg of ['vite.config.js', 'vite.config.ts', 'next.config.js', 'next.config.mjs']) {
    if (existsSync(cfg)) {
      const m = readFileSync(cfg, 'utf8').match(/\bport[:\s]+(\d{4,5})/);
      if (m) return `http://localhost:${m[1]}`;
    }
  }

  // Probe common dev ports — first one that responds wins
  const probe = await browser.newPage();
  for (const port of [3000, 5173, 4200, 8080, 8000, 4000, 3001, 8005, 1234]) {
    try {
      const res = await probe.goto(`http://localhost:${port}`, { timeout: 1500, waitUntil: 'commit' });
      if (res) { await probe.close(); return `http://localhost:${port}`; }
    } catch {}
  }
  await probe.close();
  console.warn('⚠  No running dev server detected — using http://localhost:3000');
  return 'http://localhost:3000';
}

// ── 2. Helpers ────────────────────────────────────────────────────────────
async function shot(page, name) {
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: false });
  console.log('captured', name);
}

/** Return locale-appropriate sample text based on the page's <html lang>. */
async function sampleText(page) {
  const lang = await page.evaluate(() => document.documentElement.lang || 'en');
  if (lang.startsWith('ar')) return 'الكتاب مفيد';
  if (lang.startsWith('zh')) return '你好世界';
  if (lang.startsWith('ja')) return 'こんにちは';
  if (lang.startsWith('he')) return 'שלום עולם';
  return 'The quick brown fox';
}

/** First *visible* element matching a CSS selector. */
async function find(page, selector) {
  for (const el of await page.$$(selector)) {
    if (await el.isVisible()) return el;
  }
  return null;
}

/** First visible button/control whose aria-label, title, or text matches any keyword. */
async function findByKeyword(page, ...keywords) {
  for (const kw of keywords) {
    const el = await find(page,
      `[aria-label*="${kw}" i], [title*="${kw}" i], button:has-text("${kw}"), [role=button]:has-text("${kw}")`
    );
    if (el) return el;
  }
  return null;
}

// ── 3. Run ────────────────────────────────────────────────────────────────
const browser = await chromium.launch({ headless: true });
const BASE    = await resolveBase(browser);
console.log('Testing against', BASE);

// ── Desktop 1280×800 ─────────────────────────────────────────────────────
const pg = await (await browser.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
await pg.goto(BASE, { waitUntil: 'networkidle', timeout: 15000 }).catch(() => pg.goto(BASE));
await shot(pg, '01_initial');

const text  = await sampleText(pg);
const INPUT = 'textarea, [contenteditable="true"], input[type=text], input[type=search], input:not([type])';
const input = await find(pg, INPUT);
if (input) {
  await input.click();      await shot(pg, '02_input_focused');
  await input.type(text);   await shot(pg, '03_input_filled');
}

// Primary navigation tabs
const tabs = await pg.$$('[role=tab], nav a, nav button, [aria-pressed]');
for (let i = 0; i < Math.min(tabs.length, 6); i++) {
  try {
    await tabs[i].click(); await pg.waitForTimeout(600);
    const label = (await tabs[i].textContent() || `tab${i}`).trim().replace(/\s+/g, '_').slice(0, 20);
    await shot(pg, `04_tab_${i}_${label}`);
  } catch {}
}

// Primary action  (submit / save / send / record / dictate …)
const primaryBtn = await findByKeyword(pg, 'submit', 'save', 'send', 'confirm', 'start', 'record', 'dictate');
if (primaryBtn) { await primaryBtn.click(); await pg.waitForTimeout(800); await shot(pg, '05_primary_action'); }

// Secondary action  (copy / export / download / share …)
const secondaryBtn = await findByKeyword(pg, 'copy', 'export', 'download', 'share');
if (secondaryBtn) { await secondaryBtn.click(); await pg.waitForTimeout(400); await shot(pg, '06_secondary_action'); }

// Destructive / reset action  (clear / reset / delete …)
const destructiveBtn = await findByKeyword(pg, 'clear', 'reset', 'delete', 'remove');
if (destructiveBtn) { await destructiveBtn.click(); await pg.waitForTimeout(400); await shot(pg, '07_destructive_action'); }

// Error / 404 state
await pg.goto(BASE + '/nonexistent', { timeout: 5000 }).catch(() => {});
await pg.waitForTimeout(500); await shot(pg, '08_error_state');

// ── Mobile 375×812 ───────────────────────────────────────────────────────
const mp = await (await browser.newContext({ viewport: { width: 375, height: 812 } })).newPage();
await mp.goto(BASE, { waitUntil: 'networkidle', timeout: 15000 }).catch(() => mp.goto(BASE));
await shot(mp, '09_mobile_initial');
const mInput = await find(mp, INPUT);
if (mInput) { await mInput.click(); await shot(mp, '10_mobile_input_focused'); }

// ── Keyboard navigation ───────────────────────────────────────────────────
const kp = await (await browser.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
await kp.goto(BASE, { waitUntil: 'networkidle', timeout: 15000 }).catch(() => kp.goto(BASE));
for (let i = 0; i < 5; i++) await kp.keyboard.press('Tab');
await shot(kp, '11_keyboard_focus_5tabs');

// ── Responsive Matrix — uncomment as needed ───────────────────────────────
// Small mobile 320×568:   newContext({ viewport: { width: 320,  height: 568  } })
// Large mobile 414×896:   newContext({ viewport: { width: 414,  height: 896  } })
// Tablet       768×1024:  newContext({ viewport: { width: 768,  height: 1024 } })
// Laptop       1440×900:  newContext({ viewport: { width: 1440, height: 900  } })
// Ultra-wide   2560×1440: newContext({ viewport: { width: 2560, height: 1440 } })

await browser.close();
console.log('All screenshots saved to', OUT);
