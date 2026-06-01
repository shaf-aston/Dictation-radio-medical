import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
const OUT = './ux_screenshots/after';
mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch({ headless: true });

const pg = await (await browser.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
await pg.goto('http://localhost:8005', { waitUntil: 'networkidle', timeout: 15000 });
await pg.screenshot({ path: OUT + '/01_initial_viewport.png', fullPage: false });
await pg.screenshot({ path: OUT + '/02_fullpage.png', fullPage: true });

const summary = await pg.locator('#advancedSettings > summary').first();
if (summary) { await summary.click(); await pg.waitForTimeout(350); }
await pg.screenshot({ path: OUT + '/03_settings_open.png', fullPage: false });

const mp = await (await browser.newContext({ viewport: { width: 375, height: 812 } })).newPage();
await mp.goto('http://localhost:8005', { waitUntil: 'networkidle', timeout: 15000 });
await mp.screenshot({ path: OUT + '/04_mobile.png', fullPage: false });

console.log('done');
await browser.close();
