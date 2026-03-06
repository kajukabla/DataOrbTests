#!/usr/bin/env node
import fs from 'node:fs';

const url = process.argv[2] || 'http://localhost:8081/';
const runs = Number(process.argv[3] || 3);

const { chromium } = await import('playwright');

const browser = await chromium.launch({
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
  ],
});

const allRuns = [];

for (let r = 0; r < runs; r++) {
  const context = await browser.newContext({
    viewport: { width: 1600, height: 900 },
    permissions: ['camera'],
  });
  const page = await context.newPage();
  const logs = [];
  try {
    page.on('console', (msg) => {
      logs.push(`[${msg.type()}] ${msg.text()}`);
    });
    page.on('pageerror', (err) => {
      logs.push(`[pageerror] ${err.message}`);
    });

    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForTimeout(2500);

    await page.evaluate(() => {
      document.getElementById('settingsToggle')?.click();
    });
    await page.waitForTimeout(120);
    await page.evaluate(() => {
      const sel = document.getElementById('faceDebugMode');
      if (!sel) return;
      sel.value = '2';
      sel.dispatchEvent(new Event('input', { bubbles: true }));
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitForTimeout(80);

    const btnBefore = await page.locator('#faceTrackingToggle').innerText();
    if (/start/i.test(btnBefore)) {
      await page.evaluate(() => {
        document.getElementById('faceTrackingToggle')?.click();
      });
      await page.waitForTimeout(100);
    }

    const samples = [];
    for (let i = 0; i < 40; i++) {
      const sample = await page.evaluate(() => {
        const status = document.getElementById('faceTrackingStatus')?.textContent?.trim() || '';
        const button = document.getElementById('faceTrackingToggle')?.textContent?.trim() || '';
        const mode = document.getElementById('faceDebugMode')?.value || '';
        const canvas = document.getElementById('faceDebugCanvas');
        const styleDisplay = canvas ? getComputedStyle(canvas).display : 'missing';
        const w = canvas?.width || 0;
        const h = canvas?.height || 0;
        return { t: performance.now(), status, button, mode, styleDisplay, w, h };
      });
      samples.push(sample);
      await page.waitForTimeout(250);
    }

    const shouldStop = samples.some(s => /Start Webcam Face Tracking/i.test(s.button));
    const debugHidden = samples.some(s => s.styleDisplay === 'none' || s.w === 0 || s.h === 0);

    const runOut = {
      run: r + 1,
      shouldStop,
      debugHidden,
      first: samples[0],
      last: samples[samples.length - 1],
      samples,
      logTail: logs.slice(-80),
    };
    allRuns.push(runOut);
    await page.screenshot({ path: `/tmp/face-probe-run-${r + 1}.png`, fullPage: true });
  } catch (err) {
    allRuns.push({
      run: r + 1,
      failed: true,
      error: err?.message || String(err),
      logTail: logs.slice(-80),
    });
  } finally {
    try {
      await context.close();
    } catch {}
  }
}

try {
  await browser.close();
} catch {}

const report = {
  ts: new Date().toISOString(),
  url,
  runs: allRuns,
};

const outPath = '/tmp/face-tracking-probe-report.json';
fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(`report=${outPath}`);
for (const r of allRuns) {
  if (r.failed) {
    console.log(`run#${r.run}: failed error="${r.error}"`);
  } else {
    console.log(`run#${r.run}: stopToggled=${r.shouldStop} debugHidden=${r.debugHidden} firstStatus="${r.first?.status || ''}" lastStatus="${r.last?.status || ''}"`);
  }
}
