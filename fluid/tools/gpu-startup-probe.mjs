#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

function parseArgs(argv) {
  const out = {
    runs: 20,
    timeoutMs: 45000,
    url: 'http://localhost:8081/?gpu_debug=1',
    outFile: path.resolve(process.cwd(), 'gpu-startup-report.json'),
    headless: true,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--runs' && next) { out.runs = Math.max(1, Number(next) || out.runs); i++; continue; }
    if (arg === '--timeout' && next) { out.timeoutMs = Math.max(1000, Number(next) || out.timeoutMs); i++; continue; }
    if (arg === '--url' && next) { out.url = next; i++; continue; }
    if (arg === '--out' && next) { out.outFile = path.resolve(process.cwd(), next); i++; continue; }
    if (arg === '--headed') { out.headless = false; continue; }
  }
  return out;
}

function percentile(sorted, p) {
  if (!sorted.length) return null;
  if (sorted.length === 1) return sorted[0];
  const rank = p * (sorted.length - 1);
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  if (low === high) return sorted[low];
  const weight = rank - low;
  return sorted[low] * (1 - weight) + sorted[high] * weight;
}

function parseInitLine(logs, regex) {
  for (const line of logs) {
    const m = line.match(regex);
    if (m) return m;
  }
  return null;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  let chromium = null;
  try {
    ({ chromium } = await import('playwright'));
  } catch (err) {
    console.error('Playwright is required. Install with: npm install --no-save playwright');
    console.error(err?.message || String(err));
    process.exit(1);
  }

  const browser = await chromium.launch({
    headless: opts.headless,
    args: ['--enable-unsafe-webgpu'],
  });

  const runs = [];
  for (let i = 0; i < opts.runs; i++) {
    const runId = i + 1;
    const context = await browser.newContext();
    const page = await context.newPage();
    const logs = [];
    let done = false;
    let outcome = 'timeout';

    const finish = (kind) => {
      if (done) return;
      done = true;
      outcome = kind;
    };

    page.on('console', (msg) => {
      const text = msg.text();
      logs.push(`[${msg.type()}] ${text}`);
      if (/Fluid simulation starting\.\.\./i.test(text)) finish('started');
      if (/No WebGPU device acquired|Failed to acquire WebGPU device|No WebGPU adapter/i.test(text)) finish('failed');
    });
    page.on('pageerror', (err) => logs.push(`[pageerror] ${err.message}`));

    const t0 = performance.now();
    try {
      await page.goto(opts.url, { waitUntil: 'domcontentloaded', timeout: opts.timeoutMs });
      const deadline = Date.now() + opts.timeoutMs;
      while (!done && Date.now() < deadline) {
        await page.waitForTimeout(100);
      }
      if (!done) finish('timeout');
    } catch (err) {
      logs.push(`[goto-error] ${err?.message || String(err)}`);
      finish('failed');
    }
    const elapsedMs = Number((performance.now() - t0).toFixed(1));

    let diagSummary = null;
    try {
      diagSummary = await page.evaluate(() => {
        if (typeof window.dumpGpuInitSummary !== 'function') return null;
        return window.dumpGpuInitSummary('startup-probe');
      });
    } catch {}

    const adapterMatch = parseInitLine(logs, /Init: adapter acquired \(([\d.]+)ms, mode=([^,)]+)/i);
    const deviceMatch = parseInitLine(logs, /Init: device acquired \(([\d.]+)ms, total GPU: ([\d.]+)ms, mode=([^,)]+)/i);
    runs.push({
      runId,
      outcome,
      elapsedMs,
      adapterMs: adapterMatch ? Number(adapterMatch[1]) : null,
      adapterMode: adapterMatch ? adapterMatch[2] : null,
      deviceMs: deviceMatch ? Number(deviceMatch[1]) : null,
      totalGpuMs: deviceMatch ? Number(deviceMatch[2]) : null,
      deviceMode: deviceMatch ? deviceMatch[3] : null,
      logTail: logs.slice(-80),
      diagSummary,
    });

    console.log(`#${runId} outcome=${outcome} elapsed=${elapsedMs}ms adapter=${adapterMatch ? `${adapterMatch[1]}ms(${adapterMatch[2]})` : 'n/a'} device=${deviceMatch ? `${deviceMatch[1]}ms total:${deviceMatch[2]}ms (${deviceMatch[3]})` : 'n/a'}`);
    await context.close();
  }

  await browser.close();

  const started = runs.filter(r => r.outcome === 'started');
  const failed = runs.filter(r => r.outcome === 'failed');
  const timedOut = runs.filter(r => r.outcome === 'timeout');
  const deviceTotals = started.map(r => r.totalGpuMs).filter(v => typeof v === 'number').sort((a, b) => a - b);
  const adapterTimes = started.map(r => r.adapterMs).filter(v => typeof v === 'number').sort((a, b) => a - b);

  const summary = {
    timestamp: new Date().toISOString(),
    options: opts,
    counts: {
      runs: runs.length,
      started: started.length,
      failed: failed.length,
      timedOut: timedOut.length,
    },
    startupMs: {
      adapterP50: percentile(adapterTimes, 0.5),
      adapterP95: percentile(adapterTimes, 0.95),
      totalGpuP50: percentile(deviceTotals, 0.5),
      totalGpuP95: percentile(deviceTotals, 0.95),
      maxTotalGpu: deviceTotals.length ? deviceTotals[deviceTotals.length - 1] : null,
    },
    runs,
  };

  fs.mkdirSync(path.dirname(opts.outFile), { recursive: true });
  fs.writeFileSync(opts.outFile, `${JSON.stringify(summary, null, 2)}\n`, 'utf-8');

  console.log('\nSummary');
  console.log(`runs=${summary.counts.runs} started=${summary.counts.started} failed=${summary.counts.failed} timedOut=${summary.counts.timedOut}`);
  console.log(`adapter p50=${summary.startupMs.adapterP50 ?? 'n/a'}ms p95=${summary.startupMs.adapterP95 ?? 'n/a'}ms`);
  console.log(`total GPU p50=${summary.startupMs.totalGpuP50 ?? 'n/a'}ms p95=${summary.startupMs.totalGpuP95 ?? 'n/a'}ms max=${summary.startupMs.maxTotalGpu ?? 'n/a'}ms`);
  console.log(`report=${opts.outFile}`);
}

main().catch((err) => {
  console.error(err?.stack || err?.message || String(err));
  process.exit(1);
});
