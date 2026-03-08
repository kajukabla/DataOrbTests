const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { exec } = require('node:child_process');

const PORT = 8081;
const ROOT = __dirname;
const DATA_DIR = path.join(__dirname, 'data');
const AUTO_OPEN_BROWSER = process.env.FLUID_AUTO_OPEN_BROWSER === '1';
const RUNS_DIR = path.join(DATA_DIR, 'runs');
const RATINGS_FILE = path.join(DATA_DIR, 'ratings.json');
const EXAMPLES_FILE = path.join(DATA_DIR, 'examples.json');

const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
};

function sanitizeRunName(name) {
  if (typeof name !== 'string') return null;
  const clean = name.replace(/[^a-zA-Z0-9_ -]/g, '').trim();
  if (!clean || clean.length > 64) return null;
  return clean;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try { resolve(JSON.parse(body)); }
      catch { resolve(null); }
    });
    req.on('error', reject);
  });
}

function jsonResponse(res, status, data) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  res.end(JSON.stringify(data));
}

const server = http.createServer(async (req, res) => {
  const reqPath = (req.url || '/').split('?')[0] || '/';

  // CORS preflight for /api/*
  if (req.method === 'OPTIONS' && reqPath.startsWith('/api/')) {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    res.end();
    return;
  }

  // ─── Ratings API ─────────────────────────────────────────────
  if (reqPath === '/api/ratings') {
    if (req.method === 'GET') {
      try {
        const data = fs.readFileSync(RATINGS_FILE, 'utf-8');
        jsonResponse(res, 200, JSON.parse(data));
      } catch {
        jsonResponse(res, 200, []);
      }
      return;
    }
    if (req.method === 'PUT') {
      const body = await readBody(req);
      if (!body || !Array.isArray(body.ratings)) {
        jsonResponse(res, 400, { error: 'Expected {ratings:[...]}' });
        return;
      }
      fs.mkdirSync(DATA_DIR, { recursive: true });
      fs.writeFileSync(RATINGS_FILE, JSON.stringify(body.ratings));
      jsonResponse(res, 200, { ok: true });
      return;
    }
    if (req.method === 'DELETE') {
      try { fs.unlinkSync(RATINGS_FILE); } catch {}
      jsonResponse(res, 200, { ok: true });
      return;
    }
  }

  // ─── Runs API ────────────────────────────────────────────────
  if (reqPath === '/api/runs' && req.method === 'GET') {
    try {
      fs.mkdirSync(RUNS_DIR, { recursive: true });
      const files = fs.readdirSync(RUNS_DIR)
        .filter(f => f.endsWith('.json'))
        .map(f => f.replace(/\.json$/, ''));
      jsonResponse(res, 200, files);
    } catch {
      jsonResponse(res, 200, []);
    }
    return;
  }

  if (reqPath === '/api/runs/save' && req.method === 'POST') {
    const body = await readBody(req);
    const name = sanitizeRunName(body?.name);
    if (!name) { jsonResponse(res, 400, { error: 'Invalid run name' }); return; }
    fs.mkdirSync(RUNS_DIR, { recursive: true });
    try {
      const data = fs.readFileSync(RATINGS_FILE, 'utf-8');
      fs.writeFileSync(path.join(RUNS_DIR, `${name}.json`), data);
      jsonResponse(res, 200, { ok: true });
    } catch {
      jsonResponse(res, 400, { error: 'No ratings to save' });
    }
    return;
  }

  if (reqPath === '/api/runs/load' && req.method === 'POST') {
    const body = await readBody(req);
    const name = sanitizeRunName(body?.name);
    if (!name) { jsonResponse(res, 400, { error: 'Invalid run name' }); return; }
    const runFile = path.join(RUNS_DIR, `${name}.json`);
    try {
      const data = fs.readFileSync(runFile, 'utf-8');
      fs.mkdirSync(DATA_DIR, { recursive: true });
      fs.writeFileSync(RATINGS_FILE, data);
      jsonResponse(res, 200, JSON.parse(data));
    } catch {
      jsonResponse(res, 404, { error: 'Run not found' });
    }
    return;
  }

  if (reqPath === '/api/runs/delete' && req.method === 'DELETE') {
    const body = await readBody(req);
    const name = sanitizeRunName(body?.name);
    if (!name) { jsonResponse(res, 400, { error: 'Invalid run name' }); return; }
    try { fs.unlinkSync(path.join(RUNS_DIR, `${name}.json`)); } catch {}
    jsonResponse(res, 200, { ok: true });
    return;
  }

  // ─── Examples API ──────────────────────────────────────────────
  if (reqPath === '/api/examples') {
    if (req.method === 'GET') {
      try {
        const data = fs.readFileSync(EXAMPLES_FILE, 'utf-8');
        jsonResponse(res, 200, JSON.parse(data));
      } catch {
        jsonResponse(res, 200, []);
      }
      return;
    }
    if (req.method === 'POST') {
      const body = await readBody(req);
      const name = sanitizeRunName(body?.name);
      if (!name) { jsonResponse(res, 400, { error: 'Invalid example name' }); return; }
      if (!body?.params || typeof body.params !== 'object') { jsonResponse(res, 400, { error: 'Missing params' }); return; }
      fs.mkdirSync(DATA_DIR, { recursive: true });
      let examples = [];
      try { examples = JSON.parse(fs.readFileSync(EXAMPLES_FILE, 'utf-8')); } catch {}
      examples = examples.filter(e => e.name !== name);
      examples.push({ name, params: body.params, timestamp: Date.now() });
      fs.writeFileSync(EXAMPLES_FILE, JSON.stringify(examples));
      jsonResponse(res, 200, { ok: true });
      return;
    }
    if (req.method === 'DELETE') {
      const body = await readBody(req);
      const name = sanitizeRunName(body?.name);
      if (!name) { jsonResponse(res, 400, { error: 'Invalid example name' }); return; }
      try {
        let examples = JSON.parse(fs.readFileSync(EXAMPLES_FILE, 'utf-8'));
        examples = examples.filter(e => e.name !== name);
        fs.writeFileSync(EXAMPLES_FILE, JSON.stringify(examples));
      } catch {}
      jsonResponse(res, 200, { ok: true });
      return;
    }
  }

  // ─── Log relay ───────────────────────────────────────────────
  if (req.method === 'POST' && reqPath === '/log') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const { level, args } = JSON.parse(body);
        const tag = level === 'error' ? '\x1b[31mERR\x1b[0m' :
                    level === 'warn'  ? '\x1b[33mWRN\x1b[0m' :
                                        '\x1b[36mLOG\x1b[0m';
        console.log(`[browser ${tag}] ${args.join(' ')}`);
      } catch {
        console.log(`[browser] ${body}`);
      }
      res.writeHead(200, { 'Access-Control-Allow-Origin': '*' });
      res.end('ok');
    });
    return;
  }

  if (req.method === 'OPTIONS' && reqPath === '/log') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    res.end();
    return;
  }

  if (reqPath === '/favicon.ico') {
    res.writeHead(204, { 'Cache-Control': 'no-store' });
    res.end();
    return;
  }

  const url = reqPath === '/' ? '/index.html' : reqPath;
  const filePath = path.join(ROOT, url);
  const ext = path.extname(filePath);

  console.log(`${req.method} ${url}`);

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not Found');
      return;
    }
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(data);
  });
});

server.listen(PORT, () => {
  const url = `http://localhost:${PORT}`;
  console.log(`Server running at ${url}`);
  if (AUTO_OPEN_BROWSER) {
    exec(`open ${url}`);
  }
});
