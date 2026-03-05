const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { exec } = require('node:child_process');

const PORT = 8081;
const ROOT = __dirname;
const DATA_DIR = path.join(__dirname, 'data');
const RUNS_DIR = path.join(DATA_DIR, 'runs');
const RATINGS_FILE = path.join(DATA_DIR, 'ratings.json');
const EXAMPLES_FILE = path.join(DATA_DIR, 'examples.json');
const RECORDINGS_FILE = path.join(DATA_DIR, 'recordings.json');

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
  const clean = name.replace(/[^a-zA-Z0-9_-]/g, '');
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
  // CORS preflight for /api/*
  if (req.method === 'OPTIONS' && req.url.startsWith('/api/')) {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    res.end();
    return;
  }

  // ─── Ratings API ─────────────────────────────────────────────
  if (req.url === '/api/ratings') {
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
  if (req.url === '/api/runs' && req.method === 'GET') {
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

  if (req.url === '/api/runs/save' && req.method === 'POST') {
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

  if (req.url === '/api/runs/load' && req.method === 'POST') {
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

  if (req.url === '/api/runs/delete' && req.method === 'DELETE') {
    const body = await readBody(req);
    const name = sanitizeRunName(body?.name);
    if (!name) { jsonResponse(res, 400, { error: 'Invalid run name' }); return; }
    try { fs.unlinkSync(path.join(RUNS_DIR, `${name}.json`)); } catch {}
    jsonResponse(res, 200, { ok: true });
    return;
  }

  // ─── Examples API ──────────────────────────────────────────────
  if (req.url === '/api/examples') {
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

  // ─── Recordings API ───────────────────────────────────────────
  if (req.url === '/api/recordings') {
    if (req.method === 'GET') {
      try {
        const data = fs.readFileSync(RECORDINGS_FILE, 'utf-8');
        jsonResponse(res, 200, JSON.parse(data));
      } catch {
        jsonResponse(res, 200, []);
      }
      return;
    }
    if (req.method === 'POST') {
      const body = await readBody(req);
      if (!body?.points || !Array.isArray(body.points) || body.points.length < 5) {
        jsonResponse(res, 400, { error: 'Need at least 5 points' });
        return;
      }
      fs.mkdirSync(DATA_DIR, { recursive: true });
      let recordings = [];
      try { recordings = JSON.parse(fs.readFileSync(RECORDINGS_FILE, 'utf-8')); } catch {}
      const name = body.name || `gesture-${Date.now()}`;
      recordings.push({ name, timestamp: Date.now(), points: body.points });
      fs.writeFileSync(RECORDINGS_FILE, JSON.stringify(recordings));
      jsonResponse(res, 200, { ok: true });
      return;
    }
    if (req.method === 'DELETE') {
      const body = await readBody(req);
      if (body?.all) {
        try { fs.unlinkSync(RECORDINGS_FILE); } catch {}
        jsonResponse(res, 200, { ok: true });
        return;
      }
      const name = body?.name;
      if (!name) { jsonResponse(res, 400, { error: 'Specify name or {all:true}' }); return; }
      try {
        let recordings = JSON.parse(fs.readFileSync(RECORDINGS_FILE, 'utf-8'));
        recordings = recordings.filter(r => r.name !== name);
        fs.writeFileSync(RECORDINGS_FILE, JSON.stringify(recordings));
      } catch {}
      jsonResponse(res, 200, { ok: true });
      return;
    }
  }

  // ─── Log relay ───────────────────────────────────────────────
  if (req.method === 'POST' && req.url === '/log') {
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

  if (req.method === 'OPTIONS' && req.url === '/log') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    res.end();
    return;
  }

  const url = req.url === '/' ? '/index.html' : req.url.split('?')[0];
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
  exec(`open ${url}`);
});
