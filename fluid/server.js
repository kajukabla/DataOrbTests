const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { exec } = require('node:child_process');

const PORT = 8081;
const ROOT = __dirname;

const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.css':  'text/css',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
};

const server = http.createServer((req, res) => {
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
