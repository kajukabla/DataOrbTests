// Simple dev server with log relay endpoint
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = 4567;

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
};

const server = http.createServer((req, res) => {
  // Log relay endpoint
  if (req.method === 'POST' && req.url === '/log') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const { level, args } = JSON.parse(body);
        const prefix = level === 'error' ? '\x1b[31m[ERR]\x1b[0m' :
                       level === 'warn'  ? '\x1b[33m[WRN]\x1b[0m' :
                                           '\x1b[36m[LOG]\x1b[0m';
        console.log(`${prefix} ${args.join(' ')}`);
      } catch {}
      res.writeHead(200);
      res.end();
    });
    return;
  }

  // Static file serving
  let filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url);
  const ext = path.extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Fluid Platformer server running at http://localhost:${PORT}`);
});
