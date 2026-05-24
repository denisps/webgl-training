const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function extname(path) {
  const index = path.lastIndexOf('.');
  return index >= 0 ? path.slice(index) : '';
}

function sanitizePath(urlPath) {
  const decoded = decodeURIComponent(urlPath);
  const cleaned = decoded.replace(/\\/g, '/');
  const normalized = cleaned.split('/').filter((segment) => segment && segment !== '.');
  if (normalized.some((segment) => segment === '..')) {
    throw new Error('Invalid path');
  }
  return normalized.join('/');
}

async function resolvePath(urlPath) {
  const relative = urlPath === '/' ? 'public/index.html' : sanitizePath(urlPath);
  const candidates = [];
  if (relative.endsWith('/')) {
    candidates.push(`${relative}index.html`);
  } else {
    candidates.push(relative);
    candidates.push(`${relative}/index.html`);
  }

  for (const candidate of candidates) {
    const file = Bun.file(candidate);
    if (await file.exists()) {
      return candidate;
    }
  }
  return null;
}

Bun.serve({
  port: 3000,
  async fetch(request) {
    try {
      const url = new URL(request.url);
      const filePath = await resolvePath(url.pathname);
      if (!filePath) {
        return new Response('Not Found', { status: 404 });
      }
      const file = Bun.file(filePath);
      return new Response(file, {
        headers: {
          'Content-Type': MIME_TYPES[extname(filePath)] || 'application/octet-stream',
        },
      });
    } catch {
      return new Response('Bad Request', { status: 400 });
    }
  },
});

console.log('WebGL Training server running at http://localhost:3000');
