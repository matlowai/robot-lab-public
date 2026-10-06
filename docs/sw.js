// Media cache for the course, talk and Splat Lab (GitHub Pages can't set long cache headers).
// Videos, images and splat files are cached on first use and then served from this device, including the
// Range requests browsers make while playing or seeking a video. Pages can also post a list of their media
// URLs ({type: 'warm', urls}) so the next slides are cached in the background before they are shown.
// Bump CACHE whenever a media file is replaced under the same name; old caches are deleted on activate.
const CACHE = 'robot-lab-media-v1';
const VIDEO = /\.(mp4|webm)(\?.*)?$/i;
const MEDIA = /\.(mp4|webm|jpg|jpeg|png|gif|webp|spz|ply|json)(\?.*)?$/i;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith('robot-lab-media-') && k !== CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});

async function full(url, low) {
  const cache = await caches.open(CACHE);
  let r = await cache.match(url);
  if (r) return r;
  // whole file, no Range, so it can be cached; background warm-ups ask for low network priority
  r = await fetch(url, low ? { credentials: 'same-origin', priority: 'low' } : { credentials: 'same-origin' });
  if (r.ok && r.status === 200) await cache.put(url, r.clone());
  return r;
}

function slice(resp, range) {
  return resp.arrayBuffer().then((buf) => {
    const m = /bytes=(\d*)-(\d*)/.exec(range || '');
    const size = buf.byteLength;
    let start = m && m[1] ? parseInt(m[1], 10) : 0;
    let end = m && m[2] ? parseInt(m[2], 10) : size - 1;
    if (m && !m[1] && m[2]) { start = Math.max(0, size - parseInt(m[2], 10)); end = size - 1; }  // suffix range
    end = Math.min(end, size - 1);
    if (start >= size || start > end) return new Response(null, { status: 416, headers: { 'Content-Range': `bytes */${size}` } });
    return new Response(buf.slice(start, end + 1), {
      status: 206,
      headers: {
        'Content-Type': resp.headers.get('Content-Type') || 'application/octet-stream',
        'Content-Range': `bytes ${start}-${end}/${size}`,
        'Content-Length': String(end - start + 1),
        'Accept-Ranges': 'bytes',
      },
    });
  });
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || !MEDIA.test(url.pathname)) return;
  const key = url.origin + url.pathname;
  e.respondWith((async () => {
    try {
      // Videos not cached yet stream straight from the network (no wait for the whole file on a slow link);
      // they are cached by the low-priority warm-up the pages request for upcoming slides.
      if (VIDEO.test(url.pathname)) {
        const hit = await (await caches.open(CACHE)).match(key);
        if (!hit) return fetch(req);
        const range = req.headers.get('Range');
        return range ? slice(hit.clone(), range) : hit;
      }
      const r = await full(key);
      if (!r.ok) return r;
      const range = req.headers.get('Range');
      return range ? slice(r.clone(), range) : r;
    } catch (err) {
      return fetch(req);  // offline cache miss or a fetch error: fall back to the network as usual
    }
  })());
});

const queue = [];
let draining = null;
function drain() {
  if (!draining) draining = (async () => {
    while (queue.length) { const u = queue.shift(); try { await full(u, true); } catch (err) { /* try the next one */ } }
    draining = null;
  })();
  return draining;
}

self.addEventListener('message', (e) => {
  const d = e.data || {};
  if (d.type !== 'warm' || !Array.isArray(d.urls)) return;
  const add = [];
  for (const u of d.urls) {
    try {
      const url = new URL(u, self.location.href);
      if (url.origin === self.location.origin && MEDIA.test(url.pathname)) add.push(url.origin + url.pathname);
    } catch (err) { /* skip one bad URL */ }
  }
  for (const k of add) { const i = queue.indexOf(k); if (i >= 0) queue.splice(i, 1); }
  queue.unshift(...add);  // newest request first (the slides about to be shown), one file at a time
  e.waitUntil(drain());
});
