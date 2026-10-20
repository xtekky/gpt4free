/* Offline shell for the viewer. The stream itself is always live. */
const BASE = "/remote_desktop";
const CACHE = "remote-desktop-v3";
const ASSETS = [
  `${BASE}/view`,
  `${BASE}/static/css/app.css`,
  `${BASE}/static/js/viewer.js`,
  `${BASE}/static/icon.svg`,
  `${BASE}/manifest.webmanifest`,
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== location.origin) return;
  if (url.pathname.startsWith(`${BASE}/ws`) || url.pathname.startsWith(`${BASE}/api/`)) return;

  event.respondWith(
    fetch(request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(request, copy)).catch(() => {});
        return response;
      })
      .catch(() => caches.match(request).then((cached) => cached || caches.match(`${BASE}/view`)))
  );
});
