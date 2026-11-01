/*
 * g4f desktop service worker.
 *
 * The desktop itself is a live VNC stream and is never cached. This worker
 * only keeps the /desktop/ shell around so the installed app can start and
 * explain that it is offline. It is registered by g4f-controls.js and works
 * both at /desktop/ and behind the desktop.g4f.local host route, so every URL
 * is resolved against the registration scope instead of a hard-coded prefix.
 */

const CACHE_NAME = "g4f-desktop-v3";
const CACHE_PREFIX = "g4f-desktop-";
const SHELL = [
  "./",
  "manifest.webmanifest",
  "g4f-controls.css",
  "g4f-controls.js",
  "g4f.svg",
];

const scope = self.registration.scope;
const shellUrls = SHELL.map((path) => new URL(path, scope).href);
const offlineShellUrl = new URL("./", scope).href;

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      // Add each file independently: one unavailable icon must not prevent the
      // worker from installing and taking control of the page.
      .then((cache) =>
        Promise.all(shellUrls.map((url) => cache.add(url).catch(() => null)))
      )
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) =>
        Promise.all(
          names
            .filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME)
            .map((name) => caches.delete(name))
        )
      )
      .then(() => self.clients.claim())
  );
});

function offlineResponse() {
  return new Response(
    "<!doctype html><meta charset=\"utf-8\"><title>g4f Desktop offline</title>" +
      "<body style=\"margin:0;display:grid;place-items:center;min-height:100vh;background:#0b0f14;color:#e6edf5;font:16px system-ui,sans-serif\">" +
      "<main style=\"text-align:center;padding:24px\"><h1>g4f Desktop is offline</h1>" +
      "<p>The connection to the sandbox is down. Reconnect and try again.</p></main>",
    {
      status: 503,
      headers: { "Content-Type": "text/html; charset=utf-8" },
    }
  );
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await fetch(request);
    // Cache only complete, same-origin responses. The VNC WebSocket is
    // excluded in the fetch handler; partial/range responses are not cached.
    if (response.status === 200 && response.type === "basic") {
      await cache.put(request, response.clone()).catch(() => {});
    }
    return response;
  } catch (error) {
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) return cached;
    if (request.mode === "navigate") {
      const shell = await cache.match(offlineShellUrl);
      if (shell) return shell;
    }
    return offlineResponse();
  }
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  // The bridge upgrades to a WebSocket; never proxy or cache it.
  if (url.pathname.endsWith("/websockify")) return;
  if (request.headers.has("range")) return;

  event.respondWith(networkFirst(request));
});
