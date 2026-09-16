/**
 * g4f extension — OAuth sign-in for g4f.space (PKCE authorization-code flow).
 *
 * Wire format (see g4f.dev/workers/members-worker.js):
 *   authorize: GET https://auth.g4f.space/members/oauth/authorize
 *       ?response_type=code&client_id=g4f-web&redirect_uri=...&state=...
 *       &code_challenge=<S256(verifier)>&code_challenge_method=S256
 *   token:     POST https://auth.g4f.space/members/oauth/token
 *       grant_type=authorization_code&client_id=g4f-web&client_secret=...
 *       &code=...&redirect_uri=...&code_verifier=...
 *       -> { access_token, token_type, expires_in, scope, user: {...} }
 *
 * The token endpoint returns a temporary login API key as `access_token`
 * plus the full user record (id, secret, name, tier, ...). The workspace
 * secret used for encrypted conversation sync is derived client-side:
 *     hex( SHA-256( `${user.id}:${user.secret}` ) )
 *
 * Redirect handling — the auth worker only accepts redirects on
 * localhost / 127.0.0.1 / g4f.dev / *.g4f.space (see `isValidRedirect` in
 * members-worker.js), so `https://<ext-id>.chromiumapp.org/` is NOT allowed.
 * We therefore run a tab-based flow with a loopback redirect:
 *   - authorize URL opens in a regular tab (login chooser: GitHub/Discord/…)
 *   - after sign-in the server redirects to http://127.0.0.1:<port>/oauth/callback?code=…
 *   - the extension already holds host permissions for 127.0.0.1 on any
 *     port, so chrome.tabs.onUpdated exposes the final URL and we capture
 *     the code (nothing needs to actually listen on that port).
 */

import { STORAGE_KEYS } from "./constants.js";

const AUTH_BASE = "https://auth.g4f.space";
const CLIENT_ID = "g4f-web";
// Public-by-design first-party client secret (browser clients cannot hold
// secrets; PKCE protects the code exchange).
const CLIENT_SECRET = "5594a516-0da6-4167-bcaa-132e715c54a3";
// Loopback redirect — hostname is in the worker's allow-list; the port is
// arbitrary (the tab may show a 404/refused page, we only need the URL).
const REDIRECT_URI = "http://127.0.0.1:1337/oauth/callback";
/** storage key holding a content-script handoff while the flow is pending */
const CB_KEY = "g4fOAuthCallback";

/* ------------------------------------------------------------------ */
/* PKCE helpers                                                        */
/* ------------------------------------------------------------------ */

function randomString(len = 64) {
  const bytes = new Uint8Array(len);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(36).padStart(2, "0")).join("").slice(0, len);
}

function base64Url(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function sha256Hex(str) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(str));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function codeChallenge(verifier) {
  return base64Url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)));
}

/* ------------------------------------------------------------------ */
/* Session storage (chrome.storage.local)                              */
/* ------------------------------------------------------------------ */

export async function getSession() {
  const res = await chrome.storage.local.get([
    STORAGE_KEYS.user, STORAGE_KEYS.token, STORAGE_KEYS.expires, STORAGE_KEYS.workspaceSecret,
  ]);
  const user = res[STORAGE_KEYS.user] || null;
  if (!user) return null;
  return {
    user,
    token: res[STORAGE_KEYS.token] || "",
    expires: res[STORAGE_KEYS.expires] || 0,
    workspaceSecret: res[STORAGE_KEYS.workspaceSecret] || "",
  };
}

export async function setSession(session) {
  await chrome.storage.local.set({
    [STORAGE_KEYS.user]: session.user,
    [STORAGE_KEYS.token]: session.token || "",
    [STORAGE_KEYS.expires]: session.expires || 0,
    [STORAGE_KEYS.workspaceSecret]: session.workspaceSecret || "",
  });
}

export async function clearSession() {
  await chrome.storage.local.remove([
    STORAGE_KEYS.user, STORAGE_KEYS.token, STORAGE_KEYS.expires, STORAGE_KEYS.workspaceSecret,
  ]);
}

export function isExpired(session) {
  return !session || !session.token || (session.expires > 0 && Date.now() / 1000 > session.expires - 60);
}

/** Derive the workspace secret from the user record. */
export async function deriveWorkspaceSecret(user) {
  return sha256Hex(`${user.id}:${user.secret}`);
}

/* ------------------------------------------------------------------ */
/* Sign-in / sign-out                                                  */
/* ------------------------------------------------------------------ */

/**
 * Run the full OAuth flow. Resolves with the session or throws.
 * Tab-based loopback flow (see header comment for why launchWebAuthFlow
 * cannot be used: chromiumapp.org is not in the auth worker allow-list).
 */
export async function signIn() {
  const state = randomString(24);
  const verifier = randomString(64);
  const challenge = await codeChallenge(verifier);

  const params = new URLSearchParams({
    response_type: "code",
    client_id: CLIENT_ID,
    redirect_uri: REDIRECT_URI,
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
  });

  const code = await tabFlow(`${AUTH_BASE}/members/oauth/authorize?${params}`, state);
  if (!code) throw new Error("Sign-in cancelled or timed out.");

  const session = await exchangeCode(code, verifier, REDIRECT_URI);
  await setSession(session);
  return session;
}

export async function signOut() {
  // Best-effort server-side logout; ignore failures.
  try {
    const session = await getSession();
    if (session?.token) {
      await fetch(`${AUTH_BASE}/members/api/logout`, {
        headers: { Authorization: "Bearer " + session.token },
      });
    }
  } catch { /* offline — clear locally anyway */ }
  await clearSession();
}

/* ------------------------------------------------------------------ */
/* Internals                                                           */
/* ------------------------------------------------------------------ */

function parseCode(redirectUrl, state) {
  if (!redirectUrl) return null;
  try {
    const url = new URL(redirectUrl);
    if (url.searchParams.get("state") !== state) return null;
    return url.searchParams.get("code");
  } catch {
    return null;
  }
}

/**
 * Tab-based loopback flow: open the authorize page in a new tab. After the
 * user signs in, the auth server redirects to REDIRECT_URI (127.0.0.1) with
 * ?code=…&state=…. Because the extension holds host_permissions for
 * 127.0.0.1:*, chrome.tabs.onUpdated exposes the URL even though nothing
 * is listening on that port. A session-storage handoff (CB_KEY) is polled
 * as a secondary path for robustness.
 */
async function tabFlow(authorizeUrl, state) {
  const tab = await chrome.tabs.create({ url: authorizeUrl, active: true });
  const deadline = Date.now() + 5 * 60 * 1000; // 5 min

  return new Promise((resolve) => {
    let done = false;
    const finish = (code) => {
      if (done) return;
      done = true;
      chrome.tabs.onUpdated.removeListener(onUpdated);
      chrome.tabs.onRemoved.removeListener(onRemoved);
      chrome.storage.session.remove(CB_KEY).catch(() => {});
      resolve(code);
      // Close the sign-in tab once we have the code.
      if (code && tab?.id != null) chrome.tabs.remove(tab.id).catch(() => {});
    };

    function onUpdated(tabId, change) {
      if (tabId !== tab.id || done) return;
      const url = change?.url || "";
      if (!url.includes("code=")) return;
      const code = parseCode(url, state);
      if (code) finish(code);
    }
    function onRemoved(removedId) {
      if (removedId === tab.id) finish(null);
    }

    chrome.tabs.onUpdated.addListener(onUpdated);
    chrome.tabs.onRemoved.addListener(onRemoved);

    // Secondary path: poll for a handoff written by a content script.
    const poll = setInterval(async () => {
      if (done) { clearInterval(poll); return; }
      if (Date.now() > deadline) { clearInterval(poll); finish(null); return; }
      try {
        const { [CB_KEY]: cb } = await chrome.storage.session.get(CB_KEY);
        if (cb?.code && cb.state === state) finish(cb.code);
      } catch { /* ignore */ }
    }, 1000);
  });
}

/** Exchange an authorization code for a session. */
async function exchangeCode(code, verifier, redirectUri) {
  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: CLIENT_ID,
    client_secret: CLIENT_SECRET,
    code,
    redirect_uri: redirectUri,
    code_verifier: verifier,
  });
  const res = await fetch(`${AUTH_BASE}/members/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!res.ok) {
    let msg = `Token exchange failed (HTTP ${res.status})`;
    try {
      const err = await res.json();
      if (err?.error_description) msg = err.error_description;
      else if (err?.error) msg = err.error;
    } catch { /* keep default */ }
    throw new Error(msg);
  }
  const data = await res.json();
  if (!data?.access_token || !data?.user) {
    throw new Error("Token response missing access_token/user.");
  }
  return {
    user: data.user,
    token: data.access_token,
    expires: Date.now() / 1000 + (data.expires_in || 3600),
    workspaceSecret: await deriveWorkspaceSecret(data.user),
  };
}
