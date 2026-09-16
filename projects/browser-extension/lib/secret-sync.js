/**
 * g4f extension — conversation sync against the g4f server's
 * /v1/secret/conversations* endpoints (see g4f/api/__init__.py).
 *
 *   GET    /v1/secret/conversations            -> { conversations: [indexEntry] }
 *   POST   /v1/secret/conversations            (body = conversation dict w/ id)
 *   POST   /v1/secret/conversations/sync       (body {conversations:[...]}) -> {saved, errors}
 *   GET    /v1/secret/conversations/{id}       -> conversation dict
 *   DELETE /v1/secret/conversations/{id}       -> {deleted: true}
 *
 * Auth headers (required by the server):
 *   x-user-id:          <user.id from OAuth session>
 *   x-workspace-secret: hex(SHA-256(`${user.id}:${user.secret}`))  (optional,
 *                       enables AES-256-GCM encryption server-side)
 */

import { normalizeBaseUrl } from "./constants.js";
import { getSession, isExpired } from "./oauth.js";

function requireSession() {
  return getSession().then((session) => {
    if (!session) throw new Error("Not signed in — connect your g4f.space account first.");
    if (isExpired(session)) throw new Error("Session expired — sign in again.");
    return session;
  });
}

function headers(session) {
  const h = { "Content-Type": "application/json", "x-user-id": session.user.id };
  if (session.workspaceSecret) h["x-workspace-secret"] = session.workspaceSecret;
  return h;
}

function baseUrl(settings) {
  return normalizeBaseUrl(settings.serverUrl);
}

/** GET /v1/secret/conversations — list index entries. */
export async function listRemoteConversations(settings) {
  const session = await requireSession();
  const res = await fetch(`${baseUrl(settings)}/v1/secret/conversations`, {
    headers: headers(session),
  });
  if (!res.ok) throw new Error(await parseErr(res));
  const data = await res.json();
  return Array.isArray(data?.conversations) ? data.conversations : [];
}

/** POST /v1/secret/conversations/sync — push conversations (upsert). */
export async function pushConversations(settings, conversations) {
  const session = await requireSession();
  if (!conversations?.length) return { saved: 0, errors: [] };
  const res = await fetch(`${baseUrl(settings)}/v1/secret/conversations/sync`, {
    method: "POST",
    headers: headers(session),
    body: JSON.stringify({ conversations }),
  });
  if (!res.ok) throw new Error(await parseErr(res));
  return res.json(); // { saved, errors }
}

/** GET /v1/secret/conversations/{id} — fetch one full conversation. */
export async function fetchRemoteConversation(settings, id) {
  const session = await requireSession();
  const res = await fetch(
    `${baseUrl(settings)}/v1/secret/conversations/${encodeURIComponent(id)}`,
    { headers: headers(session) }
  );
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(await parseErr(res));
  return res.json();
}

/** DELETE /v1/secret/conversations/{id} */
export async function deleteRemoteConversation(settings, id) {
  const session = await requireSession();
  const res = await fetch(
    `${baseUrl(settings)}/v1/secret/conversations/${encodeURIComponent(id)}`,
    { method: "DELETE", headers: headers(session) }
  );
  if (!res.ok && res.status !== 404) throw new Error(await parseErr(res));
  return true;
}

/**
 * Full two-way sync between local storage and the server.
 * Strategy: last-write-wins per conversation by `updatedAt`.
 * Returns { pushed, pulled, deleted, errors }.
 */
export async function syncConversations(settings, localConversations) {
  const remoteIndex = await listRemoteConversations(settings);
  const localById = new Map(localConversations.map((c) => [c.id, c]));
  const remoteById = new Map(remoteIndex.map((e) => [e.id, e]));

  const toPush = [];
  const toPullIds = [];
  const errors = [];

  // Local newer or missing remotely -> push.
  for (const c of localConversations) {
    const remote = remoteById.get(c.id);
    if (!remote || (c.updatedAt || 0) > (remote.updated || 0)) {
      toPush.push(c);
    } else if ((remote.updated || 0) > (c.updatedAt || 0)) {
      toPullIds.push(c.id);
    }
  }
  // Remote-only -> pull.
  for (const e of remoteIndex) {
    if (!localById.has(e.id)) toPullIds.push(e.id);
  }

  let pushed = 0;
  if (toPush.length) {
    const res = await pushConversations(settings, toPush);
    pushed = res.saved || 0;
    for (const err of res.errors || []) errors.push("push: " + err);
  }

  let pulled = 0;
  const pulledConversations = [];
  for (const id of toPullIds) {
    try {
      const conv = await fetchRemoteConversation(settings, id);
      if (conv) {
        pulledConversations.push(conv);
        pulled++;
      }
    } catch (e) {
      errors.push("pull " + id + ": " + (e?.message || e));
    }
  }

  return { pushed, pulled, pulledConversations, errors };
}

async function parseErr(res) {
  let msg = `HTTP ${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body?.error?.message) msg = body.error.message;
    else if (typeof body?.detail === "string") msg = body.detail;
  } catch { /* keep default */ }
  if (res.status === 401) msg += " — check sign-in / x-user-id.";
  return msg;
}
