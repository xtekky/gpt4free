/**
 * g4f extension — CDP bridge agent.
 *
 * Lets the g4f server (or any authorized client) drive this browser as a
 * CDP endpoint: the agent connects OUT to the server's relay WebSocket and
 * executes CDP commands locally via chrome.debugger on dedicated automation
 * tabs. No --remote-debugging-port flag, no separate Chrome instance.
 *
 * Protocol (JSON messages over the WebSocket):
 *   -> {type:"cdp", id, tabId?, method, params?, sessionId?}   execute command
 *   <- {type:"cdp", id, result?, error?}                       command result
 *   <- {type:"event", tabId, method, params}                   CDP event fan-out
 *   -> {type:"attach", tabId} / {type:"detach", tabId}         manage targets
 *   <- {type:"attached", tabId, targetId} / {type:"detached", tabId}
 *   -> {type:"tabs"}  <- {type:"tabs", tabs:[{id,title,url}]}  list automation tabs
 *   -> {type:"new_tab", url?}  <- {type:"tab", tabId, ...}     create target
 *   -> {type:"close_tab", tabId}                               close target
 *
 * Security: the server URL comes from the user's own settings; the relay
 * requires the same API key as the REST API (sent as ?key= or auth header).
 */

import { getSettings } from "./storage.js";
import { normalizeBaseUrl } from "./constants.js";

const AGENT_STATE_KEY = "g4fCdpAgent";

/** Automation tabs owned by the agent (never the user's own tabs). */
const managedTabs = new Set();
/** tabId -> {targetId, attached, version} */
const attached = new Map();
/** Live WebSocket to the relay. */
let ws = null;
/** Backoff / lifecycle */
let reconnectTimer = null;
let running = false;
let backoffMs = 2000;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

export async function startAgent() {
  if (running) return;
  running = true;
  backoffMs = 2000;
  await saveState({ running: true, since: Date.now() });
  connect();
}

export async function stopAgent() {
  running = false;
  await saveState({ running: false });
  stopPingLoop();
  if (reconnectTimer) clearTimeout(reconnectTimer);
  if (ws) try { ws.close(); } catch { /* ignore */ }
  ws = null;
  for (const tabId of [...managedTabs]) await closeTab(tabId);
}

export async function getAgentState() {
  const { [AGENT_STATE_KEY]: state } = await chrome.storage.local.get(AGENT_STATE_KEY);
  return state || { running: false };
}

async function saveState(patch) {
  const state = { ...(await getAgentState()), ...patch };
  await chrome.storage.local.set({ [AGENT_STATE_KEY]: state });
  return state;
}

function connect() {
  if (!running) return;
  chrome.runtime.sendMessage({ type: "g4f:cdp:connecting" }).catch(() => {});
  getSettings().then((settings) => {
    if (!running) return;
    const base = normalizeBaseUrl(settings.serverUrl).replace(/^http/, "ws");
    const url = new URL(base + "/v1/cdp/agent");
    // Security: only loopback servers may drive this browser. A remote
    // g4f instance must never be allowed to execute CDP commands locally.
    const host = url.hostname;
    if (!["localhost", "127.0.0.1", "[::1]", "::1", "0.0.0.0"].includes(host)) {
      saveState({
        connected: false,
        error: `Refused: "${host}" is not local. Set the Server URL to your local g4f server (e.g. http://localhost:1337) — a remote server must never control this browser.`,
      });
      chrome.runtime.sendMessage({ type: "g4f:cdp:disconnected" }).catch(() => {});
      return;
    }
    if (settings.apiKey) url.searchParams.set("key", settings.apiKey);
    url.searchParams.set("client", chrome.runtime.id);

    try {
      ws = new WebSocket(url.toString());
    } catch (e) {
      scheduleReconnect(e);
      return;
    }
    let opened = false;

    ws.onopen = () => {
      opened = true;
      backoffMs = 2000;
      saveState({ connected: true, since: Date.now(), error: "" });
      chrome.runtime.sendMessage({ type: "g4f:cdp:connected" }).catch(() => {});
      startPingLoop();
    };

    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      handleRelayMessage(msg).catch((e) =>
        console.warn("[g4f-cdp] handler error:", e)
      );
    };

    ws.onclose = () => {
      stopPingLoop();
      ws = null;
      saveState({
        connected: false,
        ...(opened ? {} : {
          error: "Relay unreachable — start the server with G4F_BROWSER_MODE=extension (python -m g4f --port 1337).",
        }),
      });
      chrome.runtime.sendMessage({ type: "g4f:cdp:disconnected" }).catch(() => {});
      scheduleReconnect();
    };

    ws.onerror = () => {
      try { ws?.close(); } catch { /* ignore */ }
    };
  });
}

function scheduleReconnect(err) {
  if (!running) return;
  if (err) saveState({ connected: false, error: String(err?.message || err) });
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, backoffMs);
  backoffMs = Math.min(backoffMs * 2, 60000);
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
    return true;
  }
  return false;
}

/* ------------------------------------------------------------------ */
/* Keepalive (MV3 service workers die after ~30s idle)                 */
/* ------------------------------------------------------------------ */

let pingTimer = null;
let lastPong = 0;

function startPingLoop() {
  stopPingLoop();
  lastPong = Date.now();
  // 20s ping keeps the SW alive and detects dead links within ~40s.
  pingTimer = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      stopPingLoop();
      return;
    }
    if (!send({ type: "ping", ts: Date.now() })) return;
    // If no pong for 2 intervals, the link is a zombie — force reconnect.
    if (Date.now() - lastPong > 45000) {
      console.warn("[g4f-cdp] ping timeout, forcing reconnect");
      try { ws.close(); } catch { /* ignore */ }
    }
  }, 20000);
  ws.addEventListener("message", (ev) => {
    try {
      if (JSON.parse(ev.data)?.type === "pong") lastPong = Date.now();
    } catch { /* ignore */ }
  });
}

function stopPingLoop() {
  if (pingTimer) clearInterval(pingTimer);
  pingTimer = null;
}

/* ------------------------------------------------------------------ */
/* Relay message handling                                              */
/* ------------------------------------------------------------------ */

async function handleRelayMessage(msg) {
  switch (msg.type) {
    case "cdp": {
      // Execute a CDP command on an attached target.
      const tabId = msg.tabId;
      if (!attached.has(tabId)) {
        try {
          await attachDebugger(tabId);
        } catch (e) {
          send({ type: "cdp", id: msg.id, error: `Not attached to tab ${tabId}: ${e?.message || e}` });
          return;
        }
      }
      try {
        const result = await chrome.debugger.sendCommand(
          { tabId },
          msg.method,
          msg.params || {}
        );
        send({ type: "cdp", id: msg.id, result: result ?? {} });
      } catch (e) {
        send({ type: "cdp", id: msg.id, error: String(e?.message || e) });
      }
      return;
    }

    case "attach": {
      try {
        const tabId = msg.tabId ?? (await createTab(msg.url)).tabId;
        const targetId = await attachDebugger(tabId);
        send({ type: "attached", id: msg.id, tabId, targetId });
      } catch (e) {
        send({ type: "error", id: msg.id, error: `attach failed: ${e?.message || e}` });
      }
      return;
    }

    case "detach": {
      const tabId = msg.tabId;
      try {
        await chrome.debugger.detach({ tabId });
      } catch { /* already detached */ }
      send({ type: "detached", id: msg.id, tabId });
      return;
    }

    case "new_tab": {
      try {
        const tab = await createTab(msg.url);
        // Attach immediately so the tab is ready for CDP commands.
        await attachDebugger(tab.tabId).catch(() => {});
        send({ type: "tab", id: msg.id, ...tab });
      } catch (e) {
        send({ type: "error", id: msg.id, error: `new_tab failed: ${e?.message || e}` });
      }
      return;
    }

    case "close_tab": {
      try { await closeTab(msg.tabId); } catch { /* ignore */ }
      send({ type: "closed", id: msg.id, tabId: msg.tabId });
      return;
    }

    case "tabs": {
      const tabs = [];
      for (const tabId of managedTabs) {
        try {
          const tab = await chrome.tabs.get(tabId);
          tabs.push({ tabId, title: tab.title, url: tab.url });
        } catch { /* closed */ }
      }
      send({ type: "tabs", id: msg.id, tabs });
      return;
    }

    case "ping":
      send({ type: "pong", id: msg.id, ts: msg.ts });
      return;
  }
}

/* ------------------------------------------------------------------ */
/* Tab + debugger management                                           */
/* ------------------------------------------------------------------ */

async function createTab(url = "about:blank") {
  const tab = await chrome.tabs.create({ url, active: false });
  managedTabs.add(tab.id);
  return { tabId: tab.id, title: tab.title || "", url: tab.url || url };
}

async function closeTab(tabId) {
  managedTabs.delete(tabId);
  attached.delete(tabId);
  try { await chrome.tabs.remove(tabId); } catch { /* already gone */ }
}

async function attachDebugger(tabId) {
  const target = await chrome.debugger.attach({ tabId }, "1.3");
  // target is {tabId, targetId} in MV3 when successful
  const targetId = target?.targetId || `tab-${tabId}`;
  attached.set(tabId, { targetId, attached: true });
  return targetId;
}

/* Debugger detach events (user opened devtools, tab closed, etc.) */
chrome.debugger.onDetach.addListener((source) => {
  if (source.tabId != null) attached.delete(source.tabId);
});

/* Clean up managed tabs closed by the user */
chrome.tabs.onRemoved.addListener((tabId) => {
  if (managedTabs.delete(tabId)) attached.delete(tabId);
});

/* Service worker wake-up: resume if it was running before */
chrome.runtime.onStartup?.addListener?.(() => {});
getAgentState().then((state) => {
  if (state.running) startAgent();
});
