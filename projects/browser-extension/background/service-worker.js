/**
 * g4f extension — background service worker.
 *
 * Responsibilities:
 *  - Route chat/image requests from popup & side panel (single fetch owner,
 *    survives popup closing during streaming).
 *  - Context menu integration ("Ask g4f about: …", "Summarize selection").
 *  - Side panel lifecycle (open per-window on action click).
 *  - Page text extraction via chrome.scripting on the active tab.
 */

import { MESSAGE_TYPES, DEFAULT_SETTINGS } from "../lib/constants.js";
import { getSettings, getConversations, saveConversations } from "../lib/storage.js";
import {
  chatCompletion, generateImage, listModels, listProviders, checkHealth,
} from "../lib/g4f-client.js";
import { signIn, signOut, getSession, isExpired } from "../lib/oauth.js";
import { syncConversations } from "../lib/secret-sync.js";
import { startAgent, stopAgent, getAgentState } from "../lib/cdp-agent.js";

/** port.id -> port for streaming fan-out */
const streams = new Map();
/** abort controllers by stream id */
const aborts = new Map();

/* ------------------------------------------------------------------ */
/* Install / startup                                                   */
/* ------------------------------------------------------------------ */

chrome.runtime.onInstalled.addListener(async (details) => {
  // Seed default settings on first install.
  await getSettings();

  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "g4f-ask-selection",
      title: "Ask g4f about “%s”",
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: "g4f-summarize-page",
      title: "Summarize this page with g4f",
      contexts: ["page", "frame"],
    });
    chrome.contextMenus.create({
      id: "g4f-explain-selection",
      title: "Explain “%s” in simple terms",
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: "g4f-translate-selection",
      title: "Translate “%s” to English",
      contexts: ["selection"],
    });
  });

  if (details.reason === "install") {
    // Open the options page so the user can point the extension at their server.
    chrome.runtime.openOptionsPage();
  }
});

/* ------------------------------------------------------------------ */
/* Side panel behavior                                                 */
/* ------------------------------------------------------------------ */

chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: false })
  .catch(() => {});

chrome.commands?.onCommand.addListener(async (command) => {
  if (command === "open-side-panel") {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.windowId != null) {
      chrome.sidePanel.open({ windowId: tab.windowId }).catch(() => {});
    }
  }
});

// Clicking the toolbar button opens the side panel (popup is available via
// right-click on the icon → "Open popup", or pin & use the context menu).
chrome.action.onClicked.addListener(async (tab) => {
  if (tab?.windowId != null) {
    chrome.sidePanel.open({ windowId: tab.windowId }).catch(() => {});
  }
});

/* ------------------------------------------------------------------ */
/* Context menus                                                       */
/* ------------------------------------------------------------------ */

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const selection = (info.selectionText || "").trim();
  switch (info.menuItemId) {
    case "g4f-ask-selection":
      await openPanelWithPreset({ type: "ask", text: selection, tab });
      break;
    case "g4f-explain-selection":
      await openPanelWithPreset({ type: "explain", text: selection, tab });
      break;
    case "g4f-translate-selection":
      await openPanelWithPreset({ type: "translate", text: selection, tab });
      break;
    case "g4f-summarize-page": {
      const page = await getPageText(tab);
      await openPanelWithPreset({ type: "summarize", text: page?.text || "", tab });
      break;
    }
  }
});

async function openPanelWithPreset(preset) {
  // Store the preset, then open the panel; the panel picks it up on load.
  await chrome.storage.session.set({ g4fPreset: { ...preset, ts: Date.now() } });
  const windowId = preset.tab?.windowId ?? (await chrome.windows.getCurrent())?.id;
  if (windowId != null) {
    chrome.sidePanel.open({ windowId }).catch(() => {});
  }
}

/* ------------------------------------------------------------------ */
/* Page extraction                                                     */
/* ------------------------------------------------------------------ */

async function getPageText(tab) {
  if (!tab?.id) return null;
  const url = tab.url || "";
  if (!/^https?:|^file:/.test(url)) return null;
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        // Clone-safe extraction: prefer <article>/main, fall back to body.
        const root =
          document.querySelector("article") ||
          document.querySelector("main") ||
          document.body;
        if (!root) return { text: "", title: document.title, url: location.href };
        const clone = root.cloneNode(true);
        clone
          .querySelectorAll("script,style,noscript,svg,iframe,nav,footer,header,aside,form,button")
          .forEach((n) => n.remove());
        const text = (clone.innerText || clone.textContent || "")
          .replace(/\n{3,}/g, "\n\n")
          .trim();
        return { text, title: document.title, url: location.href };
      },
    });
    return results?.[0]?.result || null;
  } catch {
    return null;
  }
}

async function getSelection(tab) {
  if (!tab?.id) return null;
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => String(window.getSelection?.() || ""),
    });
    return results?.[0]?.result || "";
  } catch {
    return "";
  }
}

/* ------------------------------------------------------------------ */
/* Message router                                                      */
/* ------------------------------------------------------------------ */

chrome.runtime.onConnect.addListener((port) => {
  if (!port.name?.startsWith("g4f:")) return;
  streams.set(port.name, port);
  port.onDisconnect.addListener(() => {
    streams.delete(port.name);
    // port.name is "g4f:<streamId>" — aborts are keyed by the plain streamId.
    const streamId = port.name.slice("g4f:".length);
    aborts.get(streamId)?.abort();
    aborts.delete(streamId);
  });
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    switch (msg?.type) {
      case MESSAGE_TYPES.MODELS: {
        try {
          const settings = await getSettings();
          sendResponse({ ok: true, data: await listModels(settings) });
        } catch (e) {
          sendResponse({ ok: false, error: String(e?.message || e) });
        }
        break;
      }
      case MESSAGE_TYPES.PROVIDERS: {
        try {
          const settings = await getSettings();
          sendResponse({ ok: true, data: await listProviders(settings) });
        } catch (e) {
          sendResponse({ ok: false, error: String(e?.message || e) });
        }
        break;
      }
      case MESSAGE_TYPES.HEALTH: {
        const settings = await getSettings();
        sendResponse(await checkHealth(settings));
        break;
      }
      case MESSAGE_TYPES.GET_PAGE: {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        sendResponse(await getPageText(tab));
        break;
      }
      case MESSAGE_TYPES.GET_SELECTION: {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        sendResponse({ text: await getSelection(tab) });
        break;
      }
      case MESSAGE_TYPES.OPEN_SIDE_PANEL: {
        const windowId = msg.windowId ?? (await chrome.windows.getCurrent())?.id;
        if (windowId != null) {
          chrome.sidePanel.open({ windowId }).catch(() => {});
          sendResponse({ ok: true });
        } else {
          sendResponse({ ok: false, error: "No window" });
        }
        break;
      }
      case MESSAGE_TYPES.CHAT: {
        await handleChat(msg);
        sendResponse({ ok: true, started: true });
        break;
      }
      case MESSAGE_TYPES.CHAT_ABORT: {
        aborts.get(msg.streamId)?.abort();
        sendResponse({ ok: true, found: aborts.has(msg.streamId) });
        break;
      }
      case MESSAGE_TYPES.IMAGE: {
        await handleImage(msg);
        sendResponse({ ok: true, started: true });
        break;
      }

      /* ---------------- account & cloud sync ---------------- */
      case MESSAGE_TYPES.OAUTH_START: {
        try {
          const session = await signIn();
          sendResponse({ ok: true, user: session.user });
        } catch (e) {
          sendResponse({ ok: false, error: String(e?.message || e) });
        }
        break;
      }
      case MESSAGE_TYPES.OAUTH_LOGOUT: {
        try {
          await signOut();
          sendResponse({ ok: true });
        } catch (e) {
          sendResponse({ ok: false, error: String(e?.message || e) });
        }
        break;
      }
      case MESSAGE_TYPES.OAUTH_STATUS: {
        try {
          const session = await getSession();
          sendResponse({
            ok: true,
            signedIn: !!session && !isExpired(session),
            user: session?.user
              ? { id: session.user.id, name: session.user.name, username: session.user.username, avatar: session.user.avatar, tier: session.user.tier }
              : null,
          });
        } catch (e) {
          sendResponse({ ok: false, error: String(e?.message || e) });
        }
        break;
      }
      case MESSAGE_TYPES.SYNC_NOW:
      case MESSAGE_TYPES.SYNC_PUSH:
      case MESSAGE_TYPES.SYNC_PULL: {
        try {
          const settings = await getSettings();
          const local = await getConversations();
          const result = await syncConversations(settings, local);
          if (result.pulledConversations?.length) {
            // Merge pulled conversations into local storage (newest first).
            const byId = new Map(local.map((c) => [c.id, c]));
            for (const c of result.pulledConversations) byId.set(c.id, c);
            const merged = [...byId.values()].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
            await saveConversations(merged);
          }
          sendResponse({ ok: true, ...result, pulledConversations: undefined });
        } catch (e) {
          sendResponse({ ok: false, error: String(e?.message || e) });
        }
        break;
      }

      /* ---------------- CDP bridge (browser-as-provider) ---------------- */
      case "g4f:cdp:start": {
        try {
          await startAgent();
          sendResponse({ ok: true, state: await getAgentState() });
        } catch (e) {
          sendResponse({ ok: false, error: String(e?.message || e) });
        }
        break;
      }
      case "g4f:cdp:stop": {
        try {
          await stopAgent();
          sendResponse({ ok: true, state: await getAgentState() });
        } catch (e) {
          sendResponse({ ok: false, error: String(e?.message || e) });
        }
        break;
      }
      case "g4f:cdp:status": {
        sendResponse({ ok: true, state: await getAgentState() });
        break;
      }
      default:
        sendResponse({ ok: false, error: "Unknown message type" });
    }
  })();
  return true; // async sendResponse
});

/* ------------------------------------------------------------------ */
/* Chat / image streaming over a dedicated port                        */
/* ------------------------------------------------------------------ */

function post(streamId, payload) {
  // Ports are registered under their full name "g4f:<streamId>".
  const port = streams.get("g4f:" + streamId);
  if (port) {
    try {
      port.postMessage(payload);
    } catch {
      /* port closed */
    }
  }
}

async function handleChat(msg) {
  const { streamId, messages, overrides = {} } = msg;
  const settings = await getSettings();
  const controller = new AbortController();
  aborts.set(streamId, controller);

  try {
    await chatCompletion(settings, messages, {
      ...overrides,
      signal: controller.signal,
      onChunk: (piece, full) => post(streamId, { type: MESSAGE_TYPES.STREAM_CHUNK, piece, full }),
    });
    post(streamId, { type: MESSAGE_TYPES.STREAM_DONE });
  } catch (e) {
    const aborted = e?.name === "AbortError";
    post(streamId, {
      type: aborted ? MESSAGE_TYPES.STREAM_DONE : MESSAGE_TYPES.STREAM_ERROR,
      error: aborted ? undefined : String(e?.message || e),
    });
  } finally {
    aborts.delete(streamId);
  }
}

async function handleImage(msg) {
  const { streamId, prompt, overrides = {} } = msg;
  const settings = await getSettings();
  const controller = new AbortController();
  aborts.set(streamId, controller);
  try {
    const urls = await generateImage(settings, prompt, {
      ...overrides,
      signal: controller.signal,
    });
    post(streamId, { type: MESSAGE_TYPES.STREAM_DONE, urls });
  } catch (e) {
    post(streamId, {
      type: MESSAGE_TYPES.STREAM_ERROR,
      error: String(e?.message || e),
    });
  } finally {
    aborts.delete(streamId);
  }
}
