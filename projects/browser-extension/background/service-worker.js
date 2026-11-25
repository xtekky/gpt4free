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
import { getSettings } from "../lib/storage.js";
import {
  chatCompletion, generateImage, listModels, listProviders, checkHealth,
} from "../lib/g4f-client.js";
import { startAgent, stopAgent, getAgentState } from "../lib/cdp-agent.js";

/** port.name -> port for streaming fan-out */
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

const TRACK_ORIGINS = {
  chatgpt: ["https://chatgpt.com/*", "https://chat.openai.com/*"],
  gemini: ["https://gemini.google.com/*"],
  google_ai: ["https://www.google.com/search*"],
  google_search: ["https://www.google.com/search*"],
};

// Opt-in: the tracker script is only registered for sources the user enabled.
async function syncTracker() {
  const { trackSources = {} } = await chrome.storage.local.get("trackSources");
  const matches = [...new Set(Object.entries(trackSources)
    .filter(([, on]) => on).flatMap(([key]) => TRACK_ORIGINS[key] || []))];
  try {
    await chrome.scripting.unregisterContentScripts({ ids: ["g4f-tracker"] });
  } catch {}
  if (matches.length) {
    await chrome.scripting.registerContentScripts([{
      id: "g4f-tracker", matches, js: ["content/tracker.js", "content/chatgpt-history.js"], runAt: "document_idle", persistAcrossSessions: true,
    }]);
  }
}
chrome.runtime.onInstalled.addListener(syncTracker);
chrome.runtime.onStartup.addListener(syncTracker);

let trackedChatWrites = Promise.resolve();
function serializeTrackedChatWrite(operation) {
  const result = trackedChatWrites.then(operation);
  trackedChatWrites = result.catch((error) => console.error("Tracked chat storage failed:", error));
  return result;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    switch (msg?.type) {
      case "g4f:tracked": {
        await serializeTrackedChatWrite(async () => {
          const { trackSources = {} } = await chrome.storage.local.get("trackSources");
          const conv = msg.conversation;
          if (conv && trackSources[conv.source]) {
            const { trackedChats = {} } = await chrome.storage.local.get("trackedChats");
            trackedChats[`${conv.source}:${conv.id}`] = conv;
            await chrome.storage.local.set({ trackedChats });
          }
        });
        sendResponse({ ok: true });
        break;
      }
      case "g4f:chatgpt:store": {
        await serializeTrackedChatWrite(async () => {
          const conv = msg.conversation;
          const key = `trackedChat:chatgpt:${conv.id}`;
          const versionKey = `chatgptVersion:${conv.id}`;
          const stored = await chrome.storage.local.get(["trackSources", versionKey]);
          if (!stored.trackSources?.chatgpt) throw new Error("ChatGPT tracking is disabled");
          if (msg.version === null || stored[versionKey] !== msg.version) {
            await chrome.storage.local.set({ [key]: conv, [versionKey]: msg.version });
          }
        });
        sendResponse({ ok: true });
        break;
      }
      case "g4f:tracked:ack": {
        await serializeTrackedChatWrite(async () => {
          const snapshots = msg.conversations || [];
          const keys = snapshots.map((c) => `trackedChat:${c.source}:${c.id}`);
          const stored = await chrome.storage.local.get(["trackedChats", ...keys]);
          const trackedChats = stored.trackedChats || {};
          const remove = [];
          for (const conv of snapshots) {
            const key = `trackedChat:${conv.source}:${conv.id}`;
            if (JSON.stringify(stored[key]) === JSON.stringify(conv)) remove.push(key);
            const legacyKey = `${conv.source}:${conv.id}`;
            if (JSON.stringify(trackedChats[legacyKey]) === JSON.stringify(conv)) delete trackedChats[legacyKey];
          }
          await chrome.storage.local.set({ trackedChats });
          if (remove.length) await chrome.storage.local.remove(remove);
        });
        sendResponse({ ok: true });
        break;
      }
      case "g4f:tracker:sync": {
        await syncTracker();
        sendResponse({ ok: true });
        break;
      }
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
        const tab = msg.tabId
          ? await chrome.tabs.get(msg.tabId).catch(() => null)
          : (await chrome.tabs.query({ active: true, currentWindow: true }))[0];
        sendResponse(await getPageText(tab));
        break;
      }
      case MESSAGE_TYPES.GET_SELECTION: {
        const tab = msg.tabId
          ? await chrome.tabs.get(msg.tabId).catch(() => null)
          : (await chrome.tabs.query({ active: true, currentWindow: true }))[0];
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
  })().catch((error) => {
    console.error("Extension message failed:", error);
    sendResponse({ ok: false, error: String(error?.message || error) });
  });
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
