/**
 * g4f extension — side panel logic.
 *
 * Two modes:
 *  1. "embed"  (default) — the full g4f.dev chat in an iframe. Conversations
 *     persist in the chat's own IndexedDB (chat-db / conversations), exactly
 *     like on the website. A postMessage bridge injects quick-action prompts
 *     and page context into the embedded chat.
 *  2. "native" — lightweight built-in chat backed by chrome.storage.local,
 *     with a conversation history drawer. Used when the embed is unavailable
 *     (offline, blocked, or by user preference).
 */

import { MESSAGE_TYPES, normalizeBaseUrl } from "../lib/constants.js";
import {
  getSettings, saveSettings, getConversations, saveConversations,
  getActiveConversationId, setActiveConversationId, newConversation,
} from "../lib/storage.js";
import { renderMarkdown } from "../lib/markdown.js";
import { toast, bindCopyHandlers, escapeHtml } from "../lib/ui.js";

const $ = (sel) => document.querySelector(sel);
const EMBED_KEY = "g4fEmbedMode"; // "embed" | "native" (chrome.storage.local)

/**
 * Chat app URL, derived from the configured server URL: the public
 * g4f.space API serves its chat UI from g4f.dev, while any other server
 * (self-hosted instances) hosts the chat itself under /chat/.
 */
function getChatUrl() {
  try {
    const u = new URL(normalizeBaseUrl(settings?.serverUrl));
    const host = u.hostname;
    if (host !== "g4f.space" && !host.endsWith(".g4f.space")) {
      return u.origin + "/chat/";
    }
  } catch { /* fall through to the public default */ }
  return "https://g4f.dev/chat/";
}

/** Origin the embedded chat posts messages from. */
function getChatOrigin() {
  return new URL(getChatUrl()).origin;
}

let settings = null;
let conversation = null;
let conversations = [];
let streamId = null;
let port = null;
let generating = false;
let imageMode = false;
/** pending prompt waiting for the embedded chat to confirm readiness */
let pendingEmbedPrompt = null;
/** resolve fn for the bridge handshake */
let embedReadyResolve = null;
/** current waiter promise for the bridge handshake */
let embedReadyWait = null;
/** true once the embedded chat announced readiness */
let embedChatReady = false;

init();

// Quick actions hand their prompt over via session storage. init() only
// runs once per panel load, so consume new presets even when the side
// panel is already open.
chrome.storage.session.onChanged.addListener((changes, area) => {
  if (area === "session" && changes.g4fPreset?.newValue) consumePreset();
});

async function init() {
  settings = await getSettings();
  bindCopyHandlers(document, () => toast($("#toast-root"), "Copied"));

  const mode = await getMode();
  applyMode(mode);

  conversations = await getConversations();
  const activeId = await getActiveConversationId();
  conversation =
    conversations.find((c) => c.id === activeId) || createConversation();

  wireEvents();
  wireEmbedBridge();
  if (mode === "native") {
    loadModels();
    checkHealth();
    renderMessages();
    renderHistory();
  }
  await consumePreset();   // context-menu handoff
  await consumeHandoff();  // popup handoff
}

/* ------------------------------------------------------------------ */
/* Mode switching (embed <-> native)                                   */
/* ------------------------------------------------------------------ */

async function getMode() {
  const { [EMBED_KEY]: mode } = await chrome.storage.local.get(EMBED_KEY);
  return mode === "native" ? "native" : "embed";
}

function applyMode(mode) {
  const embed = mode !== "native";
  $("#embed-view").hidden = !embed;
  $("#native-view").hidden = embed;
  document.body.classList.toggle("mode-chat", embed);
  document.body.classList.toggle("mode-native", !embed);
}

async function setMode(mode) {
  await chrome.storage.local.set({ [EMBED_KEY]: mode });
  applyMode(mode);
  if (mode === "native") {
    // lazy-init native view
    conversations = await getConversations();
    const activeId = await getActiveConversationId();
    conversation = conversations.find((c) => c.id === activeId) || createConversation();
    loadModels();
    checkHealth();
    renderMessages();
    renderHistory();
  }
}

/* ------------------------------------------------------------------ */
/* Embedded g4f.dev chat bridge                                        */
/* ------------------------------------------------------------------ */

function wireEmbedBridge() {
  const frame = $("#chat-frame");

  // Follow the configured server (g4f.space -> g4f.dev, else the server
  // itself). The iframe's HTML default already points at g4f.dev, so this
  // only triggers a reload for custom server URLs.
  if (frame.src !== getChatUrl()) frame.src = getChatUrl();

  $("#embed-reload").addEventListener("click", () => {
    embedChatReady = false;
    frame.src = getChatUrl(); // base URL: never re-run a #q= prompt
  });
  $("#embed-open-tab").addEventListener("click", () =>
    chrome.tabs.create({ url: getChatUrl() })
  );
  $("#mode-toggle").addEventListener("click", () => setMode("native"));
  $("#mode-toggle-native").addEventListener("click", () => setMode("embed"));
  $("#open-options").addEventListener("click", () => chrome.runtime.openOptionsPage());
  $("#open-options-native").addEventListener("click", () => chrome.runtime.openOptionsPage());

  // Surface load failures (offline / blocked) without blocking the UI.
  frame.addEventListener("error", () => showEmbedError("Could not load the chat."));
  frame.addEventListener("load", () => {
    embedChatReady = false; // a (re)loaded chat must re-announce
    $("#embed-status").hidden = true;
    // The chat app announces itself once its addons are up.
    frame.contentWindow?.postMessage({ type: "g4f-ext:hello" }, getChatUrl());
    // Fast path: the chat registers its message listener before its load
    // event, so a direct postMessage usually lands. Must not go through
    // sendToEmbed() here — embedChatReady is still false and it would
    // answer with a #q= reload, looping forever.
    if (pendingEmbedPrompt) {
      frame.contentWindow?.postMessage({ type: "g4f-ext:ask", prompt: pendingEmbedPrompt }, getChatUrl());
    }
  });

  window.addEventListener("message", (event) => {
    if (event.origin !== getChatOrigin()) return;
    const data = event.data || {};
    if (data.type === "g4f-chat:ready") {
      embedChatReady = true;
      embedReadyResolve?.();
      embedReadyResolve = null;
      if (pendingEmbedPrompt) sendToEmbed(pendingEmbedPrompt);
    }
    if (data.type === "g4f-ext:open-auth" && typeof data.url === "string") {
      openLoginPopup(data.url);
    }
    if (data.type === "g4f-ext:auth-done") {
      // The login popup stored the session (announced via the shared
      // localStorage). Close it; onRemoved refreshes the chat's login UI.
      closeLoginPopup();
    }
  });
}

/* Login popup handoff ------------------------------------------------- */
/* The embedded chat cannot open windows itself (window.open is suppressed
 * for cross-origin frames inside the side panel), so it asks us to open
 * the OAuth login window for it. */

let loginPopupWinId = null;

function openLoginPopup(rawUrl) {
  let target;
  try {
    target = new URL(rawUrl, getChatUrl());
  } catch {
    return;
  }
  const chatOrigin = new URL(getChatUrl()).origin;
  const chatHost = new URL(chatOrigin).hostname;
  const host = target.hostname;
  const hostOk =
    host === chatHost ||
    host === "g4f.dev" ||
    host === "auth.g4f.space" ||
    host.endsWith(".g4f.space") ||
    host.endsWith(".g4f.dev");
  // https for the public hosts; http only when the chat itself is http
  // (self-hosted server on localhost).
  const protoOk = target.protocol === "https:" ||
    (target.protocol === "http:" && chatOrigin.startsWith("http:"));
  if (!protoOk || !hostOk) return;

  chrome.windows.create(
    { url: target.toString(), type: "popup", width: 520, height: 760 },
    (win) => {
      if (!win || !win.id) return;
      loginPopupWinId = win.id;
      const tabId = win.tabs?.[0]?.id;
      if (tabId == null) return;
      // Auto-close the popup once the OAuth callback (?code=…) has been
      // handled and the session stored.
      const onUpdated = (tid, info) => {
        if (tid !== tabId || loginPopupWinId !== win.id) return;
        const u = info.url || "";
        if (!u.startsWith(new URL(getChatUrl()).origin + "/") || !u.includes("code=")) return;
        setTimeout(() => {
          chrome.tabs.onUpdated.removeListener(onUpdated);
          if (loginPopupWinId === win.id) {
            loginPopupWinId = null;
            chrome.windows.remove(win.id, () => void chrome.runtime.lastError);
          }
        }, 2500);
      };
      chrome.tabs.onUpdated.addListener(onUpdated);
    }
  );
}

chrome.windows.onRemoved.addListener((winId) => {
  if (winId !== loginPopupWinId) return;
  loginPopupWinId = null;
  // Nudge the embedded chat to re-check its login state.
  $("#chat-frame")?.contentWindow?.postMessage(
    { type: "g4f-ext:auth-done" },
    getChatUrl()
  );
});

/** Close the login popup if it is still open (best-effort). */
function closeLoginPopup() {
  if (loginPopupWinId == null) return;
  const winId = loginPopupWinId;
  loginPopupWinId = null;
  chrome.windows.remove(winId, () => void chrome.runtime.lastError);
}

function showEmbedError(message) {
  const el = $("#embed-status");
  el.textContent = message + " Use ⇆ for the built-in chat or ⧉ to open the chat in a tab.";
  el.hidden = false;
}

/**
 * Inject a prompt into the embedded chat. The chat's own ask flow
 * (handle_ask) reads #userInput, so we fill it and click #sendButton.
 * Falls back to a URL handoff (?prompt=…) if the bridge is not ready.
 */
function sendToEmbed(prompt) {
  const frame = $("#chat-frame");
  if (!embedChatReady) {
    // Chat not ready (or still loading): a postMessage could fire before
    // the page's listener exists. Instead, reload the frame with the
    // prompt in the #q= hash — the chat page consumes it on load.
    pendingEmbedPrompt = null;
    frame.src = getChatUrl() + "#q=" + encodeURIComponent(prompt);
    return;
  }
  try {
    frame.contentWindow?.postMessage({ type: "g4f-ext:ask", prompt }, getChatUrl());
    pendingEmbedPrompt = null;
  } catch {
    pendingEmbedPrompt = prompt;
  }
}

/** Wait briefly for the embedded chat to announce readiness. */
function waitForEmbedReady(timeoutMs = 8000) {
  if (embedChatReady) return Promise.resolve();
  if (!embedReadyWait) {
    let resolveWait;
    embedReadyWait = new Promise((resolve) => { resolveWait = resolve; });
    const timer = setTimeout(() => { embedReadyWait = null; resolveWait(); }, timeoutMs);
    embedReadyResolve = () => {
      clearTimeout(timer);
      embedReadyWait = null;
      resolveWait();
    };
  }
  return embedReadyWait;
}

/* ------------------------------------------------------------------ */
/* Conversation management (native mode)                               */
/* ------------------------------------------------------------------ */

function createConversation() {
  const c = newConversation();
  conversations.unshift(c);
  saveConversations(conversations);
  setActiveConversationId(c.id);
  return c;
}

function touchConversation() {
  conversation.updatedAt = Date.now();
  const idx = conversations.findIndex((c) => c.id === conversation.id);
  if (idx >= 0) conversations[idx] = conversation;
  saveConversations(conversations);
  setActiveConversationId(conversation.id);
  renderHistory();
}

function resetChat() {
  conversation = createConversation();
  renderMessages();
  renderHistory();
}

/* ------------------------------------------------------------------ */
/* History drawer (native mode)                                        */
/* ------------------------------------------------------------------ */

function renderHistory() {
  const list = $("#history-list");
  if (!list) return;
  list.innerHTML = "";
  const items = conversations
    .filter((c) => c.messages.some((m) => m.role === "user" && m.content))
    .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));

  if (!items.length) {
    list.innerHTML = '<div class="history-empty">No conversations yet</div>';
    return;
  }
  for (const c of items) {
    const row = document.createElement("div");
    row.className = "history-item" + (c.id === conversation.id ? " active" : "");
    const btn = document.createElement("button");
    btn.className = "history-open";
    btn.innerHTML =
      `<span class="history-title">${escapeHtml(c.title || "New chat")}</span>` +
      `<span class="history-date">${new Date(c.updatedAt || c.createdAt).toLocaleDateString()}</span>`;
    btn.addEventListener("click", async () => {
      conversation = c;
      await setActiveConversationId(c.id);
      renderMessages();
      renderHistory();
      $("#history-drawer").hidden = true;
    });
    const del = document.createElement("button");
    del.className = "history-del";
    del.textContent = "🗑";
    del.title = "Delete conversation";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      conversations = conversations.filter((x) => x.id !== c.id);
      await saveConversations(conversations);
      if (conversation.id === c.id) {
        conversation = createConversation();
        renderMessages();
      }
      renderHistory();
    });
    row.appendChild(btn);
    row.appendChild(del);
    list.appendChild(row);
  }
}

/* ------------------------------------------------------------------ */
/* Events (native mode)                                                */
/* ------------------------------------------------------------------ */

function wireEvents() {
  $("#new-chat").addEventListener("click", resetChat);
  $("#history-btn").addEventListener("click", () => {
    const drawer = $("#history-drawer");
    drawer.hidden = !drawer.hidden;
    if (!drawer.hidden) renderHistory();
  });
  $("#history-close").addEventListener("click", () => { $("#history-drawer").hidden = true; });
  $("#history-clear").addEventListener("click", async () => {
    if (!confirm("Delete all conversations?")) return;
    conversations = [];
    await saveConversations(conversations);
    conversation = createConversation();
    renderMessages();
    renderHistory();
  });

  const input = $("#input");
  const send = $("#send");
  const stop = $("#stop");

  send.addEventListener("click", () => submit());
  stop.addEventListener("click", () => {
    if (streamId) {
      chrome.runtime.sendMessage({ type: MESSAGE_TYPES.CHAT_ABORT, streamId });
    }
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && settings.sendOnEnter) {
      e.preventDefault();
      submit();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
    imageMode = input.value.startsWith("/img ");
  });

  $("#include-page").addEventListener("change", (e) =>
    saveSettings({ includePageContext: e.target.checked })
  );
  $("#include-page").checked = !!settings.includePageContext;

  // Model picker
  $("#model-select").addEventListener("change", (e) => {
    saveSettings({ model: e.target.value });
    toast($("#toast-root"), "Model: " + (e.target.value || "default"));
  });
  // Click-to-retry when the model list failed to load.
  $("#model-select").addEventListener("click", (e) => {
    if (e.target.classList.contains("error")) {
      e.preventDefault();
      loadModels();
    }
  });
}

async function loadModels() {
  const select = $("#model-select");
  select.classList.remove("error");
  select.innerHTML = '<option value="">Loading models…</option>';
  try {
    const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.MODELS });
    if (!res?.ok) throw new Error(res?.error || "No response from background");
    const models = res.data.filter((m) => !m.provider);
    select.innerHTML = '<option value="">Default model</option>';
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.id;          // full id sent to the server
      opt.textContent = m.label || m.id; // human-readable label
      if (m.id === settings.model) opt.selected = true;
      select.appendChild(opt);
    }
    if (!models.length) {
      select.innerHTML = '<option value="">No models available</option>';
      select.classList.add("error");
    } else if (settings.model) {
      select.value = settings.model; // show saved pick in the closed box
    }
  } catch (e) {
    // Keep the usable "Default model" entry and surface WHY it failed
    // (hover the select for details; the server host is included).
    console.warn("[g4f] loadModels failed:", e);
    let host = settings.serverUrl;
    try { host = new URL(normalizeBaseUrl(settings.serverUrl)).host; } catch { /* keep raw */ }
    select.innerHTML = '<option value="">Default model</option>';
    select.title = `Model list unavailable (${host}): ${e?.message || e} — click to retry`;
    select.classList.add("error");
  }
}

async function checkHealth() {
  const el = $("#conn-status");
  try {
    const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.HEALTH });
    el.classList.toggle("ok", !!res?.ok);
    el.classList.toggle("bad", !res?.ok);
    el.title = res?.ok ? "Server reachable" : "Server unreachable";
  } catch {
    el.classList.add("bad");
  }
}

/* ------------------------------------------------------------------ */
/* Submit / stream (native mode)                                       */
/* ------------------------------------------------------------------ */

async function submit() {
  const input = $("#input");
  let text = input.value.trim();
  if (!text || generating) return;

  const isImage = text.startsWith("/img ");
  if (isImage) {
    input.value = "";
    input.style.height = "auto";
    return runImage(text.slice(5).trim());
  }

  const messages = [];
  if (settings.systemPrompt) {
    messages.push({ role: "system", content: settings.systemPrompt });
  }
  for (const m of conversation.messages) {
    messages.push({ role: m.role, content: m.content });
  }

  // Context enrichment
  const extras = [];
  if ($("#include-page").checked) {
    const page = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.GET_PAGE });
    if (page?.text) {
      extras.push(
        `Current page: "${page.title}" (${page.url})\n\n---\n${page.text.slice(0, settings.pageContextLimit)}\n---`
      );
    }
  }
  if ($("#include-selection").checked) {
    const sel = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.GET_SELECTION });
    if (sel?.text?.trim()) {
      extras.push("Selected text:\n" + sel.text.trim().slice(0, settings.pageContextLimit));
    }
  }

  const userContent = extras.length ? extras.join("\n\n") + "\n\n" + text : text;
  messages.push({ role: "user", content: userContent });

  conversation.messages.push({ role: "user", content: text });
  if (conversation.title === "New chat") {
    conversation.title = text.slice(0, 42) + (text.length > 42 ? "…" : "");
  }
  appendMessageEl({ role: "user", content: text });
  touchConversation();

  input.value = "";
  input.style.height = "auto";
  setGenerating(true);

  conversation.messages.push({ role: "assistant", content: "" });
  const assistantEl = appendMessageEl({ role: "assistant", content: "" }, true);

  streamId = "panel-" + Date.now();
  port = chrome.runtime.connect({ name: "g4f:" + streamId });
  port.onMessage.addListener((msg) => onStream(msg, assistantEl));
  port.onDisconnect.addListener(() => (port = null));

  await chrome.runtime.sendMessage({
    type: MESSAGE_TYPES.CHAT,
    streamId,
    messages,
    overrides: { model: $("#model-select").value || undefined },
  });
}

function onStream(msg, assistantEl) {
  if (msg.type === MESSAGE_TYPES.STREAM_CHUNK) {
    const last = conversation.messages[conversation.messages.length - 1];
    if (last?.role === "assistant") {
      last.content = msg.full;
      const body = assistantEl.querySelector(".msg-body");
      body.innerHTML = renderMarkdown(msg.full);
      scrollBottom();
    }
  } else if (msg.type === MESSAGE_TYPES.STREAM_DONE) {
    const last = conversation.messages[conversation.messages.length - 1];
    if (last?.role === "assistant" && !last.content) {
      last.content = "*(empty response)*";
      assistantEl.querySelector(".msg-body").textContent = "(empty response)";
    }
    touchConversation();
    setGenerating(false);
  } else if (msg.type === MESSAGE_TYPES.STREAM_ERROR) {
    assistantEl.querySelector(".msg-body").innerHTML =
      `<span class="error">⚠ ${escapeHtml(msg.error || "Request failed")}</span>`;
    const last = conversation.messages[conversation.messages.length - 1];
    if (last?.role === "assistant") last.content = "Error: " + (msg.error || "");
    touchConversation();
    setGenerating(false);
  }
}

function setGenerating(on) {
  generating = on;
  $("#send").hidden = on;
  $("#stop").hidden = !on;
}

function scrollBottom() {
  const msgs = $("#messages");
  msgs.scrollTop = msgs.scrollHeight;
}

/* ------------------------------------------------------------------ */
/* Image generation (native mode)                                      */
/* ------------------------------------------------------------------ */

async function runImage(prompt) {
  if (!prompt) return;
  appendMessageEl({ role: "user", content: "🖼 " + prompt });
  const container = $("#images");
  const placeholder = document.createElement("div");
  placeholder.className = "img-placeholder";
  placeholder.innerHTML = '<span class="spinner"></span> Generating…';
  container.prepend(placeholder);

  streamId = "img-" + Date.now();
  port = chrome.runtime.connect({ name: "g4f:" + streamId });
  port.onMessage.addListener((msg) => {
    if (msg.type === MESSAGE_TYPES.STREAM_DONE) {
      placeholder.remove();
      for (const url of msg.urls || []) {
        const img = document.createElement("img");
        img.src = url;
        img.className = "generated-img";
        img.title = "Click to open";
        img.addEventListener("click", () => window.open(url, "_blank"));
        container.prepend(img);
      }
      if (!msg.urls?.length) {
        toast($("#toast-root"), "No images returned", "warn");
      }
    } else if (msg.type === MESSAGE_TYPES.STREAM_ERROR) {
      placeholder.remove();
      toast($("#toast-root"), msg.error || "Image failed", "error");
    }
  });
  await chrome.runtime.sendMessage({
    type: MESSAGE_TYPES.IMAGE,
    streamId,
    prompt,
  });
}

/* ------------------------------------------------------------------ */
/* Rendering (native mode)                                             */
/* ------------------------------------------------------------------ */

function renderMessages() {
  const root = $("#messages");
  root.innerHTML = "";
  if (!conversation.messages.length) {
    root.innerHTML = `
      <div class="welcome">
        <img src="../icons/icon48.png" alt="" width="42" height="42" />
        <h1>Free AI, in your sidebar</h1>
        <p>Ask anything. Use <strong>Ask g4f</strong> in the right-click menu on any selection, or enable “page context” to chat about the current tab.</p>
        <div class="welcome-tips">
          <span class="tip">/img a red cube — generate images</span>
          <span class="tip">Alt+Shift+S — toggle panel</span>
        </div>
      </div>`;
    return;
  }
  for (const m of conversation.messages) appendMessageEl(m);
  scrollBottom();
}

function appendMessageEl(message, streaming = false) {
  const root = $("#messages");
  root.querySelector(".welcome")?.remove();

  const wrap = document.createElement("div");
  wrap.className = `msg msg-${message.role}` + (streaming ? " streaming" : "");

  const actions = document.createElement("div");
  actions.className = "msg-actions";

  if (message.role === "assistant" && message.content) {
    const copy = document.createElement("button");
    copy.className = "copy-btn mini";
    copy.textContent = "⧉";
    copy.title = "Copy";
    copy.setAttribute("data-copy", message.content);
    actions.appendChild(copy);
  }

  const body = document.createElement("div");
  body.className = "msg-body";
  body.innerHTML = message.content ? renderMarkdown(message.content) : '<span class="spinner"></span>';

  wrap.appendChild(actions);
  wrap.appendChild(body);
  root.appendChild(wrap);
  scrollBottom();
  return wrap;
}

/* ------------------------------------------------------------------ */
/* Handoffs (context menu / popup)                                     */
/* ------------------------------------------------------------------ */

async function consumePreset() {
  const { g4fPreset } = await chrome.storage.session.get("g4fPreset");
  if (!g4fPreset) return;
  await chrome.storage.session.remove("g4fPreset");
  if (Date.now() - (g4fPreset.ts || 0) > 30000) return; // stale

  const prompts = {
    ask: (t) => t,
    explain: (t) => `Explain this in simple terms:\n\n${t}`,
    translate: (t) => `Translate the following text to English. Only output the translation:\n\n${t}`,
    summarize: (t) => `Summarize the following page content in a few bullet points:\n\n${t}`,
  };
  const fn = prompts[g4fPreset.type] || prompts.ask;
  const text = (g4fPreset.text || "").trim();
  if (!text) return;

  const prompt = fn(text.slice(0, settings.pageContextLimit));

  if ($("#embed-view").hidden === false) {
    // Embedded chat: wait for readiness, then inject.
    await waitForEmbedReady();
    sendToEmbed(prompt);
  } else {
    const input = $("#input");
    input.value = prompt;
    input.dispatchEvent(new Event("input"));
    submit();
  }
}

async function consumeHandoff() {
  const { g4fHandoff } = await chrome.storage.session.get("g4fHandoff");
  if (!g4fHandoff) return;
  await chrome.storage.session.remove("g4fHandoff");
  if (Date.now() - (g4fHandoff.ts || 0) > 60000) return;

  // Popup mini-chat results always land in the native history so they persist.
  await setMode("native");
  for (const m of g4fHandoff.messages || []) {
    conversation.messages.push(m);
    appendMessageEl(m);
  }
  touchConversation();
}
