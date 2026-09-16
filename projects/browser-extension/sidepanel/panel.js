/**
 * g4f extension — side panel logic.
 * Full conversation UI: history, streaming, model picker, context toggles,
 * image generation mode, context-menu presets and popup handoff.
 */

import { MESSAGE_TYPES, normalizeBaseUrl } from "../lib/constants.js";
import {
  getSettings, saveSettings, getConversations, saveConversations,
  getActiveConversationId, setActiveConversationId, newConversation,
} from "../lib/storage.js";
import { renderMarkdown } from "../lib/markdown.js";
import { toast, bindCopyHandlers, escapeHtml } from "../lib/ui.js";

const $ = (sel) => document.querySelector(sel);

let settings = null;
let conversation = null;
let conversations = [];
let streamId = null;
let port = null;
let generating = false;
let imageMode = false;

init();

async function init() {
  settings = await getSettings();
  bindCopyHandlers(document, () => toast($("#toast-root"), "Copied"));

  conversations = await getConversations();
  const activeId = await getActiveConversationId();
  conversation =
    conversations.find((c) => c.id === activeId) || createConversation();

  wireEvents();
  loadModels();
  checkHealth();
  renderMessages();
  refreshAccountUI();

  await consumePreset();   // context-menu handoff
  await consumeHandoff();  // popup handoff
}

/* ------------------------------------------------------------------ */
/* Account (g4f.space OAuth) & cloud sync                              */
/* ------------------------------------------------------------------ */

async function refreshAccountUI() {
  const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.OAUTH_STATUS });
  const signedIn = !!res?.signedIn;
  const user = res?.user;

  $("#account-btn").hidden = signedIn;
  $("#sync-btn").hidden = !signedIn;
  $("#account-bar").hidden = !signedIn;
  if (signedIn && user) {
    $("#account-name").textContent = user.name || user.username || user.id;
    $("#account-tier").textContent = user.tier ? String(user.tier) : "";
    const avatar = $("#account-avatar");
    if (user.avatar) { avatar.src = user.avatar; avatar.hidden = false; }
    else avatar.hidden = true;
  }
}

async function startSignIn() {
  const btn = $("#account-btn");
  btn.disabled = true;
  btn.textContent = "…";
  try {
    const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.OAUTH_START });
    if (!res?.ok) throw new Error(res?.error || "Sign-in failed");
    toast($("#toast-root"), "Signed in as " + (res.user?.name || res.user?.id), "ok");
    await refreshAccountUI();
    await syncNow(true);
  } catch (e) {
    toast($("#toast-root"), e.message || String(e), "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "⇧";
  }
}

async function syncNow(silent = false) {
  const btn = $("#sync-btn");
  btn.classList.add("spinning");
  try {
    const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.SYNC_NOW });
    if (!res?.ok) throw new Error(res?.error || "Sync failed");
    if (res.pulled > 0) {
      conversations = await getConversations();
      const activeId = await getActiveConversationId();
      conversation = conversations.find((c) => c.id === activeId) || conversation;
      renderMessages();
    }
    if (!silent) {
      toast($("#toast-root"), `Synced: pushed ${res.pushed}, pulled ${res.pulled}`, "ok");
    }
  } catch (e) {
    if (!silent) toast($("#toast-root"), e.message || String(e), "error");
  } finally {
    btn.classList.remove("spinning");
  }
}

/* ------------------------------------------------------------------ */
/* Conversation management                                             */
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
}

function resetChat() {
  conversation = createConversation();
  renderMessages();
}

/* ------------------------------------------------------------------ */
/* Events                                                              */
/* ------------------------------------------------------------------ */

function wireEvents() {
  $("#new-chat").addEventListener("click", resetChat);
  $("#open-options").addEventListener("click", () => chrome.runtime.openOptionsPage());

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
  });

  // Image mode: prefix prompt with "/img "
  input.addEventListener("input", () => {
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

  // Account & sync
  $("#account-btn").addEventListener("click", startSignIn);
  $("#account-logout").addEventListener("click", async () => {
    await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.OAUTH_LOGOUT });
    await refreshAccountUI();
    toast($("#toast-root"), "Signed out");
  });
  $("#sync-btn").addEventListener("click", () => syncNow());
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
/* Submit / stream                                                     */
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
    // Auto-push the finished conversation to the user's secret workspace.
    syncNow(true);
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
/* Image generation                                                    */
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
/* Rendering                                                           */
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

  const input = $("#input");
  input.value = fn(text.slice(0, settings.pageContextLimit));
  input.dispatchEvent(new Event("input"));
  submit();
}

async function consumeHandoff() {
  const { g4fHandoff } = await chrome.storage.session.get("g4fHandoff");
  if (!g4fHandoff) return;
  await chrome.storage.session.remove("g4fHandoff");
  if (Date.now() - (g4fHandoff.ts || 0) > 60000) return;

  for (const m of g4fHandoff.messages || []) {
    conversation.messages.push(m);
    appendMessageEl(m);
  }
  touchConversation();
}
