/**
 * g4f extension — popup.
 * Quick actions and mini-chats. Everything typed here is handed off to the
 * side panel (embedded g4f.dev chat or native lite chat) so it persists in
 * the conversation history instead of vanishing when the popup closes.
 */

import { MESSAGE_TYPES } from "../lib/constants.js";
import { getSettings } from "../lib/storage.js";
import { toast, bindCopyHandlers } from "../lib/ui.js";
import { renderMarkdown } from "../lib/markdown.js";

const $ = (sel) => document.querySelector(sel);

let settings = null;
let streamId = null;
let port = null;
let pending = []; // messages accumulated for the side-panel handoff

init();

async function init() {
  settings = await getSettings();
  bindCopyHandlers(document, () => toast($("#toast-root"), "Copied"));

  renderQuickActions();
  wireEvents();
  checkHealth();
  loadSelection();
}

/* ------------------------------------------------------------------ */
/* Quick actions                                                       */
/* ------------------------------------------------------------------ */

function renderQuickActions() {
  const wrap = $("#quick-actions");
  wrap.innerHTML = "";
  for (const action of settings.quickActions || []) {
    const btn = document.createElement("button");
    btn.className = "qa-btn";
    btn.textContent = action.label;
    btn.title = `Run "${action.label}" on the current page or selection`;
    btn.addEventListener("click", () => runQuickAction(action));
    wrap.appendChild(btn);
  }
  if (!settings.quickActions?.length) {
    wrap.innerHTML = '<div class="qa-empty">No quick actions configured</div>';
  }
}

/**
 * Quick actions run against the active tab's page/selection and open the
 * side panel, where the prompt is injected into the embedded chat.
 */
async function runQuickAction(action) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  let text = "";
  try {
    const res = await chrome.runtime.sendMessage({
      type: MESSAGE_TYPES.GET_SELECTION, tabId: tab?.id,
    });
    text = res?.text?.trim() || "";
  } catch { /* fall through to page text */ }
  if (!text) {
    try {
      const res = await chrome.runtime.sendMessage({
        type: MESSAGE_TYPES.GET_PAGE, tabId: tab?.id,
      });
      text = (res?.text || "").slice(0, settings.pageContextLimit);
    } catch { /* no context available */ }
  }

  const templates = {
    summarize: (t) => `Summarize the following content in a few bullet points:\n\n${t}`,
    explain: (t) => `Explain this in simple terms:\n\n${t}`,
    translate: (t) => `Translate the following text to English. Only output the translation:\n\n${t}`,
    ask: (t) => t,
  };
  const fn = templates[action.type] || templates.ask;
  const prompt = fn(text || action.prompt || "");

  // Hand off to the side panel; it owns the conversation.
  await chrome.storage.session.set({
    g4fPreset: { type: action.type, text: text || action.prompt || "", ts: Date.now() },
  });
  await openPanel();
}

/* ------------------------------------------------------------------ */
/* Mini chat                                                           */
/* ------------------------------------------------------------------ */

function wireEvents() {
  const input = $("#input");
  $("#send").addEventListener("click", () => sendPrompt());
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && settings.sendOnEnter) {
      e.preventDefault();
      sendPrompt();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 100) + "px";
  });

  $("#open-panel").addEventListener("click", () => openPanel());
  $("#open-options").addEventListener("click", () => chrome.runtime.openOptionsPage());
}

async function openPanel() {
  try {
    // Must be called synchronously from the user gesture when possible.
    await chrome.sidePanel.open({ windowId: chrome.windows.WINDOW_ID_CURRENT });
  } catch {
    // Fallback: let the service worker open it.
    chrome.runtime.sendMessage({ type: MESSAGE_TYPES.OPEN_SIDE_PANEL }).catch(() => {});
  }
  window.close();
}

async function sendPrompt() {
  const input = $("#input");
  const text = input.value.trim();
  if (!text || streamId) return;

  pending.push({ role: "user", content: text });
  appendMessage({ role: "user", content: text });
  input.value = "";
  input.style.height = "auto";
  setGenerating(true);

  const messages = [];
  if (settings.systemPrompt) {
    messages.push({ role: "system", content: settings.systemPrompt });
  }
  for (const m of pending) messages.push({ role: m.role, content: m.content });

  streamId = "popup-" + Date.now();
  port = chrome.runtime.connect({ name: "g4f:" + streamId });
  port.onMessage.addListener(onStream);
  port.onDisconnect.addListener(() => (port = null));

  await chrome.runtime.sendMessage({
    type: MESSAGE_TYPES.CHAT,
    streamId,
    messages,
    overrides: { model: settings.model || undefined },
  });
}

function onStream(msg) {
  if (msg.type === MESSAGE_TYPES.STREAM_CHUNK) {
    const last = pending[pending.length - 1];
    if (last?.role === "assistant") {
      last.content = msg.full;
      const el = $("#messages").lastElementChild?.querySelector(".msg-body");
      if (el) el.innerHTML = renderMarkdown(msg.full);
    }
  } else if (msg.type === MESSAGE_TYPES.STREAM_DONE) {
    const last = pending[pending.length - 1];
    if (last?.role === "assistant" && !last.content) last.content = "*(empty response)*";
    setGenerating(false);
  } else if (msg.type === MESSAGE_TYPES.STREAM_ERROR) {
    const el = $("#messages").lastElementChild?.querySelector(".msg-body");
    if (el) el.innerHTML = `<span class="error">⚠ ${msg.error || "Request failed"}</span>`;
    setGenerating(false);
  }
}

function setGenerating(on) {
  $("#send").hidden = on;
  $("#stop").hidden = !on;
}

function appendMessage(message) {
  const root = $("#messages");
  root.querySelector(".welcome")?.remove();
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${message.role}`;

  const actions = document.createElement("div");
  actions.className = "msg-actions";
  if (message.role === "assistant" && message.content) {
    const copy = document.createElement("button");
    copy.className = "copy-btn mini";
    copy.textContent = "⧉";
    copy.setAttribute("data-copy", message.content);
    actions.appendChild(copy);
  }

  const body = document.createElement("div");
  body.className = "msg-body";
  body.innerHTML = message.content ? renderMarkdown(message.content) : '<span class="spinner"></span>';

  wrap.appendChild(actions);
  wrap.appendChild(body);
  root.appendChild(wrap);
  root.scrollTop = root.scrollHeight;
}


/* ------------------------------------------------------------------ */
/* Status                                                              */
/* ------------------------------------------------------------------ */

async function checkHealth() {
  const dot = $("#health-dot");
  try {
    const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.HEALTH });
    dot.classList.toggle("ok", !!res?.ok);
    dot.classList.toggle("bad", !res?.ok);
    dot.title = res?.ok ? "Server reachable" : "Server unreachable";
  } catch {
    dot.classList.add("bad");
  }
}

async function loadSelection() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const res = await chrome.runtime.sendMessage({
      type: MESSAGE_TYPES.GET_SELECTION, tabId: tab?.id,
    });
    const text = res?.text?.trim();
    if (text) {
      const box = $("#selection-box");
      box.hidden = false;
      $("#selection-text").textContent = text.slice(0, 120) + (text.length > 120 ? "…" : "");
      $("#selection-use").addEventListener("click", () => {
        $("#input").value = text;
        $("#input").focus();
      });
    }
  } catch { /* activeTab not granted on this tab */ }
}
