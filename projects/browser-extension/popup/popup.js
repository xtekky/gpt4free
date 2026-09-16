/**
 * g4f extension — popup logic.
 * Quick actions, selection handling, mini chat preview, handoff to side panel.
 */

import { MESSAGE_TYPES } from "../lib/constants.js";
import { getSettings } from "../lib/storage.js";
import { renderMarkdown } from "../lib/markdown.js";
import { toast, bindCopyHandlers, escapeHtml } from "../lib/ui.js";

const $ = (sel) => document.querySelector(sel);

let settings = null;
let selection = "";
let pending = []; // messages accumulated in the popup preview
let streamId = null;
let port = null;

init();

async function init() {
  settings = await getSettings();
  bindCopyHandlers(document, () => {});
  renderQuickActions();
  wireEvents();
  checkHealth();
  loadSelection();
}

function renderQuickActions() {
  const root = $("#quick-actions");
  root.innerHTML = "";
  for (const qa of settings.quickActions || []) {
    const btn = document.createElement("button");
    btn.className = "qa-btn";
    btn.textContent = qa.label;
    btn.title = qa.prompt.slice(0, 120);
    btn.addEventListener("click", () => runQuickAction(qa));
    root.appendChild(btn);
  }
}

function wireEvents() {
  $("#open-options").addEventListener("click", () => chrome.runtime.openOptionsPage());
  $("#open-panel").addEventListener("click", async () => {
    // Must be called directly in the popup: the user-gesture context is lost
    // when routed through the service worker, and Chrome then rejects
    // sidePanel.open() with "may only be called in response to a user gesture".
    try {
      const win = await chrome.windows.getCurrent();
      await chrome.sidePanel.open({ windowId: win.id });
    } catch (e) {
      console.warn("[g4f] direct sidePanel.open failed, falling back to SW:", e);
      try {
        await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.OPEN_SIDE_PANEL });
      } catch { /* ignore */ }
    }
    window.close();
  });

  $("#include-page").addEventListener("change", (e) => {
    // Persist choice for next time.
    chrome.storage.sync.set({ g4fIncludePage: e.target.checked });
  });
  chrome.storage.sync.get("g4fIncludePage").then((r) => {
    $("#include-page").checked = !!r.g4fIncludePage;
  });

  const promptEl = $("#prompt");
  const sendEl = $("#send");
  sendEl.addEventListener("click", () => sendPrompt(promptEl.value));
  promptEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && settings.sendOnEnter) {
      e.preventDefault();
      sendPrompt(promptEl.value);
    }
  });
}

async function loadSelection() {
  try {
    const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.GET_SELECTION });
    selection = (res?.text || "").trim();
    if (selection) {
      $("#selection-box").hidden = false;
      $("#selection-text").textContent =
        selection.length > 300 ? selection.slice(0, 300) + "…" : selection;
    }
  } catch {
    /* activeTab not granted — ignore */
  }
}

async function checkHealth() {
  const dot = $("#status-dot");
  try {
    const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.HEALTH });
    dot.classList.toggle("ok", !!res?.ok);
    dot.classList.toggle("bad", !res?.ok);
    dot.title = res?.ok
      ? "g4f server reachable"
      : "g4f server unreachable — click to open settings";
    if (!res?.ok) {
      dot.addEventListener("click", () => chrome.runtime.openOptionsPage(), { once: true });
    }
  } catch {
    dot.classList.add("bad");
  }
}

async function runQuickAction(qa) {
  let context = selection;
  if (!context && qa.id === "summarize") {
    const page = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.GET_PAGE });
    context = page?.text || "";
  }
  if (!context) {
    toast($("#toast-root"), "No text selected / page empty", "warn");
    return;
  }
  await sendPrompt(qa.prompt + context.slice(0, settings.pageContextLimit));
}

async function sendPrompt(text) {
  text = (text || "").trim();
  if (!text || streamId) return;

  const messages = [];
  if (settings.systemPrompt) {
    messages.push({ role: "system", content: settings.systemPrompt });
  }
  if ($("#include-page").checked) {
    const page = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.GET_PAGE });
    if (page?.text) {
      messages.push({
        role: "system",
        content: `Current page: "${page.title}" (${page.url})\n\nContent:\n${page.text.slice(0, settings.pageContextLimit)}`,
      });
    }
  }
  messages.push({ role: "user", content: text });

  // Show preview area with user message
  pending.push({ role: "user", content: text });
  renderPreview();
  $("#chat-preview").hidden = false;
  $("#prompt").value = "";
  $("#send").disabled = true;

  // Open streaming port to background
  streamId = "popup-" + Date.now();
  port = chrome.runtime.connect({ name: "g4f:" + streamId });
  port.onMessage.addListener(onStreamMessage);
  port.onDisconnect.addListener(() => {
    port = null;
  });

  await chrome.runtime.sendMessage({
    type: MESSAGE_TYPES.CHAT,
    streamId,
    messages,
  });
}

function onStreamMessage(msg) {
  if (msg.type === MESSAGE_TYPES.STREAM_CHUNK) {
    const last = pending[pending.length - 1];
    if (last && last.role === "assistant") {
      last.content = msg.full;
    } else {
      pending.push({ role: "assistant", content: msg.full });
    }
    renderPreview();
  } else if (msg.type === MESSAGE_TYPES.STREAM_DONE) {
    finishStream();
  } else if (msg.type === MESSAGE_TYPES.STREAM_ERROR) {
    toast($("#toast-root"), msg.error || "Request failed", "error");
    finishStream();
  }
}

function finishStream() {
  streamId = null;
  $("#send").disabled = false;
  // Hand the conversation off to the side panel so the user keeps it.
  if (pending.some((m) => m.role === "assistant")) {
    chrome.storage.session.set({ g4fHandoff: { messages: pending, ts: Date.now() } });
  }
}

function renderPreview() {
  const root = $("#preview-messages");
  root.innerHTML = pending
    .map(
      (m) =>
        `<div class="msg msg-${m.role}"><div class="msg-body">${renderMarkdown(m.content)}</div></div>`
    )
    .join("");
  root.scrollTop = root.scrollHeight;
}
