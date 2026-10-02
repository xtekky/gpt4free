/**
 * g4f extension — options page logic.
 */

import { MESSAGE_TYPES } from "../lib/constants.js";
import { getSettings, saveSettings, resetSettings } from "../lib/storage.js";
import { toast, escapeHtml } from "../lib/ui.js";

const $ = (sel) => document.querySelector(sel);

let settings = null;

init();

async function init() {
  settings = await getSettings();
  fillForm();
  wireEvents();
  loadModelOptions();
  refreshCdpUI();
}

/* ---------------- CDP bridge (browser as provider) ---------------- */

async function refreshCdpUI() {
  const res = await chrome.runtime.sendMessage({ type: "g4f:cdp:status" });
  const state = res?.state || {};
  const btn = $("#cdp-toggle");
  const status = $("#cdp-status");
  btn.textContent = state.running ? "Disable" : "Enable";
  if (state.running) {
    status.textContent = state.connected
      ? "✓ Connected to server"
      : "Connecting…" + (state.error ? ` (${state.error})` : "");
    status.className = "conn-result " + (state.connected ? "ok" : "");
  } else {
    status.textContent = "Off";
    status.className = "conn-result";
  }
}

async function toggleCdp() {
  const running = (await chrome.runtime.sendMessage({ type: "g4f:cdp:status" }))?.state?.running;
  const res = await chrome.runtime.sendMessage({
    type: running ? "g4f:cdp:stop" : "g4f:cdp:start",
  });
  if (!res?.ok) {
    toast($("#toast-root"), res?.error || "CDP toggle failed", "error");
  }
  await refreshCdpUI();
}

/* ---------------- form ---------------- */

function fillForm() {
  $("#server-url").value = settings.serverUrl;
  $("#api-key").value = settings.apiKey;
  $("#model").value = settings.model;
  $("#provider").value = settings.provider;
  $("#system-prompt").value = settings.systemPrompt;
  $("#temperature").value = settings.temperature ?? "";
  $("#max-tokens").value = settings.maxTokens ?? "";
  $("#stream").checked = !!settings.stream;
  $("#image-model").value = settings.imageModel;
  $("#image-count").value = settings.imageCount;
  $("#page-context-limit").value = settings.pageContextLimit;
  $("#send-on-enter").checked = !!settings.sendOnEnter;
  renderQuickActions();
}

function renderQuickActions() {
  const list = $("#quick-actions-list");
  list.innerHTML = "";
  (settings.quickActions || []).forEach((qa, i) => {
    const row = document.createElement("div");
    row.className = "qa-row";
    row.innerHTML = `
      <input class="qa-label" data-i="${i}" value="${escapeHtml(qa.label)}" placeholder="Label" />
      <textarea class="qa-prompt" data-i="${i}" rows="2" placeholder="Prompt prefix…">${escapeHtml(qa.prompt)}</textarea>
      <button class="btn subtle qa-del" data-i="${i}" title="Remove">✕</button>
    `;
    list.appendChild(row);
  });
  list.querySelectorAll(".qa-del").forEach((btn) =>
    btn.addEventListener("click", (e) => {
      settings.quickActions.splice(Number(e.target.dataset.i), 1);
      renderQuickActions();
    })
  );
}

function collectForm() {
  return {
    serverUrl: $("#server-url").value.trim(),
    apiKey: $("#api-key").value.trim(),
    model: $("#model").value,
    provider: $("#provider").value,
    systemPrompt: $("#system-prompt").value,
    temperature: $("#temperature").value === "" ? null : Number($("#temperature").value),
    maxTokens: $("#max-tokens").value === "" ? null : Number($("#max-tokens").value),
    stream: $("#stream").checked,
    imageModel: $("#image-model").value,
    imageCount: Math.max(1, Math.min(4, Number($("#image-count").value) || 1)),
    pageContextLimit: Number($("#page-context-limit").value) || 6000,
    sendOnEnter: $("#send-on-enter").checked,
    quickActions: settings.quickActions,
  };
}

function wireEvents() {
  $("#save").addEventListener("click", async () => {
    // Read quick action edits back into settings
    document.querySelectorAll(".qa-label").forEach((el) => {
      settings.quickActions[Number(el.dataset.i)].label = el.value;
    });
    document.querySelectorAll(".qa-prompt").forEach((el) => {
      settings.quickActions[Number(el.dataset.i)].prompt = el.value;
    });
    settings = await saveSettings(collectForm());
    toast($("#toast-root"), "Settings saved", "ok");
  });

  $("#reset").addEventListener("click", async () => {
    settings = await resetSettings();
    fillForm();
    toast($("#toast-root"), "Reset to defaults");
  });

  $("#add-qa").addEventListener("click", () => {
    settings.quickActions.push({ id: "qa_" + Date.now(), label: "New action", prompt: "" });
    renderQuickActions();
  });

  $("#test-conn").addEventListener("click", async () => {
    const result = $("#conn-result");
    result.textContent = "Testing…";
    result.className = "conn-result";
    // Save URL/key first so the background uses fresh values.
    settings = await saveSettings({
      serverUrl: $("#server-url").value.trim(),
      apiKey: $("#api-key").value.trim(),
    });
    const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.HEALTH });
    if (res?.ok) {
      result.textContent = "✓ Connected";
      result.classList.add("ok");
      loadModelOptions();
    } else {
      result.textContent = "✗ " + (res?.error || `unreachable (HTTP ${res?.status})`);
      result.classList.add("bad");
    }
  });

  // CDP bridge toggle
  $("#cdp-toggle").addEventListener("click", toggleCdp);
}

async function loadModelOptions() {
  try {
    const res = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.MODELS });
    if (!res?.ok) throw new Error(res?.error);
    const modelSel = $("#model");
    const imgSel = $("#image-model");
    const provSel = $("#provider");
    const keepModel = modelSel.value, keepImg = imgSel.value, keepProv = provSel.value;

    modelSel.innerHTML = '<option value="">Server default</option>';
    imgSel.innerHTML = '<option value="">Server default</option>';
    provSel.innerHTML = '<option value="">Auto</option>';

    for (const m of res.data) {
      const opt = document.createElement("option");
      opt.value = m.id;          // full id sent to the server
      opt.textContent = (m.label || m.id) + (m.image ? " 🖼" : "") + (m.vision ? " 👁" : "");
      if (m.provider) continue;
      modelSel.appendChild(opt.cloneNode(true));
      if (m.image) imgSel.appendChild(opt);
    }

    try {
      const pres = await chrome.runtime.sendMessage({ type: MESSAGE_TYPES.PROVIDERS });
      if (pres?.ok) {
        for (const p of pres.data) {
          const opt = document.createElement("option");
          opt.value = p.id;
          opt.textContent = p.label;
          provSel.appendChild(opt);
        }
      }
    } catch { /* providers optional */ }

    modelSel.value = keepModel;
    imgSel.value = keepImg;
    provSel.value = keepProv;
  } catch {
    /* server offline — leave defaults */
  }
}
