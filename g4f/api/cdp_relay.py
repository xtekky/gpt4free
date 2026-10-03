"""
g4f — CDP relay: expose browser-extension agents as a CDP endpoint.

The g4f API server can use browsers provided by the g4f browser extension
(projects/browser-extension). The extension connects OUT to this relay over
a WebSocket and executes CDP commands locally via chrome.debugger on
dedicated automation tabs. This module presents the classic CDP HTTP surface
(/json, /json/new, /json/close) plus per-target WebSocket pass-through, so
g4f's own CDPSession (g4f/requests/cdp.py) can connect to it unchanged.

Wire protocol between relay and extension agent (JSON):
    -> {type:"cdp", id, tabId, method, params}   execute CDP command
    <- {type:"cdp", id, result?, error?}         command result
    -> {type:"new_tab", url?}  <- {type:"tab", tabId, url, title}
    -> {type:"close_tab", tabId} <- {type:"closed", tabId}
    -> {type:"tabs"}           <- {type:"tabs", tabs:[{tabId,title,url}]}

Enable from the extension's options page ("Browser as CDP provider") or by
setting G4F_BROWSER_MODE=extension before starting the server.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import secrets
import time
from typing import Dict, List, Optional
from urllib.parse import quote, urlparse

from fastapi import Body, Depends, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, Response

try:  # FastAPI / Starlette
    from starlette.websockets import WebSocketState
except ImportError:  # pragma: no cover
    WebSocketState = None  # type: ignore

from ..config import AppConfig

debug = logging.getLogger("g4f.cdp_relay")


def _debug_js_source() -> str:
    """Source of the g4f.dev debug panel, injected into copies on Escape."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "g4f.dev" / "dist" / "js" / "debug.js"
    try:
        return path.read_text()
    except OSError:
        return "console.warn('debug.js not found at ' + " + json.dumps(str(path)) + ");"


# Forwards clicks/inputs/selects in the served HTML copy to the live target.
_COPY_SCRIPT = r"""
let busy = false, timer;
const send = async (body, reload) => {
  if (busy) return;
  busy = true;
  clearTimeout(timer);
  try {
    await fetch(location.origin + '/browser/' + encodeURIComponent(T) + '/action', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  } catch {}
  busy = false;
  if (reload) location.reload();
};
const idx = el => { const t = el.closest('[data-index]'); return t ? [t, +t.dataset.index] : []; };
const plain = el => el.matches('input[type=checkbox],input[type=radio],input[type=file]');
document.addEventListener('click', e => {
  const [el, i] = idx(e.target);
  if (el === undefined || el.isContentEditable || el.matches('select,textarea,input:not([type=checkbox]):not([type=radio]):not([type=submit]):not([type=button])')) return;
  e.preventDefault();
  send({type: 'click', index: i}, true);
}, true);
document.addEventListener('submit', e => e.preventDefault(), true);
document.addEventListener('change', e => {
  const [el, i] = idx(e.target);
  if (el && el.matches('select')) send({type: 'select', index: i, value: el.value}, true);
}, true);
document.addEventListener('input', e => {
  const [el, i] = idx(e.target);
  if (busy || !el || el.matches('select') || plain(el)) return;
  clearTimeout(timer);
  timer = setTimeout(() => send({type: 'type', index: i,
    value: el.isContentEditable ? el.textContent : el.value}, true), 600);
}, true);
document.addEventListener('keydown', e => {
  const [el, i] = idx(e.target);
  if (e.key !== 'Enter' || !el || !el.matches('input')) return;
  e.preventDefault();
  send({type: 'type', index: i, value: el.value, submit: true}, true);
}, true);
// Escape sends a debug action: the live target injects the debug panel.
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape' || window.__g4fDebugSent) return;
  window.__g4fDebugSent = true;
  send({type: 'debug'}, false);
}, true);
"""

# PA provider studio: record click / type / select / scrape actions on the
# served HTML copy, reorder them, test them against the live target and save
# the generated .pa.py provider file into the g4f workspace.
# Injected into /browser/{target_id}/html BEFORE _COPY_SCRIPT so its capture
# listeners run first and can suppress forwarding while picking elements.
_STUDIO_SCRIPT = r"""
(() => {
  if (window.__paStudio) return;
  window.__paStudio = true;

  const $ = (sel, root) => (root || document).querySelector(sel);
  const actions = [];  // {type, selector, value, submit, attribute, wait}
  let pickMode = null; // null | 'click' | 'type' | 'select' | 'scrape'
  let dragIndex = null; // index of the step currently being dragged
  let providerName = 'StudioProvider';
  let providerUrl = (document.querySelector('link[rel="canonical"]') || {}).href || location.origin;
  let editorOpen = false;  // in-panel code editor for the generated .pa.py
  let editedCode = null;   // user-edited provider code (overrides generateCode())
  let savedPath = null;    // server path of the last successful save

  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const label = a => ({click: 'Click', type: 'Type', select: 'Select', scrape: 'Scrape', wait: 'Wait'}[a.type] || a.type);

  // --- best unique CSS selector for an element -----------------------------
  const cssPath = el => {
    if (!(el instanceof Element)) return '';
    const parts = [];
    for (let node = el; node && node.nodeType === 1; node = node.parentElement) {
      if (node.id && document.querySelectorAll('#' + CSS.escape(node.id)).length === 1) {
        parts.unshift('#' + CSS.escape(node.id));
        break;
      }
      const name = node.getAttribute('name');
      if (name && document.querySelectorAll(`${node.tagName}[name="${CSS.escape(name)}"]`).length === 1) {
        parts.unshift(`${node.tagName.toLowerCase()}[name="${CSS.escape(name)}"]`);
        break;
      }
      const testAttr = ['data-testid', 'data-test', 'data-qa'].find(a =>
        node.getAttribute(a) && document.querySelectorAll(`[${a}="${CSS.escape(node.getAttribute(a))}"]`).length === 1);
      if (testAttr) {
        parts.unshift(`[${testAttr}="${node.getAttribute(testAttr)}"]`);
        break;
      }
      let nth = 1, sib = node;
      while ((sib = sib.previousElementSibling)) nth++;
      parts.unshift(`${node.tagName.toLowerCase()}:nth-child(${nth})`);
      if (parts.length >= 6) break;
      if (parts.length >= 2 && document.querySelectorAll(parts.join(' > ')).length === 1) break;
    }
    return parts.join(' > ');
  };

  // --- outline overlay ------------------------------------------------------
  let outline;
  const showOutline = el => {
    if (!outline) { outline = document.createElement('div'); document.body.appendChild(outline); }
    const r = el.getBoundingClientRect();
    Object.assign(outline.style, {
      position: 'fixed', left: r.left + 'px', top: r.top + 'px',
      width: r.width + 'px', height: r.height + 'px',
      border: '2px solid #00e676', background: 'rgba(0,230,118,.12)',
      pointerEvents: 'none', zIndex: 2147483646, borderRadius: '3px'
    });
  };
  const hideOutline = () => { if (outline) { outline.remove(); outline = null; } };

  // --- state persistence (the copy reloads after forwarded actions) ---------
  const storeKey = 'paStudio:' + T;
  const persist = () => {
    try { sessionStorage.setItem(storeKey, JSON.stringify({actions, name: providerName, url: providerUrl, code: editedCode, path: savedPath})); } catch {}
  };
  try {
    const saved = JSON.parse(sessionStorage.getItem(storeKey) || 'null');
    if (saved) {
      (saved.actions || []).forEach(a => actions.push(a));
      if (saved.name) providerName = saved.name;
      if (saved.url) providerUrl = saved.url;
      if (saved.code) editedCode = saved.code;
      if (saved.path) savedPath = saved.path;
    }
  } catch {}

  // --- studio panel ---------------------------------------------------------
  const panel = document.createElement('div');
  panel.id = 'pa-studio';
  document.body.appendChild(panel);

  // --- drag the panel by its header (position survives copy reloads) --------
  let panelPos = null;
  try { panelPos = JSON.parse(sessionStorage.getItem(storeKey + ':pos') || 'null'); } catch {}
  const applyPos = () => {
    if (!panelPos) return;
    panel.style.left = panelPos.x + 'px';
    panel.style.top = panelPos.y + 'px';
    panel.style.right = 'auto';
    panel.style.bottom = 'auto';
  };
  applyPos();
  panel.addEventListener('pointerdown', e => {
    if (pickMode || !e.target.closest || !e.target.closest('#pa-studio h3')) return;
    const r = panel.getBoundingClientRect();
    const offX = e.clientX - r.left, offY = e.clientY - r.top;
    panelPos = {x: r.left, y: r.top};
    applyPos();
    const move = ev => {
      panelPos = {
        x: Math.max(0, Math.min(ev.clientX - offX, innerWidth - 40)),
        y: Math.max(0, Math.min(ev.clientY - offY, innerHeight - 30)),
      };
      applyPos();
    };
    const up = () => {
      document.removeEventListener('pointermove', move, true);
      document.removeEventListener('pointerup', up, true);
      try { sessionStorage.setItem(storeKey + ':pos', JSON.stringify(panelPos)); } catch {}
    };
    document.addEventListener('pointermove', move, true);
    document.addEventListener('pointerup', up, true);
    e.preventDefault();
  });

  const render = () => {
    panel.innerHTML = `
      <style>
        #pa-studio{position:fixed;right:12px;bottom:12px;width:360px;max-height:70vh;overflow:auto;
          background:#111827;color:#e5e7eb;font:12px/1.5 monospace;border:1px solid #374151;
          border-radius:8px;padding:10px;z-index:2147483647;box-shadow:0 8px 24px rgba(0,0,0,.5)}
        #pa-studio h3{margin:0 0 8px;font-size:13px;color:#00e676;cursor:move;user-select:none;-webkit-user-select:none;touch-action:none}
        #pa-studio h3:active{cursor:grabbing}
        #pa-studio button{cursor:pointer;background:#1f2937;color:#e5e7eb;border:1px solid #374151;
          border-radius:4px;padding:3px 8px;font:11px monospace;margin:1px}
        #pa-studio button:hover{background:#374151}
        #pa-studio button.primary{background:#065f46;border-color:#10b981}
        #pa-studio button.danger{color:#f87171}
        #pa-studio .step{display:flex;align-items:center;gap:4px;padding:3px 0;border-bottom:1px solid #1f2937;cursor:grab}
        #pa-studio .step:active{cursor:grabbing}
        #pa-studio .step.dragging{opacity:.35}
        #pa-studio .step.drop-target{box-shadow:inset 0 2px 0 #00e676}
        #pa-studio .step code{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#93c5fd}
        #pa-studio input{background:#0b1220;color:#e5e7eb;border:1px solid #374151;border-radius:4px;padding:2px 4px;font:11px monospace;width:100%;margin:2px 0}
        #pa-studio .row{display:flex;gap:4px;margin:4px 0;flex-wrap:wrap}
        #pa-studio .hint{color:#9ca3af;font-size:10px;margin:4px 0}
        #pa-studio .status{color:#fbbf24;min-height:14px;font-size:10px;word-break:break-all}
        #pa-studio a{color:#93c5fd}
        #pa-studio .path{color:#9ca3af;font-size:10px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        #pa-studio textarea{width:100%;height:240px;background:#0b1220;color:#d1fae5;border:1px solid #374151;
          border-radius:4px;padding:4px;font:10px/1.4 monospace;white-space:pre;overflow:auto;resize:vertical;box-sizing:border-box}
      </style>
      <h3>PA Provider Studio</h3>
      <div class="row">
        <button data-add="click" class="primary">+ Click</button>
        <button data-add="type">+ Input</button>
        <button data-add="select">+ Select</button>
        <button data-add="scrape">+ Scrape</button>
        <button data-add="wait">+ Wait</button>
      </div>
      <div class="hint">${pickMode ? 'Now click the element on the page … (Esc cancels)' : 'Pick an action type, then click the element on the page. Drag steps to reorder.'}</div>
      <div id="pa-steps"></div>
      <div class="row">
        <button id="pa-test" class="primary">▶ Test all</button>
        <button id="pa-save">💾 Save .pa.py</button>
        <button id="pa-edit">📝 ${editedCode ? 'Edit code •' : 'Edit code'}</button>
        <button id="pa-clear" class="danger">Clear</button>
      </div>
      <div class="row"><input id="pa-name" placeholder="provider name (e.g. MyChat)" value="${esc(providerName)}"></div>
      <div class="row"><input id="pa-url" placeholder="provider url" value="${esc(providerUrl)}"></div>
      <div class="status" id="pa-status"></div>
      ${savedPath ? `<div class="row"><a id="pa-open" href="vscode://file/${encodeURI(savedPath).replace(/#/g, '%23').replace(/\?/g, '%3F')}" title="Open in VS Code">📂 Open in editor</a><code class="path" title="${esc(savedPath)}">${esc(savedPath)}</code></div>` : ''}
      ${editorOpen ? `
      <div class="editor">
        <textarea id="pa-code" spellcheck="false">${esc(editedCode ?? generateCode())}</textarea>
        <div class="row">
          <button id="pa-code-save" class="primary">💾 Save</button>
          <button id="pa-code-reset" class="danger">↺ Reset to recorded</button>
          <button id="pa-code-close">✕ Close</button>
        </div>
        <div class="hint">Edit the generated Python code — saving writes this version. Use {prompt} in type steps as a placeholder for the user prompt.</div>
      </div>` : ''}`;

    const steps = $('#pa-steps');
    if (!actions.length) steps.innerHTML = '<div class="hint">No actions recorded yet.</div>';
    actions.forEach((a, i) => {
      const row = document.createElement('div');
      row.className = 'step';
      const detail = a.type === 'wait' ? `${a.wait || 1}s`
        : a.type === 'type' ? `${esc(a.selector)} = ${esc((a.value || '').slice(0, 24))}`
        : a.type === 'select' ? `${esc(a.selector)} → ${esc(a.value || '')}`
        : a.type === 'scrape' ? `${esc(a.selector)}${a.attribute ? '[' + esc(a.attribute) + ']' : ' (text)'}${a.lastValue ? ' → ' + esc(a.lastValue.slice(0, 30)) : ''}`
        : esc(a.selector);
      row.innerHTML = `<code>${i + 1}. ${label(a)} ${detail}</code>
        <button data-up="${i}" title="move up">↑</button>
        <button data-down="${i}" title="move down">↓</button>
        <button data-del="${i}" class="danger" title="delete">✕</button>`;
      steps.appendChild(row);
      // drag & drop reordering
      row.draggable = true;
      row.ondragstart = e => {
        dragIndex = i;
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', String(i));
        requestAnimationFrame(() => row.classList.add('dragging'));
      };
      row.ondragend = () => {
        dragIndex = null;
        row.classList.remove('dragging');
        panel.querySelectorAll('.step').forEach(s => s.classList.remove('drop-target'));
      };
      row.ondragover = e => {
        if (dragIndex === null) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        if (dragIndex !== i) row.classList.add('drop-target');
      };
      row.ondragleave = () => row.classList.remove('drop-target');
      row.ondrop = e => {
        e.preventDefault();
        e.stopPropagation();
        row.classList.remove('drop-target');
        const from = dragIndex;
        dragIndex = null;
        if (from === null || from === i) return;
        const [moved] = actions.splice(from, 1);
        actions.splice(i, 0, moved);
        persist(); render();
      };
      if (a.type === 'type' || a.type === 'select' || a.type === 'scrape' || a.type === 'wait') {
        const inp = document.createElement('input');
        inp.placeholder = a.type === 'type' ? 'text to type ({prompt} = user message)'
          : a.type === 'select' ? 'option value'
          : a.type === 'wait' ? 'seconds'
          : 'attribute (empty = text)';
        inp.value = a.type === 'scrape' ? (a.attribute || '') : a.type === 'wait' ? (a.wait || 1) : (a.value || '');
        inp.onchange = () => {
          if (a.type === 'scrape') a.attribute = inp.value.trim();
          else if (a.type === 'wait') a.wait = parseFloat(inp.value) || 1;
          else a.value = inp.value;
          persist(); render();
        };
        steps.appendChild(inp);
      }
      if (a.type === 'type') {
        const sub = document.createElement('label');
        sub.style.cssText = 'font-size:10px;color:#9ca3af';
        sub.innerHTML = `<input type=checkbox ${a.submit ? 'checked' : ''}> press Enter after typing`;
        sub.querySelector('input').onchange = e => { a.submit = e.target.checked; persist(); };
        steps.appendChild(sub);
      }
    });

    // dropping on empty space below the steps moves the dragged step to the end
    steps.ondragover = e => { if (dragIndex !== null) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; } };
    steps.ondrop = e => {
      if (dragIndex === null) return;
      e.preventDefault();
      const [moved] = actions.splice(dragIndex, 1);
      dragIndex = null;
      actions.push(moved);
      persist(); render();
    };
    panel.querySelectorAll('[data-add]').forEach(b => b.onclick = () => {
      const mode = b.dataset.add;
      if (mode === 'wait') { actions.push({type: 'wait', wait: 1}); persist(); render(); return; }
      pickMode = mode; render();
    });
    panel.querySelectorAll('[data-up]').forEach(b => b.onclick = () => { const i = +b.dataset.up; if (i > 0) [actions[i - 1], actions[i]] = [actions[i], actions[i - 1]]; persist(); render(); });
    panel.querySelectorAll('[data-down]').forEach(b => b.onclick = () => { const i = +b.dataset.down; if (i < actions.length - 1) [actions[i + 1], actions[i]] = [actions[i], actions[i + 1]]; persist(); render(); });
    panel.querySelectorAll('[data-del]').forEach(b => b.onclick = () => { actions.splice(+b.dataset.del, 1); persist(); render(); });
    $('#pa-clear').onclick = () => { actions.length = 0; persist(); render(); };
    $('#pa-test').onclick = testAll;
    $('#pa-save').onclick = saveProvider;
    $('#pa-edit').onclick = () => { editorOpen = !editorOpen; render(); };
    if (editorOpen) {
      $('#pa-code-save').onclick = saveProvider;
      $('#pa-code-reset').onclick = () => { editedCode = null; persist(); render(); };
      $('#pa-code-close').onclick = () => { editorOpen = false; render(); };
      $('#pa-code').oninput = e => { editedCode = e.target.value; persist(); };
    }
    $('#pa-name').onchange = e => { providerName = e.target.value; persist(); };
    $('#pa-url').onchange = e => { providerUrl = e.target.value; persist(); };
  };

  // --- element picking ------------------------------------------------------
  document.addEventListener('mousemove', e => {
    if (!pickMode) return;
    const el = document.elementFromPoint(e.clientX, e.clientY);
    if (!el || el.closest('#pa-studio')) { hideOutline(); return; }
    showOutline(el);
  }, true);
  document.addEventListener('click', e => {
    if (!pickMode) return;
    // Let clicks inside the studio panel work normally.
    if (e.target.closest && e.target.closest('#pa-studio')) return;
    e.preventDefault(); e.stopImmediatePropagation();
    const target = e.target;
    const mode = pickMode;
    pickMode = null; hideOutline();
    const selector = cssPath(target);
    if (!selector) { render(); return; }
    if (mode === 'click') actions.push({type: 'click', selector});
    else if (mode === 'type') actions.push({type: 'type', selector, value: ''});
    else if (mode === 'select') actions.push({type: 'select', selector, value: ''});
    else if (mode === 'scrape') actions.push({type: 'scrape', selector, attribute: ''});
    persist(); render();
  }, true);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && pickMode) {
      e.stopImmediatePropagation();
      pickMode = null; hideOutline(); render();
    }
  }, true);

  // --- run actions against the live target ---------------------------------
  const post = body => fetch(location.origin + '/browser/' + encodeURIComponent(T) + '/action', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  }).then(r => r.json());

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  const runAction = async a => {
    if (a.type === 'wait') { await sleep((parseFloat(a.wait) || 1) * 1000); return {ok: true}; }
    const body = {selector: a.selector};
    if (a.type === 'click') return post({...body, type: 'click'});
    if (a.type === 'type') return post({...body, type: 'type', value: a.value || '', submit: !!a.submit});
    if (a.type === 'select') return post({...body, type: 'select', value: a.value || ''});
    if (a.type === 'scrape') return post({...body, type: 'scrape', attribute: a.attribute || ''});
    return {ok: false, error: 'unknown action'};
  };

  const status = (msg, cls) => { const el = $('#pa-status'); if (el) { el.textContent = msg; el.style.color = cls || '#fbbf24'; } };

  const testAll = async () => {
    if (!actions.length) return status('Nothing to test.');
    status('Running …');
    for (let i = 0; i < actions.length; i++) {
      try {
        const res = await runAction(actions[i]);
        const inner = res && res.result ? res.result : res;
        if (!inner || inner.ok === false || (res && res.error))
          return status(`Step ${i + 1} failed: ${(inner && inner.error) || (res && res.error) || 'error'}`, '#f87171');
        if (actions[i].type === 'scrape' && inner && inner.value !== undefined)
          actions[i].lastValue = String(inner.value).slice(0, 200);
        await sleep(400);
      } catch (err) { return status(`Step ${i + 1} error: ${err}`, '#f87171'); }
    }
    status('All steps OK ✓', '#00e676');
    persist(); render();
  };

  // --- generate + save the .pa.py provider ----------------------------------
  const pyStr = s => JSON.stringify(String(s ?? ''));
  const pyLit = v => {
    if (v === undefined || v === null) return 'None';
    if (typeof v === 'boolean') return v ? 'True' : 'False';
    if (typeof v === 'number') return String(v);
    if (typeof v === 'string') return JSON.stringify(v);
    if (Array.isArray(v)) return '[' + v.map(pyLit).join(', ') + ']';
    return '{' + Object.entries(v).map(([k, val]) => `${JSON.stringify(k)}: ${pyLit(val)}`).join(', ') + '}';
  };

  const generateCode = () => {
    const name = (providerName).replace(/[^A-Za-z0-9_]/g, '') || 'StudioProvider';
    const startUrl = (document.querySelector('link[rel="canonical"]') || {}).href || location.href;
    const recorded = actions.map(({lastValue, ...rest}) => rest);
    return [
      'from __future__ import annotations',
      '',
      'import asyncio',
      '',
      'from g4f.typing import AsyncResult, Messages',
      'from g4f.requests.cdp import CDPSession',
      'from g4f.mcp.browser_dom import (',
      '    selector_click_js,',
      '    selector_exists_js,',
      '    selector_scrape_js,',
      '    selector_select_js,',
      '    selector_type_js,',
      ')',
      'from g4f.Provider.base_provider import AsyncGeneratorProvider, ProviderModelMixin',
      'from g4f.Provider.helper import format_prompt',
      '',
      '',
      `class ${name}(AsyncGeneratorProvider, ProviderModelMixin):`,
      "    'Recorded with the g4f PA Provider Studio (/browser/<target>/html).'",
      '',
      `    label = ${pyStr(name)}`,
      `    url = ${pyStr(providerUrl)}`,
      '    working = True',
      '    supports_stream = True',
      '',
      '    default_model = "auto"',
      '    models = [default_model]',
      '',
      `    _start_url = ${pyStr(startUrl)}`,
      `    _actions = ${pyLit(recorded)}`,
      '',
      '    @classmethod',
      '    async def create_async_generator(cls, model, messages, **kwargs):',
      '        prompt = format_prompt(messages)',
      '        session = CDPSession()',
      '        await session.start()',
      '        try:',
      '            await session.navigate(cls._start_url)',
      '            for action in cls._actions:',
      '                kind = action["type"]',
      '                if kind == "wait":',
      '                    await asyncio.sleep(float(action.get("wait", 1)))',
      '                    continue',
      '                selector = action.get("selector", "")',
      '                await cls._wait(session, selector)',
      '                if kind == "click":',
      '                    await session.evaluate_js(selector_click_js(selector))',
      '                elif kind == "type":',
      '                    value = str(action.get("value", "")).replace("{prompt}", prompt)',
      '                    await session.evaluate_js(selector_type_js(selector, value, bool(action.get("submit"))))',
      '                elif kind == "select":',
      '                    await session.evaluate_js(selector_select_js(selector, str(action.get("value", ""))))',
      '                elif kind == "scrape":',
      '                    result = await session.evaluate_js(selector_scrape_js(selector, str(action.get("attribute", ""))))',
      '                    value = (result or {}).get("value", "")',
      '                    if value:',
      '                        yield value',
      '                        return',
      '            await asyncio.sleep(2)',
      '            text = await session.evaluate_js("document.body ? document.body.innerText : \'\'")',
      '            yield (text or "").strip() or "No response"',
      '        finally:',
      '            await session.close()',
      '',
      '    @staticmethod',
      '    async def _wait(session, selector, timeout=10):',
      '        for _ in range(timeout * 2):',
      '            result = await session.evaluate_js(selector_exists_js(selector))',
      '            if result and result.get("ok"):',
      '                return',
      '            await asyncio.sleep(0.5)',
      '',
    ].join('\n');
  };

  const saveProvider = async () => {
    status('Saving …');
    try {
      const res = await fetch(location.origin + '/browser/' + encodeURIComponent(T) + '/save_provider', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: providerName, code: editedCode ?? generateCode()})
      });
      const data = await res.json();
      if (data.ok) {
        savedPath = data.path;
        persist();
        status(`Saved → ${data.path}`, '#00e676');
      } else {
        status(`Save failed: ${data.detail || data.error}`, '#f87171');
      }
    } catch (err) { status('Save failed: ' + err, '#f87171'); }
    render();
  };

  render();
})();
"""
AGENT_PATH = "/v1/cdp/agent"  # extension agents connect here


class AgentConnection:
    """One connected extension agent (one browser)."""

    def __init__(self, ws: WebSocket, client_id: str):
        self.ws = ws
        self.client_id = client_id
        self.pending: Dict[int, asyncio.Future] = {}
        self._id = 0
        self.tabs: List[dict] = []
        self.last_seen: float = time.time()

    async def send(self, obj: dict) -> None:
        await self.ws.send_text(json.dumps(obj))

    async def roundtrip(self, obj: dict, timeout: float = 30.0) -> dict:
        """Send a request expecting a correlated response."""
        self._id += 1
        msg_id = self._id
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending[msg_id] = fut
        try:
            await self.send({**obj, "id": msg_id})
            return await asyncio.wait_for(fut, timeout)
        finally:
            self.pending.pop(msg_id, None)

    def resolve(self, msg: dict) -> None:
        """Resolve a pending roundtrip by response id."""
        fut = self.pending.pop(msg.get("id"), None)
        if fut and not fut.done():
            if "error" in msg and msg["error"] is not None:
                fut.set_exception(RuntimeError(str(msg["error"])))
            else:
                fut.set_result(msg)


class CdpRelay:
    """
    Registry of connected agents + the CDP-over-HTTP facade.

    Presents the subset of the Chrome DevTools HTTP API that
    g4f.requests.cdp.CDPSession uses:
        GET  /json            -> list targets
        PUT  /json/new        -> create target (returns webSocketDebuggerUrl)
        GET  /json/close/{id} -> close target
    plus WebSocket pass-through at /v1/cdp/ws/{target_id}.
    """

    def __init__(self):
        self.agents: Dict[str, AgentConnection] = {}
        # target_id -> (client_id, tab_id)
        self.targets: Dict[str, tuple] = {}
        self._lock = asyncio.Lock()
        # target_id -> {title,url,html?} for tabs seen open; closed ones move to `closed`.
        self.known: Dict[str, dict] = {}
        self.closed: Dict[str, dict] = {}
        # Zombie reaper: MV3 service workers die after ~30s idle, which can
        # leave sockets that no longer respond. Track liveness via pings.
        self._reaper_task: Optional[asyncio.Task] = None

    def _ensure_reaper(self) -> None:
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self._reaper())

    async def _reaper(self) -> None:
        """Drop agents that stop answering pings (dead service workers)."""
        while self.agents:
            await asyncio.sleep(10)
            for agent in list(self.agents.values()):
                if time.time() - agent.last_seen > 45:
                    debug.warning(
                        f"CDP relay: reaping zombie agent '{agent.client_id}'"
                    )
                    try:
                        await agent.ws.close()
                    except Exception:
                        pass
                    await self.unregister(agent)

    # ---------------- agent management ----------------

    async def register(self, ws: WebSocket, client_id: str) -> AgentConnection:
        agent = AgentConnection(ws, client_id)
        async with self._lock:
            # One agent per extension id: newest wins.
            old = self.agents.get(client_id)
            self.agents[client_id] = agent
        self._ensure_reaper()
        if old and old is not agent:
            try:
                await old.ws.close()
            except Exception:
                pass
        # An agent is connected: route CDPSession traffic through the relay
        # instead of launching a local Chrome instance.
        self._set_extension_mode(True)
        debug.info(f"CDP relay: agent '{client_id}' connected")
        return agent

    async def unregister(self, agent: AgentConnection) -> None:
        async with self._lock:
            if self.agents.get(agent.client_id) is agent:
                del self.agents[agent.client_id]
            dead = [
                tid for tid, (cid, _) in self.targets.items()
                if cid == agent.client_id
            ]
            for tid in dead:
                del self.targets[tid]
            for tid in [t for t in self.known if t.rpartition(":")[0] == agent.client_id]:
                self._mark_closed(tid)
        if not self.agents:
            # Last agent gone: fall back to local browser handling.
            self._set_extension_mode(False)
        debug.info(f"CDP relay: agent '{agent.client_id}' disconnected")

    @staticmethod
    def _set_extension_mode(enabled: bool) -> None:
        """Enable/disable extension routing in g4f's CDP core."""
        try:
            from ..cookies import BrowserConfig

            current = getattr(BrowserConfig, "browser_mode", None)
            if enabled and current != "extension":
                BrowserConfig.browser_mode = "extension"
                debug.info("CDP relay: extension mode enabled (agent registered)")
            elif not enabled and current == "extension":
                BrowserConfig.browser_mode = None
                debug.info("CDP relay: extension mode disabled (no agents)")
        except Exception as e:
            debug.warning(f"CDP relay: failed to set browser_mode: {e}")

    def get_agent(self, client_id: Optional[str] = None) -> AgentConnection:
        if client_id and client_id in self.agents:
            return self.agents[client_id]
        if self.agents:
            return next(iter(self.agents.values()))
        raise RuntimeError(
            "No browser extension agent connected. "
            "Enable 'Browser as CDP provider' in the g4f extension."
        )

    # ---------------- CDP facade (used by routes below) ----------------

    def _mark_closed(self, target_id: str) -> None:
        info = self.known.pop(target_id, None)
        if info is None:
            return
        self.closed[target_id] = {**info, "closed_at": time.time()}
        while len(self.closed) > 50:
            del self.closed[next(iter(self.closed))]

    async def evaluate(self, target_id: str, expression: str, timeout: float = 30.0):
        """Run JS in a target tab via the agent and return the JSON value."""
        client_id, tab_id = self.split_target(target_id)
        agent = self.get_agent(client_id)
        res = await agent.roundtrip(
            {"type": "cdp", "tabId": tab_id, "method": "Runtime.evaluate",
             "params": {"expression": expression, "returnByValue": True,
                        "awaitPromise": True}},
            timeout=timeout,
        )
        out = res.get("result", {})
        if out.get("exceptionDetails"):
            raise RuntimeError(str(out["exceptionDetails"].get("text", "evaluation failed")))
        return out.get("result", {}).get("value")

    async def snapshot(self, target_id: str) -> Optional[str]:
        """Standalone HTML copy of an open tab (cached); cached copy for closed tabs."""
        from ..mcp.browser_dom import SNAPSHOT_JS

        try:
            html = await self.evaluate(target_id, SNAPSHOT_JS)
        except Exception as e:
            debug.warning(f"CDP relay: snapshot failed for {target_id}: {e}")
            html = None
        if html:
            info = self.known.get(target_id)
            if info is not None:
                info["html"] = html
            return html
        return (self.known.get(target_id) or self.closed.get(target_id) or {}).get("html")

    async def list_targets(self) -> List[dict]:
        targets = []
        seen = set()
        for client_id, agent in list(self.agents.items()):
            try:
                res = await agent.roundtrip({"type": "tabs"}, timeout=5)
                agent.tabs = res.get("tabs", [])
            except Exception as e:
                debug.warning(f"CDP relay: tabs query failed for {client_id}: {e}")
                agent.tabs = []
            else:
                # Only a successful query proves a tab is gone.
                prefix = f"{client_id}:"
                live = {f"{client_id}:{t['tabId']}" for t in agent.tabs}
                for tid in [k for k in self.known if k.startswith(prefix) and k not in live]:
                    self._mark_closed(tid)
            for tab in agent.tabs:
                tid = f"{client_id}:{tab['tabId']}"
                seen.add(tid)
                info = self.known.setdefault(tid, {})
                info.update(title=tab.get("title", ""), url=tab.get("url", ""))
                targets.append(
                    {
                        "id": tid,
                        "type": "page",
                        "title": tab.get("title", ""),
                        "url": tab.get("url", ""),
                        "attached": False,
                    }
                )
        return targets

    async def new_target(self, url: str = "about:blank") -> dict:
        agent = self.get_agent()
        res = await agent.roundtrip({"type": "new_tab", "url": url}, timeout=15)
        tab_id = res.get("tabId")
        target_id = f"{agent.client_id}:{tab_id}"
        async with self._lock:
            self.targets[target_id] = (agent.client_id, tab_id)
        return {
            "id": target_id,
            "type": "page",
            "title": res.get("title", ""),
            "url": res.get("url", url),
            "webSocketDebuggerUrl": f"ws://__relay__/{target_id}",
        }

    async def close_target(self, target_id: str) -> bool:
        client_id, tab_id = self.targets.get(target_id, (None, None))
        if client_id is None:
            try:
                client_id, tab_id = self.split_target(target_id)
            except RuntimeError:
                return False
        agent = self.agents.get(client_id)
        if not agent:
            return False
        if target_id in self.known:
            await self.snapshot(target_id)
        try:
            await agent.roundtrip({"type": "close_tab", "tabId": tab_id}, timeout=5)
        except Exception:
            pass
        async with self._lock:
            self.targets.pop(target_id, None)
            self._mark_closed(target_id)
        return True

    def split_target(self, target_id: str):
        """target_id 'client:tab' -> (client_id, tab_id:int)."""
        client_id, _, tab = target_id.rpartition(":")
        if not client_id or not tab.isdigit():
            raise RuntimeError(f"Invalid target id: {target_id}")
        return client_id, int(tab)


# Single relay instance for the app.
relay = CdpRelay()


async def agent_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint: extension agents connect here."""
    client_id = ws.query_params.get("client", "extension")
    await ws.accept()
    agent = await relay.register(ws, client_id)
    try:
        while True:
            raw = await ws.receive_text()
            agent.last_seen = time.time()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            mtype = msg.get("type")
            if mtype == "ping":
                # Agent keepalive — reply so it knows the link is healthy.
                agent.last_seen = time.time()
                try:
                    await agent.send({"type": "pong", "id": msg.get("id")})
                except Exception:
                    pass
                continue
            if mtype == "cdp" and "id" in msg:
                agent.resolve(msg)
            elif mtype in ("tab", "tabs", "closed", "pong", "attached", "detached"):
                # responses to new_tab / tabs / close_tab / ping roundtrips
                agent.resolve(msg)
            elif mtype == "error":
                agent.resolve(msg)
            # "attached"/"detached"/"pong" are informational for now
    except WebSocketDisconnect:
        pass
    except Exception as e:  # pragma: no cover
        debug.warning(f"CDP relay: agent error: {e}")
    finally:
        await relay.unregister(agent)


async def cdp_ws_endpoint(ws: WebSocket, target_id: str) -> None:
    """
    WebSocket endpoint: CDPSession connects here per target.
    Passes CDP commands through to the extension agent.
    """
    await ws.accept()
    try:
        client_id, tab_id = relay.split_target(target_id)
        agent = relay.get_agent(client_id)
    except RuntimeError as e:
        await ws.close(code=4404, reason=str(e))
        return

    async def reader():
        """g4f -> extension"""
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    cmd = json.loads(raw)
                except ValueError:
                    continue
                req_id = cmd.get("id")
                try:
                    result = await agent.roundtrip(
                        {"type": "cdp", "tabId": tab_id,
                         "method": cmd.get("method"),
                         "params": cmd.get("params", {})},
                        timeout=60,
                    )
                    await ws.send_text(json.dumps(
                        {"id": req_id, "result": result.get("result", {})}
                    ))
                except Exception as e:
                    await ws.send_text(json.dumps(
                        {"id": req_id, "error": str(e)}
                    ))
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    async def writer():
        """extension events -> g4f (event fan-out is not needed by CDPSession,
        which polls; keep the loop alive and ignore)."""
        try:
            while True:
                await asyncio.sleep(1)
        except Exception:
            pass

    try:
        await asyncio.gather(reader(), writer())
    except Exception:
        pass


def register_cdp_relay(app) -> None:
    """Attach relay routes to the FastAPI app."""

    def _require_debug() -> None:
        # Checked per request so a later enable_logging() takes effect.
        from .. import debug as g4f_debug

        if not g4f_debug.logging:
            raise HTTPException(status_code=404, detail="Not Found")

    @app.websocket(AGENT_PATH)
    async def _agent(ws: WebSocket) -> None:
        await agent_endpoint(ws)

    @app.get("/json")
    async def _json_list() -> List[dict]:
        return await relay.list_targets()

    @app.get("/json/list")
    async def _json_list2() -> List[dict]:
        return await relay.list_targets()

    @app.put("/json/new")
    async def _json_new(url: str = "about:blank") -> dict:
        return await relay.new_target(url)

    @app.get("/json/new")
    async def _json_new_get(url: str = "about:blank") -> dict:
        return await relay.new_target(url)

    @app.get("/json/close/{target_id}")
    async def _json_close(target_id: str) -> dict:
        ok = await relay.close_target(target_id)
        return {"ok": ok}

    @app.websocket("/v1/cdp/ws/{target_id}")
    async def _cdp(ws: WebSocket, target_id: str) -> None:
        await cdp_ws_endpoint(ws, target_id)

    @app.get("/browser", response_class=HTMLResponse, dependencies=[Depends(_require_debug)])
    async def _browser_list() -> HTMLResponse:
        await relay.list_targets()
        from ..requests.cdp import list_session_targets

        local = await list_session_targets()
        opened = {t["id"]: t for t in local["open"]}
        closed = {t["id"]: t for t in local["closed"]}
        # Relay entries win: extension-mode CDPSessions appear in both.
        opened.update(relay.known)
        closed.update(relay.closed)
        for tid in opened:
            closed.pop(tid, None)

        def rows(items: dict, label: str) -> str:
            out = []
            for tid, info in reversed(list(items.items())):
                link = f"/browser/{quote(tid, safe='')}/html"
                has_copy = label == "open" or info.get("html")
                cell = f'<a href="{link}">html copy</a>' if has_copy else "-"
                if label == "open":
                    cell += (
                        f' <form method=post action="/browser/{quote(tid, safe="")}/debug" '
                        'style=display:inline><button title="inject / show the debug panel">debug</button></form>'
                        f' <form method=post action="/browser/{quote(tid, safe="")}/close" '
                        'style=display:inline><button>close</button></form>'
                    )
                out.append(
                    f"<tr><td>{label}</td><td>{html.escape(info.get('title') or '')}</td>"
                    f"<td>{html.escape(info.get('url') or '')}</td><td>{cell}</td></tr>"
                )
            return "".join(out)

        body = (
            "<!DOCTYPE html><meta charset=utf-8><title>Browser targets</title>"
            "<style>body{font-family:sans-serif;margin:2em}td,th{padding:4px 12px;"
            "text-align:left}</style><h1>Browser targets</h1>"
            "<form method=post action=/browser/new><input name=url type=url required "
            "placeholder='https://...' size=60> <button>open new target</button></form>"
            "<table><tr><th>State</th><th>Title</th><th>URL</th><th>Copy</th></tr>"
            f"{rows(opened, 'open')}{rows(closed, 'closed')}</table>"
        )
        return HTMLResponse(body)

    def _is_not_demo_and_debug() -> bool:
        _require_debug()
        if AppConfig.demo:
            raise HTTPException(status_code=404, detail="Not Found")

    @app.post("/browser/new", dependencies=[Depends(_is_not_demo_and_debug)])
    async def _browser_new(url: str = Form(...)) -> RedirectResponse:
        from ..mcp.browser_dom import debug_js

        if urlparse(url).scheme not in ("http", "https"):
            raise HTTPException(status_code=400, detail="Only http(s) URLs are supported")
        from ..image import is_safe_url

        # Debug-only endpoint: allow local and network URLs (e.g. localhost dev servers).
        if not await asyncio.get_running_loop().run_in_executor(
            None, is_safe_url, url, True
        ):
            raise HTTPException(status_code=400, detail="Local and network URLs are not allowed")
        if relay.agents:
            target = await relay.new_target(url)
            relay.known.setdefault(target["id"], {}).update(title="", url=url)
            try:
                await relay.evaluate(target["id"], debug_js(_debug_js_source()))
            except Exception as e:
                debug.warning(f"CDP relay: debug.js injection failed for {target['id']}: {e}")
        else:
            from ..requests.cdp import CDPSession

            session = CDPSession()
            await session.start()
            await session.navigate(url)
            try:
                await session.evaluate_js(debug_js(_debug_js_source()))
            except Exception as e:
                debug.warning(f"CDP: debug.js injection failed: {e}")
        return RedirectResponse("/browser", status_code=303)

    @app.post("/browser/{target_id}/close", dependencies=[Depends(_require_debug)])
    async def _browser_close(target_id: str) -> RedirectResponse:
        from ..requests.cdp import _open_sessions

        session = _open_sessions.get(target_id)
        if session is not None and target_id not in relay.known:
            await session.close()
        else:
            await relay.close_target(target_id)
        return RedirectResponse("/browser", status_code=303)

    @app.post("/browser/{target_id}/debug", dependencies=[Depends(_require_debug)])
    async def _browser_debug(target_id: str) -> RedirectResponse:
        """Clickable handle: inject (or re-show) the debug panel on a target."""
        from ..mcp.browser_dom import debug_js

        js = debug_js(_debug_js_source())
        try:
            if target_id in relay.targets:
                await relay.evaluate(target_id, js)
            else:
                from ..requests.cdp import _open_sessions

                session = _open_sessions.get(target_id)
                if session is None:
                    raise HTTPException(status_code=404, detail="Target not found")
                await session.evaluate_js(js)
        except HTTPException:
            raise
        except Exception as e:
            debug.warning(f"CDP relay: debug.js injection failed for {target_id}: {e}")
        return RedirectResponse("/browser", status_code=303)

    @app.get("/browser/debug.js", dependencies=[Depends(_require_debug)])
    async def _browser_debug_js() -> Response:
        return Response(_debug_js_source(), media_type="application/javascript")

    @app.get("/browser/{target_id}/html", response_class=HTMLResponse, dependencies=[Depends(_require_debug)])
    async def _browser_html(target_id: str) -> HTMLResponse:
        if target_id in relay.known or target_id in relay.closed:
            page = await relay.snapshot(target_id)
        else:
            from ..requests.cdp import snapshot_session_target

            page = await snapshot_session_target(target_id)
        if not page:
            raise HTTPException(status_code=404, detail="No HTML copy available")
        nonce = secrets.token_urlsafe(16)
        script = (
            f'<script nonce="{nonce}">const T={json.dumps(target_id)},N={json.dumps(nonce)};'
            f'{_STUDIO_SCRIPT}\n{_COPY_SCRIPT}</script>'
        )
        page = page.replace("</body>", script + "</body>", 1) if "</body>" in page else page + script
        return HTMLResponse(
            page,
            headers={"Content-Security-Policy": "default-src * data: blob: 'unsafe-inline'; "
                     f"script-src 'nonce-{nonce}'; object-src 'none'; connect-src 'self'; "
                     "form-action 'none'", "Cache-Control": "no-cache"},
        )

    @app.post("/browser/{target_id}/action", dependencies=[Depends(_require_debug)])
    async def _browser_action(target_id: str, payload: dict = Body(...)) -> dict:
        from ..mcp.browser_dom import (
            click_js,
            debug_js,
            select_js,
            type_js,
            selector_click_js,
            selector_scrape_js,
            selector_select_js,
            selector_type_js,
        )

        try:
            kind = payload.get("type")
            if kind == "debug":
                js = debug_js(_debug_js_source())
            elif "selector" in payload:
                # Selector-based actions (PA provider studio): address the
                # element by CSS selector instead of data-index.
                selector = str(payload["selector"])
                if kind == "click":
                    js = selector_click_js(selector)
                elif kind == "type":
                    js = selector_type_js(selector, str(payload.get("value", "")), True, bool(payload.get("submit")))
                elif kind == "select":
                    js = selector_select_js(selector, str(payload.get("value", "")))
                elif kind == "scrape":
                    js = selector_scrape_js(selector, str(payload.get("attribute", "")))
                else:
                    raise HTTPException(status_code=400, detail="Unknown action type")
            else:
                index = int(payload["index"])
                if kind == "click":
                    js = click_js(index)
                elif kind == "type":
                    js = type_js(index, str(payload.get("value", "")), True, bool(payload.get("submit")))
                elif kind == "select":
                    js = select_js(index, str(payload.get("value", "")))
                else:
                    raise HTTPException(status_code=400, detail="Unknown action type")
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid action")
        if target_id in relay.known:
            result = await relay.evaluate(target_id, js)
        else:
            from ..requests.cdp import _open_sessions

            session = _open_sessions.get(target_id)
            if session is None or not session.is_alive:
                raise HTTPException(status_code=409, detail="Target is not open")
            result = await session.evaluate_js(js)
        await asyncio.sleep(0.3 if kind in ("type", "debug") and not payload.get("submit") else 1)  # let the page settle before the copy reloads
        return {"result": result}

    @app.post("/browser/{target_id}/save_provider", dependencies=[Depends(_is_not_demo_and_debug)])
    async def _browser_save_provider(target_id: str, payload: dict = Body(...)) -> dict:
        """Save a studio-recorded provider as a ``.pa.py`` file in the workspace."""
        from ..files import secure_filename
        from ..mcp.pa_provider import get_workspace_dir

        code = payload.get("code")
        if not code or not isinstance(code, str):
            raise HTTPException(status_code=400, detail="Missing provider code")
        if len(code) > 256 * 1024:
            raise HTTPException(status_code=413, detail="Provider code too large")
        name = secure_filename(str(payload.get("name") or "StudioProvider"), max_length=60)
        name = "".join(c for c in name if c.isalnum() or c == "_").strip("_") or "StudioProvider"
        target_dir = get_workspace_dir() / "pa-providers"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{name}.pa.py"
        counter = 1
        while path.exists() and path.read_text(encoding="utf-8") != code:
            path = target_dir / f"{name}_{counter}.pa.py"
            counter += 1
            if counter > 100:
                raise HTTPException(status_code=409, detail="Too many provider files with the same name")
        path.write_text(code, encoding="utf-8")
        return {"ok": True, "path": str(path)}
