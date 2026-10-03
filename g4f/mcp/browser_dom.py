"""JavaScript snippets shared by the /browser route and the MCP browser tools."""

from __future__ import annotations

import json

_INTERACTIVE = (
    "button,textarea,a,input,select,"
    "[contenteditable=\"true\"],[contenteditable=\"\"]"
)

# Evaluates to a standalone, script-free HTML copy with data-index on interactive elements.
SNAPSHOT_JS = r"""
(() => {
  const doc = document.cloneNode(true);
  doc.querySelectorAll('script, noscript, iframe, object, embed').forEach(el => el.remove());
  doc.querySelectorAll('*').forEach(el => {
    [...el.attributes].forEach(attr => {
      if (/^on/i.test(attr.name)) el.removeAttribute(attr.name);
    });
  });
  let index = 0;
  // Live form state is not part of the cloned markup.
  const liveFields = document.querySelectorAll('textarea,input,select');
  doc.querySelectorAll('textarea,input,select').forEach((el, i) => {
    const live = liveFields[i];
    if (!live) return;
    if (el.tagName === 'TEXTAREA') el.textContent = live.value;
    else if (el.tagName === 'SELECT') [...el.options].forEach((o, j) => o.toggleAttribute('selected', live.options[j]?.selected));
    else if (live.type === 'checkbox' || live.type === 'radio') el.toggleAttribute('checked', live.checked);
    else el.setAttribute('value', live.value);
  });
  doc.querySelectorAll(%s).forEach(el => el.setAttribute('data-index', index++));
  const pageUrl = new URL(location.href);
  pageUrl.hash = '';
  if (!doc.head.querySelector('link[rel="canonical"]')) {
    const canonical = doc.createElement('link');
    canonical.setAttribute('rel', 'canonical');
    canonical.setAttribute('href', pageUrl.href);
    doc.head.appendChild(canonical);
  }
  if (!doc.head.querySelector('base')) {
    const base = doc.createElement('base');
    base.setAttribute('href', new URL('./', location.href).href);
    doc.head.insertBefore(base, doc.head.firstChild);
  }
  return '<!DOCTYPE html>\n' + doc.documentElement.outerHTML;
})()
""" % json.dumps(_INTERACTIVE)

_FIND = (
    "const els = [...document.querySelectorAll(%s)];"
    "const el = els[%%d];"
    "if (!el) return {ok: false, error: 'No element with data-index %%d (found ' + els.length + ')'};"
) % json.dumps(_INTERACTIVE)


def click_js(index: int) -> str:
    return "(() => {" + (_FIND % (index, index)) + """
      el.scrollIntoView({block: 'center'});
      el.click();
      return {ok: true, tag: el.tagName.toLowerCase()};
    })()"""


def type_js(index: int, text: str, clear: bool = True, submit: bool = False) -> str:
    return "(() => {" + (_FIND % (index, index)) + """
      const text = %s, clear = %s, submit = %s;
      el.scrollIntoView({block: 'center'});
      el.focus();
      if (clear) {
        document.execCommand('selectAll', false, null);
      } else if (el.isContentEditable) {
        const range = document.createRange();
        range.selectNodeContents(el);
        range.collapse(false);
        const sel = getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      } else {
        el.setSelectionRange(el.value.length, el.value.length);
      }
      const current = () => el.isContentEditable ? el.innerText : el.value;
      if (!document.execCommand('insertText', false, text)) {
        if (el.isContentEditable) {
          // ProseMirror (chatgpt.com) handles text via beforeinput.
          el.dispatchEvent(new InputEvent('beforeinput', {bubbles: true, cancelable: true, data: text, inputType: 'insertText'}));
        }
        if (!current().includes(text)) {
          const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
          const setter = el.isContentEditable ? null : Object.getOwnPropertyDescriptor(proto, 'value')?.set;
          if (setter) setter.call(el, clear ? text : el.value + text);
          else el.textContent = clear ? text : el.textContent + text;
          el.dispatchEvent(new InputEvent('input', {bubbles: true, data: text, inputType: 'insertText'}));
        }
      }
      if (submit) {
        for (const type of ['keydown', 'keypress', 'keyup'])
          el.dispatchEvent(new KeyboardEvent(type, {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));
        if (el.form) el.form.requestSubmit ? el.form.requestSubmit() : el.form.submit();
      }
      return {ok: true, tag: el.tagName.toLowerCase()};
    })()""" % (json.dumps(text), json.dumps(clear), json.dumps(submit))


def select_js(index: int, value: str) -> str:
    return "(() => {" + (_FIND % (index, index)) + """
      if (!(el instanceof HTMLSelectElement)) return {ok: false, error: 'Element is not a <select>'};
      const wanted = %s;
      const opt = [...el.options].find(o => o.value === wanted) ||
                  [...el.options].find(o => o.text.trim() === wanted);
      if (!opt) return {ok: false, error: 'Option not found', options: [...el.options].map(o => o.value)};
      el.value = opt.value;
      el.dispatchEvent(new Event('input', {bubbles: true}));
      el.dispatchEvent(new Event('change', {bubbles: true}));
      return {ok: true, value: el.value};
    })()""" % json.dumps(value)
