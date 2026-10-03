"""JavaScript snippets shared by the /browser route and the MCP browser tools."""

from __future__ import annotations

import json

_INTERACTIVE = (
    "button,textarea,a,input,select,"
    "[contenteditable=\"true\"],[contenteditable=\"\"],"
    # SVG icons are common click targets (onclick on the <svg> or a wrapper).
    # SVGElement has no .click(), so click_js falls back to dispatchEvent.
    "svg,div:has(svg),span:has(svg)"
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
  // Keep hidden textareas (height: 0, overflow: hidden) visible in the copy.
  doc.querySelectorAll('textarea').forEach(el => {
    el.style.height = '';
    el.style.overflow = '';
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
      // SVGElement has no .click() (HTMLElement-only) — dispatch a bubbling event instead.
      if (typeof el.click === 'function') el.click();
      else el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, view: window}));
      return {ok: true, tag: el.tagName.toLowerCase()};
    })()"""


def type_js(index: int, text: str, clear: bool = True, submit: bool = False) -> str:
    return "(() => {" + (_FIND % (index, index)) + """
      const text = %s, clear = %s, submit = %s;
      // Resolve the real editable: chatgpt.com renders a wrapper <div> around
      // its contenteditable composer, so a picked container must be unwrapped.
      let target = el;
      if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement) && !target.isContentEditable) {
        const inner = target.querySelector('textarea, input, [contenteditable], [role="textbox"]');
        if (inner) target = inner;
      }
      target.scrollIntoView({block: 'center'});
      target.focus();
      const exec = (...args) => typeof document.execCommand === 'function' && document.execCommand(...args);
      if (clear) {
        exec('selectAll', false, null);
      } else if (target.isContentEditable) {
        const range = document.createRange();
        range.selectNodeContents(target);
        range.collapse(false);
        const sel = getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      } else if (target.setSelectionRange) {
        target.setSelectionRange(target.value.length, target.value.length);
      }
      const current = () => target.isContentEditable ? (target.innerText || '') : (target.value ?? '');
      if (!exec('insertText', false, text)) {
        if (target.isContentEditable) {
          // ProseMirror (chatgpt.com) handles text via beforeinput.
          target.dispatchEvent(new InputEvent('beforeinput', {bubbles: true, cancelable: true, data: text, inputType: 'insertText'}));
        }
        if (!current().includes(text)) {
          // The native value setter is only legal on real input fields;
          // calling it on other elements throws "Illegal invocation".
          const isField = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;
          const setter = isField ? Object.getOwnPropertyDescriptor(Object.getPrototypeOf(target), 'value')?.set : null;
          if (setter) setter.call(target, clear ? text : target.value + text);
          else target.textContent = clear ? text : target.textContent + text;
          target.dispatchEvent(new InputEvent('input', {bubbles: true, data: text, inputType: 'insertText'}));
        }
      }
      if (submit) {
        for (const type of ['keydown', 'keypress', 'keyup'])
          target.dispatchEvent(new KeyboardEvent(type, {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));
        if (target.form) target.form.requestSubmit ? target.form.requestSubmit() : target.form.submit();
      }
      return {ok: true, tag: target.tagName.toLowerCase()};
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

def debug_js(source: str) -> str:
    """Inject a debug script (e.g. the g4f.dev debug panel) into the live target."""
    return "(() => {" + """
      if (window.g4fDebug) return {ok: true, already: true};
      const s = document.createElement('script');
      s.textContent = %s;
      (document.head || document.documentElement).appendChild(s);
      return {ok: true, injected: !!window.g4fDebug};
    })()""" % json.dumps(source)


# ---------------------------------------------------------------------------
# Selector-based actions (PA provider studio)
# ---------------------------------------------------------------------------
# These mirror click_js / type_js / select_js but address the element by a
# CSS selector instead of a data-index, so recorded recipes keep working
# when the page renders a different number of elements.

_FIND_BY_SELECTOR = (
    "const el = document.querySelector(%s);"
    "if (!el) return {ok: false, error: 'No element matches selector ' + %s};"
)


def _selector_js(selector: str, body: str) -> str:
    """Wrap *body* in an IIFE that resolves ``el`` from a CSS selector."""
    return "(() => {" + (_FIND_BY_SELECTOR % (json.dumps(selector), json.dumps(selector))) + body + "})()"


def selector_click_js(selector: str) -> str:
    """Click the element matching *selector*."""
    return _selector_js(selector, """
      el.scrollIntoView({block: 'center'});
      if (typeof el.click === 'function') el.click();
      else el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, view: window}));
      return {ok: true, tag: el.tagName.toLowerCase()};
    """)


def selector_type_js(selector: str, text: str, clear: bool = True, submit: bool = False) -> str:
    """Type *text* into the element matching *selector* (same semantics as type_js)."""
    return _selector_js(selector, """
      const text = %s, clear = %s, submit = %s;
      // Resolve the real editable: chatgpt.com renders a wrapper <div> around
      // its contenteditable composer, so a picked container must be unwrapped.
      let target = el;
      if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement) && !target.isContentEditable) {
        const inner = target.querySelector('textarea, input, [contenteditable], [role="textbox"]');
        if (inner) target = inner;
      }
      target.scrollIntoView({block: 'center'});
      target.focus();
      const exec = (...args) => typeof document.execCommand === 'function' && document.execCommand(...args);
      if (clear) {
        exec('selectAll', false, null);
      } else if (target.isContentEditable) {
        const range = document.createRange();
        range.selectNodeContents(target);
        range.collapse(false);
        const sel = getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      } else if (target.setSelectionRange) {
        target.setSelectionRange(target.value.length, target.value.length);
      }
      const current = () => target.isContentEditable ? (target.innerText || '') : (target.value ?? '');
      if (!exec('insertText', false, text)) {
        if (target.isContentEditable) {
          target.dispatchEvent(new InputEvent('beforeinput', {bubbles: true, cancelable: true, data: text, inputType: 'insertText'}));
        }
        if (!current().includes(text)) {
          // The native value setter is only legal on real input fields;
          // calling it on other elements throws "Illegal invocation".
          const isField = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;
          const setter = isField ? Object.getOwnPropertyDescriptor(Object.getPrototypeOf(target), 'value')?.set : null;
          if (setter) setter.call(target, clear ? text : target.value + text);
          else target.textContent = clear ? text : target.textContent + text;
          target.dispatchEvent(new InputEvent('input', {bubbles: true, data: text, inputType: 'insertText'}));
        }
      }
      if (submit) {
        for (const type of ['keydown', 'keypress', 'keyup'])
          target.dispatchEvent(new KeyboardEvent(type, {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));
        if (target.form) target.form.requestSubmit ? target.form.requestSubmit() : target.form.submit();
      }
      return {ok: true, tag: target.tagName.toLowerCase()};
    """ % (json.dumps(text), json.dumps(clear), json.dumps(submit)))


def selector_select_js(selector: str, value: str) -> str:
    """Choose an option in the <select> matching *selector* (same semantics as select_js)."""
    return _selector_js(selector, """
      if (!(el instanceof HTMLSelectElement)) return {ok: false, error: 'Element is not a <select>'};
      const wanted = %s;
      const opt = [...el.options].find(o => o.value === wanted) ||
                  [...el.options].find(o => o.text.trim() === wanted);
      if (!opt) return {ok: false, error: 'Option not found', options: [...el.options].map(o => o.value)};
      el.value = opt.value;
      el.dispatchEvent(new Event('input', {bubbles: true}));
      el.dispatchEvent(new Event('change', {bubbles: true}));
      return {ok: true, value: el.value};
    """ % json.dumps(value))


def selector_scrape_js(selector: str, attribute: str = "") -> str:
    """Read an attribute, text or (default) full Markdown from the first
    element matching *selector*. Without an attribute the element's HTML is
    converted to Markdown (headings, links, images, lists, code, tables)."""
    return _selector_js(selector, """
      const attr = %s;
      if (attr) {
        // Parentheses are required: mixing ?? with || in one expression is a
        // JavaScript SyntaxError ("Unexpected token '||'").
        const value = el.getAttribute(attr) ?? ((attr === 'href' && el.href) || (attr === 'src' && el.currentSrc) || '');
        return {ok: true, value: String(value).trim(), tag: el.tagName.toLowerCase()};
      }
      // Minimal HTML → Markdown converter (DOM walker, no dependencies).
      const SKIP = /^(SCRIPT|STYLE|NOSCRIPT|TEMPLATE|IFRAME|SVG|CANVAS)$/;
      const BLOCK = /^(P|DIV|SECTION|ARTICLE|HEADER|FOOTER|MAIN|NAV|ASIDE|FIGURE|FIGCAPTION|DL|DT|DD|FORM|FIELDSET|ADDRESS|DETAILS|SUMMARY)$/;
      const md = (node) => {
        if (node.nodeType === Node.TEXT_NODE) return node.textContent.replace(/\\s+/g, ' ');
        if (node.nodeType !== Node.ELEMENT_NODE) return '';
        const tag = node.tagName;
        if (SKIP.test(tag)) return '';
        const inner = [...node.childNodes].map(md).join('');
        const text = inner.trim();
        switch (tag) {
          case 'BR': return '\\n';
          case 'HR': return '\\n\\n---\\n\\n';
          case 'H1': case 'H2': case 'H3': case 'H4': case 'H5': case 'H6':
            return '\\n\\n' + '#'.repeat(+tag[1]) + ' ' + text + '\\n\\n';
          case 'STRONG': case 'B': return text ? `**${text}**` : '';
          case 'EM': case 'I': return text ? `*${text}*` : '';
          case 'DEL': case 'S': return text ? `~~${text}~~` : '';
          case 'CODE': return node.closest('pre') || !text ? inner : '`' + text + '`';
          case 'PRE': return '\\n\\n```\\n' + node.textContent.replace(/\\n+$/, '') + '\\n```\\n\\n';
          case 'A': return text ? `[${text}](${node.href || node.getAttribute('href') || ''})` : '';
          case 'IMG': return `![${node.getAttribute('alt') || ''}](${node.src || node.getAttribute('src') || ''})`;
          case 'UL': case 'OL': {
            const items = [...node.children].filter((c) => c.tagName === 'LI').map((li, i) => {
              const body = [...li.childNodes].map(md).join('').trim().replace(/\\n{2,}/g, '\\n');
              return (tag === 'OL' ? (i + 1) + '. ' : '- ') + body;
            });
            return items.length ? '\\n\\n' + items.join('\\n') + '\\n\\n' : '';
          }
          case 'BLOCKQUOTE': return text ? '\\n\\n' + text.split('\\n').map((l) => '> ' + l).join('\\n') + '\\n\\n' : '';
          case 'TABLE': {
            const rows = [...node.querySelectorAll('tr')].map((tr) =>
              [...tr.children].map((c) => (c.textContent || '').trim().replace(/\\|/g, '\\\\|').replace(/\\n/g, ' ')).join(' | '));
            if (!rows.length) return '';
            const rule = rows[0].split('|').map(() => '---').join(' | ');
            return '\\n\\n' + [rows[0], rule, ...rows.slice(1)].join('\\n') + '\\n\\n';
          }
          case 'INPUT': case 'TEXTAREA': case 'SELECT':
            return node.value ? '\\n\\n' + node.value + '\\n\\n' : '';
          default: return BLOCK.test(tag) ? '\\n\\n' + text + '\\n\\n' : inner;
        }
      };
      const markdown = md(el).replace(/[ \\t]+\\n/g, '\\n').replace(/\\n{3,}/g, '\\n\\n').trim();
      return {ok: true, value: markdown, tag: el.tagName.toLowerCase()};
    """ % json.dumps(attribute or ""))


def selector_exists_js(selector: str) -> str:
    """Check whether an element matches *selector* (used for wait steps)."""
    return "(() => ({ok: !!document.querySelector(%s)}))()" % json.dumps(selector)
