/**
 * g4f extension — shared UI helpers (toast, copy buttons, spinner).
 */

export function toast(container, message, type = "info") {
  if (!container) return;
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  container.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 300);
  }, 2600);
}

export function copyToClipboard(text) {
  return navigator.clipboard.writeText(text);
}

/** Attach a delegated copy handler for `.copy-btn` elements carrying `data-copy`. */
export function bindCopyHandlers(root, onCopied) {
  root.addEventListener("click", async (e) => {
    const btn = e.target.closest(".copy-btn");
    if (!btn) return;
    const text = btn.getAttribute("data-copy") || "";
    try {
      await copyToClipboard(text);
      const original = btn.textContent;
      btn.textContent = "✓";
      setTimeout(() => (btn.textContent = original), 1200);
      onCopied?.();
    } catch {
      onCopied?.(new Error("Copy failed"));
    }
  });
}

export function spinner() {
  const span = document.createElement("span");
  span.className = "spinner";
  return span;
}

export function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
