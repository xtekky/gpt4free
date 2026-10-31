/*
 * g4f controls for the noVNC client served at /desktop/.
 *
 * The stock noVNC UI runs inside the #vnc iframe. bin/novnc-brand.sh exposes
 * that UI on the iframe's window as `g4fUI` together with a small key/text
 * bridge, so this outer page can provide the same touch-friendly pieces as the
 * Remote Desktop viewer (projects/remote-desktop):
 *
 *   Type    a text field that sends a whole string to the VM
 *   Keys    modifier and navigation buttons a phone keyboard cannot produce
 *   Keys    one-tap desktop shortcuts (Alt+Tab, Ctrl+C, Win+R, ...)
 *
 * Everything is layered over the client; normal mouse and keyboard input keeps
 * going straight to noVNC when the user interacts with the framebuffer.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const frame = $("vnc");

  const QUALITY = {
    low: { label: "low", value: 2 },
    medium: { label: "med", value: 6 },
    high: { label: "high", value: 9 },
  };
  const QUALITY_ORDER = ["low", "medium", "high"];
  const MODIFIERS = ["Control", "Alt", "Shift", "Meta"];

  let quality = "medium";
  let toolbarTimer = null;
  let toastTimer = null;

  /* The iframe is same-origin, so the bridge installed by novnc-brand.sh is
     reachable directly. A missing bridge means noVNC is still loading (or the
     build is older); callers just report "not connected" until it appears. */
  function ui() {
    try {
      const win = frame.contentWindow;
      return win && win.g4fUI ? win.g4fUI : null;
    } catch {
      return null;
    }
  }

  function state() {
    const api = ui();
    if (!api || typeof api.g4fState !== "function") return null;
    try {
      return api.g4fState();
    } catch {
      return null;
    }
  }

  function connected() {
    const current = state();
    return Boolean(current && current.connected);
  }

  function flash(text) {
    const toast = $("toast");
    toast.textContent = text;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 1600);
  }

  function qualityName(value) {
    if (!Number.isFinite(value)) return "medium";
    if (value <= 3) return "low";
    if (value <= 6) return "medium";
    return "high";
  }

  function setQuality(level) {
    const api = ui();
    if (!api) {
      flash("desktop not connected");
      return;
    }
    if (!QUALITY[level]) level = "medium";
    quality = level;
    api.g4fSetQuality(QUALITY[level].value);
    $("quality-label").textContent = QUALITY[level].label;
    flash(`quality: ${QUALITY[level].label}`);
    sync();
  }

  function cycleQuality() {
    const index = QUALITY_ORDER.indexOf(quality);
    setQuality(QUALITY_ORDER[(index + 1) % QUALITY_ORDER.length]);
  }

  /* ------------------------------------------------------------------- panels */

  const panels = {
    keyboard: $("keyboard"),
    shortcuts: $("shortcuts"),
  };

  function panelOpen() {
    return Object.values(panels).some((panel) => panel.classList.contains("open"));
  }

  function togglePanel(name, force) {
    const panel = panels[name];
    const open = force !== undefined ? force : !panel.classList.contains("open");
    if (open) {
      for (const [otherName, other] of Object.entries(panels)) {
        if (otherName !== name) {
          other.classList.remove("open");
          $(`btn-${otherName}`).classList.remove("active");
        }
      }
    }
    panel.classList.toggle("open", open);
    $(`btn-${name}`).classList.toggle("active", open);
    $(`btn-${name}`).setAttribute("aria-pressed", String(open));
    if (open) {
      showToolbar();
      if (name === "keyboard") setTimeout(() => $("text-input").focus(), 50);
    }
  }

  /* ---------------------------------------------------------------- toolbar */

  function hideToolbar() {
    $("toolbar").classList.add("hidden");
    $("handle").classList.remove("hidden");
  }

  function scheduleHide() {
    clearTimeout(toolbarTimer);
    toolbarTimer = setTimeout(() => {
      if (panelOpen()) {
        scheduleHide();
        return;
      }
      hideToolbar();
    }, 6000);
  }

  function showToolbar() {
    $("toolbar").classList.remove("hidden");
    $("handle").classList.add("hidden");
    scheduleHide();
  }

  /* ------------------------------------------------------------------ state */

  function sync() {
    const current = state();
    const isConnected = Boolean(current && current.connected);
    const viewOnly = !current || current.viewOnly;

    $("btn-control").classList.toggle("active", isConnected && !viewOnly);
    $("btn-control").setAttribute("aria-pressed", String(isConnected && !viewOnly));
    $("control-label").textContent = isConnected && !viewOnly ? "Control" : "View";

    $("btn-fit").classList.toggle("active", Boolean(current && current.scale));
    $("btn-fit").setAttribute("aria-pressed", String(Boolean(current && current.scale)));

    if (current && typeof current.quality === "number") {
      quality = qualityName(current.quality);
    }
    $("quality-label").textContent = QUALITY[quality].label;

    const modifiers = (current && current.modifiers) || [];
    for (const button of document.querySelectorAll("#keyboard .keys button[data-key]")) {
      const key = button.dataset.key;
      if (MODIFIERS.includes(key)) {
        button.classList.toggle("active", modifiers.includes(key));
      }
    }
  }

  /* ------------------------------------------------------------------- input */

  function guard() {
    const current = state();
    if (!current || !current.connected) {
      flash("desktop not connected");
      return false;
    }
    if (current.viewOnly) {
      flash("take control first");
      return false;
    }
    return true;
  }

  function sendText() {
    const input = $("text-input");
    const text = input.value;
    if (!text) return;
    const api = ui();
    if (!api || !guard()) return;
    if (!api.g4fSendText(text)) {
      flash("could not send text");
      return;
    }
    input.value = "";
    sync();
  }

  function pressKey(button) {
    const api = ui();
    const key = button.dataset.key;
    if (!api) {
      flash("desktop not connected");
      return;
    }
    if (MODIFIERS.includes(key)) {
      if (!guard()) return;
      api.g4fToggleModifier(key);
      sync();
      return;
    }
    if (!guard()) return;
    if (api.g4fSendKey(key)) {
      api.rfb.focus();
      sync();
    }
  }

  /* --------------------------------------------------------------- shortcuts */

  //: One tap shortcuts for the things a phone keyboard cannot do. ``keys`` is
  //: pressed in order and released in reverse, so modifiers wrap the key.
  const SHORTCUTS = [
    { id: "alt-tab", label: "Alt+Tab", keys: ["Alt", "Tab"] },
    { id: "alt-shift-tab", label: "Alt+Shift+Tab", keys: ["Alt", "Shift", "Tab"] },
    { id: "win", label: "Win", keys: ["Meta"] },
    { id: "win-d", label: "Show desktop", keys: ["Meta", "d"] },
    { id: "win-e", label: "Explorer", keys: ["Meta", "e"] },
    { id: "win-l", label: "Lock", keys: ["Meta", "l"] },
    { id: "win-r", label: "Run", keys: ["Meta", "r"] },
    { id: "win-tab", label: "Task view", keys: ["Meta", "Tab"] },
    { id: "ctrl-c", label: "Copy", keys: ["Control", "c"] },
    { id: "ctrl-v", label: "Paste", keys: ["Control", "v"] },
    { id: "ctrl-x", label: "Cut", keys: ["Control", "x"] },
    { id: "ctrl-z", label: "Undo", keys: ["Control", "z"] },
    { id: "ctrl-a", label: "Select all", keys: ["Control", "a"] },
    { id: "ctrl-s", label: "Save", keys: ["Control", "s"] },
    { id: "ctrl-f", label: "Find", keys: ["Control", "f"] },
    { id: "ctrl-w", label: "Close tab", keys: ["Control", "w"] },
    { id: "ctrl-shift-t", label: "Reopen tab", keys: ["Control", "Shift", "t"] },
    { id: "ctrl-shift-esc", label: "Task manager", keys: ["Control", "Shift", "Escape"] },
    { id: "alt-f4", label: "Close window", keys: ["Alt", "F4"] },
    { id: "alt-left", label: "Back", keys: ["Alt", "ArrowLeft"] },
    { id: "alt-right", label: "Forward", keys: ["Alt", "ArrowRight"] },
    { id: "escape", label: "Esc", keys: ["Escape"] },
    { id: "enter", label: "Enter", keys: ["Enter"] },
    { id: "tab", label: "Tab", keys: ["Tab"] },
    { id: "f5", label: "Refresh", keys: ["F5"] },
    { id: "f11", label: "Fullscreen", keys: ["F11"] },
    { id: "printscreen", label: "Screenshot", keys: ["PrintScreen"] },
  ];

  function runShortcut(shortcut) {
    const api = ui();
    if (!api || !guard()) return;
    if (api.g4fSendCombo(shortcut.keys)) {
      // The combo releases its own keys; drop any sticky modifiers too so the
      // next tap starts from a clean keyboard state.
      api.g4fReleaseModifiers();
      api.rfb.focus();
      flash(shortcut.label);
      sync();
    } else {
      flash(`unsupported shortcut: ${shortcut.label}`);
    }
  }

  function buildShortcuts() {
    const box = $("shortcuts-list");
    for (const shortcut of SHORTCUTS) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = shortcut.label;
      button.dataset.shortcut = shortcut.id;
      button.addEventListener("click", () => runShortcut(shortcut));
      box.appendChild(button);
    }
  }

  /* -------------------------------------------------------------------- misc */

  function toggleFullscreen() {
    const root = document.documentElement;
    const fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;
    if (fullscreenElement) {
      const exit = document.exitFullscreen || document.webkitExitFullscreen;
      if (exit) {
        const result = exit.call(document);
        if (result && result.catch) result.catch(() => {});
      }
      return;
    }
    const request = root.requestFullscreen || root.webkitRequestFullscreen;
    if (!request) {
      flash("fullscreen not supported");
      return;
    }
    const result = request.call(root);
    if (result && result.catch) result.catch(() => flash("fullscreen blocked"));
  }

  function toggleControl() {
    const api = ui();
    const current = state();
    if (!api || !current || !current.connected) {
      flash("desktop not connected");
      return;
    }
    api.g4fSetViewOnly(!current.viewOnly);
    flash(current.viewOnly ? "control enabled" : "view only");
    sync();
  }

  function disconnect() {
    const api = ui();
    if (api && api.rfb && api.connected) api.disconnect();
    flash("disconnected");
    sync();
  }

  /* -------------------------------------------------------------------- boot */

  function boot() {
    buildShortcuts();

    $("btn-control").addEventListener("click", toggleControl);
    $("btn-keyboard").addEventListener("click", () => togglePanel("keyboard"));
    $("btn-shortcuts").addEventListener("click", () => togglePanel("shortcuts"));
    $("btn-quality").addEventListener("click", cycleQuality);
    $("btn-fit").addEventListener("click", () => {
      const api = ui();
      if (!api || !api.rfb) {
        flash("desktop not connected");
        return;
      }
      api.g4fToggleFit();
      sync();
    });
    $("btn-fullscreen").addEventListener("click", toggleFullscreen);
    $("btn-disconnect").addEventListener("click", disconnect);

    $("close-keyboard").addEventListener("click", () => togglePanel("keyboard", false));
    $("close-shortcuts").addEventListener("click", () => togglePanel("shortcuts", false));
    $("send-text").addEventListener("click", sendText);

    $("text-input").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        sendText();
      } else if (event.key === "Escape") {
        event.preventDefault();
        togglePanel("keyboard", false);
      }
    });

    for (const button of document.querySelectorAll("#keyboard .keys button[data-key]")) {
      button.addEventListener("click", () => pressKey(button));
    }

    $("toolbar").addEventListener("pointerdown", showToolbar);
    $("toolbar").addEventListener("focusin", showToolbar);
    $("handle").addEventListener("click", showToolbar);
    $("handle").addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      showToolbar();
    });

    frame.addEventListener("load", sync);
    setInterval(sync, 500);
    sync();
    showToolbar();
  }

  boot();
})();
