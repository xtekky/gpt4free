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

  /* Bind an event without taking the whole page down when one control is
     missing (an older landing page, a partial deploy). A single missing node
     used to throw during boot and leave every later handler unattached, which
     made the toolbar's Keys/Type panels look completely dead. */
  function bind(id, event, handler, options) {
    const element = $(id);
    if (!element) {
      console.warn(`g4f controls: missing #${id} for ${event}`);
      return null;
    }
    element.addEventListener(event, handler, options);
    return element;
  }

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

  /* The on-screen keyboard shrinks the *visual* viewport on iOS/Android but
     leaves fixed elements anchored to the layout viewport, so the toolbar and
     the Type/Keys panels ended up behind the keyboard. Publish the overlap as
     a CSS variable and let the stylesheet lift the controls above it. */
  function updateKeyboardInset() {
    const viewport = window.visualViewport;
    const height = viewport ? viewport.height : window.innerHeight;
    const top = viewport ? viewport.offsetTop : 0;
    const inset = viewport
      ? Math.max(0, window.innerHeight - height - top)
      : 0;
    // Only lift the controls for a real keyboard (a focused text field and a
    // large overlap). Mobile URL-bar jitter is smaller and must not move the
    // toolbar around while scrolling.
    const typing = document.activeElement === $("text-input");
    const clamped = inset > 120 && typing ? Math.round(inset) : 0;
    document.documentElement.style.setProperty("--g4f-keyboard-inset", `${clamped}px`);
  }

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

  /* The controls stay usable while noVNC is still connecting or retrying, so
     say which of those is happening instead of always blaming a missing
     desktop. A missing bridge means the client has not finished loading (or an
     older image did not install it). */
  function statusMessage() {
    if (!ui()) return "desktop still loading";
    const current = state();
    if (!current) return "desktop still loading";
    if (current.reconnecting) return "desktop reconnecting";
    if (current.connecting) return "desktop connecting";
    return "desktop not connected";
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
      flash(statusMessage());
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

  function blurTextInput() {
    const input = $("text-input");
    // Do not leave the soft keyboard up when the user closes the Type panel or
    // switches to Keys: on a phone it would cover the shortcuts instead.
    if (input && document.activeElement === input) input.blur();
  }

  function togglePanel(name, force) {
    const panel = panels[name];
    const open = force !== undefined ? force : !panel.classList.contains("open");
    if (open) {
      for (const [otherName, other] of Object.entries(panels)) {
        if (otherName !== name) {
          other.classList.remove("open");
          $(`btn-${otherName}`).classList.toggle("active", false);
          $(`btn-${otherName}`).setAttribute("aria-pressed", "false");
        }
      }
    }
    panel.classList.toggle("open", open);
    $(`btn-${name}`).classList.toggle("active", open);
    $(`btn-${name}`).setAttribute("aria-pressed", String(open));
    if (open) {
      showToolbar();
      if (name === "keyboard") {
        // Focus synchronously: iOS only opens the soft keyboard when the
        // focus() happens inside the tap gesture, not in a later timer.
        const input = $("text-input");
        if (input) input.focus({ preventScroll: true });
      } else {
        blurTextInput();
      }
    } else if (name === "keyboard") {
      blurTextInput();
    }
    updateKeyboardInset();
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

    const resize = (current && current.resize) || "remote";
    const fit = Boolean(current && (current.scale || resize === "scale"));
    $("btn-fit").classList.toggle("active", fit);
    $("btn-fit").setAttribute("aria-pressed", String(fit));

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
      flash(statusMessage());
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
    // Keep the field focused so the soft keyboard stays open for the next
    // chunk on a phone; the bridge sends straight to RFB and does not need
    // the noVNC canvas to take focus.
    sync();
  }

  function pressKey(button) {
    const api = ui();
    const key = button.dataset.key;
    if (!api) {
      flash(statusMessage());
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
      // Do not call rfb.focus() here: it steals focus from the Type input and
      // dismisses the on-screen keyboard in the middle of a typing session.
      sync();
    } else {
      flash(`could not send ${key}`);
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
      // g4fSendCombo releases the combo keys itself; drop any sticky
      // modifiers too so the next tap starts from a clean keyboard state.
      api.g4fReleaseModifiers();
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
      flash(statusMessage());
      return;
    }
    api.g4fSetViewOnly(!current.viewOnly);
    flash(current.viewOnly ? "control enabled" : "view only");
    sync();
  }

  function disconnect() {
    const api = ui();
    let left = false;
    if (api && typeof api.g4fDisconnect === "function") {
      // Cancels a pending reconnect too, so Leave works even when the desktop
      // never connected (the common case when the VNC server is unreachable).
      left = Boolean(api.g4fDisconnect());
    } else if (api && typeof api.disconnect === "function" && api.rfb) {
      // Fallback for a client branded by an older image.
      api.disconnect();
      left = true;
    }
    if (!left) {
      // No usable bridge: the iframe owns the connection, so reload it to
      // stop a stuck connect/reconnect loop instead of leaving a dead page.
      if (api) {
        api.inhibitReconnect = true;
        if (api.reconnectCallback) {
          clearTimeout(api.reconnectCallback);
          api.reconnectCallback = null;
        }
      }
      try {
        frame.contentWindow.location.reload();
      } catch {
        frame.src = frame.src;
      }
    }
    flash("disconnected");
    sync();
  }

  /* ---------------------------------------------------------------- install */

  let installPrompt = null;

  function isInstalled() {
    return (
      window.matchMedia("(display-mode: standalone)").matches ||
      window.matchMedia("(display-mode: fullscreen)").matches ||
      window.navigator.standalone === true
    );
  }

  function isIos() {
    return /iphone|ipad|ipod/i.test(window.navigator.userAgent);
  }

  function showInstallButton(show) {
    const button = $("btn-install");
    if (button) button.classList.toggle("hidden", !show || isInstalled());
  }

  async function installApp() {
    if (installPrompt) {
      const prompt = installPrompt;
      installPrompt = null;
      showInstallButton(false);
      prompt.prompt();
      const choice = await prompt.userChoice.catch(() => null);
      if (!choice || choice.outcome !== "accepted") {
        showInstallButton(true);
        flash("install dismissed");
      }
      return;
    }
    if (isInstalled()) {
      flash("desktop already installed");
    } else if (isIos()) {
      flash("Share > Add to Home Screen");
    } else {
      flash("browser menu > Install app");
    }
  }

  function registerServiceWorker() {
    if (!("serviceWorker" in navigator) || !window.isSecureContext) return;
    const script = new URL("sw.js", document.baseURI);
    const scope = new URL("./", document.baseURI).pathname;
    navigator.serviceWorker.register(script, { scope }).catch(() => {});
  }

  /* -------------------------------------------------------------------- boot */

  function boot() {
    buildShortcuts();

    bind("btn-control", "click", toggleControl);
    bind("btn-keyboard", "click", () => togglePanel("keyboard"));
    bind("btn-shortcuts", "click", () => togglePanel("shortcuts"));
    bind("btn-quality", "click", cycleQuality);
    bind("btn-fit", "click", () => {
      const api = ui();
      if (!api || !api.rfb) {
        flash(statusMessage());
        return;
      }
      const mode = api.g4fToggleFit();
      flash(mode === "scale" ? "fit: local scale" : "fit: remote resize");
      sync();
    });
    bind("btn-fullscreen", "click", toggleFullscreen);
    bind("btn-install", "click", installApp);
    bind("btn-disconnect", "click", disconnect);

    if (!isInstalled() && isIos()) showInstallButton(true);
    window.addEventListener("beforeinstallprompt", (event) => {
      event.preventDefault();
      installPrompt = event;
      showInstallButton(true);
      showToolbar();
    });
    window.addEventListener("appinstalled", () => {
      installPrompt = null;
      showInstallButton(false);
      flash("desktop installed");
    });
    registerServiceWorker();

    bind("close-keyboard", "click", () => togglePanel("keyboard", false));
    bind("close-shortcuts", "click", () => togglePanel("shortcuts", false));
    bind("send-text", "click", sendText);

    bind("text-input", "keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        sendText();
      } else if (event.key === "Escape") {
        event.preventDefault();
        togglePanel("keyboard", false);
      }
    });
    bind("text-input", "focus", updateKeyboardInset);
    bind("text-input", "blur", updateKeyboardInset);

    for (const button of document.querySelectorAll("#keyboard .keys button[data-key]")) {
      button.addEventListener("click", () => pressKey(button));
    }

    bind("toolbar", "pointerdown", showToolbar);
    bind("toolbar", "focusin", showToolbar);
    bind("handle", "click", showToolbar);
    bind("handle", "keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      showToolbar();
    });

    // Escape closes whichever overlay panel is open, matching the desktop
    // viewer. The text field handles its own Escape above.
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      if (event.target === $("text-input")) return;
      if ($("keyboard").classList.contains("open")) togglePanel("keyboard", false);
      else if ($("shortcuts").classList.contains("open")) togglePanel("shortcuts", false);
    });

    // Keep the fixed controls above the soft keyboard. visualViewport is the
    // only reliable signal on iOS, where window.innerHeight does not shrink.
    updateKeyboardInset();
    window.addEventListener("resize", updateKeyboardInset);
    window.addEventListener("orientationchange", updateKeyboardInset);
    if (window.visualViewport) {
      window.visualViewport.addEventListener("resize", updateKeyboardInset);
      window.visualViewport.addEventListener("scroll", updateKeyboardInset);
    }

    frame.addEventListener("load", () => {
      updateKeyboardInset();
      sync();
    });
    setInterval(sync, 500);
    sync();
    showToolbar();
  }

  boot();
})();
