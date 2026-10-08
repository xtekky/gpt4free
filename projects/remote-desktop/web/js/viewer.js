/* Viewer page: play the host's WebRTC stream and forward touch input. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);

  const state = {
    ws: null,
    room: (params.get("room") || "").toUpperCase(),
    token: params.get("token") || "",
    pc: null,
    control: null,
    self: null,
    host: null,
    hasControl: false,
    retry: 500,
    closed: false,
    frames: 0,
    lastFps: performance.now(),
    gesture: null,
    longPressTimer: null,
    moved: false,
    drag: false,
    lastTap: 0,
    quality: params.get("q") || localStorage.getItem("rd-quality") || "auto",
    hidden: document.hidden,
    lastBytes: 0,
    lastBytesAt: 0,
    kbps: 0,
    iceServers: [],
    iceLogged: false,
  };

  //: Caps per quality level. The viewer only receives, so it asks the host to
  //: re-encode at this level; the width/height here cap the displayed picture.
  const QUALITY = {
    low: { label: "low", maxWidth: 854, maxHeight: 480, maxBitrate: 250_000, maxFramerate: 12 },
    medium: { label: "med", maxWidth: 1280, maxHeight: 720, maxBitrate: 900_000, maxFramerate: 20 },
    high: { label: "high", maxWidth: 1920, maxHeight: 1080, maxBitrate: 2_500_000, maxFramerate: 30 },
    auto: { label: "auto", maxWidth: 1280, maxHeight: 720, maxBitrate: 1_200_000, maxFramerate: 24 },
  };

  const QUALITY_ORDER = ["low", "medium", "high", "auto"];

  const log = () => {};

  const setDot = (id, cls) => {
    $(id).className = `dot ${cls}`;
  };

  const send = (message) => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(message));
      return true;
    }
    return false;
  };

  /* ---------------------------------------------------------------- websocket */

  function connect() {
    if (state.closed) return;
    setDot("ws-dot", "warn");
    $("ws-text").textContent = "connecting";

    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${scheme}//${location.host}/ws`);
    state.ws = ws;

    ws.onopen = async () => {
      state.retry = 500;
      setDot("ws-dot", "on");
      $("ws-text").textContent = "connected";
      // Refresh first: the credentials are time limited and a reconnect after
      // an hour would otherwise offer an expired relay login.
      await loadIceServers();
      send({ type: "hello", role: "viewer", room: state.room, token: state.token });
    };

    ws.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      handle(message);
    };

    ws.onclose = () => {
      setDot("ws-dot", "off");
      $("ws-text").textContent = "reconnecting";
      closePeer();
      if (state.closed) return;
      setTimeout(connect, state.retry);
      state.retry = Math.min(state.retry * 2, 8000);
    };
  }

  function handle(message) {
    switch (message.type) {
      case "joined":
        state.self = message.peer;
        state.host = message.host;
        state.control = message.control;
        hideOverlay();
        updateControl();
        send({ type: "signal", to: state.host, data: { kind: "request" } });
        break;
      case "signal":
        onSignal(message.from, message.data);
        break;
      case "control":
        state.control = message.owner;
        updateControl();
        if (message.denied) flash("Another viewer is in control");
        break;
      case "peer-left":
        if (message.role === "host") {
          showOverlay("The host stopped sharing. Waiting for it to come back...");
          closePeer();
        }
        break;
      case "error":
        showOverlay(message.message);
        break;
      default:
        break;
    }
  }

  /* ------------------------------------------------------------------ webrtc */

  let signalChain = Promise.resolve();
  const pendingIce = [];

  function newPeer() {
    const pc = new RTCPeerConnection({ iceServers: state.iceServers, iceCandidatePoolSize: 2 });
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        send({ type: "signal", to: state.host, data: { kind: "ice", candidate: event.candidate } });
      }
    };
    pc.onicecandidateerror = (event) => {
      console.warn(`ICE candidate error ${event.errorCode} ${event.errorText || ""}`);
    };
    pc.ontrack = (event) => {
      const video = $("screen");
      if (video.srcObject !== event.streams[0]) {
        video.srcObject = event.streams[0];
        video.play().catch(() => {});
        startFpsMeter();
        applyQuality();
      }
    };
    pc.onconnectionstatechange = () => {
      if (["failed", "disconnected"].includes(pc.connectionState)) {
        showOverlay("Connection lost. Reconnecting...");
        closePeer();
        setTimeout(() => send({ type: "signal", to: state.host, data: { kind: "request" } }), 800);
      }
    };
    state.pc = pc;
    return pc;
  }

  function onSignal(from, data) {
    signalChain = signalChain.then(() => processSignal(from, data)).catch(() => {});
    return signalChain;
  }

  async function processSignal(from, data) {
    if (!data || typeof data !== "object") return;
    if (data.kind === "offer") {
      await applyOffer(from, data.sdp);
    } else if (data.kind === "ice" && data.candidate) {
      await applyIce(data.candidate);
    }
  }

  async function applyOffer(from, sdp) {
    let pc = state.pc;
    if (!pc || pc.connectionState === "closed") pc = newPeer();

    // The host offers both when the viewer joins and when it asks for a stream,
    // so the same offer can arrive twice. Answering the second one would call
    // setLocalDescription while the connection is already "stable".
    if (pc.signalingState !== "stable" || pc.currentRemoteDescription) return;

    try {
      await pc.setRemoteDescription(sdp);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      send({ type: "signal", to: from, data: { kind: "answer", sdp: pc.localDescription } });
      await flushIce();
    } catch (error) {
      showOverlay(`Handshake failed: ${error.message}`);
    }
  }

  async function applyIce(candidate) {
    const pc = state.pc;
    if (!pc || pc.connectionState === "closed" || !pc.remoteDescription) {
      pendingIce.push(candidate);
      return;
    }
    try {
      await pc.addIceCandidate(candidate);
    } catch {
      /* stale candidate from a previous negotiation */
    }
  }

  async function flushIce() {
    const pc = state.pc;
    if (!pc || !pc.remoteDescription) return;
    while (pendingIce.length) {
      try {
        await pc.addIceCandidate(pendingIce.shift());
      } catch {
        /* ignore unusable candidates */
      }
    }
  }

  function closePeer() {
    pendingIce.length = 0;
    if (!state.pc) return;
    try {
      state.pc.close();
    } catch {
      /* already closed */
    }
    state.pc = null;
    $("screen").srcObject = null;
  }

  /* --------------------------------------------------------------- bandwidth */

  function qualitySettings() {
    return QUALITY[state.quality] || QUALITY.auto;
  }

  /* The viewer only receives, so it cannot cap the bitrate itself: it asks the
     host to re-encode this stream at the requested level. The CSS cap keeps the
     picture from being upscaled on a small screen. */
  function applyQuality() {
    const settings = qualitySettings();
    const video = $("screen");
    video.style.maxWidth = `${settings.maxWidth}px`;
    video.style.maxHeight = `${settings.maxHeight}px`;
    video.style.margin = "0 auto";
    if (state.host) {
      send({ type: "signal", to: state.host, data: { kind: "quality", level: state.quality } });
    }
  }

  function setQuality(level, { announce = true } = {}) {
    if (!QUALITY[level]) level = "auto";
    state.quality = level;
    try {
      localStorage.setItem("rd-quality", level);
    } catch {
      /* private mode */
    }
    $("quality-label").textContent = qualitySettings().label;
    $("btn-quality").classList.toggle("active", level !== "auto");
    applyQuality();
    if (announce) flash(`quality: ${qualitySettings().label}`);
  }

  function cycleQuality() {
    const index = QUALITY_ORDER.indexOf(state.quality);
    setQuality(QUALITY_ORDER[(index + 1) % QUALITY_ORDER.length]);
  }

  /* The host keeps encoding even when nobody is looking, so a hidden tab
     should stop the download instead of paying for frames it never shows. */
  function onVisibilityChange() {
    state.hidden = document.hidden;
    const video = $("screen");
    if (state.hidden) {
      video.pause();
    } else {
      video.play().catch(() => {});
      state.lastBytesAt = 0;
    }
  }

  async function sampleBandwidth() {
    const pc = state.pc;
    if (!pc || !pc.getStats) return;
    let bytes = 0;
    try {
      const report = await pc.getStats();
      report.forEach((entry) => {
        if (entry.type === "inbound-rtp" && entry.kind === "video" && typeof entry.bytesReceived === "number") {
          bytes += entry.bytesReceived;
        }
      });
    } catch {
      return;
    }
    const now = performance.now();
    if (state.lastBytesAt && bytes >= state.lastBytes) {
      const kbps = ((bytes - state.lastBytes) * 8) / (now - state.lastBytesAt);
      state.kbps = state.kbps ? state.kbps * 0.6 + kbps * 0.4 : kbps;
      $("fps-pill").textContent = `${Math.round((state.frames * 1000) / Math.max(1, now - state.lastFps))} fps · ${formatRate(state.kbps)}`;
    }
    state.lastBytes = bytes;
    state.lastBytesAt = now;
  }

  function formatRate(kbps) {
    if (!kbps) return "--";
    if (kbps >= 1000) return `${(kbps / 1000).toFixed(1)} Mb/s`;
    return `${Math.round(kbps)} kb/s`;
  }

  /* ------------------------------------------------------------------- input */

  function videoRect() {
    const video = $("screen");
    const box = video.getBoundingClientRect();
    const vw = video.videoWidth || 16;
    const vh = video.videoHeight || 9;
    const scale = Math.min(box.width / vw, box.height / vh);
    const width = vw * scale;
    const height = vh * scale;
    return {
      left: box.left + (box.width - width) / 2,
      top: box.top + (box.height - height) / 2,
      width,
      height,
      vw,
      vh,
    };
  }

  function normalize(clientX, clientY) {
    const rect = videoRect();
    const x = (clientX - rect.left) / rect.width;
    const y = (clientY - rect.top) / rect.height;
    return {
      x: Math.min(1, Math.max(0, x)),
      y: Math.min(1, Math.max(0, y)),
      screen: { width: rect.vw, height: rect.vh },
    };
  }

  function emit(event) {
    if (!state.hasControl) return;
    send({ type: "input", event });
  }

  function move(clientX, clientY) {
    const point = normalize(clientX, clientY);
    emit({ type: "move", x: point.x, y: point.y, screen: point.screen });
  }

  function button(action, clientX, clientY, name = "left") {
    const point = normalize(clientX, clientY);
    emit({ type: "button", action, button: name, x: point.x, y: point.y, screen: point.screen });
  }

  function onTouchStart(event) {
    if (event.touches.length === 2) {
      cancelLongPress();
      const [a, b] = event.touches;
      state.gesture = {
        kind: "scroll",
        x: (a.clientX + b.clientX) / 2,
        y: (a.clientY + b.clientY) / 2,
      };
      return;
    }
    if (event.touches.length !== 1) return;
    const touch = event.touches[0];
    state.moved = false;
    state.drag = false;
    state.gesture = { kind: "pointer", x: touch.clientX, y: touch.clientY, startX: touch.clientX, startY: touch.clientY };
    move(touch.clientX, touch.clientY);
    state.longPressTimer = setTimeout(() => {
      state.longPressTimer = null;
      if (!state.moved) {
        button("click", touch.clientX, touch.clientY, "right");
        state.gesture = null;
        flash("right click");
      }
    }, 550);
  }

  function onTouchMove(event) {
    if (!state.gesture) return;
    event.preventDefault();
    if (state.gesture.kind === "scroll" && event.touches.length === 2) {
      const [a, b] = event.touches;
      const x = (a.clientX + b.clientX) / 2;
      const y = (a.clientY + b.clientY) / 2;
      const dx = x - state.gesture.x;
      const dy = y - state.gesture.y;
      state.gesture.x = x;
      state.gesture.y = y;
      if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
        emit({ type: "scroll", dx: -dx / 4, dy: -dy / 4 });
      }
      return;
    }
    const touch = event.touches[0];
    if (!touch) return;
    if (Math.hypot(touch.clientX - state.gesture.startX, touch.clientY - state.gesture.startY) > 8) {
      state.moved = true;
      cancelLongPress();
      // A finger that keeps moving is a drag, not a tap: hold the left button
      // down so windows and selections can be moved, and release it on touchend.
      if (!state.drag) {
        state.drag = true;
        button("down", touch.clientX, touch.clientY, "left");
      }
    }
    move(touch.clientX, touch.clientY);
  }

  function onTouchEnd(event) {
    cancelLongPress();
    const gesture = state.gesture;
    const dragged = state.drag;
    state.gesture = null;
    state.drag = false;
    if (!gesture || gesture.kind !== "pointer") return;
    const touch = event.changedTouches[0];
    if (!touch) return;
    if (dragged) {
      button("up", touch.clientX, touch.clientY, "left");
      return;
    }
    if (state.moved) return;
    // Two quick taps in the same spot are a double click, which is how a
    // remote desktop opens files and folders.
    const now = performance.now();
    const near = Math.hypot(touch.clientX - gesture.startX, touch.clientY - gesture.startY) < 24;
    if (now - state.lastTap < 300 && near) {
      state.lastTap = 0;
      button("click", touch.clientX, touch.clientY, "left");
      button("click", touch.clientX, touch.clientY, "left");
      flash("double click");
      return;
    }
    state.lastTap = now;
    button("click", touch.clientX, touch.clientY, "left");
  }

  function cancelLongPress() {
    if (state.longPressTimer) {
      clearTimeout(state.longPressTimer);
      state.longPressTimer = null;
    }
  }

  function onWheel(event) {
    event.preventDefault();
    emit({ type: "scroll", dx: event.deltaX / 40, dy: event.deltaY / 40 });
  }

  function onKeyDown(event) {
    if (!state.hasControl) return;
    if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) return;
    if (event.key.length === 1 && !event.ctrlKey && !event.altKey && !event.metaKey) return;
    event.preventDefault();
    emit({ type: "key", action: "down", key: event.key });
  }

  function onKeyUp(event) {
    if (!state.hasControl) return;
    if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) return;
    if (event.key.length === 1 && !event.ctrlKey && !event.altKey && !event.metaKey) return;
    emit({ type: "key", action: "up", key: event.key });
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
    if (!state.hasControl) {
      flash("take control first");
      return;
    }
    emit({ type: "key", action: "combo", keys: shortcut.keys });
    flash(shortcut.label);
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

  function toggleShortcuts(force) {
    const panel = $("shortcuts");
    const open = force !== undefined ? force : !panel.classList.contains("open");
    panel.classList.toggle("open", open);
    $("btn-shortcuts").classList.toggle("active", open);
    if (open) showToolbar();
  }

  /* ---------------------------------------------------------------------- ui */

  function showOverlay(text) {
    $("overlay-text").textContent = text;
    $("overlay").classList.remove("hidden");
  }

  function hideOverlay() {
    $("overlay").classList.add("hidden");
  }

  function flash(text) {
    const pill = $("control-pill");
    const previous = pill.textContent;
    pill.textContent = text;
    setTimeout(() => {
      pill.textContent = previous;
      updateControl();
    }, 1200);
  }

  function updateControl() {
    state.hasControl = Boolean(state.self && state.control === state.self);
    $("control-pill").textContent = state.hasControl ? "you have control" : "view only";
    $("btn-control").classList.toggle("active", state.hasControl);
  }

  function renderStats() {
    const now = performance.now();
    const fps = Math.round((state.frames * 1000) / Math.max(1, now - state.lastFps));
    $("fps-pill").textContent = state.kbps ? `${fps} fps · ${formatRate(state.kbps)}` : `${fps} fps`;
  }

  function startFpsMeter() {
    const video = $("screen");
    if (!video.requestVideoFrameCallback) return;
    const tick = () => {
      state.frames += 1;
      const now = performance.now();
      if (now - state.lastFps >= 1000) {
        renderStats();
        state.frames = 0;
        state.lastFps = now;
      }
      video.requestVideoFrameCallback(tick);
    };
    video.requestVideoFrameCallback(tick);
  }

  function toggleKeyboard(force) {
    const keyboard = $("keyboard");
    const open = force !== undefined ? force : !keyboard.classList.contains("open");
    keyboard.classList.toggle("open", open);
    $("btn-keyboard").classList.toggle("active", open);
    if (open) $("text-input").focus();
  }

  function sendText() {
    const input = $("text-input");
    const text = input.value;
    if (!text) return;
    emit({ type: "text", text });
    input.value = "";
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      document.documentElement.requestFullscreen().catch(() => {});
    }
  }

  function toggleFit() {
    const video = $("screen");
    const contain = video.style.objectFit !== "cover";
    video.style.objectFit = contain ? "cover" : "contain";
    $("btn-fit").classList.toggle("active", contain);
  }

  /* The toolbar sits on top of the picture, so it slides away while the user
     works and comes back on a tap in the bottom strip. */
  let toolbarTimer = null;

  function showToolbar() {
    $("toolbar").classList.remove("hidden");
    clearTimeout(toolbarTimer);
    toolbarTimer = setTimeout(() => {
      if (!state.gesture && !$("shortcuts").classList.contains("open")) $("toolbar").classList.add("hidden");
    }, 4000);
  }

  function onStageTap(event) {
    if (event.target.closest("#toolbar, #keyboard, #shortcuts, #hud, #overlay")) return;
    if ($("toolbar").classList.contains("hidden")) {
      showToolbar();
      return;
    }
    if (event.clientY > innerHeight - 90) showToolbar();
  }

  function disconnect() {
    state.closed = true;
    closePeer();
    if (state.ws) state.ws.close();
    showOverlay("Disconnected.");
  }

  /* -------------------------------------------------------------------- boot */

  /* Without a relay the phone on cellular only learns its own carrier address
     and the host's LAN address, so ICE never pairs. Fetch the relay list first.
     The credentials are time limited, so they are refreshed periodically; a
     peer that is already connected keeps working until it is rebuilt. */
  async function loadIceServers() {
    try {
      const response = await fetch("/api/status");
      const data = await response.json();
      if (Array.isArray(data.ice_servers)) {
        state.iceServers = data.ice_servers.filter((entry) => entry && entry.urls);
      }
    } catch {
      state.iceServers = [];
    }
    if (!state.iceLogged) {
      state.iceLogged = true;
      const urls = state.iceServers.flatMap((entry) => (Array.isArray(entry.urls) ? entry.urls : [entry.urls]));
      console.info(urls.length ? `ICE servers: ${urls.join(", ")}` : "ICE servers: none (LAN only)");
    }
  }

  function boot() {
    const video = $("screen");
    video.addEventListener("touchstart", onTouchStart, { passive: true });
    video.addEventListener("touchmove", onTouchMove, { passive: false });
    video.addEventListener("touchend", onTouchEnd, { passive: true });
    video.addEventListener("touchcancel", onTouchEnd, { passive: true });
    video.addEventListener("wheel", onWheel, { passive: false });
    video.addEventListener("contextmenu", (event) => event.preventDefault());
    video.addEventListener("click", onStageTap);

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);

    $("connect").addEventListener("click", () => {
      const code = $("room").value.trim().toUpperCase();
      if (code.length < 4) {
        $("overlay-error").textContent = "Enter the 6 character room code.";
        $("overlay-error").hidden = false;
        return;
      }
      state.room = code;
      $("overlay-error").hidden = true;
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        send({ type: "hello", role: "viewer", room: state.room, token: state.token });
      } else {
        connect();
      }
    });

    $("room").addEventListener("keydown", (event) => {
      if (event.key === "Enter") $("connect").click();
    });

    $("btn-control").addEventListener("click", () => {
      send({ type: "control", action: state.hasControl ? "release" : "request" });
    });
    $("btn-keyboard").addEventListener("click", () => toggleKeyboard());
    $("close-keyboard").addEventListener("click", () => toggleKeyboard(false));
    $("btn-shortcuts").addEventListener("click", () => toggleShortcuts());
    $("close-shortcuts").addEventListener("click", () => toggleShortcuts(false));
    $("btn-quality").addEventListener("click", cycleQuality);
    $("send-text").addEventListener("click", sendText);
    $("text-input").addEventListener("keydown", (event) => {
      if (event.key === "Enter") sendText();
    });
    $("btn-fit").addEventListener("click", toggleFit);
    $("btn-fullscreen").addEventListener("click", toggleFullscreen);
    $("btn-disconnect").addEventListener("click", disconnect);

    $("toolbar").addEventListener("pointerdown", showToolbar);
    showToolbar();

    buildShortcuts();
    setQuality(state.quality, { announce: false });
    document.addEventListener("visibilitychange", onVisibilityChange);
    setInterval(sampleBandwidth, 2000);

    document.querySelectorAll("#keyboard .keys button[data-key]").forEach((element) => {
      element.addEventListener("click", () => {
        const key = element.dataset.key;
        emit({ type: "key", action: "click", key });
      });
    });

    if (state.room) {
      $("room").value = state.room;
      connect();
    } else {
      showOverlay("Enter the room code shown on the host computer.");
      loadIceServers();
    }

    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    }
  }

  boot();
})();
