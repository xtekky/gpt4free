"use strict";

/*
 * Mobile client for the remote desktop agent.
 *
 * Binary websocket frames are: 0x01 | uint32 sequence | JPEG bytes.
 * Text messages are JSON, see server.py for the schema.
 */

const FRAME_MAGIC = 0x01;
const HEADER_SIZE = 5;
const TAP_SLOP_PX = 12;
const LONG_PRESS_MS = 450;
const MOVE_INTERVAL_MS = 25;

const canvas = document.getElementById("screen");
const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
const hint = document.getElementById("hint");
const latency = document.getElementById("latency");
const statusEl = document.getElementById("status");
const softKeys = document.getElementById("soft-keys");

const btnConnect = document.getElementById("btn-connect");
const btnControl = document.getElementById("btn-control");
const btnKeyboard = document.getElementById("btn-keyboard");
const btnFit = document.getElementById("btn-fit");
const btnFullscreen = document.getElementById("btn-fullscreen");

const state = {
  socket: null,
  connected: false,
  control: false,
  allowInput: true,
  remote: { width: 0, height: 0 },
  sequence: -1,
  received: 0,
  dropped: 0,
  lastFrameAt: 0,
  fps: 0,
  reconnects: 0,
  reconnectTimer: null,
  decodePending: false,
  pointer: { x: 0, y: 0 },
  lastMoveAt: 0,
  lastSentMove: null,
  gesture: null,
  keysDown: new Set(),
};

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = kind || "";
}

function token() {
  const params = new URLSearchParams(location.search);
  const fromUrl = params.get("token");
  if (fromUrl) {
    try { sessionStorage.setItem("rd-token", fromUrl); } catch (e) { /* ignore */ }
    return fromUrl;
  }
  try { return sessionStorage.getItem("rd-token") || ""; } catch (e) { return ""; }
}

// -- websocket ----------------------------------------------------------

function wsUrl() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}/ws?token=${encodeURIComponent(token())}`;
}

function connect() {
  if (state.socket) return;
  clearTimeout(state.reconnectTimer);
  setStatus("connecting…", "warn");
  const socket = new WebSocket(wsUrl());
  socket.binaryType = "arraybuffer";
  state.socket = socket;

  socket.onopen = () => {
    state.connected = true;
    state.reconnects = 0;
    hint.classList.add("hidden");
    btnConnect.textContent = "Disconnect";
    btnConnect.classList.remove("primary");
    setStatus("connected", "live");
  };

  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleText(event.data);
    } else {
      handleBinary(event.data);
    }
  };

  socket.onclose = (event) => {
    const wasConnected = state.connected;
    state.connected = false;
    state.control = false;
    state.socket = null;
    btnConnect.textContent = "Connect";
    btnConnect.classList.add("primary");
    btnControl.disabled = true;
    btnControl.classList.remove("active");
    btnKeyboard.disabled = true;
    if (event.code === 4401) {
      setStatus("bad token", "error");
      hint.textContent = "Access token rejected. Open the URL with ?token=… again.";
      hint.classList.remove("hidden");
      return;
    }
    if (event.code === 4429) {
      setStatus("too many clients", "error");
      hint.textContent = "Another device is already connected. Try again later.";
      hint.classList.remove("hidden");
      return;
    }
    setStatus("disconnected", "error");
    if (wasConnected) scheduleReconnect();
  };

  socket.onerror = () => setStatus("socket error", "error");
}

function scheduleReconnect() {
  if (state.reconnectTimer) return;
  const delay = Math.min(10000, 500 * Math.pow(2, state.reconnects++));
  setStatus(`reconnecting in ${Math.round(delay / 1000)}s`, "warn");
  state.reconnectTimer = setTimeout(() => {
    state.reconnectTimer = null;
    connect();
  }, delay);
}

function disconnect() {
  clearTimeout(state.reconnectTimer);
  state.reconnectTimer = null;
  if (state.socket) {
    const socket = state.socket;
    state.socket = null;
    socket.close();
  }
  hint.classList.remove("hidden");
  setStatus("idle");
}

function send(message) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return false;
  state.socket.send(JSON.stringify(message));
  return true;
}

function handleText(raw) {
  let message;
  try { message = JSON.parse(raw); } catch (e) { return; }
  switch (message.t) {
    case "hello":
      state.remote.width = message.width || 0;
      state.remote.height = message.height || 0;
      state.allowInput = message.allow_input !== false;
      state.control = !!message.control;
      btnControl.disabled = !state.allowInput;
      btnKeyboard.disabled = !state.allowInput || !state.control;
      if (message.control) btnControl.classList.add("active");
      hint.classList.add("hidden");
      setStatus(`live ${state.remote.width}×${state.remote.height}`, "live");
      break;
    case "control":
      state.control = message.action === "granted";
      btnControl.classList.toggle("active", state.control);
      btnControl.textContent = state.control ? "Release" : "Control";
      btnKeyboard.disabled = !state.control;
      if (state.control) {
        setStatus("control granted", "live");
      } else if (message.action === "denied") {
        setStatus(`in use by ${message.client || "another device"}`, "warn");
      } else {
        setStatus("view only", "warn");
      }
      break;
    case "error":
      setStatus(message.message || "error", "error");
      break;
    default:
      break;
  }
}

function handleBinary(buffer) {
  const view = new DataView(buffer);
  if (view.byteLength <= HEADER_SIZE || view.getUint8(0) !== FRAME_MAGIC) return;
  const sequence = view.getUint32(1);
  if (state.sequence >= 0 && sequence > state.sequence + 1) {
    state.dropped += sequence - state.sequence - 1;
  }
  state.sequence = sequence;
  state.received += 1;
  const blob = new Blob([new Uint8Array(buffer, HEADER_SIZE)]);
  scheduleDecode(blob);
}

function scheduleDecode(blob) {
  if (state.decodePending) {
    // Skip this frame rather than building a backlog of decodes.
    state.dropped += 1;
    return;
  }
  state.decodePending = true;
  createImageBitmap(blob).then(
    (bitmap) => {
      draw(bitmap);
      bitmap.close();
      state.decodePending = false;
    },
    () => { state.decodePending = false; }
  );
}

function draw(bitmap) {
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  }
  ctx.drawImage(bitmap, 0, 0);
  const now = performance.now();
  if (state.lastFrameAt) {
    const delta = now - state.lastFrameAt;
    if (delta > 0) state.fps = state.fps ? state.fps * 0.8 + (1000 / delta) * 0.2 : 1000 / delta;
  }
  state.lastFrameAt = now;
  latency.textContent = `${state.fps.toFixed(1)} fps · ${state.received} frames`;
}

// -- pointer input ------------------------------------------------------

function toNormalized(event) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  return {
    x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
  };
}

function sendMouse(action, extra) {
  if (!state.control || !state.allowInput) return;
  send(Object.assign({ t: "mouse", action: action, x: state.pointer.x, y: state.pointer.y }, extra || {}));
}

function sendMove(point) {
  state.pointer = point;
  if (!state.control) return;
  const now = performance.now();
  if (now - state.lastMoveAt < MOVE_INTERVAL_MS) return;
  state.lastMoveAt = now;
  const key = `${point.x.toFixed(3)},${point.y.toFixed(3)}`;
  if (key === state.lastSentMove) return;
  state.lastSentMove = key;
  sendMouse("move");
}

canvas.addEventListener("pointerdown", (event) => {
  const point = toNormalized(event);
  if (!point) return;
  canvas.setPointerCapture(event.pointerId);
  state.pointer = point;
  state.gesture = {
    id: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    startedAt: performance.now(),
    moved: false,
    longPress: false,
    timer: setTimeout(() => {
      if (!state.gesture || state.gesture.moved) return;
      state.gesture.longPress = true;
      sendMouse("click", { button: "right", clicks: 1 });
    }, LONG_PRESS_MS),
  };
  sendMouse("down", { button: "left" });
});

canvas.addEventListener("pointermove", (event) => {
  const point = toNormalized(event);
  if (!point) return;
  if (state.gesture && state.gesture.id === event.pointerId) {
    const dx = event.clientX - state.gesture.startX;
    const dy = event.clientY - state.gesture.startY;
    if (Math.hypot(dx, dy) > TAP_SLOP_PX) state.gesture.moved = true;
  }
  sendMove(point);
});

function endGesture(event, cancelled) {
  const gesture = state.gesture;
  if (!gesture || gesture.id !== event.pointerId) return;
  clearTimeout(gesture.timer);
  state.gesture = null;
  const point = toNormalized(event);
  if (point) state.pointer = point;
  const held = performance.now() - gesture.startedAt;
  if (cancelled) {
    sendMouse("up", { button: "left" });
    return;
  }
  if (gesture.longPress) {
    sendMouse("up", { button: "left" });
    return;
  }
  if (!gesture.moved && held < LONG_PRESS_MS) {
    sendMouse("up", { button: "left" });
    sendMouse("click", { button: "left", clicks: 1 });
    return;
  }
  sendMouse("up", { button: "left" });
}

canvas.addEventListener("pointerup", (event) => endGesture(event, false));
canvas.addEventListener("pointercancel", (event) => endGesture(event, true));

canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    const point = toNormalized(event);
    if (point) state.pointer = point;
    sendMouse("scroll", { dx: Math.sign(event.deltaX), dy: Math.sign(event.deltaY) });
  },
  { passive: false }
);

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

// -- soft keyboard ------------------------------------------------------

softKeys.addEventListener("keydown", (event) => {
  if (!state.control) return;
  event.preventDefault();
  if (state.keysDown.has(event.key)) return;
  state.keysDown.add(event.key);
  send({ t: "key", action: "down", key: event.key });
});

softKeys.addEventListener("keyup", (event) => {
  if (!state.control) return;
  event.preventDefault();
  state.keysDown.delete(event.key);
  send({ t: "key", action: "up", key: event.key });
});

softKeys.addEventListener("input", () => {
  if (!state.control) return;
  const value = softKeys.value;
  if (value) send({ t: "text", text: value });
  softKeys.value = "";
});

softKeys.addEventListener("blur", () => {
  for (const key of state.keysDown) {
    send({ t: "key", action: "up", key: key });
  }
  state.keysDown.clear();
});

// -- toolbar ------------------------------------------------------------

btnConnect.addEventListener("click", () => {
  if (state.socket) disconnect();
  else connect();
});

btnControl.addEventListener("click", () => {
  if (state.control) send({ t: "control", action: "release" });
  else send({ t: "control", action: "request" });
});

btnKeyboard.addEventListener("click", () => {
  if (!state.control) return;
  softKeys.value = "";
  softKeys.focus();
  setStatus("keyboard ready", "live");
});

btnFit.addEventListener("click", () => {
  document.body.classList.toggle("fill");
  btnFit.classList.toggle("active", document.body.classList.contains("fill"));
});

btnFullscreen.addEventListener("click", () => {
  const root = document.documentElement;
  if (document.fullscreenElement) document.exitFullscreen();
  else if (root.requestFullscreen) root.requestFullscreen().catch(() => {});
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    for (const key of state.keysDown) send({ t: "key", action: "up", key: key });
    state.keysDown.clear();
  }
});

window.addEventListener("beforeunload", () => {
  if (state.socket) state.socket.close();
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  });
}

if (token() && new URLSearchParams(location.search).get("autoconnect") !== "0") {
  connect();
} else if (!token()) {
  hint.textContent = "No access token. Reopen the link shown on the desktop.";
}
