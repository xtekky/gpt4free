/* Host page: capture the screen with getDisplayMedia() and publish it over WebRTC. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);

  const state = {
    ws: null,
    room: null,
    token: params.get("token") || "",
    stream: null,
    peers: new Map(),
    viewerIds: new Set(),
    viewers: 0,
    control: null,
    urls: [],
    retry: 500,
    closed: false,
    capture: null,
    quality: new Map(),
    iceServers: [],
    iceLogged: false,
  };

  //: Encoder caps the host applies per viewer. A viewer on mobile data asks
  //: for "low" and the host re-encodes instead of pushing a full 1080p stream.
  const QUALITY = {
    low: { maxWidth: 854, maxHeight: 480, maxBitrate: 250_000, maxFramerate: 12 },
    medium: { maxWidth: 1280, maxHeight: 720, maxBitrate: 900_000, maxFramerate: 20 },
    high: { maxWidth: 1920, maxHeight: 1080, maxBitrate: 2_500_000, maxFramerate: 30 },
    auto: { maxWidth: 1280, maxHeight: 720, maxBitrate: 1_200_000, maxFramerate: 24 },
  };

  const log = (message) => {
    const line = `${new Date().toLocaleTimeString()}  ${message}`;
    const box = $("log");
    box.textContent = `${line}\n${box.textContent}`.slice(0, 4000);
  };

  const setDot = (id, cls) => {
    const dot = $(id);
    dot.className = `dot ${cls}`;
  };

  const wsUrl = () => {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    return `${scheme}//${location.host}/ws`;
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

    const ws = new WebSocket(wsUrl());
    state.ws = ws;

    ws.onopen = () => {
      state.retry = 500;
      setDot("ws-dot", "on");
      $("ws-text").textContent = "connected";
      send({ type: "hello", role: "host", token: state.token });
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
      closeAllPeers();
      if (state.closed) return;
      setTimeout(connect, state.retry);
      state.retry = Math.min(state.retry * 2, 8000);
    };

    ws.onerror = () => log("websocket error");
  }

  function handle(message) {
    switch (message.type) {
      case "room":
        state.room = message.room;
        state.control = message.control;
        showRoom(message.room);
        log(`room ${message.room} open`);
        break;
      case "viewer-joined":
        state.viewers = message.viewers;
        state.control = message.control;
        state.viewerIds.add(message.peer);
        log(`viewer ${message.peer} joined (${message.viewers} total)`);
        updateViewers();
        updateControl();
        if (state.stream) offerTo(message.peer);
        break;
      case "peer-left":
        state.viewers = message.viewers;
        state.control = message.control;
        state.viewerIds.delete(message.peer);
        dropPeer(message.peer);
        log(`${message.role} ${message.peer} left`);
        updateViewers();
        updateControl();
        break;
      case "signal":
        onSignal(message.from, message.data);
        break;
      case "control":
        state.control = message.owner;
        updateControl();
        log(message.owner ? `control granted to ${message.owner}` : "control released");
        break;
      case "error":
        log(`error: ${message.code} - ${message.message}`);
        if (message.code === "bad-token") {
          $("share-error").textContent =
            "The access token is missing or wrong. Reopen this page from the URL printed in the terminal.";
          $("share-error").hidden = false;
        }
        break;
      default:
        break;
    }
  }

  /* ------------------------------------------------------------------ capture */

  async function startShare() {
    $("share-error").hidden = true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      fail("This browser does not support getDisplayMedia(). Use Chrome, Edge or Firefox on the desktop.");
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: { ideal: 30, max: 60 }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
        preferCurrentTab: false,
        surfaceSwitching: "include",
      });
    } catch (error) {
      if (error && error.name === "NotAllowedError") {
        log("capture cancelled by the user");
      } else {
        fail(`Screen capture failed: ${error && error.message ? error.message : error}`);
      }
      return;
    }

    state.stream = stream;
    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings ? track.getSettings() : {};
    $("capture-info").textContent =
      `Capturing ${settings.width || "?"}x${settings.height || "?"} @ ${Math.round(settings.frameRate || 0)} fps` +
      (track.label ? ` - ${track.label}` : "");
    track.addEventListener("ended", () => {
      log("capture ended by the user");
      stopShare();
    });

    $("share").disabled = true;
    $("stop").disabled = false;
    log("screen capture started");

    for (const viewerId of state.peers.keys()) offerTo(viewerId);
    if (state.viewers > 0 && state.peers.size === 0) {
      log("waiting for viewers to reconnect");
    }
  }

  function stopShare() {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    closeAllPeers();
    $("share").disabled = false;
    $("stop").disabled = true;
    $("capture-info").textContent = "";
    log("screen capture stopped");
  }

  function fail(message) {
    $("share-error").textContent = message;
    $("share-error").hidden = false;
    log(message);
  }

  /* ------------------------------------------------------------------ webrtc */

  let signalChain = Promise.resolve();
  const negotiations = new Map();

  function newPeer(viewerId) {
    const pc = new RTCPeerConnection({ iceServers: state.iceServers, iceCandidatePoolSize: 2 });
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        send({ type: "signal", to: viewerId, data: { kind: "ice", candidate: event.candidate } });
      }
    };
    pc.onicecandidateerror = (event) => {
      log(`peer ${viewerId}: ICE candidate error ${event.errorCode} ${event.errorText || ""}`);
    };
    pc.onconnectionstatechange = () => {
      log(`peer ${viewerId}: ${pc.connectionState}`);
      if (["failed", "closed", "disconnected"].includes(pc.connectionState)) dropPeer(viewerId);
    };
    state.peers.set(viewerId, pc);
    return pc;
  }

  /* Re-encode the outgoing video for one viewer. The sender is the only side
     that can cap the bitrate, so the viewer's quality request lands here. */
  async function applyQuality(viewerId) {
    const pc = state.peers.get(viewerId);
    if (!pc) return;
    const settings = QUALITY[state.quality.get(viewerId)] || QUALITY.auto;
    for (const sender of pc.getSenders()) {
      if (!sender.track || sender.track.kind !== "video") continue;
      const parameters = sender.getParameters();
      if (!parameters.encodings || !parameters.encodings.length) parameters.encodings = [{}];
      const encoding = parameters.encodings[0];
      encoding.maxBitrate = settings.maxBitrate;
      encoding.maxFramerate = settings.maxFramerate;
      encoding.scaleResolutionDownBy = scaleFor(sender.track, settings);
      try {
        await sender.setParameters(parameters);
      } catch (error) {
        log(`quality for ${viewerId} failed: ${error.message}`);
      }
    }
    log(`viewer ${viewerId} quality: ${state.quality.get(viewerId) || "auto"}`);
  }

  /* Downscaling is what actually shrinks the frame; a bitrate cap alone still
     pays for full-resolution pixels. */
  function scaleFor(track, settings) {
    const { width = 0, height = 0 } = track.getSettings();
    if (!width || !height) return 1;
    const scale = Math.max(width / settings.maxWidth, height / settings.maxHeight);
    return scale > 1 ? Math.min(scale, 4) : 1;
  }

  async function negotiate(viewerId) {
    if (!state.stream) return;
    let pc = state.peers.get(viewerId);
    if (!pc || pc.connectionState === "closed") pc = newPeer(viewerId);

    const senders = pc.getSenders().map((sender) => sender.track);
    let added = false;
    for (const track of state.stream.getTracks()) {
      if (!senders.includes(track)) {
        pc.addTrack(track, state.stream);
        added = true;
      }
    }

    // A viewer asks for a stream right after joining, which races with the
    // offer sent on "viewer-joined". Only one negotiation may be in flight,
    // and a connected peer with all tracks attached needs no new offer.
    if (pc.signalingState !== "stable") return;
    if (!added && pc.connectionState === "connected") return;

    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await applyQuality(viewerId);
      send({ type: "signal", to: viewerId, data: { kind: "offer", sdp: pc.localDescription } });
      log(`offer sent to ${viewerId}`);
    } catch (error) {
      log(`offer to ${viewerId} failed: ${error.message}`);
      dropPeer(viewerId);
    }
  }

  // The state guard above only holds if the previous negotiation has finished,
  // so offers are queued per viewer instead of running concurrently.
  function offerTo(viewerId) {
    const previous = negotiations.get(viewerId) || Promise.resolve();
    const next = previous.then(() => negotiate(viewerId)).catch(() => {});
    negotiations.set(viewerId, next);
    return next;
  }

  function onSignal(from, data) {
    signalChain = signalChain.then(() => processSignal(from, data)).catch(() => {});
    return signalChain;
  }

  async function processSignal(from, data) {
    if (!data || typeof data !== "object") return;
    if (data.kind === "request") {
      await offerTo(from);
      return;
    }
    if (data.kind === "quality") {
      state.quality.set(from, QUALITY[data.level] ? data.level : "auto");
      await applyQuality(from);
      return;
    }
    let pc = state.peers.get(from);
    if (!pc) {
      if (data.kind !== "offer") return;
      pc = newPeer(from);
    }
    try {
      if (data.kind === "answer") {
        if (pc.signalingState !== "have-local-offer") return;
        await pc.setRemoteDescription(data.sdp);
      } else if (data.kind === "offer") {
        if (pc.signalingState !== "stable") return;
        await pc.setRemoteDescription(data.sdp);
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        send({ type: "signal", to: from, data: { kind: "answer", sdp: pc.localDescription } });
      } else if (data.kind === "ice" && data.candidate) {
        if (!pc.remoteDescription) return;
        await pc.addIceCandidate(data.candidate);
      }
    } catch (error) {
      log(`signal from ${from} failed: ${error.message}`);
    }
  }

  function dropPeer(viewerId) {
    const pc = state.peers.get(viewerId);
    if (!pc) return;
    state.peers.delete(viewerId);
    negotiations.delete(viewerId);
    state.quality.delete(viewerId);
    try {
      pc.close();
    } catch {
      /* already closed */
    }
  }

  function closeAllPeers() {
    for (const viewerId of [...state.peers.keys()]) dropPeer(viewerId);
  }

  /* --------------------------------------------------------------------- ui */

  function showRoom(code) {
    $("room-card").hidden = false;
    $("room-code").textContent = code;
    const base = state.urls.find((url) => !/127\.0\.0\.1|localhost/.test(url)) || state.urls[0] || location.origin;
    const link = `${base}/view?room=${code}${state.token ? `&token=${encodeURIComponent(state.token)}` : ""}`;
    $("view-url").textContent = link;
    const qr = $("qr");
    qr.src = `/api/qr.svg?url=${encodeURIComponent(link)}`;
    qr.hidden = false;
  }

  function updateViewers() {
    $("viewer-text").textContent = `${state.viewers} viewer${state.viewers === 1 ? "" : "s"}`;
    setDot("viewer-dot", state.viewers > 0 ? "on" : "");
  }

  function updateControl() {
    $("control-pill").textContent = state.control ? `control: ${state.control}` : "control: nobody";
  }

  /* The relay list decides whether a phone on cellular can reach us at all, so
     it is fetched before any peer is built and refreshed with the status poll.
     Only the URLs are compared: the TURN credentials are re-minted on every
     request, so comparing them would rebuild the peers on every poll. */
  function applyIceServers(servers) {
    if (!Array.isArray(servers)) return;
    const next = servers.filter((entry) => entry && entry.urls);
    const urlsOf = (list) => JSON.stringify(list.map((entry) => entry.urls));
    const changed = urlsOf(next) !== urlsOf(state.iceServers);
    state.iceServers = next;
    if (!state.iceLogged) {
      state.iceLogged = true;
      const urls = next.flatMap((entry) => (Array.isArray(entry.urls) ? entry.urls : [entry.urls]));
      log(urls.length ? `ICE servers: ${urls.join(", ")}` : "ICE servers: none (LAN only)");
    }
    if (changed && state.peers.size) {
      log("ICE configuration changed, rebuilding peers");
      closeAllPeers();
    }
  }

  async function refreshStatus() {
    try {
      const response = await fetch("/api/status");
      const data = await response.json();
      state.urls = data.urls || [];
      applyIceServers(data.ice_servers);
      if (state.room) showRoom(state.room);
      const input = $("input-text");
      if (data.input) {
        input.textContent = "input injection ready";
        setDot("input-dot", "on");
      } else if (data.allow_input) {
        input.textContent = data.input_error ? "input unavailable" : "input idle";
        setDot("input-dot", "warn");
      } else {
        input.textContent = "input disabled";
        setDot("input-dot", "off");
      }
    } catch (error) {
      log(`status request failed: ${error.message}`);
    }
  }

  /* ------------------------------------------------------------------- boot */

  function boot() {
    if (!window.isSecureContext) $("secure-warning").hidden = false;

    $("share").addEventListener("click", startShare);
    $("stop").addEventListener("click", stopShare);
    $("copy-link").addEventListener("click", () => copy($("view-url").textContent));
    $("copy-code").addEventListener("click", () => copy($("room-code").textContent));

    window.addEventListener("beforeunload", () => {
      state.closed = true;
      stopShare();
    });

    connect();
    refreshStatus();
    setInterval(refreshStatus, 15000);
    updateViewers();
    updateControl();
  }

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      log("copied to clipboard");
    } catch {
      log("clipboard unavailable");
    }
  }

  boot();
})();
