# Remote Desktop

Stream this computer's screen to a phone and control it from there — mouse,
keyboard and scrolling — over the local network. No app store, no account, no
cloud relay: the desktop runs a small Python agent that serves a mobile web app
(PWA) and the video stream from the same origin.

```
┌──────────────────────────┐            ┌───────────────────────────┐
│  Desktop (agent)         │            │  Phone (browser / PWA)    │
│                          │            │                           │
│  mss ──► JPEG frames ────┼── WS ─────►│  canvas + createImageBitmap│
│  pynput ◄── input events ┼── WS ◄─────┤  touch / soft keyboard    │
│  FastAPI + uvicorn       │            │  index.html + app.js      │
└──────────────────────────┘            └───────────────────────────┘
        http://<lan-ip>:8765/?token=…
```

## Features

- Live screen stream over WebSocket, JPEG encoded, downscaled to fit a phone.
- Full remote control: pointer move/click/right-click/scroll and keyboard input.
- Single control owner — extra devices can watch but not type, so two people
  cannot fight over the cursor.
- Token-protected. A QR code is printed on startup so pairing is one scan.
- Capture pauses automatically when nobody is connected.
- Installable as a PWA (fullscreen, offline shell).

## Install

```bash
cd projects/remote-desktop
pip install -r requirements.txt
```

Linux needs an X11 or Wayland session the agent can read; `pynput` may require
`python3-xlib` or membership of the `input` group for keyboard injection.

## Run

```bash
python -m remote_desktop
```

The agent prints a local URL, a LAN URL and a QR code:

```
  Local URL : http://127.0.0.1:8765/?token=…
  Phone URL : http://192.168.1.24:8765/?token=…
```

Open the phone URL (or scan the QR code) on a device on the same network. The
page connects automatically and asks for control.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Interface to bind |
| `--port` | `8765` | Port to listen on |
| `--token` | generated | Shared access token |
| `--fps` | `12` | Target frames per second (1–60) |
| `--quality` | `60` | JPEG quality (10–95) |
| `--max-width` | `1280` | Downscale frames to this width |
| `--monitor` | `1` | Monitor index, `1` is primary |
| `--no-input` | off | View only, never inject input |
| `--no-auto-control` | off | Require an explicit control request |
| `--no-cursor` | off | Do not composite the mouse cursor |
| `--log-level` | `info` | `debug`, `info`, `warning`, `error` |

Every option also has an `RD_*` environment variable (`RD_PORT`, `RD_FPS`,
`RD_TOKEN`, `RD_ALLOW_INPUT`, …). CLI flags win over the environment.

Lower `--fps` and `--max-width` if the stream stutters on a weak Wi-Fi link;
raise them on a fast local network.

## Using the phone client

| Gesture | Effect |
| --- | --- |
| Tap | Left click |
| Long press | Right click |
| Drag | Move the pointer |
| Two-finger / wheel scroll | Scroll |
| **Control** | Request or release the input lock |
| **Keyboard** | Open the soft keyboard; keystrokes go to the desktop |
| **Fit** | Toggle between native size and fill-the-screen |
| **Full** | Fullscreen |

## HTTP API

All endpoints require the token, as `Authorization: Bearer <token>`,
`X-RD-Token: <token>` or `?token=<token>`.

| Endpoint | Description |
| --- | --- |
| `GET /api/status` | Screen size, monitors, client count, control owner, counters |
| `GET /api/screen` | A single JPEG snapshot |
| `POST /api/control` | `{"action": "request"\|"release", "client": "<id>"}` |
| `WS /ws` | Frame stream (binary) plus JSON control messages |

Binary frames are `0x01 | uint32 sequence | JPEG`. Text messages are JSON:
`{"t": "hello"}`, `{"t": "control"}`, `{"t": "error"}`, `{"t": "pong"}` from the
server; `{"t": "mouse"|"key"|"text"|"control"|"ping"}` from the client. Pointer
coordinates are normalised to `0..1` so they survive downscaling.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite uses fake capture and input backends, so it runs headless.

## Notes

- The agent binds `0.0.0.0` by default. Anyone on the same network who has the
  token can control the machine — treat the token like a password, and prefer
  `--no-input` when you only need to watch.
- Only one monitor is streamed at a time; pick it with `--monitor`.
- Clipboard sync, file transfer and audio are not implemented.
- The phone must be able to reach the desktop directly. For remote access over
  the internet, put the agent behind a VPN or a tunnel such as WireGuard or
  Tailscale rather than exposing the port.
