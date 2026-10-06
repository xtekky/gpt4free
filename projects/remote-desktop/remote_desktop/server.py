"""FastAPI application: REST status endpoints, the websocket stream and the PWA."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .capture import CaptureUnavailableError, ScreenCapture
from .config import Settings
from .input import InputController, InputUnavailableError
from .session import SessionHub

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

#: Upper bound on client messages per second, as a cheap abuse guard.
MAX_MESSAGES_PER_SECOND = 500


def lan_ip() -> str:
    """Best-effort local address a phone on the same network can reach."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def qr_ascii(url: str) -> str:
    """Render *url* as a QR code using half-block characters."""
    try:
        import qrcode
    except ImportError:  # pragma: no cover - optional dependency
        return ""
    code = qrcode.QRCode(border=1)
    code.add_data(url)
    code.make(fit=True)
    matrix = code.get_matrix()
    lines = []
    for row in range(0, len(matrix), 2):
        top = matrix[row]
        bottom = matrix[row + 1] if row + 1 < len(matrix) else [False] * len(top)
        lines.append(
            "".join(
                "\u2588" if t and b else "\u2580" if t else "\u2584" if b else " "
                for t, b in zip(top, bottom)
            )
        )
    return "\n".join(lines)


def _token_from(headers, query) -> str | None:
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    header_token = headers.get("x-rd-token")
    if header_token:
        return header_token
    return query.get("token")


def _token_ok(settings: Settings, headers, query) -> bool:
    supplied = _token_from(headers, query)
    if not supplied:
        return False
    return secrets.compare_digest(str(supplied), settings.token)


def build_hub(settings: Settings) -> SessionHub:
    """Create the capture/input backends, degrading gracefully when unavailable."""
    capture = None
    controller = None
    try:
        capture = ScreenCapture(
            monitor=settings.monitor,
            max_width=settings.max_width,
            quality=settings.quality,
            with_cursor=settings.with_cursor,
        )
    except CaptureUnavailableError as exc:
        logger.error("screen capture unavailable: %s", exc)
    if settings.allow_input:
        try:
            controller = InputController(
                screen_size=lambda: capture.screen_size() if capture else (0, 0),
                rate_limit=settings.input_rate,
                enabled=True,
            )
        except InputUnavailableError as exc:
            logger.error("input injection unavailable: %s", exc)
    return SessionHub(settings, capture=capture, controller=controller)


def create_app(settings: Settings, hub: SessionHub | None = None) -> FastAPI:
    """Build the ASGI app. *hub* can be injected for tests."""
    settings.resolve_token()
    hub = hub or build_hub(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await hub.shutdown()

    app = FastAPI(title="Remote Desktop Agent", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.hub = hub

    def require_token(request: Request) -> None:
        if not _token_ok(settings, request.headers, request.query_params):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    @app.get("/api/status")
    async def api_status(request: Request) -> JSONResponse:
        require_token(request)
        return JSONResponse(hub.status())

    @app.get("/api/screen")
    async def api_screen(request: Request) -> Response:
        require_token(request)
        if hub.capture is None:
            raise HTTPException(status_code=503, detail="screen capture unavailable")
        try:
            frame = await asyncio.to_thread(hub.capture.grab)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"capture failed: {exc}") from exc
        return Response(
            content=frame.data,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/control")
    async def api_control(request: Request) -> JSONResponse:
        require_token(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        action = (body or {}).get("action", "request")
        client_id = (body or {}).get("client")
        client = hub.clients.get(client_id) if client_id else None
        if client is None:
            raise HTTPException(status_code=404, detail="unknown client")
        if action == "release":
            hub.release_control(client)
            await hub.broadcast_text(
                {"t": "control", "action": "released", "client": client.id}
            )
            return JSONResponse({"ok": True, "control": hub.control_owner})
        if not hub.request_control(client):
            return JSONResponse(
                {"ok": False, "control": hub.control_owner}, status_code=409
            )
        await hub.broadcast_text(
            {"t": "control", "action": "granted", "client": client.id}
        )
        return JSONResponse({"ok": True, "control": hub.control_owner})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        if not _token_ok(settings, websocket.headers, websocket.query_params):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        client = await hub.add_client(websocket)
        if client is None:
            await websocket.send_text(
                json.dumps({"t": "error", "message": "too many clients"})
            )
            await websocket.close(code=4429)
            return

        writer = asyncio.create_task(hub.writer_loop(client))
        window_start = time.monotonic()
        window_count = 0
        try:
            await websocket.send_text(json.dumps(hub.hello(client)))
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw = message.get("text")
                if raw is None:
                    continue
                now = time.monotonic()
                if now - window_start >= 1.0:
                    window_start, window_count = now, 0
                window_count += 1
                if window_count > MAX_MESSAGES_PER_SECOND:
                    continue
                try:
                    payload = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(payload, dict):
                    continue
                kind = payload.get("t")
                if kind == "ping":
                    await websocket.send_text(json.dumps({"t": "pong"}))
                    continue
                if kind in ("mouse", "key", "text"):
                    if not client.has_control:
                        await websocket.send_text(
                            json.dumps({"t": "error", "message": "no control"})
                        )
                        continue
                    hub.touch_control(client)
                    if hub.controller is not None:
                        hub.controller.handle(payload)
                    continue
                if kind == "control":
                    action = payload.get("action", "request")
                    if action == "release":
                        hub.release_control(client)
                        await hub.broadcast_text(
                            {"t": "control", "action": "released", "client": client.id}
                        )
                    elif hub.request_control(client):
                        await hub.broadcast_text(
                            {"t": "control", "action": "granted", "client": client.id}
                        )
                    else:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "t": "control",
                                    "action": "denied",
                                    "client": hub.control_owner,
                                }
                            )
                        )
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # pragma: no cover - transport specific
            logger.debug("websocket for %s ended: %s", client.id, exc)
        finally:
            writer.cancel()
            await hub.remove_client(client)

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    else:  # pragma: no cover - only when the package is installed without the web assets
        logger.warning("web assets not found at %s", WEB_DIR)

    return app
