"""FastAPI application: static web app, WebRTC signaling and status API."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings
from .input_bridge import InputBridge
from .rooms import ROLE_HOST, ROLE_VIEWER, Peer, RoomError, RoomRegistry

logger = logging.getLogger(__name__)

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"

#: How often idle rooms are swept.
PRUNE_INTERVAL = 30.0


class Connection:
    """Serialises writes to one websocket so tasks cannot interleave frames."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self._lock = asyncio.Lock()

    async def send_json(self, message: dict) -> None:
        async with self._lock:
            await self.websocket.send_json(message)

    async def send_bytes(self, payload: bytes) -> None:
        async with self._lock:
            await self.websocket.send_bytes(payload)


def local_addresses() -> list[str]:
    """Every non-loopback IPv4 address this machine can be reached on."""
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:  # pragma: no cover - depends on host configuration
        pass
    # The UDP trick finds the address of the interface used for the default route.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            addresses.add(probe.getsockname()[0])
    except OSError:  # pragma: no cover - offline machines
        pass
    return sorted(addresses)


def share_urls(settings: Settings) -> list[str]:
    """URLs a phone can open to reach this server."""
    if settings.public_url:
        return [settings.public_url.rstrip("/")]
    hosts = local_addresses() or ["127.0.0.1"]
    return [f"http://{address}:{settings.port}" for address in hosts]


def qr_svg(payload: str) -> bytes:
    """Render ``payload`` as an SVG QR code."""
    import qrcode
    import qrcode.image.svg

    code = qrcode.QRCode(border=2, box_size=10)
    code.add_data(payload)
    code.make(fit=True)
    image = code.make_image(image_factory=qrcode.image.svg.SvgPathImage)
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer)
    return buffer.getvalue()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application."""
    settings = (settings or Settings()).clamp()
    registry = RoomRegistry(
        max_viewers=settings.max_viewers,
        room_ttl=settings.room_ttl,
        auto_control=settings.auto_control,
    )
    bridge = InputBridge(enabled=settings.allow_input, rate=settings.input_rate)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def sweeper() -> None:
            while True:
                await asyncio.sleep(PRUNE_INTERVAL)
                for room_id in registry.prune():
                    logger.info("pruned idle room %s", room_id)

        task = asyncio.create_task(sweeper())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Remote Desktop", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry
    app.state.bridge = bridge

    if WEB_ROOT.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_ROOT)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse("/host")

    @app.get("/host", include_in_schema=False)
    async def host_page() -> FileResponse:
        return FileResponse(WEB_ROOT / "host.html")

    @app.get("/view", include_in_schema=False)
    async def view_page() -> FileResponse:
        return FileResponse(WEB_ROOT / "view.html")

    @app.get("/manifest.webmanifest", include_in_schema=False)
    async def manifest() -> FileResponse:
        return FileResponse(WEB_ROOT / "manifest.webmanifest", media_type="application/manifest+json")

    @app.get("/sw.js", include_in_schema=False)
    async def service_worker() -> FileResponse:
        return FileResponse(WEB_ROOT / "sw.js", media_type="application/javascript")

    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse(
            {
                "version": __version__,
                "input": bridge.available,
                "input_error": bridge.error,
                "input_events": bridge.events,
                "allow_input": settings.allow_input,
                "auto_control": settings.auto_control,
                "max_viewers": settings.max_viewers,
                "urls": share_urls(settings),
                "ice_servers": settings.ice_servers(),
                **registry.stats(),
            }
        )

    @app.get("/api/qr.svg", include_in_schema=False)
    async def qr(url: str) -> Response:
        if len(url) > 512:
            raise HTTPException(status_code=400, detail="url too long")
        return Response(content=qr_svg(url), media_type="image/svg+xml")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        connection = Connection(websocket)
        peer = Peer(
            peer_id=secrets.token_hex(4),
            role="unknown",
            room_id="",
            send_json=connection.send_json,
            send_binary=connection.send_bytes,
        )
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is None:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await connection.send_json({"type": "error", "code": "bad-json", "message": "Malformed message"})
                    continue
                if not isinstance(payload, dict):
                    continue
                peer.touch()
                await _dispatch(payload, peer, registry, bridge, settings, connection)
        except WebSocketDisconnect:
            pass
        finally:
            await _cleanup(peer, registry)

    return app


async def _dispatch(
    message: dict,
    peer: Peer,
    registry: RoomRegistry,
    bridge: InputBridge,
    settings: Settings,
    connection: Connection,
) -> None:
    kind = message.get("type")
    if kind == "hello":
        await _handle_hello(message, peer, registry, settings, connection)
    elif kind == "signal":
        await _handle_signal(message, peer, registry)
    elif kind == "control":
        await _handle_control(message, peer, registry)
    elif kind == "input":
        await _handle_input(message, peer, registry, bridge)
    elif kind == "ping":
        await connection.send_json({"type": "pong", "t": message.get("t")})
    else:
        await connection.send_json({"type": "error", "code": "unknown-type", "message": f"Unknown type {kind!r}"})


async def _handle_hello(
    message: dict,
    peer: Peer,
    registry: RoomRegistry,
    settings: Settings,
    connection: Connection,
) -> None:
    if peer.role != "unknown":
        await connection.send_json({"type": "error", "code": "already-joined", "message": "Already joined"})
        return
    if settings.token and not secrets.compare_digest(str(message.get("token", "")), settings.token):
        await connection.send_json({"type": "error", "code": "bad-token", "message": "Invalid access token"})
        return

    role = message.get("role")
    if role == ROLE_HOST:
        room = registry.create_room(peer)
        peer.role = ROLE_HOST
        logger.info("room %s opened by host %s", room.id, peer.id)
        await connection.send_json(
            {
                "type": "room",
                "room": room.id,
                "peer": peer.id,
                "role": ROLE_HOST,
                "control": room.control_owner,
                "viewers": room.viewer_count,
            }
        )
        return

    if role == ROLE_VIEWER:
        room_id = str(message.get("room", "")).strip().upper()
        try:
            room = registry.join(room_id, peer)
        except RoomError as exc:
            await connection.send_json({"type": "error", "code": exc.code, "message": exc.message})
            return
        peer.role = ROLE_VIEWER
        logger.info("viewer %s joined room %s", peer.id, room.id)
        await connection.send_json(
            {
                "type": "joined",
                "room": room.id,
                "peer": peer.id,
                "role": ROLE_VIEWER,
                "host": room.host.id if room.host else None,
                "control": room.control_owner,
                "viewers": room.viewer_count,
            }
        )
        await _broadcast(
            room,
            {
                "type": "viewer-joined",
                "peer": peer.id,
                "viewers": room.viewer_count,
                "control": room.control_owner,
            },
            skip=peer.id,
        )
        return

    await connection.send_json({"type": "error", "code": "bad-role", "message": "role must be host or viewer"})


async def _handle_signal(message: dict, peer: Peer, registry: RoomRegistry) -> None:
    room = registry.get(peer.room_id)
    if room is None:
        return
    data = message.get("data")
    if not isinstance(data, dict):
        return
    target = message.get("to")
    envelope = {"type": "signal", "from": peer.id, "data": data}
    if target:
        recipient = room.get(str(target))
        if recipient is not None:
            await recipient.send(envelope)
        return
    await _broadcast(room, envelope, skip=peer.id)


async def _handle_control(message: dict, peer: Peer, registry: RoomRegistry) -> None:
    room = registry.get(peer.room_id)
    if room is None:
        return
    action = message.get("action")
    if action == "request":
        if registry.grant_control(room, peer.id):
            await _broadcast(room, {"type": "control", "owner": room.control_owner})
        else:
            await peer.send({"type": "control", "owner": room.control_owner, "denied": True})
    elif action == "release":
        if registry.release_control(room, peer.id):
            await _broadcast(room, {"type": "control", "owner": None})
    elif action == "grant" and peer.role == ROLE_HOST:
        target = str(message.get("target", ""))
        if registry.grant_control(room, target):
            await _broadcast(room, {"type": "control", "owner": room.control_owner})


async def _handle_input(message: dict, peer: Peer, registry: RoomRegistry, bridge: InputBridge) -> None:
    room = registry.get(peer.room_id)
    if room is None or room.control_owner != peer.id:
        return
    event = message.get("event")
    if isinstance(event, dict):
        bridge.handle(event)


async def _broadcast(room, message: dict, skip: str | None = None) -> None:
    for peer in room.peers:
        if peer.id == skip:
            continue
        await peer.send(message)


async def _cleanup(peer: Peer, registry: RoomRegistry) -> None:
    if not peer.room_id:
        return
    room = registry.leave(peer)
    if room is None:
        logger.info("room closed after %s left", peer.id)
        return
    await _broadcast(
        room,
        {"type": "peer-left", "peer": peer.id, "role": peer.role, "viewers": room.viewer_count, "control": room.control_owner},
    )
