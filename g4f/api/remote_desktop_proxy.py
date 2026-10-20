"""Reverse proxy for a locally running remote-desktop server.

``projects/remote-desktop`` serves a WebRTC screen-sharing app (host page,
viewer page, status API and a signaling WebSocket). Proxying it through the
g4f API lets the whole app be reached on the same origin and port as g4f:

    /remote_desktop            ->  http://127.0.0.1:8765/remote_desktop
    /remote_desktop/view       ->  http://127.0.0.1:8765/remote_desktop/view
    /remote_desktop/ws         ->  ws://127.0.0.1:8765/remote_desktop/ws

The target is configured with ``REMOTE_DESKTOP_BASE_URL`` or
``REMOTE_DESKTOP_HOST`` / ``REMOTE_DESKTOP_PORT`` and defaults to
``http://127.0.0.1:8765``. Only that configured instance is proxied, so the
route cannot be used as an open proxy.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .local_gateway_proxy import LocalGatewayProxy, env_root_url

logger = logging.getLogger(__name__)

PREFIX = "remote_desktop"

#: Path the remote-desktop app serves the host page on.
ROOT_PATH = PREFIX

#: Path of the signaling WebSocket on the remote-desktop app.
WS_PATH = f"/{PREFIX}/ws"


def remote_desktop_root_url() -> str:
    """Return the base URL of the local remote-desktop server."""
    return env_root_url(
        "REMOTE_DESKTOP_BASE_URL",
        "REMOTE_DESKTOP_HOST",
        "REMOTE_DESKTOP_PORT",
        "127.0.0.1",
        "8765",
        strip_v1=False,
    )


def _websocket_url(root_url: str) -> str:
    """Turn the HTTP root URL into the signaling WebSocket URL."""
    parsed = urlsplit(root_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, WS_PATH, "", ""))


_proxy = LocalGatewayProxy(
    PREFIX,
    remote_desktop_root_url,
    label="Remote desktop",
    target_prefix=ROOT_PATH,
)


async def _bridge_websocket(websocket: WebSocket, target_url: str) -> None:
    """Relay frames between the browser and the remote-desktop signaling socket."""
    await websocket.accept()
    session = aiohttp.ClientSession()
    try:
        upstream = await session.ws_connect(target_url, max_msg_size=0)
    except (aiohttp.ClientError, OSError) as e:
        logger.warning(
            "Remote desktop proxy: %s is not reachable: %s", target_url, e
        )
        await session.close()
        await websocket.close(code=1011)
        return

    async def browser_to_upstream() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    await upstream.send_bytes(data)
                    continue
                text = message.get("text")
                if text is not None:
                    await upstream.send_str(text)
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def upstream_to_browser() -> None:
        try:
            async for message in upstream:
                if message.type == aiohttp.WSMsgType.TEXT:
                    await websocket.send_text(message.data)
                elif message.type == aiohttp.WSMsgType.BINARY:
                    await websocket.send_bytes(message.data)
                elif message.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass

    tasks = [
        asyncio.create_task(browser_to_upstream()),
        asyncio.create_task(upstream_to_browser()),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await upstream.close()
        await session.close()


def register_remote_desktop_routes(app: FastAPI) -> None:
    """Register the ``/remote_desktop`` reverse proxy on ``app``."""

    @app.websocket(WS_PATH)
    async def remote_desktop_ws(websocket: WebSocket) -> None:
        await _bridge_websocket(websocket, _websocket_url(remote_desktop_root_url()))

    _proxy.register(app)