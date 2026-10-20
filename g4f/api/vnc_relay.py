"""
g4f — VNC relay: serve the noVNC web client and bridge it to a VNC server.

The sandbox desktop runs a VNC server (TigerVNC/Xvnc) that speaks raw TCP,
while browsers talk WebSocket; websockify normally bridges the two. This
module folds the same bridge into the g4f API server, so the desktop and the
API share a single port:

    /desktop/              the noVNC client, served from G4F_NOVNC_DIR
    /desktop/websockify    the WebSocket <-> VNC-server TCP bridge noVNC connects to

Both routes are the image's only bridge: the sandbox image installs neither
the Debian ``novnc`` package nor websockify, and its client files come from the
upstream noVNC release. Both routes are opt-in and disabled unless
``G4F_VNC_PORT`` is set (the VNC server's TCP port). ``G4F_VNC_HOST`` defaults
to ``127.0.0.1`` and ``G4F_NOVNC_DIR`` to ``/usr/share/novnc``. The bridge only
forwards bytes, so it works with any RFB-speaking server.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse

logger = logging.getLogger(__name__)

#: Read size for the VNC-server -> browser direction.
RELAY_CHUNK_SIZE = 1 << 16


def vnc_relay_config() -> Optional[Tuple[str, int, str]]:
    """
    Return ``(host, port, novnc_dir)`` when the relay is enabled, else ``None``.

    Enabled by ``G4F_VNC_PORT``; a missing or non-numeric value disables it.
    """
    raw_port = os.environ.get("G4F_VNC_PORT")
    if not raw_port:
        return None
    try:
        port = int(raw_port)
    except ValueError:
        logger.warning(
            "G4F_VNC_PORT=%r is not a number; the VNC relay stays disabled",
            raw_port,
        )
        return None
    host = os.environ.get("G4F_VNC_HOST", "127.0.0.1")
    novnc_dir = os.environ.get("G4F_NOVNC_DIR", "/usr/share/novnc")
    return host, port, novnc_dir


def register_vnc_routes(app: FastAPI) -> None:
    """
    Mount the noVNC client and the WebSocket bridge on ``app`` when enabled.

    Called by :func:`g4f.api.create_app`; a no-op unless ``G4F_VNC_PORT`` is
    set. The mounts are path-scoped (``/desktop``,
    ``/desktop/websockify``), so they never shadow the API or the optional GUI
    at ``/``.
    """
    config = vnc_relay_config()
    if config is None:
        return
    host, port, novnc_dir = config

    # The websocket route must be registered before the /desktop mount:
    # Starlette matches routes in order and a Mount matches websocket scopes by
    # path prefix too, so the mount would swallow /desktop/websockify and
    # StaticFiles would reject the handshake with 403.
    @app.websocket("/desktop/websockify")
    async def websockify(websocket: WebSocket) -> None:
        """Bridge one browser WebSocket to the VNC server's TCP port."""
        await websocket.accept()
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except OSError as e:
            logger.warning("VNC relay: connect to %s:%s failed: %s", host, port, e)
            await websocket.close(code=1011)
            return

        async def browser_to_vnc() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    data = message.get("bytes")
                    if data is None:
                        text = message.get("text")
                        if text is None:
                            continue
                        # noVNC's base64 sub-protocol sends text frames.
                        data = text.encode("utf-8", "surrogateescape")
                    writer.write(data)
                    await writer.drain()
            except (WebSocketDisconnect, RuntimeError):
                pass

        async def vnc_to_browser() -> None:
            try:
                while True:
                    data = await reader.read(RELAY_CHUNK_SIZE)
                    if not data:
                        break
                    await websocket.send_bytes(data)
            except (WebSocketDisconnect, RuntimeError):
                pass

        tasks = [
            asyncio.create_task(browser_to_vnc()),
            asyncio.create_task(vnc_to_browser()),
        ]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    if os.path.isdir(novnc_dir):
        @app.api_route("/desktop", methods=["GET"])
        async def serve_desktop():
            return RedirectResponse(url="/desktop/")

        from fastapi.staticfiles import StaticFiles

        app.mount(
            "/desktop",
            StaticFiles(directory=novnc_dir, html=True),
            name="desktop",
        )
        logger.info("Serving noVNC from %s at /desktop/", novnc_dir)
    else:
        logger.warning(
            "G4F_VNC_PORT is set but noVNC directory %s does not exist; "
            "the client is not served",
            novnc_dir,
        )
