"""Client registry, frame fan-out and control arbitration.

One background task captures frames and pushes them into a small bounded queue
per client. When a client cannot keep up the *oldest* frame is dropped, which
keeps latency flat instead of letting it grow without bound.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .capture import Frame, pack_frame

logger = logging.getLogger(__name__)

#: A client that holds control but sends nothing for this long loses it.
CONTROL_IDLE_TIMEOUT = 30.0


class Client:
    """A connected viewer, optionally holding the input lock."""

    def __init__(self, client_id: str, websocket, queue_size: int = 3) -> None:
        self.id = client_id
        self.websocket = websocket
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)
        self.has_control = False
        self.alive = True
        self.connected_at = time.monotonic()
        self.last_activity = time.monotonic()
        self.frames_sent = 0
        self.frames_dropped = 0

    def offer(self, payload: bytes) -> None:
        """Queue a frame, discarding the oldest one when the client lags."""
        try:
            self.queue.put_nowait(payload)
            return
        except asyncio.QueueFull:
            pass
        try:
            self.queue.get_nowait()
            self.frames_dropped += 1
        except asyncio.QueueEmpty:  # pragma: no cover - race with the writer
            pass
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:  # pragma: no cover - race with the writer
            self.frames_dropped += 1


class SessionHub:
    """Owns the capture loop, the connected clients and the control lock."""

    def __init__(self, settings, capture=None, controller=None) -> None:
        self.settings = settings
        self.capture = capture
        self.controller = controller
        self.clients: dict[str, Client] = {}
        self.capture_error: str | None = None
        self._sequence = 0
        self._counter = 0
        self._running = False
        self._capture_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None
        self._control_owner: str | None = None
        self._last_frame: Frame | None = None

    # -- introspection ---------------------------------------------------
    @property
    def streaming(self) -> bool:
        return self._running

    @property
    def control_owner(self) -> str | None:
        return self._control_owner

    @property
    def last_frame(self) -> Frame | None:
        return self._last_frame

    def screen_size(self) -> tuple[int, int]:
        if self.capture is None:
            return (0, 0)
        try:
            return self.capture.screen_size()
        except Exception:  # pragma: no cover - backend specific
            return (0, 0)

    def monitors(self) -> list[dict]:
        if self.capture is None:
            return []
        try:
            return self.capture.monitors()
        except Exception:  # pragma: no cover - backend specific
            return []

    def status(self) -> dict:
        width, height = self.screen_size()
        return {
            "ok": self.capture is not None,
            "clients": len(self.clients),
            "max_clients": self.settings.max_clients,
            "streaming": self._running,
            "fps": self.settings.fps,
            "quality": self.settings.quality,
            "max_width": self.settings.max_width,
            "monitor": self.settings.monitor,
            "monitors": self.monitors(),
            "screen": {"width": width, "height": height},
            "allow_input": self.settings.allow_input,
            "auto_control": self.settings.auto_control,
            "control": self._control_owner,
            "frames": self._sequence,
            "input": (
                None
                if self.controller is None
                else {
                    "enabled": self.controller.enabled,
                    "applied": self.controller.applied,
                    "dropped": self.controller.dropped,
                }
            ),
            "capture_error": self.capture_error,
        }

    def hello(self, client: Client) -> dict:
        width, height = self.screen_size()
        return {
            "t": "hello",
            "client": client.id,
            "width": width,
            "height": height,
            "fps": self.settings.fps,
            "quality": self.settings.quality,
            "max_width": self.settings.max_width,
            "monitor": self.settings.monitor,
            "monitors": self.monitors(),
            "allow_input": self.settings.allow_input,
            "control": client.has_control,
            "control_owner": self._control_owner,
        }

    # -- client lifecycle ------------------------------------------------
    async def add_client(self, websocket) -> Client | None:
        if len(self.clients) >= self.settings.max_clients:
            return None
        self._counter += 1
        client = Client(f"c{self._counter}", websocket, self.settings.queue_size)
        self.clients[client.id] = client
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
            self._idle_task = None
        if self.settings.auto_control and self._control_owner is None:
            self.request_control(client)
        self._ensure_capture()
        logger.info("client %s connected (%d total)", client.id, len(self.clients))
        return client

    async def remove_client(self, client: Client) -> None:
        self.clients.pop(client.id, None)
        if self._control_owner == client.id:
            self.release_control(client)
        if self.controller is not None:
            self.controller.release_all()
        logger.info("client %s disconnected (%d left)", client.id, len(self.clients))
        if not self.clients:
            self._schedule_idle_stop()

    async def broadcast_text(self, message: dict) -> None:
        for client in list(self.clients.values()):
            try:
                await client.websocket.send_json(message)
            except Exception:  # pragma: no cover - client already gone
                client.alive = False

    # -- control lock ----------------------------------------------------
    def request_control(self, client: Client) -> bool:
        if not self.settings.allow_input:
            return False
        owner = self.clients.get(self._control_owner) if self._control_owner else None
        if owner is not None and owner is not client:
            if time.monotonic() - owner.last_activity < CONTROL_IDLE_TIMEOUT:
                return False
            owner.has_control = False
        self._control_owner = client.id
        client.has_control = True
        client.last_activity = time.monotonic()
        return True

    def release_control(self, client: Client) -> None:
        client.has_control = False
        if self._control_owner == client.id:
            self._control_owner = None

    def touch_control(self, client: Client) -> None:
        client.last_activity = time.monotonic()

    # -- capture loop ----------------------------------------------------
    def _ensure_capture(self) -> None:
        if self.capture is None:
            return
        if self._capture_task is None or self._capture_task.done():
            self._capture_task = asyncio.create_task(self._capture_loop())

    def _schedule_idle_stop(self) -> None:
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._idle_stop())

    async def _idle_stop(self) -> None:
        try:
            await asyncio.sleep(self.settings.idle_timeout)
        except asyncio.CancelledError:
            return
        if not self.clients:
            logger.info("no clients for %.0fs, pausing capture", self.settings.idle_timeout)
            self._running = False

    async def _capture_loop(self) -> None:
        interval = 1.0 / self.settings.fps
        self._running = True
        next_tick = time.monotonic()
        try:
            while self._running:
                next_tick += interval
                try:
                    frame = await asyncio.to_thread(self.capture.grab)
                except Exception as exc:
                    self.capture_error = str(exc)
                    logger.warning("capture failed: %s", exc)
                    await asyncio.sleep(1.0)
                    next_tick = time.monotonic()
                    continue
                self.capture_error = None
                self._sequence += 1
                self._last_frame = frame
                payload = pack_frame(self._sequence, frame.data)
                for client in list(self.clients.values()):
                    client.offer(payload)
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_tick = time.monotonic()
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False
            self._capture_task = None

    async def writer_loop(self, client: Client) -> None:
        """Drain a client's queue onto its socket."""
        try:
            while True:
                payload = await client.queue.get()
                await client.websocket.send_bytes(payload)
                client.frames_sent += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("writer for %s stopped: %s", client.id, exc)
            client.alive = False

    async def shutdown(self) -> None:
        self._running = False
        for task in (self._capture_task, self._idle_task):
            if task is not None and not task.done():
                task.cancel()
        if self.controller is not None:
            self.controller.release_all()
        if self.capture is not None:
            self.capture.close()
