"""Screen capture and JPEG encoding.

``mss`` handles the platform specific grabbing, Pillow does the encoding. Both
are imported defensively so the rest of the agent can still be imported on
machines without a display.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import mss
    from PIL import Image

    HAS_CAPTURE = True
    CAPTURE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the host environment
    HAS_CAPTURE = False
    CAPTURE_IMPORT_ERROR = exc

#: First byte of every binary websocket frame, so the client can tell frames
#: apart from anything else that might travel over the same socket.
FRAME_MAGIC = b"\x01"
FRAME_HEADER_SIZE = 5


class CaptureUnavailableError(RuntimeError):
    """Raised when the screen cannot be captured on this machine."""


@dataclass(frozen=True)
class Frame:
    """A single encoded screen frame."""

    data: bytes
    width: int
    height: int
    timestamp: float


def pack_frame(sequence: int, data: bytes) -> bytes:
    """Prefix a JPEG payload with the magic byte and a 32 bit sequence number."""
    return FRAME_MAGIC + (sequence & 0xFFFFFFFF).to_bytes(4, "big") + data


def unpack_frame(payload: bytes) -> tuple[int, bytes]:
    """Inverse of :func:`pack_frame`; returns ``(sequence, jpeg_bytes)``."""
    if len(payload) < FRAME_HEADER_SIZE or payload[:1] != FRAME_MAGIC:
        raise ValueError("not a frame payload")
    return int.from_bytes(payload[1:FRAME_HEADER_SIZE], "big"), payload[FRAME_HEADER_SIZE:]


class ScreenCapture:
    """Grabs a monitor and encodes it as JPEG.

    ``mss`` instances are not thread safe, so one grabber is kept per thread.
    """

    def __init__(
        self,
        monitor: int = 1,
        max_width: int = 1280,
        quality: int = 60,
        with_cursor: bool = True,
    ) -> None:
        if not HAS_CAPTURE:
            raise CaptureUnavailableError(
                f"screen capture needs 'mss' and 'pillow': {CAPTURE_IMPORT_ERROR}"
            )
        self.monitor = int(monitor)
        self.max_width = max(160, int(max_width))
        self.quality = max(10, min(95, int(quality)))
        self.with_cursor = bool(with_cursor)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._monitors: list[dict] | None = None

    def _grabber(self):
        grabber = getattr(self._local, "grabber", None)
        if grabber is None:
            try:
                grabber = mss.MSS(with_cursor=self.with_cursor)
            except (TypeError, ValueError):
                # Cursor compositing is not available on every backend.
                grabber = mss.MSS()
            self._local.grabber = grabber
        return grabber

    def monitors(self) -> list[dict]:
        """Monitor geometry, index 0 being the virtual "all screens" display."""
        with self._lock:
            if self._monitors is None:
                self._monitors = [dict(monitor) for monitor in self._grabber().monitors]
            return self._monitors

    def _monitor_index(self, monitors: list[dict]) -> int:
        if not 0 <= self.monitor < len(monitors):
            raise CaptureUnavailableError(
                f"monitor {self.monitor} is not available "
                f"(found {len(monitors) - 1} screen(s))"
            )
        return self.monitor

    def screen_size(self) -> tuple[int, int]:
        """Native pixel size of the captured monitor."""
        monitors = self.monitors()
        monitor = monitors[self._monitor_index(monitors)]
        return int(monitor["width"]), int(monitor["height"])

    def grab(self) -> Frame:
        """Capture the configured monitor and return it as a JPEG frame."""
        grabber = self._grabber()
        monitors = self.monitors()
        shot = grabber.grab(monitors[self._monitor_index(monitors)])
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        if image.width > self.max_width:
            ratio = self.max_width / image.width
            image = image.resize(
                (self.max_width, max(1, round(image.height * ratio))),
                Image.BILINEAR,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        return Frame(
            data=buffer.getvalue(),
            width=image.width,
            height=image.height,
            timestamp=time.time(),
        )

    def close(self) -> None:
        grabber = getattr(self._local, "grabber", None)
        if grabber is not None:
            try:
                grabber.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self._local.grabber = None
