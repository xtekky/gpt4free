"""Stand-ins for the real capture and input backends, so tests need no display."""

from __future__ import annotations

import io
import time

from remote_desktop.capture import Frame


class FakeShot:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.bgra = bytes(width * height * 4)


class FakeGrabber:
    """Mimics the small part of the ``mss`` API the agent relies on."""

    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width = width
        self.height = height
        self.closed = False
        self.grabs = 0
        self.monitors = [
            {"left": 0, "top": 0, "width": width * 2, "height": height * 2},
            {"left": 0, "top": 0, "width": width, "height": height},
        ]

    def grab(self, monitor):
        self.grabs += 1
        return FakeShot(monitor["width"], monitor["height"])

    def close(self):
        self.closed = True


class FakeCapture:
    """A ScreenCapture replacement that returns a valid JPEG without a screen."""

    def __init__(self, width: int = 800, height: int = 600) -> None:
        from PIL import Image

        self.size = (width, height)
        self.closed = False
        self.grabs = 0
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), (20, 30, 40)).save(buffer, format="JPEG")
        self.jpeg = buffer.getvalue()

    def screen_size(self) -> tuple[int, int]:
        return self.size

    def monitors(self) -> list[dict]:
        return [
            {"left": 0, "top": 0, "width": self.size[0], "height": self.size[1]},
            {"left": 0, "top": 0, "width": self.size[0], "height": self.size[1]},
        ]

    def grab(self) -> Frame:
        self.grabs += 1
        return Frame(
            data=self.jpeg,
            width=self.size[0],
            height=self.size[1],
            timestamp=time.time(),
        )

    def close(self) -> None:
        self.closed = True


class FakeController:
    """Records input events instead of injecting them."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.applied = 0
        self.dropped = 0
        self.events: list[dict] = []
        self.releases = 0

    def handle(self, message: dict) -> bool:
        if not self.enabled:
            return False
        self.events.append(message)
        self.applied += 1
        return True

    def release_all(self) -> None:
        self.releases += 1
