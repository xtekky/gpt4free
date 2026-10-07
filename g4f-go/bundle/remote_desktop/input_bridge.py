"""Local input injection for the host page.

A browser cannot move the operating system pointer, so the host page forwards
normalised events to this bridge, which replays them with ``pynput``. The
bridge is only reachable from the machine running the server.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time

logger = logging.getLogger(__name__)

#: Browser ``KeyboardEvent.key`` names mapped to ``pynput.keyboard.Key`` members.
SPECIAL_KEYS = {
    "enter": "enter",
    "return": "enter",
    "backspace": "backspace",
    "tab": "tab",
    "escape": "esc",
    "esc": "esc",
    " ": "space",
    "space": "space",
    "spacebar": "space",
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "delete": "delete",
    "del": "delete",
    "insert": "insert",
    "home": "home",
    "end": "end",
    "pageup": "page_up",
    "pagedown": "page_down",
    "shift": "shift",
    "control": "ctrl",
    "ctrl": "ctrl",
    "alt": "alt",
    "altgraph": "alt_gr",
    "meta": "cmd",
    "os": "cmd",
    "capslock": "caps_lock",
    "numlock": "num_lock",
    "scrolllock": "scroll_lock",
    "printscreen": "print_screen",
    "pause": "pause",
    "contextmenu": "menu",
}
SPECIAL_KEYS.update({f"f{number}": f"f{number}" for number in range(1, 21)})

BUTTONS = ("left", "right", "middle")


class Backend:
    """Thin wrapper so tests can substitute a fake pynput."""

    def __init__(self, mouse, keyboard, button, key) -> None:
        self.mouse = mouse
        self.keyboard = keyboard
        self.button = button
        self.key = key


def virtual_screen_size() -> tuple[int, int] | None:
    """Best-effort size of the whole desktop in physical pixels."""
    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            width = int(user32.GetSystemMetrics(78))
            height = int(user32.GetSystemMetrics(79))
            if width > 0 and height > 0:
                return width, height
        except Exception:  # pragma: no cover - platform specific
            return None
    return None


def clamp01(value) -> float:
    """Coerce anything into a ``0.0..1.0`` float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def resolve_key(backend: Backend, name):
    """Turn a browser key name into something pynput understands."""
    if name is None:
        return None
    text = str(name)
    lowered = text.lower()
    if lowered in SPECIAL_KEYS:
        return getattr(backend.key, SPECIAL_KEYS[lowered], None)
    if len(text) == 1:
        return text
    return None


class InputBridge:
    """Replays normalised browser events on the local machine."""

    def __init__(self, enabled: bool = True, rate: float = 240.0, backend: Backend | None = None) -> None:
        self.enabled = enabled
        self.rate = rate
        self.events = 0
        self.error: str | None = None
        self._backend = backend
        self._screen: tuple[int, int] | None = None
        self._last_move = 0.0

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._backend is not None:
            return True
        return self._load() is not None

    def _load(self) -> Backend | None:
        if not self.enabled:
            return None
        if self._backend is not None:
            return self._backend
        try:
            from pynput import keyboard, mouse
        except Exception as exc:
            self.error = f"pynput unavailable: {exc}"
            logger.warning("input injection disabled: %s", self.error)
            return None
        self._backend = Backend(mouse.Controller(), keyboard.Controller(), mouse.Button, keyboard.Key)
        return self._backend

    def handle(self, event: dict) -> bool:
        """Apply one event; returns ``False`` when it was ignored or failed."""
        if not isinstance(event, dict):
            return False
        backend = self._load()
        if backend is None:
            return False
        kind = event.get("type")
        try:
            if kind == "move":
                self._move(backend, event)
            elif kind == "button":
                self._button(backend, event)
            elif kind == "scroll":
                self._scroll(backend, event)
            elif kind == "key":
                self._key(backend, event)
            elif kind == "text":
                backend.keyboard.type(str(event.get("text", "")))
            else:
                return False
        except Exception as exc:
            logger.debug("input event %r failed: %s", kind, exc)
            return False
        self.events += 1
        return True

    def _screen_size(self, event: dict) -> tuple[int, int]:
        if self._screen is None:
            self._screen = virtual_screen_size()
        if self._screen is None:
            reported = event.get("screen") or {}
            try:
                width = int(reported.get("width") or 0)
                height = int(reported.get("height") or 0)
            except (TypeError, ValueError):
                width = height = 0
            if width > 0 and height > 0:
                self._screen = (width, height)
        return self._screen or (1920, 1080)

    def _move(self, backend: Backend, event: dict) -> None:
        now = time.monotonic()
        if self.rate > 0 and now - self._last_move < 1.0 / self.rate:
            return
        self._last_move = now
        width, height = self._screen_size(event)
        x = clamp01(event.get("x")) * (width - 1)
        y = clamp01(event.get("y")) * (height - 1)
        backend.mouse.position = (round(x), round(y))

    def _button(self, backend: Backend, event: dict) -> None:
        name = str(event.get("button", "left")).lower()
        button = getattr(backend.button, name, None) or backend.button.left
        action = str(event.get("action", "click")).lower()
        if action == "down":
            backend.mouse.press(button)
        elif action == "up":
            backend.mouse.release(button)
        else:
            backend.mouse.click(button)

    def _scroll(self, backend: Backend, event: dict) -> None:
        dx = float(event.get("dx", 0.0))
        dy = float(event.get("dy", 0.0))
        backend.mouse.scroll(dx, dy)

    def _key(self, backend: Backend, event: dict) -> None:
        action = str(event.get("action", "click")).lower()
        if action == "combo":
            self._combo(backend, event)
            return
        key = resolve_key(backend, event.get("key"))
        if key is None:
            raise ValueError(f"unsupported key {event.get('key')!r}")
        if action == "down":
            backend.keyboard.press(key)
        elif action == "up":
            backend.keyboard.release(key)
        else:
            backend.keyboard.press(key)
            backend.keyboard.release(key)

    def _combo(self, backend: Backend, event: dict) -> None:
        """Press every key of a shortcut, then release them in reverse order."""
        names = event.get("keys")
        if not isinstance(names, (list, tuple)) or not names:
            raise ValueError("combo requires a non-empty key list")
        keys = []
        for name in names:
            key = resolve_key(backend, name)
            if key is None:
                raise ValueError(f"unsupported key {name!r} in combo")
            keys.append(key)
        for key in keys:
            backend.keyboard.press(key)
        for key in reversed(keys):
            backend.keyboard.release(key)
