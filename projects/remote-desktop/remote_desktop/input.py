"""Mouse and keyboard injection via ``pynput``.

The browser sends *normalised* pointer coordinates (0..1) so the mapping stays
correct no matter how far the streamed frame was downscaled.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from pynput.keyboard import Controller as KeyboardController
    from pynput.keyboard import Key
    from pynput.mouse import Button
    from pynput.mouse import Controller as MouseController

    HAS_INPUT = True
    INPUT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the host environment
    HAS_INPUT = False
    INPUT_IMPORT_ERROR = exc


class InputUnavailableError(RuntimeError):
    """Raised when synthetic input cannot be created on this machine."""


def _build_key_map() -> dict[str, object]:
    """Map browser ``KeyboardEvent.key`` names onto pynput keys."""
    names = {
        "enter": "enter",
        "return": "enter",
        "backspace": "backspace",
        "tab": "tab",
        "escape": "esc",
        "esc": "esc",
        "space": "space",
        "spacebar": "space",
        "delete": "delete",
        "del": "delete",
        "insert": "insert",
        "home": "home",
        "end": "end",
        "pageup": "page_up",
        "page_up": "page_up",
        "pagedown": "page_down",
        "page_down": "page_down",
        "arrowup": "up",
        "arrowdown": "down",
        "arrowleft": "left",
        "arrowright": "right",
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "control": "ctrl",
        "ctrl": "ctrl",
        "controlleft": "ctrl_l",
        "controlright": "ctrl_r",
        "shift": "shift",
        "shiftleft": "shift_l",
        "shiftright": "shift_r",
        "alt": "alt",
        "altleft": "alt_l",
        "altright": "alt_gr",
        "meta": "cmd",
        "os": "cmd",
        "super": "cmd",
        "win": "cmd",
        "capslock": "caps_lock",
        "numlock": "num_lock",
        "scrolllock": "scroll_lock",
        "printscreen": "print_screen",
        "pause": "pause",
        "menu": "menu",
    }
    for index in range(1, 21):
        names[f"f{index}"] = f"f{index}"

    mapping: dict[str, object] = {}
    for name, attribute in names.items():
        key = getattr(Key, attribute, None)
        if key is not None:
            mapping[name] = key
    return mapping


_KEY_MAP = _build_key_map() if HAS_INPUT else {}

_BUTTONS = (
    {"left": Button.left, "right": Button.right, "middle": Button.middle}
    if HAS_INPUT
    else {}
)


class InputController:
    """Applies mouse/keyboard events coming from a remote client."""

    def __init__(
        self,
        screen_size: Callable[[], tuple[int, int]],
        rate_limit: float = 120.0,
        enabled: bool = True,
    ) -> None:
        if not HAS_INPUT:
            raise InputUnavailableError(
                f"input injection needs 'pynput': {INPUT_IMPORT_ERROR}"
            )
        self._screen_size = screen_size
        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._lock = threading.RLock()
        self._min_interval = 1.0 / rate_limit if rate_limit > 0 else 0.0
        self._last_move = 0.0
        self._pressed_keys: set = set()
        self._pressed_buttons: set = set()
        self.enabled = bool(enabled)
        self.dropped = 0
        self.applied = 0

    # -- helpers ---------------------------------------------------------
    def _throttle(self) -> bool:
        if self._min_interval <= 0:
            return True
        now = time.monotonic()
        if now - self._last_move < self._min_interval:
            self.dropped += 1
            return False
        self._last_move = now
        return True

    def _to_pixels(self, message: dict) -> tuple[int, int]:
        width, height = self._screen_size()
        x = min(1.0, max(0.0, float(message.get("x", 0.0))))
        y = min(1.0, max(0.0, float(message.get("y", 0.0))))
        return int(round(x * max(0, width - 1))), int(round(y * max(0, height - 1)))

    def _resolve_key(self, name):
        if not name:
            return None
        if len(name) == 1:
            return name
        return _KEY_MAP.get(name.lower())

    # -- public API ------------------------------------------------------
    def handle(self, message: dict) -> bool:
        """Apply one client message. Returns ``True`` when it was applied."""
        if not self.enabled:
            return False
        kind = message.get("t")
        try:
            if kind == "mouse":
                return self._handle_mouse(message)
            if kind == "key":
                return self._handle_key(message)
            if kind == "text":
                return self._handle_text(message)
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("input event failed: %s", exc)
            return False
        return False

    def _handle_mouse(self, message: dict) -> bool:
        action = message.get("action")
        with self._lock:
            if action == "move":
                if not self._throttle():
                    return False
                self._mouse.position = self._to_pixels(message)
            elif action in ("down", "up"):
                button = _BUTTONS.get(message.get("button", "left"))
                if button is None:
                    return False
                if action == "down":
                    self._mouse.press(button)
                    self._pressed_buttons.add(button)
                else:
                    self._mouse.release(button)
                    self._pressed_buttons.discard(button)
            elif action == "click":
                button = _BUTTONS.get(message.get("button", "left"))
                if button is None:
                    return False
                clicks = max(1, min(3, int(message.get("clicks", 1))))
                self._mouse.click(button, clicks)
            elif action == "scroll":
                self._mouse.scroll(
                    int(message.get("dx", 0)), int(message.get("dy", 0))
                )
            else:
                return False
        self.applied += 1
        return True

    def _handle_key(self, message: dict) -> bool:
        key = self._resolve_key(message.get("key"))
        if key is None:
            return False
        action = message.get("action", "press")
        with self._lock:
            if action == "down":
                self._keyboard.press(key)
                self._pressed_keys.add(key)
            elif action == "up":
                self._keyboard.release(key)
                self._pressed_keys.discard(key)
            else:
                self._keyboard.press(key)
                self._keyboard.release(key)
        self.applied += 1
        return True

    def _handle_text(self, message: dict) -> bool:
        text = message.get("text") or ""
        if not isinstance(text, str) or not text:
            return False
        with self._lock:
            self._keyboard.type(text[:1024])
        self.applied += 1
        return True

    def release_all(self) -> None:
        """Release everything still held, so a disconnect cannot leave keys stuck."""
        with self._lock:
            for key in list(self._pressed_keys):
                try:
                    self._keyboard.release(key)
                except Exception:  # pragma: no cover - best effort
                    pass
            self._pressed_keys.clear()
            for button in list(self._pressed_buttons):
                try:
                    self._mouse.release(button)
                except Exception:  # pragma: no cover - best effort
                    pass
            self._pressed_buttons.clear()
