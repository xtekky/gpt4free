"""Input mapping, throttling and cleanup."""

from __future__ import annotations

import pytest

from remote_desktop.input import HAS_INPUT, InputController

pytestmark = pytest.mark.skipif(not HAS_INPUT, reason="pynput is not installed")


class RecordingMouse:
    def __init__(self):
        self.position = (0, 0)
        self.presses = []
        self.releases = []
        self.clicks = []
        self.scrolls = []

    def press(self, button):
        self.presses.append(button)

    def release(self, button):
        self.releases.append(button)

    def click(self, button, clicks):
        self.clicks.append((button, clicks))

    def scroll(self, dx, dy):
        self.scrolls.append((dx, dy))


class RecordingKeyboard:
    def __init__(self):
        self.presses = []
        self.releases = []
        self.typed = []

    def press(self, key):
        self.presses.append(key)

    def release(self, key):
        self.releases.append(key)

    def type(self, text):
        self.typed.append(text)


@pytest.fixture
def controller():
    instance = InputController(screen_size=lambda: (1000, 500), rate_limit=0)
    instance._mouse = RecordingMouse()
    instance._keyboard = RecordingKeyboard()
    return instance


def test_normalised_coordinates_map_to_pixels(controller):
    controller.handle({"t": "mouse", "action": "move", "x": 0.5, "y": 0.5})
    assert controller._mouse.position == (500, 250)


def test_coordinates_are_clamped(controller):
    controller.handle({"t": "mouse", "action": "move", "x": 5.0, "y": -3.0})
    assert controller._mouse.position == (999, 0)


def test_click_and_scroll(controller):
    controller.handle({"t": "mouse", "action": "click", "button": "right", "clicks": 2})
    controller.handle({"t": "mouse", "action": "scroll", "dx": 0, "dy": -3})
    assert len(controller._mouse.clicks) == 1
    assert controller._mouse.clicks[0][1] == 2
    assert controller._mouse.scrolls == [(0, -3)]


def test_unknown_button_is_ignored(controller):
    assert controller.handle({"t": "mouse", "action": "click", "button": "sideways"}) is False
    assert controller._mouse.clicks == []


def test_named_keys_are_mapped(controller):
    controller.handle({"t": "key", "action": "press", "key": "ArrowUp"})
    assert len(controller._keyboard.presses) == 1
    assert controller._keyboard.presses[0] == controller._keyboard.releases[0]


def test_single_character_keys_pass_through(controller):
    controller.handle({"t": "key", "action": "press", "key": "a"})
    assert controller._keyboard.presses == ["a"]


def test_unknown_key_is_ignored(controller):
    assert controller.handle({"t": "key", "action": "press", "key": "Hyper"}) is False
    assert controller._keyboard.presses == []


def test_text_is_typed_and_clipped(controller):
    controller.handle({"t": "text", "text": "hello"})
    assert controller._keyboard.typed == ["hello"]

    controller.handle({"t": "text", "text": "x" * 5000})
    assert len(controller._keyboard.typed[1]) == 1024


def test_empty_text_is_ignored(controller):
    assert controller.handle({"t": "text", "text": ""}) is False


def test_release_all_clears_held_state(controller):
    controller.handle({"t": "key", "action": "down", "key": "Shift"})
    controller.handle({"t": "mouse", "action": "down", "button": "left"})
    controller.release_all()
    assert controller._keyboard.releases
    assert controller._mouse.releases
    assert controller._pressed_keys == set()
    assert controller._pressed_buttons == set()


def test_disabled_controller_ignores_everything(controller):
    controller.enabled = False
    assert controller.handle({"t": "mouse", "action": "move", "x": 0.1, "y": 0.1}) is False
    assert controller.applied == 0


def test_move_throttling_counts_drops():
    controller = InputController(screen_size=lambda: (100, 100), rate_limit=0.0001)
    controller._mouse = RecordingMouse()
    controller._keyboard = RecordingKeyboard()
    for _ in range(50):
        controller.handle({"t": "mouse", "action": "move", "x": 0.5, "y": 0.5})
    assert controller.dropped > 0
    assert controller.applied < 50


def test_unknown_message_type_is_ignored(controller):
    assert controller.handle({"t": "teleport"}) is False
