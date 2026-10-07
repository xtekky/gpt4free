"""Tests for :mod:`remote_desktop.input_bridge`."""

from __future__ import annotations

import pytest

from remote_desktop.input_bridge import (
    SPECIAL_KEYS,
    InputBridge,
    clamp01,
    resolve_key,
    virtual_screen_size,
)


def test_clamp01_coerces_and_bounds():
    assert clamp01(0.5) == 0.5
    assert clamp01(-3) == 0.0
    assert clamp01(9) == 1.0
    assert clamp01("0.25") == 0.25
    assert clamp01(None) == 0.0
    assert clamp01("nope") == 0.0


def test_resolve_key_maps_specials(fake_backend):
    assert resolve_key(fake_backend, "Enter") == "KEY:enter"
    assert resolve_key(fake_backend, "ArrowLeft") == "KEY:left"
    assert resolve_key(fake_backend, "F5") == "KEY:f5"
    assert resolve_key(fake_backend, " ") == "KEY:space"


def test_resolve_key_passes_single_characters(fake_backend):
    assert resolve_key(fake_backend, "a") == "a"
    assert resolve_key(fake_backend, "7") == "7"


def test_resolve_key_rejects_unknown_names(fake_backend):
    assert resolve_key(fake_backend, "Unidentified") is None
    assert resolve_key(fake_backend, None) is None


def test_special_keys_cover_common_browser_names():
    for name in ("enter", "backspace", "tab", "escape", "arrowup", "delete", "control", "meta"):
        assert name in SPECIAL_KEYS


def test_virtual_screen_size_is_none_off_windows():
    size = virtual_screen_size()
    assert size is None or (size[0] > 0 and size[1] > 0)


def test_disabled_bridge_ignores_events(fake_backend):
    bridge = InputBridge(enabled=False, backend=fake_backend)
    assert bridge.available is False
    assert bridge.handle({"type": "move", "x": 0.5, "y": 0.5}) is False
    assert bridge.events == 0


def test_move_scales_to_screen(fake_backend):
    bridge = InputBridge(rate=0.0, backend=fake_backend)
    bridge._screen = (1000, 500)
    assert bridge.handle({"type": "move", "x": 0.5, "y": 1.0}) is True
    assert fake_backend.mouse.position == (500, 499)


def test_move_uses_reported_screen_when_unknown(fake_backend):
    bridge = InputBridge(rate=0.0, backend=fake_backend)
    bridge.handle({"type": "move", "x": 0.0, "y": 0.0, "screen": {"width": 800, "height": 600}})
    assert fake_backend.mouse.position == (0, 0)
    assert bridge._screen == (800, 600)


def test_move_rate_limit_drops_bursts(fake_backend):
    bridge = InputBridge(rate=1.0, backend=fake_backend)
    bridge._screen = (100, 100)
    bridge.handle({"type": "move", "x": 0.1, "y": 0.1})
    bridge.handle({"type": "move", "x": 0.9, "y": 0.9})
    assert fake_backend.mouse.position == (10, 10)


def test_button_click_press_and_release(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    bridge.handle({"type": "button", "action": "click", "button": "left"})
    bridge.handle({"type": "button", "action": "down", "button": "right"})
    bridge.handle({"type": "button", "action": "up", "button": "right"})
    assert fake_backend.mouse.clicks == ["BUTTON:left"]
    assert fake_backend.mouse.pressed == ["BUTTON:right"]
    assert fake_backend.mouse.released == ["BUTTON:right"]


def test_button_unknown_name_falls_back_to_left(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    bridge.handle({"type": "button", "action": "click", "button": "back"})
    assert fake_backend.mouse.clicks == ["BUTTON:left"]


def test_scroll_passes_deltas(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    bridge.handle({"type": "scroll", "dx": 1.5, "dy": -2})
    assert fake_backend.mouse.scrolls == [(1.5, -2.0)]


def test_scroll_ignores_garbage(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    assert bridge.handle({"type": "scroll", "dx": "x", "dy": None}) is False
    assert fake_backend.mouse.scrolls == []


def test_key_click_presses_and_releases(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    bridge.handle({"type": "key", "action": "click", "key": "Enter"})
    assert fake_backend.keyboard.pressed == ["KEY:enter"]
    assert fake_backend.keyboard.released == ["KEY:enter"]


def test_key_down_and_up(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    bridge.handle({"type": "key", "action": "down", "key": "Shift"})
    bridge.handle({"type": "key", "action": "up", "key": "Shift"})
    assert fake_backend.keyboard.pressed == ["KEY:shift"]
    assert fake_backend.keyboard.released == ["KEY:shift"]


def test_key_unknown_is_ignored(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    assert bridge.handle({"type": "key", "action": "click", "key": "Unidentified"}) is False

def test_combo_presses_in_order_and_releases_in_reverse(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    assert bridge.handle({"type": "key", "action": "combo", "keys": ["Alt", "Tab"]}) is True
    assert fake_backend.keyboard.pressed == ["KEY:alt", "KEY:tab"]
    assert fake_backend.keyboard.released == ["KEY:tab", "KEY:alt"]

def test_combo_supports_three_keys_and_characters(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    assert bridge.handle({"type": "key", "action": "combo", "keys": ["Control", "Shift", "t"]}) is True
    assert fake_backend.keyboard.pressed == ["KEY:ctrl", "KEY:shift", "t"]
    assert fake_backend.keyboard.released == ["t", "KEY:shift", "KEY:ctrl"]

def test_combo_rejects_empty_and_missing_keys(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    assert bridge.handle({"type": "key", "action": "combo", "keys": []}) is False
    assert bridge.handle({"type": "key", "action": "combo"}) is False
    assert bridge.handle({"type": "key", "action": "combo", "keys": "Alt+Tab"}) is False
    assert fake_backend.keyboard.pressed == []

def test_combo_rejects_unknown_key_without_pressing_anything(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    assert bridge.handle({"type": "key", "action": "combo", "keys": ["Alt", "Unidentified"]}) is False
    assert fake_backend.keyboard.pressed == []
    assert fake_backend.keyboard.released == []

def test_combo_counts_as_one_event(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    bridge.handle({"type": "key", "action": "combo", "keys": ["Meta", "d"]})
    assert bridge.events == 1


def test_text_is_typed(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    assert bridge.handle({"type": "text", "text": "hello"}) is True
    assert fake_backend.keyboard.typed == ["hello"]


def test_unknown_event_type_is_ignored(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    assert bridge.handle({"type": "teleport"}) is False
    assert bridge.handle("not a dict") is False


def test_backend_errors_do_not_propagate(fake_backend):
    def boom(_text):
        raise RuntimeError("nope")

    fake_backend.keyboard.type = boom
    bridge = InputBridge(backend=fake_backend)
    assert bridge.handle({"type": "text", "text": "x"}) is False


def test_events_counter_increments(fake_backend):
    bridge = InputBridge(backend=fake_backend)
    bridge.handle({"type": "text", "text": "a"})
    bridge.handle({"type": "text", "text": "b"})
    assert bridge.events == 2


def test_available_true_with_injected_backend(fake_backend):
    assert InputBridge(backend=fake_backend).available is True


@pytest.mark.parametrize("rate", [0.0, 60.0, 240.0])
def test_rate_values_are_accepted(fake_backend, rate):
    bridge = InputBridge(rate=rate, backend=fake_backend)
    assert bridge.handle({"type": "move", "x": 0.5, "y": 0.5}) is True
