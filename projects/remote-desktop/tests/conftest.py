"""Shared fixtures: a fake pynput backend so input tests never touch the OS."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote_desktop.input_bridge import Backend  # noqa: E402


class FakeMouse:
    def __init__(self) -> None:
        self.position = None
        self.pressed: list = []
        self.released: list = []
        self.clicks: list = []
        self.scrolls: list = []

    def press(self, button) -> None:
        self.pressed.append(button)

    def release(self, button) -> None:
        self.released.append(button)

    def click(self, button) -> None:
        self.clicks.append(button)

    def scroll(self, dx, dy) -> None:
        self.scrolls.append((dx, dy))


class FakeKeyboard:
    def __init__(self) -> None:
        self.typed: list[str] = []
        self.pressed: list = []
        self.released: list = []

    def type(self, text: str) -> None:
        self.typed.append(text)

    def press(self, key) -> None:
        self.pressed.append(key)

    def release(self, key) -> None:
        self.released.append(key)


class FakeButton:
    left = "BUTTON:left"
    right = "BUTTON:right"
    middle = "BUTTON:middle"


class FakeKey:
    """Any attribute resolves to a stable marker, like pynput's ``Key`` enum."""

    def __getattr__(self, name: str) -> str:
        return f"KEY:{name}"


@pytest.fixture
def fake_backend() -> Backend:
    return Backend(FakeMouse(), FakeKeyboard(), FakeButton, FakeKey())
