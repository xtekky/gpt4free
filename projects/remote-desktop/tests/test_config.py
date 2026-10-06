"""Settings defaults, environment overrides and clamping."""

from __future__ import annotations

import argparse

import pytest

from remote_desktop.config import Settings


def test_defaults_are_sane():
    settings = Settings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8765
    assert settings.allow_input is True
    assert settings.auto_control is True
    assert settings.token == ""


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RD_PORT", "9000")
    monkeypatch.setenv("RD_FPS", "24")
    monkeypatch.setenv("RD_ALLOW_INPUT", "false")
    monkeypatch.setenv("RD_INPUT_RATE", "60.5")

    settings = Settings()
    assert settings.port == 9000
    assert settings.fps == 24
    assert settings.allow_input is False
    assert settings.input_rate == 60.5


def test_env_values_that_do_not_parse_fall_back(monkeypatch):
    monkeypatch.setenv("RD_PORT", "not-a-number")
    monkeypatch.setenv("RD_INPUT_RATE", "nope")
    settings = Settings()
    assert settings.port == 8765
    assert settings.input_rate == 120.0


def test_clamp_bounds_every_field():
    settings = Settings(
        fps=999,
        quality=1,
        max_width=10,
        max_clients=0,
        queue_size=99,
        input_rate=-5.0,
        idle_timeout=0.0,
    ).clamp()
    assert settings.fps == 60
    assert settings.quality == 10
    assert settings.max_width == 160
    assert settings.max_clients == 1
    assert settings.queue_size == 16
    assert settings.input_rate == 0.0
    assert settings.idle_timeout == 5.0


def test_resolve_token_generates_once():
    settings = Settings()
    first = settings.resolve_token()
    assert first
    assert settings.resolve_token() == first


def test_resolve_token_keeps_configured_value():
    settings = Settings(token="secret")
    assert settings.resolve_token() == "secret"


def test_from_args_applies_flags_and_negations():
    args = argparse.Namespace(
        host="127.0.0.1",
        port=1234,
        token="abc",
        fps=30,
        quality=80,
        max_width=1920,
        monitor=2,
        log_level="debug",
        no_input=True,
        no_auto_control=True,
        no_cursor=True,
    )
    settings = Settings.from_args(args)
    assert (settings.host, settings.port, settings.token) == ("127.0.0.1", 1234, "abc")
    assert (settings.fps, settings.quality, settings.max_width) == (30, 80, 1920)
    assert settings.monitor == 2
    assert settings.log_level == "debug"
    assert settings.allow_input is False
    assert settings.auto_control is False
    assert settings.with_cursor is False


def test_from_args_ignores_unset_options():
    args = argparse.Namespace(
        host=None,
        port=None,
        token=None,
        fps=None,
        quality=None,
        max_width=None,
        monitor=None,
        log_level=None,
        no_input=False,
        no_auto_control=False,
        no_cursor=False,
    )
    settings = Settings.from_args(args)
    assert settings.port == 8765
    assert settings.allow_input is True


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_env_bool_truthy(monkeypatch, value):
    monkeypatch.setenv("RD_ALLOW_INPUT", value)
    assert Settings().allow_input is True


@pytest.mark.parametrize("value", ["0", "false", "No", "off"])
def test_env_bool_falsy(monkeypatch, value):
    monkeypatch.setenv("RD_ALLOW_INPUT", value)
    assert Settings().allow_input is False
