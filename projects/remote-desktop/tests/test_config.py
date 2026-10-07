"""Tests for :mod:`remote_desktop.config`."""

from __future__ import annotations

import argparse

import pytest

from remote_desktop.config import Settings, clamp


def test_defaults_without_environment(monkeypatch):
    for name in list(Settings.__dataclass_fields__):
        monkeypatch.delenv(f"RD_{name.upper()}", raising=False)
    settings = Settings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8765
    assert settings.token == ""
    assert settings.allow_input is True
    assert settings.auto_control is True
    assert settings.max_viewers == 4


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("RD_PORT", "9000")
    monkeypatch.setenv("RD_TOKEN", "secret")
    monkeypatch.setenv("RD_ALLOW_INPUT", "off")
    monkeypatch.setenv("RD_MAX_VIEWERS", "7")
    monkeypatch.setenv("RD_ROOM_TTL", "45.5")
    settings = Settings()
    assert settings.port == 9000
    assert settings.token == "secret"
    assert settings.allow_input is False
    assert settings.max_viewers == 7
    assert settings.room_ttl == 45.5


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " off "])
def test_falsey_booleans(monkeypatch, raw):
    monkeypatch.setenv("RD_ALLOW_INPUT", raw)
    assert Settings().allow_input is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "anything"])
def test_truthy_booleans(monkeypatch, raw):
    monkeypatch.setenv("RD_ALLOW_INPUT", raw)
    assert Settings().allow_input is True


def test_invalid_numbers_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("RD_PORT", "not-a-number")
    monkeypatch.setenv("RD_ROOM_TTL", "soon")
    settings = Settings()
    assert settings.port == 8765
    assert settings.room_ttl == 120.0


def test_empty_environment_value_uses_default(monkeypatch):
    monkeypatch.setenv("RD_HOST", "")
    assert Settings().host == "0.0.0.0"


def test_clamp_bounds_values():
    settings = Settings(port=99999, max_viewers=0, room_ttl=1.0, input_rate=-5.0).clamp()
    assert settings.port == 65535
    assert settings.max_viewers == 1
    assert settings.room_ttl == 10.0
    assert settings.input_rate == 0.0


def test_clamp_helper():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10


def test_resolve_token_generates_once():
    settings = Settings(token="")
    first = settings.resolve_token()
    assert first
    assert settings.resolve_token() == first


def test_resolve_token_keeps_configured_value():
    settings = Settings(token="keep-me")
    assert settings.resolve_token() == "keep-me"


def test_from_args_overrides_only_given_options():
    args = argparse.Namespace(
        host=None,
        port=1234,
        token=None,
        log_level=None,
        public_url=None,
        no_input=True,
        no_auto_control=False,
        open_browser=True,
    )
    settings = Settings.from_args(args)
    assert settings.port == 1234
    assert settings.host == "0.0.0.0"
    assert settings.allow_input is False
    assert settings.auto_control is True
    assert settings.open_browser is True


def test_from_args_tolerates_missing_attributes():
    settings = Settings.from_args(argparse.Namespace())
    assert settings.port == 8765
    assert settings.allow_input is True
