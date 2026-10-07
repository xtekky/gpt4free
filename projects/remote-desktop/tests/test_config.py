"""Tests for :mod:`remote_desktop.config`."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac

import pytest

from remote_desktop.config import Settings, clamp, turn_credentials


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


def test_ice_servers_defaults_to_stun_only(monkeypatch):
    monkeypatch.delenv("RD_ICE_SERVERS", raising=False)
    monkeypatch.delenv("RD_TURN_URL", raising=False)
    servers = Settings().ice_servers()
    assert len(servers) == 1
    assert servers[0]["urls"] == ["stun:stun.l.google.com:19302"]
    assert "credential" not in servers[0]


def test_ice_servers_adds_turn_with_credentials(monkeypatch):
    monkeypatch.delenv("RD_ICE_SERVERS", raising=False)
    monkeypatch.setenv("RD_TURN_URL", "turn:turn.example.com:3478,turns:turn.example.com:5349")
    monkeypatch.setenv("RD_TURN_SECRET", "s3cret")
    servers = Settings().ice_servers()
    assert servers[0]["urls"] == ["stun:stun.l.google.com:19302"]
    turn = servers[1]
    assert turn["urls"] == ["turn:turn.example.com:3478", "turns:turn.example.com:5349"]
    assert turn["username"].split(":")[0].isdigit()
    assert turn["credential"]


def test_ice_servers_without_secret_has_no_credentials(monkeypatch):
    monkeypatch.delenv("RD_ICE_SERVERS", raising=False)
    monkeypatch.setenv("RD_TURN_URL", "turn:turn.example.com:3478")
    monkeypatch.delenv("RD_TURN_SECRET", raising=False)
    turn = Settings().ice_servers()[1]
    assert "username" not in turn
    assert "credential" not in turn


def test_ice_servers_json_wins(monkeypatch):
    monkeypatch.setenv("RD_ICE_SERVERS", '[{"urls": ["turn:custom:3478"], "username": "u", "credential": "c"}]')
    monkeypatch.setenv("RD_TURN_URL", "turn:ignored:3478")
    servers = Settings().ice_servers()
    assert servers == [{"urls": ["turn:custom:3478"], "username": "u", "credential": "c"}]


def test_ice_servers_ignores_malformed_json(monkeypatch):
    monkeypatch.setenv("RD_ICE_SERVERS", "{not json")
    monkeypatch.delenv("RD_TURN_URL", raising=False)
    assert Settings().ice_servers()[0]["urls"] == ["stun:stun.l.google.com:19302"]


def test_ice_servers_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RD_ICE_SERVERS", "[]")
    assert Settings().ice_servers() == []


def test_ice_servers_uses_custom_stun(monkeypatch):
    monkeypatch.delenv("RD_ICE_SERVERS", raising=False)
    monkeypatch.setenv("RD_STUN_URL", "stun:stun.example.com:3478,stun:stun2.example.com:3478")
    monkeypatch.delenv("RD_TURN_URL", raising=False)
    assert Settings().ice_servers() == [
        {"urls": ["stun:stun.example.com:3478", "stun:stun2.example.com:3478"]}
    ]


def test_turn_credentials_are_time_limited_and_signed():
    username, credential = turn_credentials("topsecret", 600, now=1_000_000.0)
    expiry, _, nonce = username.partition(":")
    assert int(expiry) == 1_000_600
    assert nonce
    expected = base64.b64encode(
        hmac.new(b"topsecret", username.encode("utf-8"), hashlib.sha1).digest()
    ).decode("ascii")
    assert credential == expected


def test_turn_credentials_change_per_call():
    assert turn_credentials("s", 60) != turn_credentials("s", 60)


def test_clamp_bounds_turn_ttl():
    assert Settings(turn_ttl=1).clamp().turn_ttl == 60
    assert Settings(turn_ttl=999_999).clamp().turn_ttl == 86400
