"""Runtime configuration for the remote desktop server."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")

def clamp(value, low, high):
    """Constrain ``value`` to the inclusive ``low..high`` range."""
    return max(low, min(high, value))


def turn_credentials(secret: str, ttl: int, now: float | None = None) -> tuple[str, str]:
    """Mint a short-lived TURN username/password pair from a shared secret.

    This is the "TURN REST API" scheme coturn enables with ``use-auth-secret``:
    the username carries the expiry, and the password is its HMAC-SHA1 digest
    keyed with the shared secret. Nothing has to be stored server side.
    """
    expiry = int((time.time() if now is None else now) + ttl)
    username = f"{expiry}:{secrets.token_hex(4)}"
    digest = hmac.new(secret.encode("utf-8"), username.encode("utf-8"), hashlib.sha1).digest()
    return username, base64.b64encode(digest).decode("ascii")


@dataclass
class Settings:
    """Everything the server needs to run.

    Defaults come from ``RD_*`` environment variables so the server can be
    started without arguments; the CLI overrides them afterwards.
    """

    host: str = field(default_factory=lambda: _env_str("RD_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("RD_PORT", 8765))
    token: str = field(default_factory=lambda: _env_str("RD_TOKEN", ""))
    allow_input: bool = field(default_factory=lambda: _env_bool("RD_ALLOW_INPUT", True))
    auto_control: bool = field(default_factory=lambda: _env_bool("RD_AUTO_CONTROL", True))
    max_viewers: int = field(default_factory=lambda: _env_int("RD_MAX_VIEWERS", 4))
    room_ttl: float = field(default_factory=lambda: _env_float("RD_ROOM_TTL", 120.0))
    input_rate: float = field(default_factory=lambda: _env_float("RD_INPUT_RATE", 240.0))
    log_level: str = field(default_factory=lambda: _env_str("RD_LOG_LEVEL", "info"))
    open_browser: bool = field(default_factory=lambda: _env_bool("RD_OPEN_BROWSER", False))
    public_url: str = field(default_factory=lambda: _env_str("RD_PUBLIC_URL", ""))
    ice_servers_json: str = field(default_factory=lambda: _env_str("RD_ICE_SERVERS", ""))
    stun_url: str = field(default_factory=lambda: _env_str("RD_STUN_URL", "stun:stun.l.google.com:19302"))
    turn_url: str = field(default_factory=lambda: _env_str("RD_TURN_URL", ""))
    turn_secret: str = field(default_factory=lambda: _env_str("RD_TURN_SECRET", ""))
    turn_ttl: int = field(default_factory=lambda: _env_int("RD_TURN_TTL", 3600))

    def resolve_token(self) -> str:
        """Return the access token, generating one when none was configured."""
        if not self.token:
            self.token = secrets.token_urlsafe(16)
        return self.token

    def clamp(self) -> "Settings":
        """Keep values inside ranges the server can actually serve."""
        self.port = int(clamp(int(self.port), 1, 65535))
        self.max_viewers = int(clamp(int(self.max_viewers), 1, 32))
        self.room_ttl = float(clamp(float(self.room_ttl), 10.0, 86400.0))
        self.input_rate = float(clamp(float(self.input_rate), 0.0, 2000.0))
        self.turn_ttl = int(clamp(int(self.turn_ttl), 60, 86400))
        return self

    def ice_servers(self) -> list[dict]:
        """Return the ICE server list handed to the browsers.

        Without a relay the two peers only ever learn their LAN addresses, so a
        phone on cellular can never reach the host. ``RD_ICE_SERVERS`` wins when
        set (raw JSON, for exotic setups); otherwise a STUN server plus, when
        configured, a TURN server with freshly minted credentials is returned.
        """
        if self.ice_servers_json.strip():
            try:
                parsed = json.loads(self.ice_servers_json)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                return [entry for entry in parsed if isinstance(entry, dict)]

        servers: list[dict] = []
        if self.stun_url.strip():
            servers.append({"urls": [url.strip() for url in self.stun_url.split(",") if url.strip()]})
        if self.turn_url.strip():
            entry: dict = {"urls": [url.strip() for url in self.turn_url.split(",") if url.strip()]}
            if self.turn_secret:
                username, credential = turn_credentials(self.turn_secret, self.turn_ttl)
                entry["username"] = username
                entry["credential"] = credential
            servers.append(entry)
        return servers

    @classmethod
    def from_args(cls, args) -> "Settings":
        """Build settings from an ``argparse.Namespace``, ignoring unset options."""
        settings = cls()
        for name in ("host", "port", "token", "log_level", "public_url", "turn_url", "turn_secret", "stun_url"):
            value = getattr(args, name, None)
            if value is not None:
                setattr(settings, name, value)
        if getattr(args, "no_input", False):
            settings.allow_input = False
        if getattr(args, "no_auto_control", False):
            settings.auto_control = False
        if getattr(args, "open_browser", False):
            settings.open_browser = True
        return settings.clamp()
