"""Runtime configuration for the remote desktop agent."""

from __future__ import annotations

import os
import secrets
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


@dataclass
class Settings:
    """Everything the agent needs to run.

    Defaults come from ``RD_*`` environment variables so the agent can be
    started without arguments; the CLI overrides them afterwards.
    """

    host: str = field(default_factory=lambda: _env_str("RD_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("RD_PORT", 8765))
    token: str = field(default_factory=lambda: _env_str("RD_TOKEN", ""))
    fps: int = field(default_factory=lambda: _env_int("RD_FPS", 12))
    quality: int = field(default_factory=lambda: _env_int("RD_QUALITY", 60))
    max_width: int = field(default_factory=lambda: _env_int("RD_MAX_WIDTH", 1280))
    monitor: int = field(default_factory=lambda: _env_int("RD_MONITOR", 1))
    allow_input: bool = field(default_factory=lambda: _env_bool("RD_ALLOW_INPUT", True))
    auto_control: bool = field(default_factory=lambda: _env_bool("RD_AUTO_CONTROL", True))
    with_cursor: bool = field(default_factory=lambda: _env_bool("RD_WITH_CURSOR", True))
    max_clients: int = field(default_factory=lambda: _env_int("RD_MAX_CLIENTS", 4))
    queue_size: int = field(default_factory=lambda: _env_int("RD_QUEUE_SIZE", 3))
    input_rate: float = field(default_factory=lambda: _env_float("RD_INPUT_RATE", 120.0))
    idle_timeout: float = field(default_factory=lambda: _env_float("RD_IDLE_TIMEOUT", 60.0))
    log_level: str = field(default_factory=lambda: _env_str("RD_LOG_LEVEL", "info"))

    def resolve_token(self) -> str:
        """Return the access token, generating one when none was configured."""
        if not self.token:
            self.token = secrets.token_urlsafe(16)
        return self.token

    def clamp(self) -> "Settings":
        """Keep values inside ranges the capture pipeline can actually serve."""
        self.fps = max(1, min(60, int(self.fps)))
        self.quality = max(10, min(95, int(self.quality)))
        self.max_width = max(160, min(7680, int(self.max_width)))
        self.max_clients = max(1, min(32, int(self.max_clients)))
        self.queue_size = max(1, min(16, int(self.queue_size)))
        self.input_rate = max(0.0, min(1000.0, float(self.input_rate)))
        self.idle_timeout = max(5.0, float(self.idle_timeout))
        return self

    @classmethod
    def from_args(cls, args) -> "Settings":
        """Build settings from an ``argparse.Namespace``, ignoring unset options."""
        settings = cls()
        for name in (
            "host",
            "port",
            "token",
            "fps",
            "quality",
            "max_width",
            "monitor",
            "log_level",
        ):
            value = getattr(args, name, None)
            if value is not None:
                setattr(settings, name, value)
        if getattr(args, "no_input", False):
            settings.allow_input = False
        if getattr(args, "no_auto_control", False):
            settings.auto_control = False
        if getattr(args, "no_cursor", False):
            settings.with_cursor = False
        return settings.clamp()
