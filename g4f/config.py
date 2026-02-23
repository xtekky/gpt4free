from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import lru_cache
from typing import Optional

@lru_cache(maxsize=1)
def get_config_dir() -> Path:
    """Get platform-appropriate config directory."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".config"

DEFAULT_PORT = 1337
DEFAULT_TIMEOUT = 600
DEFAULT_STREAM_TIMEOUT = 30

PACKAGE_NAME = "g4f"
CONFIG_DIR = get_config_dir() / PACKAGE_NAME
COOKIES_DIR = CONFIG_DIR / "cookies"
CUSTOM_COOKIES_DIR = "./har_and_cookies"
ORGANIZATION = "gpt4free"
GITHUB_REPOSITORY = f"xtekky/{ORGANIZATION}"
STATIC_DOMAIN = f"{PACKAGE_NAME}.dev"
STATIC_URL = f"https://{STATIC_DOMAIN}/"
REFFERER_URL = f"https://{STATIC_DOMAIN}/"
DIST_DIR = f"./{STATIC_DOMAIN}/dist"
DEFAULT_MODEL = "openai/gpt-oss-120b"
JSDELIVR_URL = "https://cdn.jsdelivr.net/"
DOWNLOAD_URL = f"{JSDELIVR_URL}gh/{ORGANIZATION}/{STATIC_DOMAIN}/"
GITHUB_URL = f"https://raw.githubusercontent.com/{ORGANIZATION}/{STATIC_DOMAIN}/refs/heads/main/"

class AppConfig:
    ignored_providers: Optional[list[str]] = None
    g4f_api_key: Optional[str] = None
    ignore_cookie_files: bool = False
    model: str = None
    provider: str = None
    media_provider: str = None
    proxy: str = None
    gui: bool = False
    demo: bool = False
    timeout: int = DEFAULT_TIMEOUT
    stream_timeout: int = DEFAULT_STREAM_TIMEOUT
    disable_custom_api_key: bool = False

    @classmethod
    def set_config(cls, **data):
        for key, value in data.items():
            if value is not None:
                setattr(cls, key, value)

    @classmethod
    def load_from_env(cls):
        cls.g4f_api_key = os.environ.get("G4F_API_KEY", cls.g4f_api_key)
        cls.timeout = int(os.environ.get("G4F_TIMEOUT", cls.timeout))
        cls.stream_timeout = int(os.environ.get("G4F_STREAM_TIMEOUT", cls.stream_timeout))
        cls.proxy = os.environ.get("G4F_PROXY", cls.proxy)
        cls.model = os.environ.get("G4F_MODEL", cls.model)
        cls.provider = os.environ.get("G4F_PROVIDER", cls.provider)
        cls.disable_custom_api_key = os.environ.get("G4F_DISABLE_CUSTOM_API_KEY", str(cls.disable_custom_api_key)).lower() in ("true", "1", "yes")