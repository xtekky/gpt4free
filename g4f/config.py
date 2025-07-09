import os
import sys
from pathlib import Path

# Platform-appropriate directories
def get_config_dir() -> Path:
    """Get platform-appropriate config directory."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    else:  # Linux and other UNIX-like
        return Path.home() / ".config"

CONFIG_DIR = get_config_dir() / "g4f"
COOKIES_DIR = CONFIG_DIR / "cookies"
CUSTOM_COOKIES_DIR = "./har_and_cookies"
PACKAGE_NAME = "g4f"
ORGANIZATION = "gpt4free"
GITHUB_REPOSITORY = f"xtekky/{ORGANIZATION}"
STATIC_DOMAIN = f"g4f.dev"
STATIC_URL = f"https://{STATIC_DOMAIN}/"
DIST_DIR = f"./{STATIC_DOMAIN}/dist"
DOWNLOAD_URL = f"https://raw.githubusercontent.com/{ORGANIZATION}/{STATIC_DOMAIN}/refs/heads/main/"