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

PACKAGE_NAME = "g4f"
CONFIG_DIR = get_config_dir() / PACKAGE_NAME
COOKIES_DIR = CONFIG_DIR / "cookies"
CUSTOM_COOKIES_DIR = "./har_and_cookies"
ORGANIZATION = "gpt4free"
GITHUB_REPOSITORY = f"xtekky/{ORGANIZATION}"
STATIC_DOMAIN = f"{PACKAGE_NAME}.dev"
STATIC_URL = f"https://{STATIC_DOMAIN}/"
DIST_DIR = f"./{STATIC_DOMAIN}/dist"
DOWNLOAD_URL = f"https://raw.githubusercontent.com/{ORGANIZATION}/{STATIC_DOMAIN}/refs/heads/main/"