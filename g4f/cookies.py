from __future__ import annotations

import os
import time
import json
from typing import Optional, List

try:
    from platformdirs import user_config_dir
    has_platformdirs = True
except ImportError:
    has_platformdirs = False

try:
    from browser_cookie3 import (
        chrome, chromium, opera, opera_gx,
        brave, edge, vivaldi, firefox,
        _LinuxPasswordManager, BrowserCookieError
    )

    def g4f(domain_name: str) -> list:
        """
        Load cookies from the 'g4f' browser (if exists).
        """
        if not has_platformdirs:
            return []
        user_data_dir = user_config_dir("g4f")
        cookie_file = os.path.join(user_data_dir, "Default", "Cookies")
        return [] if not os.path.exists(cookie_file) else chrome(cookie_file, domain_name)

    BROWSERS = [
        g4f, firefox,
        chrome, chromium, opera, opera_gx,
        brave, edge, vivaldi,
    ]
    has_browser_cookie3 = True
except ImportError:
    has_browser_cookie3 = False
    BROWSERS: List = []

from .typing import Dict, Cookies
from .errors import MissingRequirementsError
from .config import COOKIES_DIR, CUSTOM_COOKIES_DIR
from . import debug

class CookiesConfig:
    cookies: Dict[str, Cookies] = {}
    cookies_dir: str = CUSTOM_COOKIES_DIR if os.path.exists(CUSTOM_COOKIES_DIR) else str(COOKIES_DIR)


DOMAINS = (
    ".bing.com",
    ".meta.ai",
    ".google.com",
    "www.whiterabbitneo.com",
    "huggingface.co",
    ".huggingface.co",
    "chat.reka.ai",
    "chatgpt.com",
    ".cerebras.ai",
    "github.com",
)

if has_browser_cookie3 and os.environ.get("DBUS_SESSION_BUS_ADDRESS") == "/dev/null":
    _LinuxPasswordManager.get_password = lambda a, b: b"secret"


def get_cookies(domain_name: str, raise_requirements_error: bool = True,
                single_browser: bool = False, cache_result: bool = True) -> Dict[str, str]:
    """Load cookies for a given domain from all supported browsers."""
    if domain_name in CookiesConfig.cookies:
        return CookiesConfig.cookies[domain_name]

    cookies = load_cookies_from_browsers(domain_name, raise_requirements_error, single_browser)
    if cache_result:
        CookiesConfig.cookies[domain_name] = cookies
    return cookies


def set_cookies(domain_name: str, cookies: Cookies = None) -> None:
    """Set or remove cookies for a given domain in the cache."""
    if cookies:
        CookiesConfig.cookies[domain_name] = cookies
    else:
        CookiesConfig.cookies.pop(domain_name, None)


def load_cookies_from_browsers(domain_name: str,
                               raise_requirements_error: bool = True,
                               single_browser: bool = False) -> Cookies:
    """Helper to load cookies from all supported browsers."""
    if not has_browser_cookie3:
        if raise_requirements_error:
            raise MissingRequirementsError('Install "browser_cookie3" package')
        return {}

    cookies = {}
    for cookie_fn in BROWSERS:
        try:
            cookie_jar = cookie_fn(domain_name=domain_name)
            if cookie_jar:
                debug.log(f"Read cookies from {cookie_fn.__name__} for {domain_name}")
            for cookie in cookie_jar:
                if cookie.name not in cookies and (not cookie.expires or cookie.expires > time.time()):
                    cookies[cookie.name] = cookie.value
            if single_browser and cookie_jar:
                break
        except BrowserCookieError:
            pass
        except KeyboardInterrupt:
            debug.error("Cookie loading interrupted by user.")
            break
        except Exception as e:
            debug.error(f"Error reading cookies from {cookie_fn.__name__} for {domain_name}: {e}")
    return cookies


def set_cookies_dir(dir_path: str) -> None:
    CookiesConfig.cookies_dir = dir_path


def get_cookies_dir() -> str:
    return CookiesConfig.cookies_dir


def _parse_har_file(path: str) -> Dict[str, Dict[str, str]]:
    """Parse a HAR file and return cookies by domain."""
    cookies_by_domain = {}
    try:
        with open(path, "rb") as file:
            har_file = json.load(file)
        debug.log(f"Read .har file: {path}")

        def get_domain(entry: dict) -> Optional[str]:
            headers = entry["request"].get("headers", [])
            host_values = [h["value"] for h in headers if h["name"].lower() in ("host", ":authority")]
            if not host_values:
                return None
            host = host_values.pop()
            return next((d for d in DOMAINS if d in host), None)

        for entry in har_file.get("log", {}).get("entries", []):
            domain = get_domain(entry)
            if domain:
                v_cookies = {c["name"]: c["value"] for c in entry["request"].get("cookies", [])}
                if v_cookies:
                    cookies_by_domain[domain] = v_cookies
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    return cookies_by_domain


def _parse_json_cookie_file(path: str) -> Dict[str, Dict[str, str]]:
    """Parse a JSON cookie export file."""
    cookies_by_domain = {}
    try:
        with open(path, "rb") as file:
            cookie_file = json.load(file)
        if not isinstance(cookie_file, list):
            return {}
        debug.log(f"Read cookie file: {path}")
        for c in cookie_file:
            if isinstance(c, dict) and "domain" in c:
                cookies_by_domain.setdefault(c["domain"], {})[c["name"]] = c["value"]
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    return cookies_by_domain


def read_cookie_files(dir_path: Optional[str] = None, domains_filter: Optional[List[str]] = None) -> None:
    """
    Load cookies from .har and .json files in a directory.
    """
    dir_path = dir_path or CookiesConfig.cookies_dir
    if not os.access(dir_path, os.R_OK):
        debug.log(f"Read cookies: {dir_path} dir is not readable")
        return

    # Optionally load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(dir_path, ".env"), override=True)
        debug.log(f"Read cookies: Loaded env vars from {dir_path}/.env")
    except ImportError:
        debug.error("Warning: 'python-dotenv' is not installed. Env vars not loaded.")

    har_files, json_files = [], []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".har"):
                har_files.append(os.path.join(root, file))
            elif file.endswith(".json"):
                json_files.append(os.path.join(root, file))
        break  # Do not recurse

    CookiesConfig.cookies.clear()

    # Load cookies from files
    for path in har_files:
        for domain, cookies in _parse_har_file(path).items():
            if not domains_filter or domain in domains_filter:
                CookiesConfig.cookies[domain] = cookies
                debug.log(f"Cookies added: {len(cookies)} from {domain}")

    for path in json_files:
        for domain, cookies in _parse_json_cookie_file(path).items():
            if not domains_filter or domain in domains_filter:
                CookiesConfig.cookies[domain] = cookies
                debug.log(f"Cookies added: {len(cookies)} from {domain}")