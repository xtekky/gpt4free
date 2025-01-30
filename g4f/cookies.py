from __future__ import annotations

import os
import time
import json

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

        Args:
            domain_name (str): The domain for which to load cookies.

        Returns:
            list: List of cookies.
        """
        if not has_platformdirs:
            return []
        user_data_dir = user_config_dir("g4f")
        cookie_file = os.path.join(user_data_dir, "Default", "Cookies")
        return [] if not os.path.exists(cookie_file) else chrome(cookie_file, domain_name)

    browsers = [
        g4f,
        chrome, chromium, firefox, opera, opera_gx,
        brave, edge, vivaldi,
    ]
    has_browser_cookie3 = True
except ImportError:
    has_browser_cookie3 = False
    browsers = []

from .typing import Dict, Cookies
from .errors import MissingRequirementsError
from . import debug

class CookiesConfig():
    cookies: Dict[str, Cookies] = {}
    cookies_dir: str = "./har_and_cookies"

DOMAINS = [
    ".bing.com",
    ".meta.ai",
    ".google.com",
    "www.whiterabbitneo.com",
    "huggingface.co",
    "chat.reka.ai",
    "chatgpt.com",
    ".cerebras.ai",
    "github.com",
    "huggingface.co",
    ".huggingface.co"
]

if has_browser_cookie3 and os.environ.get('DBUS_SESSION_BUS_ADDRESS') == "/dev/null":
    _LinuxPasswordManager.get_password = lambda a, b: b"secret"

def get_cookies(domain_name: str = '', raise_requirements_error: bool = True, single_browser: bool = False) -> Dict[str, str]:
    """
    Load cookies for a given domain from all supported browsers and cache the results.

    Args:
        domain_name (str): The domain for which to load cookies.

    Returns:
        Dict[str, str]: A dictionary of cookie names and values.
    """
    if domain_name in CookiesConfig.cookies:
        return CookiesConfig.cookies[domain_name]

    cookies = load_cookies_from_browsers(domain_name, raise_requirements_error, single_browser)
    CookiesConfig.cookies[domain_name] = cookies
    return cookies

def set_cookies(domain_name: str, cookies: Cookies = None) -> None:
    if cookies:
        CookiesConfig.cookies[domain_name] = cookies
    elif domain_name in CookiesConfig.cookies:
        CookiesConfig.cookies.pop(domain_name)

def load_cookies_from_browsers(domain_name: str, raise_requirements_error: bool = True, single_browser: bool = False) -> Cookies:
    """
    Helper function to load cookies from various browsers.

    Args:
        domain_name (str): The domain for which to load cookies.

    Returns:
        Dict[str, str]: A dictionary of cookie names and values.
    """
    if not has_browser_cookie3:
        if raise_requirements_error:
            raise MissingRequirementsError('Install "browser_cookie3" package')
        return {}
    cookies = {}
    for cookie_fn in browsers:
        try:
            cookie_jar = cookie_fn(domain_name=domain_name)
            if len(cookie_jar) and debug.logging:
                print(f"Read cookies from {cookie_fn.__name__} for {domain_name}")
            for cookie in cookie_jar:
                if cookie.name not in cookies:
                    if not cookie.expires or cookie.expires > time.time():
                        cookies[cookie.name] = cookie.value
            if single_browser and len(cookie_jar):
                break
        except BrowserCookieError:
            pass
        except Exception as e:
            if debug.logging:
                print(f"Error reading cookies from {cookie_fn.__name__} for {domain_name}: {e}")
    return cookies

def set_cookies_dir(dir: str) -> None:
    CookiesConfig.cookies_dir = dir

def get_cookies_dir() -> str:
    return CookiesConfig.cookies_dir

def read_cookie_files(dirPath: str = None):
    dirPath = CookiesConfig.cookies_dir if dirPath is None else dirPath
    if not os.access(dirPath, os.R_OK):
        debug.log(f"Read cookies: {dirPath} dir is not readable")
        return

    def get_domain(v: dict) -> str:
        host = [h["value"] for h in v['request']['headers'] if h["name"].lower() in ("host", ":authority")]
        if not host:
            return
        host = host.pop()
        for d in DOMAINS:
            if d in host:
                return d

    harFiles = []
    cookieFiles = []
    for root, _, files in os.walk(dirPath):
        for file in files:
            if file.endswith(".har"):
                harFiles.append(os.path.join(root, file))
            elif file.endswith(".json"):
                cookieFiles.append(os.path.join(root, file))

    CookiesConfig.cookies = {}
    for path in harFiles:
        with open(path, 'rb') as file:
            try:
                harFile = json.load(file)
            except json.JSONDecodeError:
                # Error: not a HAR file!
                continue
            debug.log(f"Read .har file: {path}")
            new_cookies = {}
            for v in harFile['log']['entries']:
                domain = get_domain(v)
                if domain is None:
                    continue
                v_cookies = {}
                for c in v['request']['cookies']:
                    v_cookies[c['name']] = c['value']
                if len(v_cookies) > 0:
                    CookiesConfig.cookies[domain] = v_cookies
                    new_cookies[domain] = len(v_cookies)
            for domain, new_values in new_cookies.items():
                debug.log(f"Cookies added: {new_values} from {domain}")
    for path in cookieFiles:
        with open(path, 'rb') as file:
            try:
                cookieFile = json.load(file)
            except json.JSONDecodeError:
                # Error: not a json file!
                continue
            if not isinstance(cookieFile, list) or not isinstance(cookieFile[0], dict) or "domain" not in cookieFile[0]:
                continue
            debug.log(f"Read cookie file: {path}")
            new_cookies = {}
            for c in cookieFile:
                if isinstance(c, dict) and "domain" in c:
                    if c["domain"] not in new_cookies:
                        new_cookies[c["domain"]] = {}
                    new_cookies[c["domain"]][c["name"]] = c["value"]
            for domain, new_values in new_cookies.items():
                debug.log(f"Cookies added: {len(new_values)} from {domain}")
                CookiesConfig.cookies[domain] = new_values