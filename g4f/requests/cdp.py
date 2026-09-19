"""
CDP Clients — Lightweight Chrome DevTools Protocol (CDP) automation.

This module provides two CDP client implementations for browser automation:

──────────────────────────────────────────────────────────────────────
CDPSession (Async) — for high-throughput providers like Cloudflare.
──────────────────────────────────────────────────────────────────────
  • Fully async (asyncio + aiohttp WebSocket).
  • Background receiver loop for event-driven communication.
  • Best for providers that stream responses and need concurrency.

  Example:
      session = CDPSession(port=9222, headless=False)
      await session.start()
      try:
          await session.navigate("https://example.com")
          title = await session.evaluate_js("document.title")
      finally:
          await session.close()

──────────────────────────────────────────────────────────────────────
Common features:
  • Auto-detects Chrome/Chromium/Edge path via BrowserConfig or system PATH.
  • Stores browser profiles in g4f cookies directory (no project root pollution).
  • Offscreen windowed mode (--window-position=-2000,-2000) bypasses Turnstile.
  • Android app: creates dedicated automation WebViews through its DevTools
    socket (browser_mode="webview", auto-detected — no Chrome needed). Each
    target is shown in front of the app UI with a close button; WebView
    debugging is disabled again when the last target is closed.
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import platform
import subprocess
import time
import urllib.request
from typing import Optional, Dict, Any, List, AsyncIterator
import hashlib
from urllib.parse import quote_plus, urlparse
import datetime

try:
    import aiohttp
except ImportError:
    pass

from ..cookies import BrowserConfig
from ..files import secure_filename
from .. import debug

try:
    from PIL import Image
    has_pillow = True
except ImportError:
    has_pillow = False

logger = logging.getLogger(__name__)

from pathlib import Path

def get_screenshot_dir(datekey: str = None) -> str:
    """Get the screenshot directory, creating it if necessary."""
    try:
        from g4f.image.copy_images import get_media_dir
        media_dir = get_media_dir()
    except ImportError:
        import tempfile
        media_dir = os.path.join(tempfile.gettempdir(), "g4f_media")
    screenshots_dir = os.path.join(media_dir, "screenshots")
    if datekey:
        screenshots_dir = os.path.join(screenshots_dir, datekey)
    os.makedirs(screenshots_dir, exist_ok=True)
    return screenshots_dir


def find_chrome_path() -> Optional[str]:
    """Search for Google Chrome or Chromium binary depending on OS."""
    try:
        from g4f.cookies import BrowserConfig

        if BrowserConfig.executable_path and os.path.exists(
            BrowserConfig.executable_path
        ):
            return BrowserConfig.executable_path
    except ImportError:
        pass

    for name in [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
        "msedge",
        "helium",
    ]:
        path = shutil.which(name)
        if path:
            return path

    sys_name = platform.system().lower()
    if sys_name == "linux":
        for path in [
            "/usr/bin/google-chrome",
            "/opt/google/chrome/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/microsoft-edge",
            "/opt/helium/helium",
        ]:
            if os.path.exists(path):
                return path
    elif sys_name in ("macos", "darwin"):
        for path in [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]:
            if os.path.exists(path):
                return path
    elif sys_name == "windows":
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Helium\Application\helium.exe",
            r"C:\Program Files (x86)\Helium\Application\helium.exe",
            r"C:\Program Files\Helium\helium.exe",
        ]
        for path in paths:
            if os.path.exists(path):
                return path
    return None


import threading
import atexit

_shared_browser_process = None
_shared_browser_port = None
_shared_browser_lock = threading.Lock()
_shared_browser_refcount = 0  # Track active CDP sessions for parallel tabs
_shared_browser_idle_timer = None  # Timer to shut down browser after idle period
_SHARED_BROWSER_IDLE_TIMEOUT = 60  # seconds to keep browser alive with zero tabs


def _terminate_shared_browser():
    """Terminate the shared browser process and reset state."""
    global _shared_browser_process, _shared_browser_port
    if _shared_browser_process:
        try:
            _shared_browser_process.terminate()
        except Exception:
            pass
        _shared_browser_process = None
    _shared_browser_port = None


def _schedule_idle_shutdown():
    """Schedule browser termination after idle timeout (call under lock)."""
    global _shared_browser_idle_timer
    if _shared_browser_idle_timer:
        _shared_browser_idle_timer.cancel()
    _shared_browser_idle_timer = threading.Timer(
        _SHARED_BROWSER_IDLE_TIMEOUT, _terminate_shared_browser
    )
    _shared_browser_idle_timer.daemon = True
    _shared_browser_idle_timer.start()
    debug.log(f"CDP: Browser idle shutdown scheduled in {_SHARED_BROWSER_IDLE_TIMEOUT}s")


def _cancel_idle_shutdown():
    """Cancel any pending idle shutdown timer (call under lock)."""
    global _shared_browser_idle_timer
    if _shared_browser_idle_timer:
        _shared_browser_idle_timer.cancel()
        _shared_browser_idle_timer = None


def _cleanup_shared_browser():
    global _shared_browser_refcount, _shared_browser_idle_timer
    if _shared_browser_idle_timer:
        _shared_browser_idle_timer.cancel()
        _shared_browser_idle_timer = None
    _terminate_shared_browser()
    _shared_browser_refcount = 0


atexit.register(_cleanup_shared_browser)


def acquire_shared_browser_ref():
    """Increment the shared browser reference count (call when opening a new tab)."""
    global _shared_browser_refcount
    with _shared_browser_lock:
        _cancel_idle_shutdown()
        _shared_browser_refcount += 1
        debug.log(f"CDP: Acquired browser tab (#{_shared_browser_refcount} active)")


def release_shared_browser_ref() -> bool:
    """Decrement the shared browser reference count.
    Returns True if the refcount reached 0 (browser kept alive for reuse)."""
    global _shared_browser_refcount
    with _shared_browser_lock:
        _shared_browser_refcount = max(0, _shared_browser_refcount - 1)
        debug.log(f"CDP: Released browser tab (#{_shared_browser_refcount} remaining)")
        if _shared_browser_refcount == 0:
            _schedule_idle_shutdown()
        return _shared_browser_refcount == 0


def find_running_cdp_port(host: str) -> Optional[int]:
    """Scan running processes for an active Chrome/Helium instance with remote debugging enabled."""
    try:
        import psutil

        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                proc_name = (proc.info.get("name") or "").lower()
                if any(
                    n in proc_name
                    for n in ("chrome", "chromium", "edge", "helium", "app")
                ):
                    for arg in cmdline:
                        if arg.startswith("--remote-debugging-port="):
                            try:
                                port = int(arg.split("=")[1])
                                # Verify if it's reachable and working
                                with urllib.request.urlopen(
                                    f"http://{host}:{port}/json", timeout=0.5
                                ) as response:
                                    if response.status == 200:
                                        return port
                            except Exception:
                                pass
            except Exception:
                pass
    except Exception:
        pass
    return None


def get_shared_browser(
    host: str,
    preferred_port: int,
    headless: bool = True,
    proxy: Optional[str] = None,
    browser_args: Optional[List[str]] = None,
) -> int:
    """
    Ensure a single shared browser instance is running and return its port.
    If a browser is already running anywhere on the system, we use it directly.

    ``proxy``/``browser_args`` only take effect when the shared browser is
    first launched — later callers reusing the shared process are ignored,
    since Chrome does not support changing its proxy at runtime.
    """
    global _shared_browser_process, _shared_browser_port

    with _shared_browser_lock:
        if preferred_port is not None:
            try:
                with urllib.request.urlopen(
                    f"http://{host}:{preferred_port}/json", timeout=0.5
                ) as response:
                    if response.status == 200:
                        return preferred_port
            except Exception:
                pass
        # 1. If we already started a shared browser in this thread, check if it's still alive/reachable
        if _shared_browser_port is not None:
            try:
                with urllib.request.urlopen(
                    f"http://{host}:{_shared_browser_port}/json", timeout=0.5
                ) as response:
                    if response.status == 200:
                        return _shared_browser_port
            except Exception:
                # Browser died or became unreachable, clean up
                if _shared_browser_process:
                    try:
                        _shared_browser_process.terminate()
                    except Exception:
                        pass
                    _shared_browser_process = None
                _shared_browser_port = None

        # 2. Check if a browser is already running anywhere on the system with CDP remote debugging
        running_port = find_running_cdp_port(host)
        if running_port is not None:
            _shared_browser_port = running_port
            return _shared_browser_port

        # 3. Otherwise, launch a new shared Chromium process on a free port
        chrome_path = find_chrome_path()
        if not chrome_path:
            raise RuntimeError("Google Chrome / Chromium / Edge executable not found.")

        # Find a free port dynamically
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        # Use standard user config directory for profile caching (like other g4f browsers)
        try:
            from platformdirs import user_config_dir

            user_data_dir = user_config_dir("g4f-cdp")
        except ImportError:
            import tempfile

            user_data_dir = os.path.join(
                tempfile.gettempdir(), "g4f_chrome_profile_cdp"
            )
        os.makedirs(user_data_dir, exist_ok=True)

        # Remove stale SingletonLock / SingletonSocket / SingletonCookie left
        # behind by a previous Chrome crash — otherwise the new process refuses
        # to start with "Failed to create …/SingletonLock: File exists".
        for lock_name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
            lock_path = os.path.join(user_data_dir, lock_name)
            try:
                if os.path.islink(lock_path) or os.path.exists(lock_path):
                    os.remove(lock_path)
            except Exception:
                pass

        cmd = [
            chrome_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            "--window-size=1280,720",
            "--no-default-browser-check",
            "--disable-suggestions-ui",
            "--no-first-run",
            "--disable-infobars",
            "--disable-popup-blocking",
            "--hide-crash-restore-bubble",
            "--disable-features=PrivacySandboxSettings4",
            "--disable-blink-features=AutomationControlled",
            "--remote-allow-origins=*",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ]
        if headless:
            cmd.append("--headless=new")
        if proxy:
            cmd.append(f"--proxy-server={proxy}")
        if browser_args:
            cmd.extend(browser_args)

        debug.log(f"CDP: Launching Chrome: {' '.join(cmd)}")
        _shared_browser_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait up to 20 seconds for readiness
        for _ in range(40):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(
                    f"http://{host}:{port}/json", timeout=1
                ) as response:
                    if response.status == 200:
                        _shared_browser_port = port
                        debug.log(f"CDP: Shared Chrome ready on port {port}")
                        return _shared_browser_port
            except Exception:
                pass

        # Chrome failed to become ready — capture stderr for diagnostics
        if _shared_browser_process:
            stderr_output = ""
            try:
                _shared_browser_process.terminate()
                # Read any stderr output before killing
                import threading

                def _read_stderr(proc, container):
                    try:
                        container.append(proc.stderr.read().decode("utf-8", errors="replace")[:2000])
                    except Exception:
                        pass

                err_list = []
                t = threading.Thread(target=_read_stderr, args=(_shared_browser_process, err_list))
                t.daemon = True
                t.start()
                t.join(timeout=2)
                if err_list:
                    stderr_output = err_list[0]
            except Exception:
                pass
            _shared_browser_process = None
        raise RuntimeError(
            f"Failed to start shared Chrome on port {port}"
            + (f": {stderr_output}" if stderr_output else "")
        )

# ──────────────────────────────────────────────────────────────────────
# Android WebView support — drive the app's own WebView via CDP.
# ──────────────────────────────────────────────────────────────────────

def _is_android() -> bool:
    """Return True when running under Android (e.g. the Chaquopy app)."""
    if os.path.exists("/system/build.prop"):
        return True
    try:
        import java  # Chaquopy java bridge  # noqa: F401
        return True
    except ImportError:
        return False

def _enable_webview_debugging() -> bool:
    """Enable remote debugging for all WebViews in this app (process-wide)."""
    try:
        from java import jclass

        WebView = jclass("android.webkit.WebView")
        WebView.setWebContentsDebuggingEnabled(True)
        return True
    except Exception as e:
        debug.log(f"CDP: failed to enable WebView debugging: {e}")
        return False

def _disable_webview_debugging() -> bool:
    """Disable remote debugging for all WebViews in this app (process-wide).

    Called when the last automation target has been closed, so the DevTools
    socket is not exposed while no automation is running.
    """
    try:
        from java import jclass

        WebView = jclass("android.webkit.WebView")
        WebView.setWebContentsDebuggingEnabled(False)
        debug.log("CDP: WebView debugging disabled (no targets left)")
        return True
    except Exception as e:
        debug.log(f"CDP: failed to disable WebView debugging: {e}")
        return False

def _find_webview_devtools_socket() -> str:
    """
    Find the abstract Unix socket name of the WebView DevTools server.

    The WebView listens on ``@webview_devtools_remote_<pid>`` in the app's
    own process. Since Android 10 apps can no longer read /proc/net/unix,
    so the pid based name is preferred and the scan is only a fallback.
    """
    own = f"webview_devtools_remote_{os.getpid()}"
    candidates = []
    try:
        with open("/proc/net/unix") as fp:
            for line in fp:
                name = line.split()[-1].lstrip("@")
                if name.startswith("webview_devtools_remote_"):
                    candidates.append(name)
    except Exception:
        pass
    if own in candidates:
        return own
    if candidates:
        return candidates[0]
    return own

def _webview_devtools_request(socket_name: str, path: str, timeout: float = 5.0):
    """HTTP GET against the WebView DevTools server over its abstract Unix socket."""
    import http.client
    import socket as _socket

    sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect("\0" + socket_name)
        conn = http.client.HTTPConnection("localhost")
        conn.sock = sock  # reuse the abstract socket connection
        conn.request("GET", path, headers={"Host": "localhost", "Connection": "close"})
        body = conn.getresponse().read()
        return json.loads(body.decode("utf-8", errors="replace"))
    finally:
        try:
            sock.close()
        except Exception:
            pass

# DevTools target ids claimed by active webview-mode sessions, so parallel
# sessions don't attach to the same freshly created automation WebView.
_webview_claimed_targets = set()

async def _webview_bridge_call(socket_name: str, target: dict, expression: str, timeout: float = 10.0):
    """Evaluate a JS expression on a WebView target and return its value.

    Used to reach the app's automation bridge (window.G4FAutomation) on the
    chat UI page, which creates and closes dedicated automation WebViews.
    """
    import aiohttp

    ws_path = urlparse(target.get("webSocketDebuggerUrl", "")).path
    if not ws_path:
        ws_path = f"/devtools/page/{target.get('id')}"
    connector = aiohttp.UnixConnector(path="\0" + socket_name)
    async with aiohttp.ClientSession(connector=connector) as session:
        ws = await session.ws_connect(f"ws://localhost{ws_path}")
        try:
            await ws.send_str(json.dumps({
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": expression, "returnByValue": True},
            }))
            deadline = time.time() + timeout
            while time.time() < deadline:
                msg = await asyncio.wait_for(ws.receive(), timeout=deadline - time.time())
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                data = json.loads(msg.data)
                if data.get("id") == 1:
                    if "error" in data:
                        raise RuntimeError(f"WebView bridge error: {data['error']}")
                    return data.get("result", {}).get("result", {}).get("value")
        finally:
            await ws.close()
    raise TimeoutError("WebView bridge call timed out")


class CDPSession:
    def __init__(
        self,
        port: Optional[int] = None,
        host: Optional[str] = None,
        user_data_dir: Optional[str] = None,
        headless: Optional[bool] = None,
        proxy: Optional[str] = None,
        browser_args: Optional[List[str]] = None,
    ):
        if port is None:
            port = BrowserConfig.port
        if host is None:
            host = BrowserConfig.host
        if host is None:
            host = "127.0.0.1"
        self.port = port
        self.host = host
        if headless is None:
            headless = BrowserConfig.headless
        self.headless = headless
        self.proxy = proxy
        self.browser_args = browser_args
        self.user_data_dir = (
            user_data_dir  # Ignored if using shared pool, but kept for compatibility
        )
        self.process = None
        self.ws = None
        self.session = None
        self.target_id = None
        self.id_counter = 0
        self._receive_task = None
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._event_handlers: Dict[str, List[asyncio.Future]] = {}
        self._event_queues: Dict[str, List[asyncio.Queue]] = {}
        self._closing = False
        self._connection_lost = False
        # True when this session runs through the browser-extension relay
        # (g4f/api/cdp_relay.py) instead of a local Chrome CDP port.
        self._via_extension = False
        # True when this session is attached to the Android app's WebView
        # (browser_mode="webview") instead of a local Chrome CDP port.
        self._via_webview = False
        self._webview_initial_url: Optional[str] = None
        # WebView automation target bookkeeping: the DevTools socket name, the
        # bridge id of the dedicated automation WebView created for this
        # session (None when falling back to the chat UI page) and the target
        # id of the chat UI page used for bridge calls.
        self._webview_socket: Optional[str] = None
        self._webview_automation_id: Optional[str] = None
        self._webview_control_id: Optional[str] = None

        # Network event loggers
        self.network_requests: List[dict] = []
        self.network_responses: List[dict] = []

    @property
    def is_alive(self) -> bool:
        """Return True if the WebSocket is still connected and not closing."""
        return not self._closing and not self._connection_lost and self.ws is not None and not self.ws.closed

    async def start(self):
        """Connect a CDP target: Android WebView, extension relay or shared Chrome."""
        browser_mode = getattr(BrowserConfig, "browser_mode", None)
        # Extension mode: route through the g4f browser extension relay
        # (g4f/api/cdp_relay.py) instead of a local Chrome CDP port.
        if browser_mode == "extension":
            return await self._start_via_extension()

        # WebView mode: attach to the Android app's own WebView through its
        # DevTools socket — there is no installable Chrome on Android. Forced
        # with G4F_BROWSER_MODE=webview, auto-detected on Android when no
        # explicit CDP port is configured.
        if browser_mode == "webview" or (
            browser_mode is None and self.port is None and _is_android()
        ):
            return await self._start_via_webview()

        if self.port is None:
            self.port = get_shared_browser(
                self.host, self.port, self.headless, self.proxy, self.browser_args
            )

        # Acquire a reference so the shared browser stays alive for this tab
        acquire_shared_browser_ref()

        # Create a new tab target
        ws_url = None
        for _ in range(10):
            try:
                req = urllib.request.Request(
                    f"http://{self.host}:{self.port}/json/new", method="PUT"
                )
                with urllib.request.urlopen(req, timeout=2) as response:
                    target = json.loads(response.read().decode("utf-8"))
                    ws_url = target.get("webSocketDebuggerUrl")
                    self.target_id = target.get("id")
                    if ws_url:
                        break
            except Exception:
                await asyncio.sleep(0.5)

        if not ws_url:
            release_shared_browser_ref()
            raise RuntimeError(f"Failed to create new tab target on port {self.port}")

        await self.connect(ws_url)

    async def connect(self, ws_url: str):
        """Connect to the target WebSocket debugger."""
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(ws_url)
        self._closing = False

        # Start receiver loop
        self._receive_task = asyncio.create_task(self._receiver_loop())

        # Enable essential domains
        await self.call("Page.enable")
        await self.call("DOM.enable")
        await self.call("Runtime.enable")
        await self.call("Network.enable")
        await self.call("Emulation.setFocusEmulationEnabled", enabled=True)

        # Force a desktop-sized viewport — the OS window size hint
        # (--window-size) is not always honored by the window manager, which
        # can leave the page narrow enough to trigger a site's mobile layout.
        try:
            await self.call(
                "Emulation.setDeviceMetricsOverride",
                width=1280,
                height=800,
                deviceScaleFactor=1,
                mobile=False,
            )
        except Exception:
            pass

        # Anti-detect: Override User-Agent to remove "HeadlessChrome"
        user_agent = await self.evaluate_js("navigator.userAgent")
        if user_agent and "HeadlessChrome" in user_agent:
            clean_ua = user_agent.replace("HeadlessChrome", "Chrome")
            await self.call("Network.setUserAgentOverride", userAgent=clean_ua)

        # Anti-detect: Inject Stealth Script
        stealth_js = """
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        window.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) return 'Intel Inc.';
            if (parameter === 37446) return 'Intel Iris OpenGL Engine';
            return originalGetParameter.call(this, parameter);
        };
        """
        await self.call("Page.addScriptToEvaluateOnNewDocument", source=stealth_js)

    async def _start_via_extension(self):
        """
        Connect through the g4f browser extension relay.

        Instead of http://host:port/json/new + a direct Chrome WebSocket,
        ask the relay (running inside the g4f API server) to create a tab in
        the extension's browser, then use the relay's pass-through WebSocket
        /v1/cdp/ws/{target_id}. Everything else (call/evaluate/event loop)
        works unchanged because the relay speaks plain CDP WebSocket.
        """
        import aiohttp

        api_host = os.environ.get("G4F_API_HOST", "127.0.0.1")
        api_port = os.environ.get("G4F_API_PORT", "1337")
        base = f"http://{api_host}:{api_port}"

        # 1. Create a tab in the extension's browser via the relay.
        import urllib.request

        req = urllib.request.Request(
            f"{base}/json/new", method="PUT",
            data=b"", headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            target = json.loads(response.read().decode("utf-8"))
        target_id = target.get("id")
        if not target_id:
            raise RuntimeError("CDP relay: failed to create extension tab")

        self.target_id = target_id
        self._via_extension = True

        # 2. Connect to the relay's pass-through WebSocket for this target.
        ws_url = f"ws://{api_host}:{api_port}/v1/cdp/ws/{quote_plus(target_id)}"
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(ws_url)
        self._closing = False
        self._receive_task = asyncio.create_task(self._receiver_loop())

        # 3. Enable essential domains (same as local mode).
        await self.call("Page.enable")
        await self.call("DOM.enable")
        await self.call("Runtime.enable")
        await self.call("Network.enable")

    async def _start_via_webview(self):
        """
        Run automation in a dedicated WebView of the Android app.

        The app's chat UI WebView exposes a JS bridge (window.G4FAutomation)
        that creates additional WebViews on demand. Each one appears as its
        own "page" target on the app's DevTools socket
        ``@webview_devtools_remote_<pid>``, is shown in front of the chat UI
        (with a close button) and can be attached via a WebSocket (aiohttp
        UnixConnector). Everything else (call/evaluate/events) works
        unchanged, because the WebView speaks plain CDP.

        Without the bridge (older app builds) this falls back to attaching to
        the chat UI page itself.
        """
        import aiohttp

        if not _enable_webview_debugging():
            raise RuntimeError(
                "CDP: could not enable WebView debugging — not running inside the Android app?"
            )

        socket_name = _find_webview_devtools_socket()
        # The DevTools socket appears shortly after debugging is enabled.
        targets = None
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                targets = _webview_devtools_request(socket_name, "/json/list")
                if isinstance(targets, list) and targets:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        if not targets or not isinstance(targets, list):
            raise RuntimeError(
                f"CDP: no WebView DevTools targets on socket @{socket_name}"
            )
        self._webview_socket = socket_name

        pages = [t for t in targets if t.get("type") == "page"]
        if not pages:
            raise RuntimeError(f"CDP: no page target in WebView DevTools: {targets}")
        # The app's chat UI acts as the control page for the automation bridge.
        control = next((t for t in pages if "127.0.0.1" in t.get("url", "")), None)

        # Prefer a dedicated automation WebView (a new CDP target) created via
        # the app's JS bridge: it is shown in front of the chat UI and keeps
        # the chat UI itself untouched. Parallel sessions each get their own.
        target = None
        if control is not None:
            target = await self._create_webview_target(socket_name, control, targets)

        if target is None:
            # Fallback: attach to the app's chat UI page directly.
            target = control if control is not None else pages[0]
            debug.log("CDP: no automation bridge — attaching to the chat UI page")

        ws_path = urlparse(target.get("webSocketDebuggerUrl", "")).path
        if not ws_path:
            ws_path = f"/devtools/page/{target.get('id')}"
        self.target_id = target.get("id")
        self._webview_initial_url = target.get("url")
        self._via_webview = True
        debug.log(
            f"CDP: attached to Android WebView target {self.target_id} "
            f"({self._webview_initial_url}) via @{socket_name}"
        )

        connector = aiohttp.UnixConnector(path="\0" + socket_name)
        self.session = aiohttp.ClientSession(connector=connector)
        self.ws = await self.session.ws_connect(f"ws://localhost{ws_path}")
        self._closing = False
        self._receive_task = asyncio.create_task(self._receiver_loop())

        # Enable essential domains (no viewport/UA overrides — the WebView is
        # a real, visible browser and the app already configured it).
        await self.call("Page.enable")
        await self.call("DOM.enable")
        await self.call("Runtime.enable")
        await self.call("Network.enable")

    async def _create_webview_target(
        self, socket_name: str, control: dict, known_targets: list
    ) -> Optional[dict]:
        """Create a dedicated automation WebView (new CDP target) via the app's
        JS bridge and return its DevTools target. Returns None when the bridge
        is unavailable or the target never shows up."""
        known_ids = {t.get("id") for t in known_targets}
        try:
            automation_id = await _webview_bridge_call(
                socket_name, control,
                "(window.G4FAutomation && window.G4FAutomation.createTarget('about:blank')) || null",
            )
        except Exception as e:
            debug.log(f"CDP: WebView bridge createTarget failed: {e}")
            return None
        if not isinstance(automation_id, str) or not automation_id:
            return None

        # Wait for the new WebView to appear as a page target on the socket.
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                fresh = _webview_devtools_request(socket_name, "/json/list")
                new_pages = [
                    t for t in (fresh or [])
                    if isinstance(t, dict)
                    and t.get("type") == "page"
                    and t.get("id") not in known_ids
                    and t.get("id") not in _webview_claimed_targets
                ]
                if new_pages:
                    target = new_pages[0]
                    _webview_claimed_targets.add(target.get("id"))
                    self._webview_automation_id = automation_id
                    self._webview_control_id = control.get("id")
                    debug.log(
                        f"CDP: created automation WebView target {target.get('id')} "
                        f"(bridge id {automation_id})"
                    )
                    return target
            except Exception:
                pass
            await asyncio.sleep(0.25)

        # Target never showed up — destroy the stray WebView again.
        debug.log("CDP: automation WebView target did not appear — cleaning up")
        try:
            await _webview_bridge_call(
                socket_name, control,
                "window.G4FAutomation && window.G4FAutomation.closeTarget(%s)"
                % json.dumps(automation_id),
            )
        except Exception:
            pass
        return None

    async def _close_webview_target(self):
        """Tear down this webview-mode session.

        Dedicated automation WebViews (created through the app's JS bridge)
        are destroyed via the bridge — the app removes them from the UI and
        disables WebView debugging when the last one is closed. Fallback
        sessions attached to the chat UI only navigate back, and debugging is
        disabled when no other page target is left.
        """
        if self._webview_automation_id:
            _webview_claimed_targets.discard(self.target_id)
            control = None
            try:
                targets = _webview_devtools_request(self._webview_socket, "/json/list")
                pages = [
                    t for t in (targets or [])
                    if isinstance(t, dict) and t.get("type") == "page"
                ]
                control = next(
                    (t for t in pages if t.get("id") == self._webview_control_id),
                    next((t for t in pages if "127.0.0.1" in t.get("url", "")), None),
                )
            except Exception:
                pass  # Socket gone — debugging already disabled (last target closed)
            if control is not None:
                try:
                    await _webview_bridge_call(
                        self._webview_socket, control,
                        "window.G4FAutomation && window.G4FAutomation.closeTarget(%s)"
                        % json.dumps(self._webview_automation_id),
                    )
                except Exception as e:
                    debug.log(f"CDP: WebView bridge closeTarget failed: {e}")
        else:
            # Fallback: the target is the app's chat UI — never close it.
            # Navigate back to the page the WebView showed before automation.
            if (
                self._webview_initial_url
                and self.ws
                and not self.ws.closed
                and not self._connection_lost
            ):
                try:
                    await asyncio.wait_for(
                        self.call("Page.navigate", url=self._webview_initial_url),
                        timeout=3.0,
                    )
                except Exception:
                    pass
            # Disable WebView debugging when this was the only page left.
            try:
                targets = _webview_devtools_request(self._webview_socket, "/json/list")
                pages = [
                    t for t in (targets or [])
                    if isinstance(t, dict) and t.get("type") == "page"
                ]
                if len(pages) <= 1:
                    _disable_webview_debugging()
            except Exception:
                pass  # Socket gone — debugging already disabled

    async def _receiver_loop(self):
        """Listen for WebSocket messages."""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if "id" in data:
                        req_id = data["id"]
                        if req_id in self._pending_requests:
                            fut = self._pending_requests[req_id]
                            if not fut.done():
                                if "error" in data:
                                    fut.set_exception(RuntimeError(data["error"]))
                                else:
                                    fut.set_result(data.get("result", {}))
                    elif "method" in data:
                        method = data["method"]
                        params = data.get("params", {})

                        # Intercept network events
                        if method == "Network.requestWillBeSent":
                            self.network_requests.append(params)
                        elif method == "Network.responseReceived":
                            self.network_responses.append(params)

                        # Resolve any futures waiting for this event
                        if method in self._event_handlers:
                            for fut in self._event_handlers[method]:
                                if not fut.done():
                                    fut.set_result(params)
                            self._event_handlers[method].clear()

                        if method in self._event_queues:
                            for q in self._event_queues[method]:
                                q.put_nowait({"_method": method, **params})
        except Exception as e:
            if not self._closing:
                logger.error(f"CDP receiver loop error: {e}")
        finally:
            self._connection_lost = True

    async def call(self, method: str, **params) -> dict:
        """Call a CDP method and wait for its result."""
        if not self.ws:
            raise RuntimeError("CDPSession is not connected")
        if self._connection_lost or self.ws.closed:
            raise ConnectionError("CDPSession connection lost (browser closed?)")

        self.id_counter += 1
        req_id = self.id_counter

        fut = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = fut

        payload = {"id": req_id, "method": method, "params": params}
        try:
            await self.ws.send_json(payload)
        except Exception as e:
            self._connection_lost = True
            self._pending_requests.pop(req_id, None)
            raise ConnectionError(f"CDPSession connection lost during send: {e}")

        try:
            return await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            raise TimeoutError(f"CDP call {method} timed out after 30 seconds")
        finally:
            self._pending_requests.pop(req_id, None)

    async def wait_for_event(self, method: str, timeout: float = 30.0) -> dict:
        """Wait for a specific CDP event to fire (one-time)."""
        fut = asyncio.get_running_loop().create_future()
        if method not in self._event_handlers:
            self._event_handlers[method] = []
        self._event_handlers[method].append(fut)

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._event_handlers[method].remove(fut)
            raise TimeoutError(f"Timeout waiting for event {method}")

    def add_event_handler(self, method: str, queue: asyncio.Queue):
        """Add a persistent event listener that pushes events to an asyncio.Queue."""
        if method not in self._event_queues:
            self._event_queues[method] = []
        self._event_queues[method].append(queue)

    def remove_event_handler(self, method: str, queue: asyncio.Queue):
        """Remove a persistent event listener."""
        if method in self._event_queues and queue in self._event_queues[method]:
            self._event_queues[method].remove(queue)

    async def evaluate_js(self, expression: str) -> Any:
        """Execute JavaScript and return the value."""
        res = await self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True
        )
        if "result" not in res or "value" not in res["result"] and res["result"].get("type") != "undefined":
            raise RuntimeError(f"JavaScript evaluation failed: {res}")
        return res.get("result", {}).get("value")

    async def get_cookies(self) -> dict:
        """Retrieve all cookies from the browser as a name-value dict."""
        cookies = await self.get_cookies_list()
        return {c["name"]: c["value"] for c in cookies}

    async def get_cookies_list(self, urls: Optional[List[str]] = None) -> List[dict]:
        """Retrieve full cookie objects from the browser session."""
        params = {}
        if urls:
            params["urls"] = urls
        res = await self.call("Network.getCookies", **params)
        return res.get("cookies", [])

    async def set_cookies(self, cookies: List[dict]):
        """Set cookies in the browser session."""
        for cookie in cookies:
            params = {
                "name": cookie.get("name"),
                "value": cookie.get("value"),
                "domain": cookie.get("domain"),
                "path": cookie.get("path"),
                "secure": cookie.get("secure"),
                "httpOnly": cookie.get("httpOnly"),
                "sameSite": cookie.get("sameSite"),
                "expires": cookie.get("expires"),
            }
            params = {k: v for k, v in params.items() if v is not None}
            if "domain" not in params and "url" not in params:
                params["url"] = "https://deepinfra.com"
            await self.call("Network.setCookie", **params)

    async def get_user_agent(self) -> str:
        """Retrieve the current browser user agent."""
        return await self.evaluate_js("navigator.userAgent")

    async def navigate(self, url: str):
        """Navigate to a URL and wait for it to load."""

        await self.call("Page.navigate", url=url)
        await self.evaluate_js("(async () => await new Promise(resolve => window.addEventListener('load', resolve)))()")

    async def reload(self):
        """Reload the current page and wait for it to load."""

        await self.call("Page.reload")
        await self.evaluate_js("(async () => await new Promise(resolve => window.addEventListener('load', resolve)))()")

    async def wait_for_network_idle(
        self, idle_time: float = 0.5, timeout: float = 15.0
    ) -> bool:
        """Wait until network activity settles (no requests for *idle_time* seconds).

        Uses Network.requestWillBeSent / Network.loadingFinished events to track
        in-flight requests. Returns True if the network went idle, False on timeout.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self.add_event_handler("Network.requestWillBeSent", queue)
        self.add_event_handler("Network.loadingFinished", queue)
        self.add_event_handler("Network.loadingFailed", queue)

        # Count currently in-flight requests via JS-free CDP approach:
        # Every requestWillBeSent increments, every loadingFinished/loadingFailed decrements.
        pending = 0
        deadline = time.monotonic() + timeout
        last_activity = time.monotonic()

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False

                idle_remaining = idle_time - (time.monotonic() - last_activity)
                wait_for = min(remaining, max(0.05, idle_remaining))

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=wait_for)
                    method = event.get("_method", "")
                    if method == "Network.requestWillBeSent":
                        pending += 1
                        last_activity = time.monotonic()
                    elif method in ("Network.loadingFinished", "Network.loadingFailed"):
                        pending = max(0, pending - 1)
                        last_activity = time.monotonic()
                except asyncio.TimeoutError:
                    pass

                if pending == 0 and (time.monotonic() - last_activity) >= idle_time:
                    return True
        finally:
            self.remove_event_handler("Network.requestWillBeSent", queue)
            self.remove_event_handler("Network.loadingFinished", queue)
            self.remove_event_handler("Network.loadingFailed", queue)

    async def mouse_move(self, x: int, y: int):
        """Simulate a mouse movement to the given coordinates."""
        await self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)

    async def click(self, x: int, y: int, delay: float = 0.05):
        """Simulate a realistic mouse click at the given coordinates."""
        await self.mouse_move(x, y)
        await asyncio.sleep(0.02)
        await self.call(
            "Input.dispatchMouseEvent",
            type="mousePressed",
            button="left",
            clickCount=1,
            x=x,
            y=y,
        )
        await asyncio.sleep(delay)
        await self.call(
            "Input.dispatchMouseEvent",
            type="mouseReleased",
            button="left",
            clickCount=1,
            x=x,
            y=y,
        )

    async def click_turnstile_checkbox(self) -> bool:
        """Find the Cloudflare Turnstile iframe on the page and click its center."""
        js_code = """
        (() => {
            const iframes = document.querySelectorAll('iframe');
            let cfIframe = null;
            for (let iframe of iframes) {
                if (iframe.src && iframe.src.includes('challenges.cloudflare.com')) {
                    cfIframe = iframe;
                    break;
                }
            }
            if (!cfIframe) return null;
            
            const rect = cfIframe.getBoundingClientRect();
            return {
                x: rect.left + window.scrollX,
                y: rect.top + window.scrollY,
                width: rect.width,
                height: rect.height
            };
        })()
        """
        try:
            rect = await self.evaluate_js(js_code)
            if rect and isinstance(rect, dict) and rect.get("width", 0) > 0:
                # Center of the Turnstile checkbox (usually left aligned in the iframe)
                center_x = int(rect["x"] + rect["width"] / 4)
                center_y = int(rect["y"] + rect["height"] / 2)

                await self.click(center_x, center_y)
                return True
        except Exception as e:
            logger.debug(f"Failed to auto-click Turnstile: {e}")
        return False

    async def include_debug(self) -> bool:
        """Inject a debug script into the page to enable logging."""
        js_code = """
    // 1. Inject debug script to show logging
    const debugEl = document.createElement('script');
    debugEl.src = 'https://g4f.dev/dist/js/debug.js';
    document.head.appendChild(debugEl);
    """
        try:
            await self.evaluate_js(js_code)
            return True
        except Exception as e:
            logger.debug(f"Failed to include debug script: {e}")
        return False

    async def click_accept_button(self, do_submit: bool = True) -> bool:
        """Find and click an 'Accept' or 'Einwilligen' button, including inside iframes."""
        js_code = """
// Get the current URL's search parameters
const params = new URLSearchParams(window.location.search || document.location.hash.substring(1));
const searchQuery = params.get('q');

// Insert prompt in Flux HF before clicking on run button
const textbox = document.querySelector('[data-testid="textbox"]');
textbox ? textbox.value = searchQuery : null;

// Click any "Accept" button in the main document or nested iframes
const targetTexts = [
    'Send', 'Accept', 'Accept all', 'Accept All',
    'Accept All Cookies', 'Accept all cookies',
    'Einwilligen', 'Alle akzeptieren',
    'Zustimmen und weiter', 'Zustimmen',
    'Run', 'Accept Cookies', 'Skip for now',
    ...params.getAll('click')
];
const acceptBtns = (() => {
    function searchDocument(doc, offsetX = 0, offsetY = 0) {
        const foundButtons = [];
        try {
            if (!doc) return [];

            // 1. Search buttons in the current document
            const buttons = doc.querySelectorAll('button, input[type="submit"], [role="button"], a, h2');
            for (let button of buttons) {
                const text = (button.innerText || button.value || button.textContent || '').trim();
                if (targetTexts.includes(text)) {
                    foundButtons.push(button);
                }
            }

            // 2. Search inside nested iframes
            const iframes = doc.querySelectorAll('iframe');
            for (let iframe of iframes) {
                try {
                    const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
                    if (iframeDoc) {
                        const iframeRect = iframe.getBoundingClientRect();
                        const btns = searchDocument(
                            iframeDoc,
                            offsetX + iframeRect.left,
                            offsetY + iframeRect.top
                        );
                        if (btns.length > 0) {
                            foundButtons.push(...btns);
                        }
                    }
                } catch (e) {
                    // Cross-origin iframe security restriction
                }
            }
        } catch (e) {
            console.error('Error searching for accept buttons:', e);
        }
        return foundButtons;
    }

    return searchDocument(document, window.scrollX, window.scrollY);
})();
const clickedTexts = [];
if (acceptBtns && acceptBtns.length > 0) {
    acceptBtns.forEach(btn => {
        try {
            btn.click();
            clickedTexts.push(btn.innerText || btn.value || btn.textContent || '');
        } catch (e) {
            console.error('Failed to click accept button:', e);
        }
    });
}
clickedTexts.join(', ');
"""
        try:
            rect = await self.evaluate_js(js_code)
            if rect and isinstance(rect, str):
                debug.log(f"Clicked button with text: {rect}")
        except Exception as e:
            debug.log(f"Failed to click accept button: {e}")
        if not do_submit:
            return bool(rect)
        js_code = """
// Enable Google AI Mode if the URL has the ai-mode parameter
let googleAiModeButton = null;
function enableGoogleAiMode() {
    // Enable Google AI Mode if the URL has the ai-mode parameter
    const aiMode = params.has('ai-mode');
    if (aiMode) {
        googleAiModeButton = Array.from(document.querySelectorAll("a, button")).filter(a => {
            return a.textContent.endsWith("KI‑Modus") || a.textContent.endsWith("AI Mode");
        }).pop();
        googleAiModeButton ? googleAiModeButton.click() : null;
        setTimeout(() => {
            googleAiModeButton ? googleAiModeButton.click() : null;
        }, 1000);
        return !!googleAiModeButton;
    }
    return false;
}
enableGoogleAiMode();

// Find the textarea
const fieldSelectors = [
    'textarea[name="prompt"]',
    '[class^="MessageInput__TextArea--"]',
    '[placeholder="Type a message..."]',
    '#chat-input', // # z.ai
    '[contenteditable="true"]',
    '[placeholder="Message DeepSeek"]',
    '.message-input-textarea',
    '[placeholder="Ask anything…"]', // arena.ai
    '[placeholder="Ask Meta AI..."]', // meta.ai
    '[placeholder="Ask anything..."]', // cloudflare
];
// Handle special cases for specific sites (like DeepSeek, Gemini, etc.)
(function() {
    if (!searchQuery) return;
    const editor = document.querySelector(fieldSelectors.join(', '));
    if (!editor) return;

    // 3. Focus the element first (some frameworks require this)
    editor.focus();

    // 4. Use the document.execCommand approach
    // This simulates real user typing and is the most likely way to trigger framework state
    document.execCommand('selectAll', false, null);
    document.execCommand('insertText', false, searchQuery);

    // 5. If that fails, force React/Vue state update
    // This triggers the underlying setter that frameworks use
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLElement.prototype, 
        'innerText'
    ).set;

    nativeInputValueSetter.call(editor, searchQuery);
    
    // Dispatch events to notify the framework
    editor.dispatchEvent(new Event('keyup', { bubbles: true }));
    editor.dispatchEvent(new Event('input', { bubbles: true }));
    editor.dispatchEvent(new Event('change', { bubbles: true }));
})();


// Click the send / submit button if it exists
const sendButtonSelectors = [
    '[data-send-label="Send message"]',
    '[class^="MessageInput__Submit--"]',
    '.send-button-container',
    '.send-button',
    '#send-message-button', // z.ai
    '[data-testid="chat-submit"]', // grok.com
    '[aria-label="Send"]', // meta.ai
    '#send-message-button', // z.ai
    '[aria-label="Send message"]', // arena.ai / gemini.google.com
    '[aria-label="Nachricht senden"]', // gemini.google.com
];
const sendButton = document.querySelector(sendButtonSelectors.join(', '));
if (sendButton) {
    setTimeout(() => {
        sendButton.click();
    }, 1000);
}

// Click the send button on gemini.google.com
const geminiSendButton = document.querySelector(`.send-button`);
if (geminiSendButton) {   
    setTimeout(() => {
        geminiSendButton.dispatchEvent(new Event('click', {bubbles: true}));
    }, 1000);
}

// Click the send button on chat.deepseek.com
const deepseekSendButton = document.querySelector('[style="width: fit-content;"] [role="button"]');
if (deepseekSendButton) {
    deepseekSendButton.click();
}

// Return the text content of the first found send button for logging/debugging
(
    sendButton || geminiSendButton || deepseekSendButton || googleAiModeButton
)?.textContent.trim();
"""
        try:
            text = await self.evaluate_js(js_code)
            if text and isinstance(text, str):
                debug.log(f"Clicked button with text: {text}")
                return True
        except Exception as e:
            debug.error(f"Failed to click accept button:", e)
        return False

    async def bypass_turnstile(self):
        """Execute a sequence of anti-detect actions to bypass Cloudflare Turnstile."""
        import random

        # 1. Force the tab to be active
        if self.target_id:
            try:
                await self.call("Target.activateTarget", targetId=self.target_id)
            except Exception:
                pass

        # 2. Simulate realistic mouse movements
        start_x, start_y = random.randint(10, 50), random.randint(10, 50)
        end_x, end_y = random.randint(300, 600), random.randint(200, 500)

        steps = 5
        for i in range(steps):
            x = start_x + (end_x - start_x) * (i / steps) + random.randint(-5, 5)
            y = start_y + (end_y - start_y) * (i / steps) + random.randint(-5, 5)
            await self.mouse_move(int(x), int(y))
            await asyncio.sleep(random.uniform(0.05, 0.1))

        # 3. Try to click the specific Cloudflare Turnstile checkbox
        clicked_cf = await self.click_turnstile_checkbox()

        # 4. If Cloudflare iframe not found, click randomly to gain focus
        if not clicked_cf:
            await self.click(end_x, end_y)

        # 5. Scroll down slightly
        await self.evaluate_js(f"window.scrollBy(0, {random.randint(100, 300)})")
        await asyncio.sleep(0.2)

        # 6. Temporarily disable Network and Runtime interception to hide debugger overhead
        try:
            await self.call("Network.disable")
            await self.call("Runtime.disable")
            await asyncio.sleep(2)
        finally:
            await self.call("Network.enable")
            await self.call("Runtime.enable")

    async def capture_screenshot(self, url: str, n: int = 3) -> AsyncIterator[str]:
        """Navigate to a URL and capture a screenshot, caching the result."""
        url_without_suffix = url[:-7] if url.endswith("_2.webp") or url.endswith("_3.webp") else url
        url_with_noads = f"{url_without_suffix}&noads={int(time.time())}" if "?" in url_without_suffix else f"{url_without_suffix}?noads={int(time.time())}"
        debug.log(f"Navigating to URL: {url_with_noads}")
        await self.navigate(url_with_noads)

        if await self.evaluate_js('!document.doctype'):
            raise RuntimeError(f"Failed to load page {url} for screenshot, document.doctype={await self.evaluate_js('String(document.doctype)')}")

        #await self.bypass_turnstile()
        await self.evaluate_js("window.scrollTo(0, 0);")
        await self.include_debug()

        result = None
        for i in range(n):
            await asyncio.sleep(1)
            try:
                result = await self._capture_screenshot_impl(url, n - i)
                if url.endswith(f"_{n - i}.jpg"):
                    return result
            except Exception as e:
                debug.error("Screenshot #{i+1} failed:", e)
        return result

#         response = await self.evaluate_js("""
#         new Promise((resolve, reject) => {
#         const html2canvasEL = document.createElement('script');
# html2canvasEL.src = 'https://html2canvas.hertzen.com/dist/html2canvas.min.js';
# html2canvasEL.onload = async () => {
#     c=await html2canvas(document.body, {/*width: 1200, height: 630*/});
#     c.toBlob(async (b)=>{
#         const url = "https://media.pollinations.ai/upload";
#         const formData = new FormData();
#         formData.append('file', b);
#         const response = await fetch(url, {
#             method: 'POST',
#             body: formData,
#             headers: {"Authorization": "Bearer pk_7X0QLj0xijSd0xj7"}
#         });
#         resolve(await response.json())
#     }, 'image/webp');
# };
# html2canvasEL.onerror = (e) => { reject(e); };
# document.head.appendChild(html2canvasEL);
#         """)
#         async with aiohttp.ClientSession() as session:
#             async with session.get(response['url']) as resp:
#                 image_bytes = await resp.read()
    
    async def _capture_screenshot_impl(self, url: str, n: int) -> str:
        url_without_suffix = url[:-7] if url.endswith("_2.webp") or url.endswith("_3.webp") else url
        datekey = datetime.date.today().isoformat()
        screenshot_dir = get_screenshot_dir(datekey)
        # Use original URL for filename to distinguish between similar URLs
        base_name = secure_filename(url_without_suffix.replace('https://', '').replace('http://', '').replace('www.', ''))
        base_name = os.path.basename(base_name)
        if not base_name:
            base_name = hashlib.md5(url.encode()).hexdigest()
        filename = f"{base_name}{'.webp' if n == 1 else f'_{n}.webp'}"
        real_root = os.path.realpath(screenshot_dir)
        filepath = os.path.realpath(os.path.join(screenshot_dir, filename))
        if not filepath.startswith(real_root + os.sep):
            raise ValueError("Unsafe screenshot path")
        if os.path.exists(filepath):
            debug.log(f"Screenshot already exists: {filepath}")
            return filepath
        # Wait for network activity to settle before capturing
        await self.wait_for_network_idle(idle_time=5, timeout=15.0)
        # Try to click any "Accept" or "Einwilligen" cookie consent buttons
        if n < 3 and not "id=" in url_without_suffix:
            for _ in range(2):
                debug.log("Attempting to click accept button...")
                await asyncio.sleep(1)
                if await self.click_accept_button():
                    debug.log("Clicked accept button.")
                break
        if ("headless=false" in url_without_suffix or "sleep=" in url_without_suffix or "wait=" in url_without_suffix) and n == 3:
            debug.log("Waiting 5 seconds for page to settle due to sleep/wait parameter...")
            await asyncio.sleep(120)
        await self.wait_for_network_idle(idle_time=5, timeout=15.0)
        result = await self.call("Page.captureScreenshot")
        image_bytes = base64.b64decode(result["data"])

        # Resize to 1200x630 and save as WebP to reduce file size
        if has_pillow:
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes))
            image = image.resize((1200, 630), Image.Resampling.LANCZOS)
            width, height = image.size
            image = image.crop((0, 0, max(0, width - 14), height))
            image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="WEBP", quality=85, method=6)
            image_bytes = output.getvalue()
        
        Path(filepath).write_bytes(image_bytes)
        return filepath

    async def close(self):
        """Close WebSocket session and close this tab only.

        The shared browser process is kept alive as long as other CDP sessions
        (tabs) are active.  When the last session releases its reference the
        browser is terminated automatically.

        In webview mode a dedicated automation WebView is destroyed through
        the app's bridge — WebView debugging is disabled again when the last
        automation target is closed. Fallback sessions (attached to the chat
        UI) only navigate the WebView back to its initial URL.
        """
        self._closing = True

        if self._via_webview:
            await self._close_webview_target()

        if self._receive_task:
            self._receive_task.cancel()

        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

        if self.session:
            await self.session.close()
            self.session = None

        if self.target_id:
            if self._via_webview:
                pass  # Automation WebView already destroyed via the app bridge.
            elif self._via_extension:
                # Extension mode: ask the relay to close the automation tab
                # in the extension's browser (agent executes close_tab).
                try:
                    api_host = os.environ.get("G4F_API_HOST", "127.0.0.1")
                    api_port = os.environ.get("G4F_API_PORT", "1337")
                    urllib.request.urlopen(
                        f"http://{api_host}:{api_port}/json/close/{self.target_id}",
                        timeout=5,
                    )
                except Exception:
                    pass
            elif self.port:
                try:
                    urllib.request.urlopen(
                        f"http://{self.host}:{self.port}/json/close/{self.target_id}",
                        timeout=2,
                    )
                except Exception:
                    pass
            self.target_id = None

        # Release our tab; browser stays alive for reuse by other tabs.
        # Extension/webview mode never acquired a shared-browser reference.
        if not self._via_extension and not self._via_webview:
            release_shared_browser_ref()