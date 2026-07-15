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
SyncCDPSession (Sync) — for Turnstile-solving providers like DeepInfra.
──────────────────────────────────────────────────────────────────────
  • Synchronous blocking recv() loop — waits as long as the browser needs.
  • No async timeouts — more reliable for slow/interactive pages.
  • Run from an async context via run_in_executor().
  • Requires: pip install websocket-client

  Example:
      def run_sync():
          session = SyncCDPSession(port=12345, headless=False)
          session.start_chrome()
          try:
              session.navigate("https://example.com")
              title = session.evaluate_js("document.title")
              return title
          finally:
              session.close()

      title = await asyncio.get_event_loop().run_in_executor(None, run_sync)

──────────────────────────────────────────────────────────────────────
Common features:
  • Auto-detects Chrome/Chromium/Edge path via BrowserConfig or system PATH.
  • Stores browser profiles in g4f cookies directory (no project root pollution).
  • Offscreen windowed mode (--window-position=-2000,-2000) bypasses Turnstile.
"""

import asyncio
import json
import logging
import os
import random
import shutil
import platform
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List

try:
    import aiohttp
except ImportError:
    pass

logger = logging.getLogger(__name__)

def find_chrome_path() -> Optional[str]:
    """Search for Google Chrome or Chromium binary depending on OS."""
    try:
        from g4f.cookies import BrowserConfig
        if BrowserConfig.executable_path and os.path.exists(BrowserConfig.executable_path):
            return BrowserConfig.executable_path
    except ImportError:
        pass

    for name in ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'chrome', 'msedge', 'helium']:
        path = shutil.which(name)
        if path:
            return path
            
    sys_name = platform.system().lower()
    if sys_name == 'linux':
        for path in [
            '/usr/bin/google-chrome', 
            '/opt/google/chrome/google-chrome', 
            '/usr/bin/chromium-browser',
            '/usr/bin/microsoft-edge',
            '/opt/helium/helium'
        ]:
            if os.path.exists(path):
                return path
    elif sys_name in ('macos', 'darwin'):
        for path in [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge'
        ]:
            if os.path.exists(path):
                return path
    elif sys_name == 'windows':
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Helium\Application\helium.exe",
            r"C:\Program Files (x86)\Helium\Application\helium.exe",
            r"C:\Program Files\Helium\helium.exe"
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

def _cleanup_shared_browser():
    global _shared_browser_process
    if _shared_browser_process:
        try:
            _shared_browser_process.terminate()
        except Exception:
            pass
        _shared_browser_process = None

atexit.register(_cleanup_shared_browser)

def find_running_cdp_port(host: str) -> Optional[int]:
    """Scan running processes for an active Chrome/Helium instance with remote debugging enabled."""
    try:
        import psutil
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                proc_name = (proc.info.get('name') or '').lower()
                if any(n in proc_name for n in ('chrome', 'chromium', 'edge', 'helium', 'app')):
                    for arg in cmdline:
                        if arg.startswith('--remote-debugging-port='):
                            try:
                                port = int(arg.split('=')[1])
                                # Verify if it's reachable and working
                                with urllib.request.urlopen(f"http://{host}:{port}/json", timeout=0.5) as response:
                                    if response.status == 200:
                                        return port
                            except Exception:
                                pass
            except Exception:
                pass
    except Exception:
        pass
    return None

def get_shared_browser(host: str, preferred_port: int, headless: bool = True) -> int:
    """
    Ensure a single shared browser instance is running and return its port.
    If a browser is already running anywhere on the system, we use it directly.
    """
    global _shared_browser_process, _shared_browser_port
    
    with _shared_browser_lock:
        # 1. If we already started a shared browser in this thread, check if it's still alive/reachable
        if _shared_browser_port is not None:
            try:
                with urllib.request.urlopen(f"http://{host}:{_shared_browser_port}/json", timeout=0.5) as response:
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
            s.bind(('', 0))
            port = s.getsockname()[1]

        # Use standard user config directory for profile caching (like other g4f browsers)
        try:
            from platformdirs import user_config_dir
            user_data_dir = user_config_dir("g4f-cdp")
        except ImportError:
            import tempfile
            user_data_dir = os.path.join(tempfile.gettempdir(), "g4f_chrome_profile_cdp")
        os.makedirs(user_data_dir, exist_ok=True)

        cmd = [
            chrome_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            "--window-position=-2000,-2000",
            "--window-size=1024,768",
            "--no-default-browser-check",
            "--disable-suggestions-ui",
            "--no-first-run",
            "--disable-infobars",
            "--disable-popup-blocking",
            "--hide-crash-restore-bubble",
            "--disable-features=PrivacySandboxSettings4",
            "--remote-allow-origins=*"
        ]
        if headless:
            cmd.append("--headless=new")
        
        _shared_browser_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait up to 20 seconds for readiness
        for _ in range(40):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://{host}:{port}/json", timeout=1) as response:
                    if response.status == 200:
                        _shared_browser_port = port
                        return _shared_browser_port
            except Exception:
                pass

        if _shared_browser_process:
            try:
                _shared_browser_process.terminate()
            except Exception:
                pass
            _shared_browser_process = None
        raise RuntimeError(f"Failed to start shared Chrome on port {port}")

class CDPSession:
    def __init__(self, port: Optional[int] = None, host: Optional[str] = None, user_data_dir: Optional[str] = None, headless: bool = True):
        try:
            from g4f.cookies import BrowserConfig
            if port is None:
                port = BrowserConfig.port
            if host is None:
                host = BrowserConfig.host
        except ImportError:
            pass
            
        if port is None:
            port = 9222
        if host is None:
            host = '127.0.0.1'
            
        self.port = port
        self.host = host
        self.headless = headless
        self.user_data_dir = user_data_dir  # Ignored if using shared pool, but kept for compatibility
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
        
        # Network event loggers
        self.network_requests: List[dict] = []
        self.network_responses: List[dict] = []

    async def start(self):
        """Launch/get shared Chrome and connect via CDP targeting a new tab."""
        self.port = get_shared_browser(self.host, self.port, self.headless)
        
        # Create a new tab target
        ws_url = None
        for _ in range(10):
            try:
                req = urllib.request.Request(f"http://{self.host}:{self.port}/json/new", method="PUT")
                with urllib.request.urlopen(req, timeout=2) as response:
                    target = json.loads(response.read().decode('utf-8'))
                    ws_url = target.get('webSocketDebuggerUrl')
                    self.target_id = target.get('id')
                    if ws_url:
                        break
            except Exception:
                await asyncio.sleep(0.5)
                
        if not ws_url:
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
                                q.put_nowait(params)
        except Exception as e:
            if not self._closing:
                logger.error(f"CDP receiver loop error: {e}")

    async def call(self, method: str, **params) -> dict:
        """Call a CDP method and wait for its result."""
        if not self.ws:
            raise RuntimeError("CDPSession is not connected")
            
        self.id_counter += 1
        req_id = self.id_counter
        
        fut = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = fut
        
        payload = {
            "id": req_id,
            "method": method,
            "params": params
        }
        await self.ws.send_json(payload)
        
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
        res = await self.call("Runtime.evaluate", expression=expression, returnByValue=True)
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
                "expires": cookie.get("expires")
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
        fut = asyncio.get_running_loop().create_future()
        if "Page.loadEventFired" not in self._event_handlers:
            self._event_handlers["Page.loadEventFired"] = []
        self._event_handlers["Page.loadEventFired"].append(fut)
        
        await self.call("Page.navigate", url=url)
        
        try:
            await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            if fut in self._event_handlers.get("Page.loadEventFired", []):
                self._event_handlers["Page.loadEventFired"].remove(fut)
            logger.warning(f"Timeout waiting for Page.loadEventFired when navigating to {url}")

    async def mouse_move(self, x: int, y: int):
        """Simulate a mouse movement to the given coordinates."""
        await self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)

    async def click(self, x: int, y: int, delay: float = 0.05):
        """Simulate a realistic mouse click at the given coordinates."""
        await self.mouse_move(x, y)
        await asyncio.sleep(0.02)
        await self.call("Input.dispatchMouseEvent", type="mousePressed", button="left", clickCount=1, x=x, y=y)
        await asyncio.sleep(delay)
        await self.call("Input.dispatchMouseEvent", type="mouseReleased", button="left", clickCount=1, x=x, y=y)

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
            
        # 3. Click randomly to gain focus
        await self.click(end_x, end_y)
        
        # 4. Scroll down slightly
        await self.evaluate_js(f"window.scrollBy(0, {random.randint(100, 300)})")
        await asyncio.sleep(0.2)
        
        # 5. Temporarily disable Network interception to hide debugger overhead
        try:
            await self.call("Network.disable")
            await asyncio.sleep(2)
        finally:
            await self.call("Network.enable")

    async def close(self):
        """Close WebSocket session and close the specific target tab."""
        self._closing = True
        
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
            
        if self.target_id and self.port:
            try:
                urllib.request.urlopen(f"http://{self.host}:{self.port}/json/close/{self.target_id}", timeout=2)
            except Exception:
                pass
            self.target_id = None




class SyncCDPSession:
    """
    Synchronous (blocking) CDP client for use in Turnstile-solving providers
    (e.g. DeepInfra) where async timeout-based approach is unreliable.

    Unlike CDPSession which uses asyncio+aiohttp and futures with timeouts,
    this class uses a simple blocking while-loop recv() model — it will wait
    as long as the browser needs without the risk of a premature TimeoutError.

    Requires: pip install websocket-client

    Example Usage (run from an async context via executor):
        def run_sync():
            session = SyncCDPSession(port=12345, headless=False)
            session.start_chrome()
            try:
                session.navigate("https://example.com")
                title = session.evaluate_js("document.title")
                return title
            finally:
                session.close()

        result = await asyncio.get_event_loop().run_in_executor(None, run_sync)
    """

    def __init__(self, port: int = 9222, host: str = '127.0.0.1', user_data_dir: Optional[str] = None, headless: bool = False):
        self.port = port
        self.host = host
        self.headless = headless
        self.user_data_dir = user_data_dir  # Ignored if using shared pool, but kept for compatibility
        self.process = None
        self.ws = None
        self.target_id = None
        self.id_counter = 0
        
        # Network event loggers
        self.network_requests: List[dict] = []
        self.network_responses: List[dict] = []

    def start_chrome(self):
        """Launch/get shared Chrome and connect via CDP targeting a new tab."""
        self.port = get_shared_browser(self.host, self.port, self.headless)
        
        # Create a new tab target
        ws_url = None
        for _ in range(10):
            try:
                req = urllib.request.Request(f"http://{self.host}:{self.port}/json/new", method="PUT")
                with urllib.request.urlopen(req, timeout=2) as response:
                    target = json.loads(response.read().decode('utf-8'))
                    ws_url = target.get('webSocketDebuggerUrl')
                    self.target_id = target.get('id')
                    if ws_url:
                        break
            except Exception:
                time.sleep(0.5)
                
        if not ws_url:
            raise RuntimeError(f"Failed to create new tab target on port {self.port}")
            
        self._connect(ws_url)

    def _connect(self, ws_url: str):
        """Connect via WebSocket to the target."""
        try:
            from websocket import create_connection
        except ImportError:
            raise ImportError('Install "websocket-client" package: pip install websocket-client')

        self.ws = create_connection(ws_url)
        self.ws.settimeout(60)  # Prevent infinite hang if Chrome crashes mid-call

        # Enable essential CDP domains
        self.call("Page.enable")
        self.call("DOM.enable")
        self.call("Runtime.enable")
        self.call("Network.enable")
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)

    def call(self, method: str, **params) -> dict:
        """Send a CDP command and block until the matching response arrives, logging events."""
        self.id_counter += 1
        payload = {"id": self.id_counter, "method": method, "params": params}
        self.ws.send(json.dumps(payload))

        # Blocking loop with a 60s socket timeout — won't hang forever if browser exits
        while True:
            response = json.loads(self.ws.recv())
            if "id" in response:
                if response.get("id") == self.id_counter:
                    if "error" in response:
                        raise RuntimeError(f"CDP error in {method}: {response['error']}")
                    return response.get("result", {})
            else:
                # Event
                event_method = response.get("method")
                event_params = response.get("params", {})
                if event_method == "Network.requestWillBeSent":
                    self.network_requests.append(event_params)
                elif event_method == "Network.responseReceived":
                    self.network_responses.append(event_params)

    def evaluate_js(self, expression: str) -> Any:
        """Execute JS on the page and return the primitive result value."""
        res = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return res.get("result", {}).get("value")

    def get_cookies(self) -> dict:
        """Retrieve all cookies from the browser as a name-value dict."""
        cookies = self.get_cookies_list()
        return {c["name"]: c["value"] for c in cookies}

    def get_cookies_list(self, urls: Optional[List[str]] = None) -> List[dict]:
        """Retrieve full cookie objects from the browser session."""
        params = {}
        if urls:
            params["urls"] = urls
        res = self.call("Network.getCookies", **params)
        return res.get("cookies", [])

    def set_cookies(self, cookies: List[dict]):
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
                "expires": cookie.get("expires")
            }
            params = {k: v for k, v in params.items() if v is not None}
            if "domain" not in params and "url" not in params:
                params["url"] = "https://deepinfra.com"
            self.call("Network.setCookie", **params)

    def navigate(self, url: str):
        """Navigate to a URL and wait for initial load."""
        self.call("Page.navigate", url=url)
        time.sleep(2.0)

    def click(self, x: int = 200, y: int = 400):
        """
        Simulate a real mouse click at (x, y) on the page.

        Gives the page window focus, which signals Cloudflare that a real user
        is present. This significantly speeds up Turnstile token generation.
        Call this once after navigate(), before polling for the token.
        """
        self.call("Input.dispatchMouseEvent",
                  type="mousePressed", x=x, y=y,
                  button="left", clickCount=1)
        self.call("Input.dispatchMouseEvent",
                  type="mouseReleased", x=x, y=y,
                  button="left", clickCount=1)

    def close(self):
        """Close WebSocket session and close the specific target tab."""
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

        if self.target_id and self.port:
            try:
                urllib.request.urlopen(f"http://{self.host}:{self.port}/json/close/{self.target_id}", timeout=2)
            except Exception:
                pass
            self.target_id = None

