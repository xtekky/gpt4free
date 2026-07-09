import os
import shutil
import platform
import subprocess
import tempfile
import time
import json
import urllib.request
import urllib.error

def find_chrome_path():
    """Search for Google Chrome or Chromium binary depending on OS."""
    # Respect g4f's custom BrowserConfig.executable_path if configured
    try:
        from g4f.cookies import BrowserConfig
        if BrowserConfig.executable_path and os.path.exists(BrowserConfig.executable_path):
            return BrowserConfig.executable_path
    except ImportError:
        pass

    # First, search using shutil.which
    for name in ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'chrome']:
        path = shutil.which(name)
        if path:
            return path
            
    # System default paths
    sys_name = platform.system().lower()
    if sys_name == 'linux':
        for path in ['/usr/bin/google-chrome', '/opt/google/chrome/google-chrome', '/usr/bin/chromium-browser']:
            if os.path.exists(path):
                return path
    elif sys_name in ('macos', 'darwin'):
        path = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
        if os.path.exists(path):
            return path
    elif sys_name == 'windows':
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        for path in paths:
            if os.path.exists(path):
                return path
    return None

class Turnstile:
    def __init__(self, port=9222, user_data_dir=None):
        self.port = port
        
        # Respect g4f's central cookies/cache directory if available
        if user_data_dir is None:
            try:
                from g4f.cookies import get_cookies_dir
                cookies_dir = get_cookies_dir()
                if cookies_dir:
                    user_data_dir = os.path.join(cookies_dir, "chrome_profile_light")
            except ImportError:
                pass
                
        # Default fallback to a persistent directory in temp folder
        if user_data_dir is None:
            user_data_dir = os.path.join(tempfile.gettempdir(), "g4f_chrome_profile_light")
            
        self.user_data_dir = user_data_dir
        self.process = None
        self.ws = None
        self.id_counter = 0

    def start_chrome(self):
        """Launch Chrome with CDP remote debugging port."""
        chrome_path = find_chrome_path()
        if not chrome_path:
            raise RuntimeError("Google Chrome / Chromium executable not found.")
            
        print(f"[Turnstile] Launching Chrome: {chrome_path} on port {self.port}")
        
        # Create an isolated profile directory
        os.makedirs(self.user_data_dir, exist_ok=True)
        
        # Launch arguments matching DrissionPage to minimize detection
        cmd = [
            chrome_path,
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.user_data_dir}",
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
        
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.connect()  # Wait for Chrome to be ready and connect via WebSocket
    
    def connect(self):
        # Wait for CDP port readiness and retrieve the WebSocket URL
        ws_url = None
        for i in range(40):  # Up to 20 seconds
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json", timeout=2) as req:
                    targets = json.loads(req.read().decode('utf-8'))
                    for target in targets:
                        if target.get('type') in ('page', 'webview'):
                            ws_url = target.get('webSocketDebuggerUrl')
                            break
                    if ws_url:
                        break
            except (urllib.error.URLError, ConnectionResetError, ConnectionRefusedError):
                pass
                
        if not ws_url:
            self.close()
            raise RuntimeError(f"Failed to connect to Chrome debugging port 127.0.0.1:{self.port}")
            
        print(f"[Turnstile] Connected to CDP WebSocket: {ws_url}")
        try:
            from websocket import create_connection
        except ImportError:
            from g4f.errors import MissingRequirementsError
            raise MissingRequirementsError('Install "websocket-client" package | pip install websocket-client')
            
        self.ws = create_connection(ws_url)
        
        # Enable necessary CDP domains
        self.call_cdp("Page.enable")
        self.call_cdp("DOM.enable")
        self.call_cdp("Runtime.enable")
        self.call_cdp("Emulation.setFocusEmulationEnabled", enabled=True)

    def call_cdp(self, method, **params):
        """Call CDP method and wait for response."""
        self.id_counter += 1
        payload = {
            "id": self.id_counter,
            "method": method,
            "params": params
        }
        self.ws.send(json.dumps(payload))
        
        while True:
            response = json.loads(self.ws.recv())
            if response.get("id") == self.id_counter:
                if "error" in response:
                    raise RuntimeError(f"CDP Error calling {method}: {response['error']}")
                return response.get("result", {})

    def evaluate_js(self, expression):
        """Execute JS code on the page and return the result."""
        res = self.call_cdp("Runtime.evaluate", expression=expression, returnByValue=True)
        return res.get("result", {}).get("value")

    def get_token(self, model: str) -> str:
        """Retrieve Turnstile token for the model."""
        if not self.ws:
            self.start_chrome()
            
        target_url = f"https://deepinfra.com/{model}"
        print(f"[Turnstile] Navigating to {target_url}...")
        self.call_cdp("Page.navigate", url=target_url)
        
        # Give some time to load
        time.sleep(2.0)
        
        # Inject completions request blocker ONLY in main frame (to avoid tampering with fetch inside Cloudflare's iframe)
        fetch_blocker_js = """
        const origFetch = window.fetch;
        window.fetch = async function(...args) {
            let url = args[0];
            if (typeof url === 'string' && url.includes('/chat/completions')) {
                return new Response('{}', {status: 200});
            }
            return origFetch.apply(this, args);
        };
        """
        self.evaluate_js(fetch_blocker_js)
        
        # Try to click "Accept" on cookies consent popup if present
        self.evaluate_js("""
        (() => {
            const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Accept');
            if (btn) btn.click();
        })()
        """)
        
        # Wait for textarea readiness, focus and input text
        print("[Turnstile] Waiting for active textarea...")
        text_entered = False
        for _ in range(40):  # Up to 20 seconds
            try:
                ready = self.evaluate_js("""
                (() => {
                    const ta = document.querySelector('textarea');
                    if (!ta) return 'no_textarea';
                    if (ta.disabled) return 'disabled';
                    ta.click();
                    ta.focus();
                    ta.scrollIntoView({ block: 'center' });
                    return 'ready';
                })()
                """)
                
                if ready == 'ready':
                    print("[Turnstile] Textarea found, focusing and entering text...")
                    
                    # Retrieve textarea nodeId for native focusing
                    doc = self.call_cdp('DOM.getDocument')
                    root_id = doc['root']['nodeId']
                    textarea = self.call_cdp('DOM.querySelector', nodeId=root_id, selector='textarea')
                    
                    # Native focus via CDP
                    self.call_cdp('DOM.focus', nodeId=textarea['nodeId'])
                    
                    # Enter text via native CDP command
                    self.call_cdp("Input.insertText", text="Test Prompt")
                    
                    # Give React some time to process input before pressing Enter
                    time.sleep(0.5)
                    
                    # Simulate Enter keypress via native CDP events (matching DrissionPage implementation)
                    self.call_cdp("Input.dispatchKeyEvent", 
                                  type="keyDown", 
                                  windowsVirtualKeyCode=13, 
                                  key="Enter", 
                                  code="Enter", 
                                  text="\r", 
                                  unmodifiedText="\r")
                    self.call_cdp("Input.dispatchKeyEvent", 
                                  type="keyUp", 
                                  windowsVirtualKeyCode=13, 
                                  key="Enter", 
                                  code="Enter", 
                                  text="\r", 
                                  unmodifiedText="\r")
                    
                    text_entered = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
            
        if not text_entered:
            print("[-] Turnstile initiation error: textarea not found or disabled.")
            return ""
            
        # Poll page for Turnstile token presence
        print("[Turnstile] Waiting for Cloudflare Turnstile solve...")
        token_js = "document.querySelector('[name=cf-turnstile-response]') ? document.querySelector('[name=cf-turnstile-response]').value : ''"
        token = ""
        for i in range(120):  # Up to 60 seconds
            try:
                token = self.evaluate_js(token_js)
                if token:
                    print(f"[Turnstile] Token generated on check {i+1}!")
                    break
            except Exception:
                pass
            time.sleep(0.5)
            
        return token
 
    def close(self):
        """Close connection and terminate Chrome process."""
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
            
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
