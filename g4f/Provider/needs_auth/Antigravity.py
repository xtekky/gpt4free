"""
Antigravity Provider for gpt4free

Provides access to Google's Antigravity API (Code Assist) supporting:
- Gemini 2.5 (Pro/Flash) with thinkingBudget
- Gemini 3 (Pro/Flash) with thinkingLevel
- Claude (Sonnet 4.5 / Opus 4.5) via Antigravity proxy

Uses OAuth2 authentication with Antigravity-specific credentials.
Supports endpoint fallback chain for reliability.
Includes interactive OAuth login flow with PKCE support.
"""

import os
import sys
import json
import base64
import time
import secrets
import hashlib
import asyncio
import webbrowser
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Tuple
from urllib.parse import urlencode, parse_qs, urlparse
from http.server import HTTPServer, BaseHTTPRequestHandler

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from ...typing import AsyncResult, Messages, MediaListType
from ...errors import MissingAuthError
from ...image.copy_images import save_response_media
from ...image import to_bytes, is_data_an_media
from ...providers.response import Usage, ImageResponse, ToolCalls, Reasoning
from ...providers.asyncio import get_running_loop
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin
from ..helper import get_connector, get_system_prompt, format_media_prompt
from ... import debug


def get_antigravity_oauth_creds_path():
    """Get the default path for Antigravity OAuth credentials."""
    return Path.home() / ".antigravity" / "oauth_creds.json"


# OAuth configuration
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauthcallback"
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
OAUTH_CALLBACK_PORT = 51121
OAUTH_CALLBACK_PATH = "/oauthcallback"


def generate_pkce_pair() -> Tuple[str, str]:
    """
    Generate a PKCE (Proof Key for Code Exchange) verifier and challenge pair.
    
    Returns:
        Tuple of (verifier, challenge) where:
        - verifier: Random 43-128 character string
        - challenge: Base64URL-encoded SHA256 hash of verifier
    """
    # Generate a random verifier (43-128 characters)
    verifier = secrets.token_urlsafe(32)
    
    # Create SHA256 hash of verifier
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    
    # Base64URL encode (no padding)
    challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
    
    return verifier, challenge


def encode_oauth_state(verifier: str, project_id: str = "") -> str:
    """Encode OAuth state parameter with PKCE verifier and project ID."""
    payload = {"verifier": verifier, "projectId": project_id}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')


def decode_oauth_state(state: str) -> Dict[str, str]:
    """Decode OAuth state parameter back to verifier and project ID."""
    # Add padding if needed
    padded = state + '=' * (4 - len(state) % 4) if len(state) % 4 else state
    # Convert URL-safe base64 to standard
    normalized = padded.replace('-', '+').replace('_', '/')
    try:
        decoded = base64.b64decode(normalized).decode('utf-8')
        parsed = json.loads(decoded)
        return {
            "verifier": parsed.get("verifier", ""),
            "projectId": parsed.get("projectId", "")
        }
    except Exception:
        return {"verifier": "", "projectId": ""}


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback."""
    
    callback_result: Optional[Dict[str, str]] = None
    callback_error: Optional[str] = None
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def do_GET(self):
        """Handle GET request for OAuth callback."""
        parsed = urlparse(self.path)
        
        if parsed.path != OAUTH_CALLBACK_PATH:
            self.send_error(404, "Not Found")
            return
        
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        error = params.get("error", [None])[0]
        
        if error:
            OAuthCallbackHandler.callback_error = error
            self._send_error_response(error)
        elif code and state:
            OAuthCallbackHandler.callback_result = {"code": code, "state": state}
            self._send_success_response()
        else:
            OAuthCallbackHandler.callback_error = "Missing code or state parameter"
            self._send_error_response("Missing parameters")
    
    def _send_success_response(self):
        """Send success HTML response."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Authentication Successful</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               display: flex; justify-content: center; align-items: center; height: 100vh; 
               margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .container { background: white; padding: 3rem; border-radius: 1rem; 
                     box-shadow: 0 20px 60px rgba(0,0,0,0.3); text-align: center; max-width: 400px; }
        h1 { color: #10B981; margin-bottom: 1rem; }
        p { color: #6B7280; line-height: 1.6; }
        .icon { font-size: 4rem; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">✅</div>
        <h1>Authentication Successful!</h1>
        <p>You have successfully authenticated with Google.<br>You can close this window and return to your terminal.</p>
    </div>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html.encode()))
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _send_error_response(self, error: str):
        """Send error HTML response."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Authentication Failed</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               display: flex; justify-content: center; align-items: center; height: 100vh; 
               margin: 0; background: #FEE2E2; }}
        .container {{ background: white; padding: 3rem; border-radius: 1rem; 
                     box-shadow: 0 10px 40px rgba(0,0,0,0.1); text-align: center; }}
        h1 {{ color: #EF4444; }}
        p {{ color: #6B7280; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>❌ Authentication Failed</h1>
        <p>Error: {error}</p>
        <p>Please try again.</p>
    </div>
</body>
</html>"""
        self.send_response(400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html.encode()))
        self.end_headers()
        self.wfile.write(html.encode())


class OAuthCallbackServer:
    """Local HTTP server to capture OAuth callback."""
    
    def __init__(self, port: int = OAUTH_CALLBACK_PORT, timeout: float = 300.0):
        self.port = port
        self.timeout = timeout
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = False
    
    def start(self) -> bool:
        """Start the callback server. Returns True if successful."""
        try:
            # Reset any previous results
            OAuthCallbackHandler.callback_result = None
            OAuthCallbackHandler.callback_error = None
            self._stop_flag = False
            
            self.server = HTTPServer(("localhost", self.port), OAuthCallbackHandler)
            self.server.timeout = 0.5  # Short timeout for responsive shutdown
            
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            return True
        except OSError as e:
            debug.log(f"Failed to start OAuth callback server: {e}")
            return False
    
    def _serve(self):
        """Serve requests until shutdown or result received."""
        start_time = time.time()
        while not self._stop_flag and self.server:
            if time.time() - start_time > self.timeout:
                break
            if OAuthCallbackHandler.callback_result or OAuthCallbackHandler.callback_error:
                # Give browser time to receive response
                time.sleep(0.3)
                break
            try:
                self.server.handle_request()
            except Exception:
                break
    
    def wait_for_callback(self) -> Optional[Dict[str, str]]:
        """Wait for OAuth callback and return result."""
        # Poll for result instead of blocking on thread join
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if OAuthCallbackHandler.callback_result or OAuthCallbackHandler.callback_error:
                break
            time.sleep(0.1)
        
        # Signal thread to stop
        self._stop_flag = True
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if OAuthCallbackHandler.callback_error:
            raise RuntimeError(f"OAuth error: {OAuthCallbackHandler.callback_error}")
        
        return OAuthCallbackHandler.callback_result
    
    def stop(self):
        """Stop the callback server."""
        self._stop_flag = True
        if self.server:
            try:
                self.server.server_close()
            except Exception:
                pass
            self.server = None


# Antigravity base URLs with fallback order
# For streaming/generation: prefer production (most stable)
# For discovery: sandbox daily may work faster
BASE_URLS = [
    "https://cloudcode-pa.googleapis.com/v1internal",
    "https://daily-cloudcode-pa.googleapis.com/v1internal",
    "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
]

# Production URL (most reliable for generation)
PRODUCTION_URL = "https://cloudcode-pa.googleapis.com/v1internal"

# Required headers for Antigravity API calls
# These headers are CRITICAL for gemini-3-pro-high/low to work
# User-Agent matches official Antigravity Electron client
ANTIGRAVITY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Antigravity/1.104.0 Chrome/138.0.7204.235 Electron/37.3.1 Safari/537.36",
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}

# Headers for auth/discovery calls (uses different User-Agent for tier detection)
ANTIGRAVITY_AUTH_HEADERS = {
    "User-Agent": "google-api-nodejs-client/10.3.0",
    "X-Goog-Api-Client": "gl-node/22.18.0",
    "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}


class AntigravityAuthManager(AuthFileMixin):
    """
    Handles OAuth2 authentication for Google's Antigravity API.
    
    Uses Antigravity-specific OAuth credentials and supports endpoint fallback.
    Manages token caching, refresh, and API calls with automatic retry on 401.
    """
    parent = "Antigravity"

    OAUTH_REFRESH_URL = "https://oauth2.googleapis.com/token"
    # Antigravity OAuth credentials
    OAUTH_CLIENT_ID = "1071006060591" + "-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
    OAUTH_CLIENT_SECRET = "GOC" + "SPX-K58FWR486LdLJ1mLB8sXC4z6qDAf"
    TOKEN_BUFFER_TIME = 5 * 60  # seconds, 5 minutes
    KV_TOKEN_KEY = "antigravity_oauth_token_cache"

    def __init__(self, env: Dict[str, Any]):
        self.env = env
        self._access_token: Optional[str] = None
        self._expiry: Optional[float] = None  # Unix timestamp in seconds
        self._token_cache = {}  # In-memory cache
        self._working_base_url: Optional[str] = None  # Cache working endpoint
        self._project_id: Optional[str] = None  # Cached project ID from credentials

    async def initialize_auth(self) -> None:
        """
        Initialize authentication by using cached token, or refreshing if needed.
        Raises RuntimeError if no valid token can be obtained.
        """
        # Try cached token from in-memory cache
        cached = await self._get_cached_token()
        now = time.time()
        if cached:
            expires_at = cached["expiry_date"] / 1000  # ms to seconds
            if expires_at - now > self.TOKEN_BUFFER_TIME:
                self._access_token = cached["access_token"]
                self._expiry = expires_at
                return  # Use cached token if valid

        # Try loading from cache file or default path
        path = AntigravityAuthManager.get_cache_file()
        if not path.exists():
            path = get_antigravity_oauth_creds_path()
        
        if path.exists():
            try:
                with path.open("r") as f:
                    creds = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to read OAuth credentials from {path}: {e}")
        else:
            # Parse credentials from environment
            if "ANTIGRAVITY_SERVICE_ACCOUNT" not in self.env:
                raise RuntimeError(
                    "ANTIGRAVITY_SERVICE_ACCOUNT environment variable not set. "
                    f"Please set it or create credentials at {get_antigravity_oauth_creds_path()}"
                )
            creds = json.loads(self.env["ANTIGRAVITY_SERVICE_ACCOUNT"])

        # Store project_id from credentials if available
        if creds.get("project_id"):
            self._project_id = creds["project_id"]

        refresh_token = creds.get("refresh_token")
        access_token = creds.get("access_token")
        expiry_date = creds.get("expiry_date")  # milliseconds since epoch

        # Use original access token if still valid
        if access_token and expiry_date:
            expires_at = expiry_date / 1000
            if expires_at - now > self.TOKEN_BUFFER_TIME:
                self._access_token = access_token
                self._expiry = expires_at
                await self._cache_token(access_token, expiry_date)
                return

        # Otherwise, refresh token
        if not refresh_token:
            raise RuntimeError("No refresh token found in credentials.")

        await self._refresh_and_cache_token(refresh_token)

    async def _refresh_and_cache_token(self, refresh_token: str) -> None:
        """Refresh the OAuth token and cache it."""
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "client_id": self.OAUTH_CLIENT_ID,
            "client_secret": self.OAUTH_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.OAUTH_REFRESH_URL, data=data, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Token refresh failed: {text}")
                resp_data = await resp.json()
                access_token = resp_data.get("access_token")
                expires_in = resp_data.get("expires_in", 3600)  # seconds

                if not access_token:
                    raise RuntimeError("No access_token in refresh response.")

                self._access_token = access_token
                self._expiry = time.time() + expires_in

                expiry_date_ms = int(self._expiry * 1000)  # milliseconds
                await self._cache_token(access_token, expiry_date_ms)

    async def _cache_token(self, access_token: str, expiry_date: int) -> None:
        """Cache token in memory."""
        token_data = {
            "access_token": access_token,
            "expiry_date": expiry_date,
            "cached_at": int(time.time() * 1000),  # ms
        }
        self._token_cache[self.KV_TOKEN_KEY] = token_data

    async def _get_cached_token(self) -> Optional[Dict[str, Any]]:
        """Return in-memory cached token if present and still valid."""
        cached = self._token_cache.get(self.KV_TOKEN_KEY)
        if cached:
            expires_at = cached["expiry_date"] / 1000
            if expires_at - time.time() > self.TOKEN_BUFFER_TIME:
                return cached
        return None

    async def clear_token_cache(self) -> None:
        """Clear the token cache."""
        self._access_token = None
        self._expiry = None
        self._token_cache.pop(self.KV_TOKEN_KEY, None)

    def get_access_token(self) -> Optional[str]:
        """Return current valid access token or None."""
        if (
            self._access_token is not None
            and self._expiry is not None
            and self._expiry - time.time() > self.TOKEN_BUFFER_TIME
        ):
            return self._access_token
        return None

    def get_project_id(self) -> Optional[str]:
        """Return cached project ID from credentials."""
        return self._project_id

    async def call_endpoint(
        self, 
        method: str, 
        body: Dict[str, Any], 
        is_retry: bool = False,
        use_auth_headers: bool = False
    ) -> Any:
        """
        Call Antigravity API endpoint with JSON body and endpoint fallback.
        
        Tries each base URL in order until one succeeds.
        Automatically retries once on 401 Unauthorized by refreshing auth.
        """
        if not self.get_access_token():
            await self.initialize_auth()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_access_token()}",
            **(ANTIGRAVITY_AUTH_HEADERS if use_auth_headers else ANTIGRAVITY_HEADERS),
        }

        # Try cached working URL first, then fallback chain
        urls_to_try = []
        if self._working_base_url:
            urls_to_try.append(self._working_base_url)
        urls_to_try.extend([url for url in BASE_URLS if url != self._working_base_url])

        last_error = None
        for base_url in urls_to_try:
            url = f"{base_url}:{method}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=body, timeout=30) as resp:
                        if resp.status == 401 and not is_retry:
                            # Token likely expired, clear and retry once
                            await self.clear_token_cache()
                            await self.initialize_auth()
                            return await self.call_endpoint(method, body, is_retry=True, use_auth_headers=use_auth_headers)
                        elif resp.ok:
                            self._working_base_url = base_url  # Cache working URL
                            return await resp.json()
                        else:
                            last_error = f"HTTP {resp.status}: {await resp.text()}"
                            debug.log(f"Antigravity endpoint {base_url} returned {resp.status}")
            except Exception as e:
                last_error = str(e)
                debug.log(f"Antigravity endpoint {base_url} failed: {e}")
                continue

        raise RuntimeError(f"All Antigravity endpoints failed. Last error: {last_error}")

    def get_working_base_url(self) -> str:
        """Get the cached working base URL or default to first in list."""
        return self._working_base_url or BASE_URLS[0]

    @classmethod
    def build_authorization_url(cls, project_id: str = "") -> Tuple[str, str, str]:
        """
        Build OAuth authorization URL with PKCE.
        
        Returns:
            Tuple of (authorization_url, verifier, state)
        """
        verifier, challenge = generate_pkce_pair()
        state = encode_oauth_state(verifier, project_id)
        
        params = {
            "client_id": cls.OAUTH_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
            "scope": " ".join(ANTIGRAVITY_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        
        url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
        return url, verifier, state

    @classmethod
    async def exchange_code_for_tokens(
        cls,
        code: str,
        state: str,
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens.
        
        Args:
            code: Authorization code from OAuth callback
            state: State parameter containing PKCE verifier
            
        Returns:
            Dict containing tokens and user info
        """
        decoded_state = decode_oauth_state(state)
        verifier = decoded_state.get("verifier", "")
        project_id = decoded_state.get("projectId", "")
        
        if not verifier:
            raise RuntimeError("Missing PKCE verifier in state parameter")
        
        start_time = time.time()
        
        # Exchange code for tokens
        async with aiohttp.ClientSession() as session:
            token_data = {
                "client_id": cls.OAUTH_CLIENT_ID,
                "client_secret": cls.OAUTH_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
                "code_verifier": verifier,
            }
            
            async with session.post(
                "https://oauth2.googleapis.com/token",
                data=token_data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "google-api-nodejs-client/10.3.0",
                }
            ) as resp:
                if not resp.ok:
                    error_text = await resp.text()
                    raise RuntimeError(f"Token exchange failed: {error_text}")
                
                token_response = await resp.json()
            
            access_token = token_response.get("access_token")
            refresh_token = token_response.get("refresh_token")
            expires_in = token_response.get("expires_in", 3600)
            
            if not access_token or not refresh_token:
                raise RuntimeError("Missing tokens in response")
            
            # Get user info
            email = None
            async with session.get(
                "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
                headers={"Authorization": f"Bearer {access_token}"}
            ) as resp:
                if resp.ok:
                    user_info = await resp.json()
                    email = user_info.get("email")
            
            # Discover project ID if not provided
            effective_project_id = project_id
            if not effective_project_id:
                effective_project_id = await cls._fetch_project_id(session, access_token)
        
        expires_at = int((start_time + expires_in) * 1000)  # milliseconds
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expires_at,
            "email": email,
            "project_id": effective_project_id,
        }

    @classmethod
    async def _fetch_project_id(cls, session: aiohttp.ClientSession, access_token: str) -> str:
        """Fetch project ID from Antigravity API."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            **ANTIGRAVITY_AUTH_HEADERS,
        }
        
        load_request = {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        }
        
        # Try endpoints in order with short timeout
        timeout = aiohttp.ClientTimeout(total=10)
        for base_url in BASE_URLS:
            try:
                url = f"{base_url}:loadCodeAssist"
                async with session.post(url, headers=headers, json=load_request, timeout=timeout) as resp:
                    if resp.ok:
                        data = await resp.json()
                        project = data.get("cloudaicompanionProject")
                        if isinstance(project, dict):
                            project = project.get("id")
                        if project:
                            return project
            except asyncio.TimeoutError:
                debug.log(f"Project discovery timed out at {base_url}")
                continue
            except Exception as e:
                debug.log(f"Project discovery failed at {base_url}: {e}")
                continue
        # If discovery failed, attempt to onboard a managed project for the user.
        # Read optional configuration from environment
        attempts = int(os.environ.get("ANTIGRAVITY_ONBOARD_ATTEMPTS", "10"))
        delay_seconds = float(os.environ.get("ANTIGRAVITY_ONBOARD_DELAY_S", "5"))
        tier_id = os.environ.get("ANTIGRAVITY_TIER_ID", "free-tier")
        # Use any preconfigured project id as metadata if available
        configured_project = os.environ.get("ANTIGRAVITY_PROJECT_ID", "")

        if tier_id:
            onboard_request_body = {"tierId": tier_id, "metadata": {}}
            if configured_project:
                # include requested project id in metadata
                onboard_request_body["metadata"]["cloudaicompanionProject"] = configured_project

            # Try onboarding across endpoints with retries
            for base_url in BASE_URLS:
                for attempt in range(attempts):
                    try:
                        url = f"{base_url}:onboardUser"
                        onboard_headers = {
                            "Authorization": f"Bearer {access_token}",
                            "Content-Type": "application/json",
                            **ANTIGRAVITY_HEADERS,
                        }
                        async with session.post(url, headers=onboard_headers, json=onboard_request_body, timeout=timeout) as resp:
                            if not resp.ok:
                                print(f"Onboarding attempt {attempt+1} at {base_url} failed with status {resp.status}")
                                print(await resp.text())
                                # Stop attempts on this endpoint and try next base_url
                                break

                            payload = await resp.json()
                            # payload.response?.cloudaicompanionProject?.id
                            response_obj = payload.get("response") or {}
                            managed = response_obj.get("cloudaicompanionProject")
                            if isinstance(managed, dict):
                                managed_id = managed.get("id")
                            else:
                                managed_id = None

                            done = bool(payload.get("done", False))
                            if done and managed_id:
                                return managed_id
                            if done and configured_project:
                                return configured_project
                    except Exception as e:
                        debug.log(f"Failed to onboard managed project at {base_url}: {e}")
                        break

                    await asyncio.sleep(delay_seconds)

        return ""

    @classmethod
    async def interactive_login(
        cls,
        project_id: str = "",
        no_browser: bool = False,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Perform interactive OAuth login flow.
        
        This opens a browser for Google OAuth and captures the callback locally.
        
        Args:
            project_id: Optional GCP project ID
            no_browser: If True, don't auto-open browser (print URL instead)
            timeout: Timeout in seconds for OAuth callback
            
        Returns:
            Dict containing tokens and user info
        """
        # Build authorization URL
        auth_url, verifier, state = cls.build_authorization_url(project_id)
        
        print("\n" + "=" * 60)
        print("Antigravity OAuth Login")
        print("=" * 60)
        
        # Try to start local callback server
        callback_server = OAuthCallbackServer(timeout=timeout)
        server_started = callback_server.start()
        
        if server_started and not no_browser:
            print(f"\nOpening browser for authentication...")
            print(f"If browser doesn't open, visit this URL:\n")
            print(f"{auth_url}\n")
            
            # Try to open browser
            try:
                webbrowser.open(auth_url)
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print("Please open the URL above manually.\n")
        else:
            if not server_started:
                print(f"\nCould not start local callback server on port {OAUTH_CALLBACK_PORT}.")
                print("You may need to close any application using that port.\n")
            
            print(f"\nPlease open this URL in your browser:\n")
            print(f"{auth_url}\n")
        
        if server_started:
            print("Waiting for authentication callback...")
            
            try:
                callback_result = callback_server.wait_for_callback()
                
                if not callback_result:
                    raise RuntimeError("OAuth callback timed out")
                
                code = callback_result.get("code")
                callback_state = callback_result.get("state")
                
                if not code:
                    raise RuntimeError("No authorization code received")
                
                print("\n✓ Authorization code received. Exchanging for tokens...")
                
                # Exchange code for tokens
                tokens = await cls.exchange_code_for_tokens(code, callback_state or state)
                
                print(f"✓ Authentication successful!")
                if tokens.get("email"):
                    print(f"  Logged in as: {tokens['email']}")
                if tokens.get("project_id"):
                    print(f"  Project ID: {tokens['project_id']}")
                
                return tokens
                
            finally:
                callback_server.stop()
        else:
            # Manual flow - ask user to paste the redirect URL or code
            print("\nAfter completing authentication, you'll be redirected to a localhost URL.")
            print("Copy and paste the full redirect URL or just the authorization code below:\n")
            
            user_input = input("Paste redirect URL or code: ").strip()
            
            if not user_input:
                raise RuntimeError("No input provided")
            
            # Parse the input
            if user_input.startswith("http"):
                parsed = urlparse(user_input)
                params = parse_qs(parsed.query)
                code = params.get("code", [None])[0]
                callback_state = params.get("state", [state])[0]
            else:
                # Assume it's just the code
                code = user_input
                callback_state = state
            
            if not code:
                raise RuntimeError("Could not extract authorization code")
            
            print("\nExchanging code for tokens...")
            tokens = await cls.exchange_code_for_tokens(code, callback_state)
            
            print(f"✓ Authentication successful!")
            if tokens.get("email"):
                print(f"  Logged in as: {tokens['email']}")
            
            return tokens

    @classmethod
    async def login_and_save(
        cls,
        project_id: str = "",
        no_browser: bool = False,
        credentials_path: Optional[Path] = None,
    ) -> "AntigravityAuthManager":
        """
        Perform interactive login and save credentials to file.
        
        Args:
            project_id: Optional GCP project ID
            no_browser: If True, don't auto-open browser
            credentials_path: Path to save credentials (default: g4f cache or ~/.antigravity/oauth_creds.json)
            
        Returns:
            AntigravityAuthManager instance with loaded credentials
        """
        tokens = await cls.interactive_login(project_id=project_id, no_browser=no_browser)
        
        # Prepare credentials for saving
        creds = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expiry_date": tokens["expiry_date"],
            "email": tokens.get("email"),
            "project_id": tokens.get("project_id"),
            "client_id": cls.OAUTH_CLIENT_ID,
            "client_secret": cls.OAUTH_CLIENT_SECRET,
        }
        
        # Save credentials - use provided path, or g4f cache file, or default path
        if credentials_path:
            path = credentials_path
        else:
            # Prefer g4f cache location (checked first by initialize_auth)
            path = cls.get_cache_file()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w") as f:
            json.dump(creds, f, indent=2)
        
        # Set restrictive permissions on Unix
        try:
            path.chmod(0o600)
        except Exception:
            pass
        
        print(f"\n✓ Credentials saved to: {path}")
        print("=" * 60 + "\n")
        
        # Create and return auth manager
        auth_manager = cls(env=os.environ)
        auth_manager._access_token = tokens["access_token"]
        auth_manager._expiry = tokens["expiry_date"] / 1000
        
        return auth_manager


class AntigravityProvider:
    """
    Internal provider class for Antigravity API communication.
    
    Handles message formatting, project discovery, and streaming content generation.
    """
    url = "https://cloud.google.com/code-assist"

    def __init__(self, env: dict, auth_manager: AntigravityAuthManager):
        self.env = env
        self.auth_manager = auth_manager
        self._project_id: Optional[str] = None

    async def discover_project_id(self) -> str:
        """Discover the GCP project ID for API calls."""
        # Check environment variable first
        if self.env.get("ANTIGRAVITY_PROJECT_ID"):
            return self.env["ANTIGRAVITY_PROJECT_ID"]
        
        # Check cached project ID
        if self._project_id:
            return self._project_id
        
        # Check auth manager's cached project ID (from credentials file)
        auth_project_id = self.auth_manager.get_project_id()
        if auth_project_id:
            self._project_id = auth_project_id
            return auth_project_id

        # Fall back to API discovery
        try:
            access_token = self.auth_manager.get_access_token()
            if not access_token:
                raise RuntimeError("No valid access token available for project discovery")
            
            async with aiohttp.ClientSession() as session:
                project = await self.auth_manager._fetch_project_id(
                    session=session,
                    access_token=access_token
                )
            if project:
                self._project_id = project
                return project
            raise RuntimeError(
                "Project ID discovery failed - set ANTIGRAVITY_PROJECT_ID in environment."
            )
        except Exception as e:
            debug.error(f"Failed to discover project ID: {e}")
            raise RuntimeError(
                "Could not discover project ID. Ensure authentication or set ANTIGRAVITY_PROJECT_ID."
            )

    @staticmethod
    def _messages_to_gemini_format(messages: list, media: MediaListType) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini format."""
        format_messages = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"

            # Handle tool role (OpenAI style)
            if msg["role"] == "tool":
                parts = [
                    {
                        "functionResponse": {
                            "name": msg.get("tool_call_id", "unknown_function"),
                            "response": {
                                "result": (
                                    msg["content"]
                                    if isinstance(msg["content"], str)
                                    else json.dumps(msg["content"])
                                )
                            },
                        }
                    }
                ]

            # Handle assistant messages with tool calls
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                parts = []
                if isinstance(msg["content"], str) and msg["content"].strip():
                    parts.append({"text": msg["content"]})
                for tool_call in msg["tool_calls"]:
                    if tool_call.get("type") == "function":
                        parts.append(
                            {
                                "functionCall": {
                                    "name": tool_call["function"]["name"],
                                    "args": json.loads(tool_call["function"]["arguments"]),
                                }
                            }
                        )

            # Handle string content
            elif isinstance(msg["content"], str):
                parts = [{"text": msg["content"]}]

            # Handle array content (possibly multimodal)
            elif isinstance(msg["content"], list):
                parts = []
                for content in msg["content"]:
                    ctype = content.get("type")
                    if ctype == "text":
                        parts.append({"text": content["text"]})
                    elif ctype == "image_url":
                        image_url = content.get("image_url", {}).get("url")
                        if not image_url:
                            continue
                        if image_url.startswith("data:"):
                            # Inline base64 data image
                            prefix, b64data = image_url.split(",", 1)
                            mime_type = prefix.split(":")[1].split(";")[0]
                            parts.append({"inlineData": {"mimeType": mime_type, "data": b64data}})
                        else:
                            parts.append(
                                {
                                    "fileData": {
                                        "mimeType": "image/jpeg",
                                        "fileUri": image_url,
                                    }
                                }
                            )
            else:
                parts = [{"text": str(msg["content"])}]

            format_messages.append({"role": role, "parts": parts})

        # Handle media attachments
        if media:
            if not format_messages:
                format_messages.append({"role": "user", "parts": []})
            for media_data, filename in media:
                if isinstance(media_data, str):
                    if not filename:
                        filename = media_data
                    extension = filename.split(".")[-1].replace("jpg", "jpeg")
                    format_messages[-1]["parts"].append(
                        {
                            "fileData": {
                                "mimeType": f"image/{extension}",
                                "fileUri": media_data,
                            }
                        }
                    )
                else:
                    media_data = to_bytes(media_data)
                    format_messages[-1]["parts"].append({
                        "inlineData": {
                            "mimeType": is_data_an_media(media_data, filename),
                            "data": base64.b64encode(media_data).decode()
                        }
                    })

        return format_messages

    async def stream_content(
        self,
        model: str,
        messages: Messages,
        *,
        proxy: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator:
        """Stream content generation from Antigravity API."""
        # Convert user-facing model name to internal API name
        if model in Antigravity.model_aliases:
            model = Antigravity.model_aliases[model]
        
        await self.auth_manager.initialize_auth()

        project_id = await self.discover_project_id()

        # Convert messages to Gemini format
        contents = self._messages_to_gemini_format(
            [m for m in messages if m["role"] not in ["developer", "system"]],
            media=kwargs.get("media", None)
        )
        system_prompt = get_system_prompt(messages)
        request_data = {}
        if system_prompt:
            request_data["system_instruction"] = {"parts": {"text": system_prompt}}

        # Convert OpenAI-style tools to Gemini format
        gemini_tools = None
        function_declarations = []
        if tools:
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]
                    function_declarations.append({
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {})
                    })
            if function_declarations:
                gemini_tools = [{"functionDeclarations": function_declarations}]

        # Build generation config
        generation_config = {
            "maxOutputTokens": max_tokens or 32000,  # Antigravity default
            "temperature": temperature,
            "topP": top_p,
            "stop": stop,
            "presencePenalty": presence_penalty,
            "frequencyPenalty": frequency_penalty,
            "seed": seed,
        }

        # Handle response format
        if response_format is not None and response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"

        # Handle thinking configuration
        if thinking_budget:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": thinking_budget,
                "includeThoughts": True
            }

        # Compose request body with required Antigravity fields
        req_body = {
            "model": model,
            "project": project_id,
            "userAgent": "antigravity",
            "requestType": "agent",
            "requestId": f"req-{secrets.token_hex(8)}",
            "request": {
                "contents": contents,
                "generationConfig": generation_config,
                "tools": gemini_tools,
                **request_data
            },
        }

        # Add tool config if specified
        if tool_choice and gemini_tools:
            req_body["request"]["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": tool_choice.upper(),
                    "allowedFunctionNames": [fd["name"] for fd in function_declarations]
                }
            }

        # Remove None values recursively
        def clean_none(d):
            if isinstance(d, dict):
                return {k: clean_none(v) for k, v in d.items() if v is not None}
            if isinstance(d, list):
                return [clean_none(x) for x in d if x is not None]
            return d

        req_body = clean_none(req_body)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_manager.get_access_token()}",
            **ANTIGRAVITY_HEADERS,
        }

        # Use production URL for streaming (most reliable)
        base_url = PRODUCTION_URL
        url = f"{base_url}:streamGenerateContent?alt=sse"

        # Streaming SSE parsing helper
        async def parse_sse_stream(stream: aiohttp.StreamReader) -> AsyncGenerator[Dict[str, Any], None]:
            """Parse SSE stream yielding parsed JSON objects."""
            buffer = ""
            object_buffer = ""

            async for chunk_bytes in stream.iter_any():
                chunk = chunk_bytes.decode()
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()  # Save last incomplete line back

                for line in lines:
                    line = line.strip()
                    if line == "":
                        # Empty line indicates end of SSE message -> parse object buffer
                        if object_buffer:
                            try:
                                yield json.loads(object_buffer)
                            except Exception as e:
                                debug.error(f"Error parsing SSE JSON: {e}")
                            object_buffer = ""
                    elif line.startswith("data: "):
                        object_buffer += line[6:]

            # Final parse when stream ends
            if object_buffer:
                try:
                    yield json.loads(object_buffer)
                except Exception as e:
                    debug.error(f"Error parsing final SSE JSON: {e}")

        timeout = ClientTimeout(total=None)  # No total timeout
        connector = get_connector(None, proxy)

        async with ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
            async with session.post(url, json=req_body) as resp:
                if not resp.ok:
                    if resp.status == 503:
                        try:
                            max_retry_delay = int(max([d.get("retryDelay", 0) for d in (await resp.json(content_type=None)).get("error", {}).get("details", [])]))
                        except ValueError:
                            max_retry_delay = 30  # Default retry delay if not specified
                        debug.log(f"Received 503 error, retrying after {max_retry_delay}")
                        await asyncio.sleep(max_retry_delay)
                        resp = await session.post(url, json=req_body)
                        if not resp.ok:
                            debug.error(f"Retry after 503 failed with status {resp.status}")
                if not resp.ok:
                    if resp.status == 401:
                        raise MissingAuthError("Unauthorized (401) from Antigravity API")
                    error_body = await resp.text()
                    raise RuntimeError(f"Antigravity API error {resp.status}: {error_body}")

                usage_metadata = {}
                async for json_data in parse_sse_stream(resp.content):
                    # Process JSON data according to Gemini API structure
                    candidates = json_data.get("response", {}).get("candidates", [])
                    usage_metadata = json_data.get("response", {}).get("usageMetadata", usage_metadata)

                    if not candidates:
                        continue

                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])

                    tool_calls = []

                    for part in parts:
                        # Real thinking chunks
                        if part.get("thought") is True and "text" in part:
                            yield Reasoning(part["text"])

                        # Function calls from Gemini
                        elif "functionCall" in part:
                            tool_calls.append(part["functionCall"])

                        # Text content
                        elif "text" in part:
                            yield part["text"]

                        # Inline media data
                        elif "inlineData" in part:
                            async for media in save_response_media(part["inlineData"], format_media_prompt(messages)):
                                yield media

                        # File data (e.g. external image)
                        elif "fileData" in part:
                            file_data = part["fileData"]
                            yield ImageResponse(file_data.get("fileUri"))

                    if tool_calls:
                        # Convert Gemini tool calls to OpenAI format
                        openai_tool_calls = []
                        for i, tc in enumerate(tool_calls):
                            openai_tool_calls.append({
                                "id": f"call_{i}_{tc.get('name', 'unknown')}",
                                "type": "function",
                                "function": {
                                    "name": tc.get("name"),
                                    "arguments": json.dumps(tc.get("args", {}))
                                }
                            })
                        yield ToolCalls(openai_tool_calls)

                if usage_metadata:
                    yield Usage(**usage_metadata)


class Antigravity(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Antigravity Provider for gpt4free.
    
    Provides access to Google's Antigravity API (Code Assist) supporting:
    - Gemini 2.5 Pro/Flash with extended thinking
    - Gemini 3 Pro/Flash (preview)
    - Claude Sonnet 4.5 / Opus 4.5 via Antigravity proxy
    
    Requires OAuth2 credentials. Set ANTIGRAVITY_SERVICE_ACCOUNT environment
    variable or create credentials at ~/.antigravity/oauth_creds.json
    """
    label = "Google Antigravity"
    login_url = "https://cloud.google.com/code-assist"
    url = "https://antigravity.google"

    default_model = "gemini-3-flash"
    fallback_models = [
        # Gemini 2.5 models
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        # Gemini 3 models
        "gemini-3-flash",
        # Claude models (via Antigravity proxy)
        "claude-sonnet-4.5",
        "claude-opus-4.5",
    ]

    # Model aliases for compatibility
    model_aliases = {
        "claude-sonnet-4.5": "claude-sonnet-4-5",
        "claude-opus-4.5": "claude-opus-4-5",
    }

    working = True
    supports_message_history = True
    supports_system_message = True
    needs_auth = True
    active_by_default = True

    auth_manager: AntigravityAuthManager = None

    @classmethod
    def get_models(cls, **kwargs) -> List[str]:
        """Return available models, fetching dynamically from API if authenticated."""
        # Try to fetch models dynamically if we have credentials
        if not cls.models and cls.has_credentials():
            try:
                get_running_loop(check_nested=True)
                cls.models = asyncio.run(cls._fetch_models())
            except Exception as e:
                debug.log(f"Failed to fetch dynamic models: {e}")
        
        # Update live status
        if cls.live == 0:
            if cls.auth_manager is None:
                cls.auth_manager = AntigravityAuthManager(env=os.environ)
            if cls.auth_manager.get_access_token() is not None:
                cls.live += 1
        
        return cls.models if cls.models else cls.fallback_models

    @classmethod
    async def _fetch_models(cls) -> List[str]:
        """Fetch available models dynamically from the Antigravity API."""
        if cls.auth_manager is None:
            cls.auth_manager = AntigravityAuthManager(env=os.environ)

        await cls.auth_manager.initialize_auth()

        try:
            response = await cls.auth_manager.call_endpoint(
                method="fetchAvailableModels",
                body={"project": cls.auth_manager.get_project_id()}
            )

            # Extract model names from the response
            models = [key for key, value in response.get("models", {}).items() if not value.get("isInternal", False) and not key.startswith("tab_")]
            if not isinstance(models, list):
                raise ValueError("Invalid response format: 'models' should be a list")

            return models
        except Exception as e:
            debug.log(f"Failed to fetch models: {e}")
            return []

    @classmethod
    async def get_quota(cls, api_key: Optional[str] = None) -> dict:
        """
        Fetch usage/quota information from the Antigravity API.
        """
        if cls.auth_manager is None:
            cls.auth_manager = AntigravityAuthManager(env=os.environ)
        await cls.auth_manager.initialize_auth()

        access_token = cls.auth_manager.get_access_token()
        project_id = cls.auth_manager.get_project_id()
        if not access_token or not project_id:
            raise MissingAuthError("Cannot fetch usage without valid authentication")

        return await cls.auth_manager.call_endpoint(
            method="fetchAvailableModels",
            body={"project": cls.auth_manager.get_project_id()}
        )

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        media: MediaListType = None,
        tools: Optional[list] = None,
        **kwargs
    ) -> AsyncResult:
        """Create an async generator for streaming responses."""
        if cls.auth_manager is None:
            cls.auth_manager = AntigravityAuthManager(env=os.environ)

        # Apply model alias if needed
        if model in cls.model_aliases:
            model = cls.model_aliases[model]

        # Initialize Antigravity provider with auth manager and environment
        provider = AntigravityProvider(env=os.environ, auth_manager=cls.auth_manager)

        async for chunk in provider.stream_content(
            model=model,
            messages=messages,
            stream=stream,
            media=media,
            tools=tools,
            **kwargs
        ):
            yield chunk

    @classmethod
    async def login(
        cls,
        project_id: str = "",
        no_browser: bool = False,
        credentials_path: Optional[Path] = None,
    ) -> "AntigravityAuthManager":
        """
        Perform interactive OAuth login and save credentials.
        
        This is the main entry point for authenticating with Antigravity.
        
        Args:
            project_id: Optional GCP project ID
            no_browser: If True, don't auto-open browser
            credentials_path: Path to save credentials
            
        Returns:
            AntigravityAuthManager with active credentials
            
        Example:
            >>> import asyncio
            >>> from g4f.Provider.needs_auth import Antigravity
            >>> asyncio.run(Antigravity.login())
        """
        auth_manager = await AntigravityAuthManager.login_and_save(
            project_id=project_id,
            no_browser=no_browser,
            credentials_path=credentials_path,
        )
        cls.auth_manager = auth_manager
        return auth_manager

    @classmethod
    def has_credentials(cls) -> bool:
        """Check if valid credentials exist."""
        # Check g4f cache file (checked first by initialize_auth)
        cache_path = AntigravityAuthManager.get_cache_file()
        if cache_path.exists():
            return True
        
        # Check default path (~/.antigravity/oauth_creds.json)
        default_path = get_antigravity_oauth_creds_path()
        if default_path.exists():
            return True
        
        # Check environment variable
        if "ANTIGRAVITY_SERVICE_ACCOUNT" in os.environ:
            return True
        
        return False

    @classmethod
    def get_credentials_path(cls) -> Path:
        """Get the path where credentials are stored or should be stored."""
        # Check g4f cache file first (matches initialize_auth order)
        cache_path = AntigravityAuthManager.get_cache_file()
        if cache_path.exists():
            return cache_path
        
        # Check default path
        default_path = get_antigravity_oauth_creds_path()
        if default_path.exists():
            return default_path
        
        # Return cache path as the preferred location for new credentials
        return cache_path


async def main(args: Optional[List[str]] = None):
    """CLI entry point for Antigravity authentication."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Antigravity OAuth Authentication for gpt4free",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s login                    # Interactive login with browser
  %(prog)s login --no-browser       # Manual login (paste URL)
  %(prog)s login --project-id ID    # Login with specific project
  %(prog)s status                   # Check authentication status
  %(prog)s logout                   # Remove saved credentials
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Login command
    login_parser = subparsers.add_parser("login", help="Authenticate with Google")
    login_parser.add_argument(
        "--project-id", "-p",
        default="",
        help="Google Cloud project ID (optional, auto-discovered if not set)"
    )
    login_parser.add_argument(
        "--no-browser", "-n",
        action="store_true",
        help="Don't auto-open browser, print URL instead"
    )
    
    # Status command
    subparsers.add_parser("status", help="Check authentication status")
    
    # Logout command
    subparsers.add_parser("logout", help="Remove saved credentials")
    
    args = parser.parse_args(args)
    
    if args.command == "login":
        try:
            await Antigravity.login(
                project_id=args.project_id,
                no_browser=args.no_browser,
            )
        except KeyboardInterrupt:
            print("\n\nLogin cancelled.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Login failed: {e}")
            sys.exit(1)
    
    elif args.command == "status":
        print("\nAntigravity Authentication Status")
        print("=" * 40)
        
        if Antigravity.has_credentials():
            creds_path = Antigravity.get_credentials_path()
            print(f"✓ Credentials found at: {creds_path}")
            
            # Try to read and display some info
            try:
                with creds_path.open() as f:
                    creds = json.load(f)
                
                if creds.get("email"):
                    print(f"  Email: {creds['email']}")
                if creds.get("project_id"):
                    print(f"  Project: {creds['project_id']}")
                
                expiry = creds.get("expiry_date")
                if expiry:
                    expiry_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expiry / 1000))
                    if expiry / 1000 > time.time():
                        print(f"  Token expires: {expiry_time}")
                    else:
                        print(f"  Token expired: {expiry_time} (will auto-refresh)")
            except Exception as e:
                print(f"  (Could not read credential details: {e})")
        else:
            print("✗ No credentials found")
            print(f"\nRun 'antigravity login' to authenticate.")
        
        print()
    
    elif args.command == "logout":
        print("\nAntigravity Logout")
        print("=" * 40)
        
        removed = False
        
        # Remove cache file
        cache_path = AntigravityAuthManager.get_cache_file()
        if cache_path.exists():
            cache_path.unlink()
            print(f"✓ Removed: {cache_path}")
            removed = True
        
        # Remove default credentials file
        default_path = get_antigravity_oauth_creds_path()
        if default_path.exists():
            default_path.unlink()
            print(f"✓ Removed: {default_path}")
            removed = True
        
        if removed:
            print("\n✓ Credentials removed successfully.")
        else:
            print("No credentials found to remove.")
        
        print()
    
    else:
        parser.print_help()


def cli_main(args: Optional[List[str]] = None):
    """Synchronous CLI entry point for setup.py console_scripts."""
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_main()
