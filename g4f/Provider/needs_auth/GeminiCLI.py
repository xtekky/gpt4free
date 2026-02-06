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
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin
from ..helper import get_connector, get_system_prompt, format_media_prompt
from ... import debug


def get_oauth_creds_path():
    return Path.home() / ".gemini" / "oauth_creds.json"


# OAuth configuration for GeminiCLI
GEMINICLI_REDIRECT_URI = "http://localhost:51122/oauthcallback"
GEMINICLI_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
GEMINICLI_OAUTH_CALLBACK_PORT = 51122
GEMINICLI_OAUTH_CALLBACK_PATH = "/oauthcallback"


def generate_pkce_pair() -> Tuple[str, str]:
    """Generate a PKCE verifier and challenge pair."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode('ascii')).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
    return verifier, challenge


def encode_oauth_state(verifier: str) -> str:
    """Encode OAuth state parameter with PKCE verifier."""
    payload = {"verifier": verifier}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')


def decode_oauth_state(state: str) -> Dict[str, str]:
    """Decode OAuth state parameter back to verifier."""
    padded = state + '=' * (4 - len(state) % 4) if len(state) % 4 else state
    normalized = padded.replace('-', '+').replace('_', '/')
    try:
        decoded = base64.b64decode(normalized).decode('utf-8')
        parsed = json.loads(decoded)
        return {"verifier": parsed.get("verifier", "")}
    except Exception:
        return {"verifier": ""}


class GeminiCLIOAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback."""
    
    callback_result: Optional[Dict[str, str]] = None
    callback_error: Optional[str] = None
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def do_GET(self):
        """Handle GET request for OAuth callback."""
        parsed = urlparse(self.path)
        
        if parsed.path != GEMINICLI_OAUTH_CALLBACK_PATH:
            self.send_error(404, "Not Found")
            return
        
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        error = params.get("error", [None])[0]
        
        if error:
            GeminiCLIOAuthCallbackHandler.callback_error = error
            self._send_error_response(error)
        elif code and state:
            GeminiCLIOAuthCallbackHandler.callback_result = {"code": code, "state": state}
            self._send_success_response()
        else:
            GeminiCLIOAuthCallbackHandler.callback_error = "Missing code or state parameter"
            self._send_error_response("Missing parameters")
    
    def _send_success_response(self):
        """Send success HTML response."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Authentication Successful</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               display: flex; justify-content: center; align-items: center; height: 100vh; 
               margin: 0; background: linear-gradient(135deg, #4285f4 0%, #34a853 100%); }
        .container { background: white; padding: 3rem; border-radius: 1rem; 
                     box-shadow: 0 20px 60px rgba(0,0,0,0.3); text-align: center; max-width: 400px; }
        h1 { color: #10B981; margin-bottom: 1rem; }
        p { color: #6B7280; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <div style="font-size: 4rem; margin-bottom: 1rem;">✅</div>
        <h1>Authentication Successful!</h1>
        <p>You have successfully authenticated with Google GeminiCLI.<br>You can close this window and return to your terminal.</p>
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


class GeminiCLIOAuthCallbackServer:
    """Local HTTP server to capture OAuth callback."""
    
    def __init__(self, port: int = GEMINICLI_OAUTH_CALLBACK_PORT, timeout: float = 300.0):
        self.port = port
        self.timeout = timeout
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = False
    
    def start(self) -> bool:
        """Start the callback server. Returns True if successful."""
        try:
            GeminiCLIOAuthCallbackHandler.callback_result = None
            GeminiCLIOAuthCallbackHandler.callback_error = None
            self._stop_flag = False
            
            self.server = HTTPServer(("localhost", self.port), GeminiCLIOAuthCallbackHandler)
            self.server.timeout = 0.5
            
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
            if GeminiCLIOAuthCallbackHandler.callback_result or GeminiCLIOAuthCallbackHandler.callback_error:
                time.sleep(0.3)
                break
            try:
                self.server.handle_request()
            except Exception:
                break
    
    def wait_for_callback(self) -> Optional[Dict[str, str]]:
        """Wait for OAuth callback and return result."""
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if GeminiCLIOAuthCallbackHandler.callback_result or GeminiCLIOAuthCallbackHandler.callback_error:
                break
            time.sleep(0.1)
        
        self._stop_flag = True
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if GeminiCLIOAuthCallbackHandler.callback_error:
            raise RuntimeError(f"OAuth error: {GeminiCLIOAuthCallbackHandler.callback_error}")
        
        return GeminiCLIOAuthCallbackHandler.callback_result
    
    def stop(self):
        """Stop the callback server."""
        self._stop_flag = True
        if self.server:
            try:
                self.server.server_close()
            except Exception:
                pass
            self.server = None

class AuthManager(AuthFileMixin):
    """
    Handles OAuth2 authentication and Google Code Assist API communication.
    Manages token caching, refresh, and API calls.

    Requires environment dict-like object with keys:
        - GCP_SERVICE_ACCOUNT: JSON string with OAuth2 credentials, containing:
            access_token, expiry_date (ms timestamp), refresh_token
        - Optionally supports cache storage via a KV storage interface implementing:
            get(key) -> value or None,
            put(key, value, expiration_seconds),
            delete(key)
    """
    parent = "GeminiCLI"

    OAUTH_REFRESH_URL = "https://oauth2.googleapis.com/token"
    OAUTH_CLIENT_ID = "681255809395" + "-oo8ft2oprdrnp9e3aqf6av3hmdib135j" + ".apps.googleusercontent.com"
    OAUTH_CLIENT_SECRET = "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"
    TOKEN_BUFFER_TIME = 5 * 60  # seconds, 5 minutes
    KV_TOKEN_KEY = "oauth_token_cache"

    def __init__(self, env: Dict[str, Any]):
        self.env = env
        self._access_token: Optional[str] = None
        self._expiry: Optional[float] = None  # Unix timestamp in seconds
        self._token_cache = {}  # Example in-memory cache; replace with KV store for production

    async def initialize_auth(self) -> None:
        """
        Initialize authentication by using cached token, or refreshing if needed.
        Raises RuntimeError if no valid token can be obtained.
        """
        # Try cached token from KV store or in-memory cache
        cached = await self._get_cached_token()
        now = time.time()
        if cached:
            expires_at = cached["expiry_date"] / 1000  # ms to seconds
            if expires_at - now > self.TOKEN_BUFFER_TIME:
                self._access_token = cached["access_token"]
                self._expiry = expires_at
                return  # Use cached token if valid

        path = AuthManager.get_cache_file()
        if not path.exists():
            path = get_oauth_creds_path()
        if path.exists():
            try:
                with path.open("r") as f:
                    creds = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to read OAuth credentials from {path}: {e}")
        else:
            # Parse credentials from environment
            if "GCP_SERVICE_ACCOUNT" not in self.env:
                raise RuntimeError("GCP_SERVICE_ACCOUNT environment variable not set.")
            creds = json.loads(self.env["GCP_SERVICE_ACCOUNT"])

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
            raise RuntimeError("No refresh token found in GCP_SERVICE_ACCOUNT.")

        await self._refresh_and_cache_token(refresh_token)

    async def _refresh_and_cache_token(self, refresh_token: str) -> None:
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
        # Cache token in KV store or fallback to memory cache
        token_data = {
            "access_token": access_token,
            "expiry_date": expiry_date,
            "cached_at": int(time.time() * 1000),  # ms
        }
        self._token_cache[self.KV_TOKEN_KEY] = token_data

    async def _get_cached_token(self) -> Optional[Dict[str, Any]]:
        # Return in-memory cached token if present and still valid
        cached = self._token_cache.get(self.KV_TOKEN_KEY)
        if cached:
            expires_at = cached["expiry_date"] / 1000
            if expires_at - time.time() > self.TOKEN_BUFFER_TIME:
                return cached
        return None

    async def clear_token_cache(self) -> None:
        self._access_token = None
        self._expiry = None

    def get_access_token(self) -> Optional[str]:
        # Return current valid access token or None
        if (
            self._access_token is not None
            and self._expiry is not None
            and self._expiry - time.time() > self.TOKEN_BUFFER_TIME
        ):
            return self._access_token
        return None

    async def call_endpoint(self, method: str, body: Dict[str, Any], is_retry=False) -> Any:
        """
        Call Google Code Assist API endpoint with JSON body.

        Automatically retries once on 401 Unauthorized by refreshing auth.
        """
        if not self.get_access_token():
            await self.initialize_auth()

        url = f"https://cloudcode-pa.googleapis.com/v1internal:{method}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_access_token()}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body) as resp:
                if resp.status == 401 and not is_retry:
                    # Token likely expired, clear and retry once
                    await self.clear_token_cache()
                    await self.initialize_auth()
                    return await self.call_endpoint(method, body, is_retry=True)
                elif not resp.ok:
                    text = await resp.text()
                    raise RuntimeError(f"API call failed with status {resp.status}: {text}")

                return await resp.json()

class GeminiCLIProvider():
    url = "https://cloud.google.com/code-assist"
    base_url = "https://cloudcode-pa.googleapis.com/v1internal"

    # Required for authentication and token management; Expects a compatible AuthManager instance
    auth_manager: AuthManager
    env: dict

    def __init__(self, env: dict, auth_manager: AuthManager):
        self.env = env
        self.auth_manager = auth_manager

        # Cache for discovered project ID
        self._project_id: Optional[str] = None

    async def discover_project_id(self) -> str:
        if self.env.get("GEMINI_PROJECT_ID"):
            return self.env["GEMINI_PROJECT_ID"]
        if self._project_id:
            return self._project_id

        try:
            load_response = await self.auth_manager.call_endpoint(
                "loadCodeAssist",
                {
                    "cloudaicompanionProject": "default-project",
                    "metadata": {"duetProject": "default-project"},
                },
            )
            project = load_response.get("cloudaicompanionProject")
            if project:
                self._project_id = project
                return project
            raise RuntimeError(
                "Project ID discovery failed - set GEMINI_PROJECT_ID in environment."
            )
        except Exception as e:
            debug.error(f"Failed to discover project ID: {e}")
            raise RuntimeError(
                "Could not discover project ID. Ensure authentication or set GEMINI_PROJECT_ID."
            )

    @staticmethod
    def _messages_to_gemini_format(messages: list, media: MediaListType) -> Dict[str, Any]:
        format_messages = []
        for msg in messages:
            # Convert a ChatMessage dict to GeminiFormattedMessage dict
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
                ],

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
                                        "mimeType": "image/jpeg",  # Could improve by validation
                                        "fileUri": image_url,
                                    }
                                }
                            )
            else:
                parts = [{"text": str(msg["content"])}]
            format_messages.append({"role": role, "parts": parts})
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
                                "fileUri": image_url,
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
        await self.auth_manager.initialize_auth()

        project_id = await self.discover_project_id()

        # Convert messages to Gemini format
        contents = self._messages_to_gemini_format([m for m in messages if m["role"] not in ["developer", "system"]], media=kwargs.get("media", None))
        system_prompt = get_system_prompt(messages)
        requestData = {}
        if system_prompt:
            requestData["system_instruction"] = {"parts": {"text": system_prompt}}

        # Convert OpenAI-style tools to Gemini format
        gemini_tools = None
        if tools:
            function_declarations = []
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

        # Compose request body
        req_body = {
            "model": model,
            "project": project_id,
            "request": {
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                    "stop": stop,
                    "presencePenalty": presence_penalty,
                    "frequencyPenalty": frequency_penalty,
                    "seed": seed,
                    "responseMimeType": None if response_format is None else ("application/json" if response_format.get("type") == "json_object" else None),
                    "thinkingConfig": {
                        "thinkingBudget": thinking_budget,
                        "includeThoughts": True
                    } if thinking_budget else None,
                },
                "tools": gemini_tools,
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": tool_choice.upper(),
                        "allowedFunctionNames": [fd["name"] for fd in function_declarations]
                    }
                } if tool_choice and gemini_tools else None,
                **requestData
            },
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
        }

        url = f"{self.base_url}:streamGenerateContent?alt=sse"

        # Streaming SSE parsing helper
        async def parse_sse_stream(stream: aiohttp.StreamReader) -> AsyncGenerator[Dict[str, Any], None]:
            """Parse SSE stream yielding parsed JSON objects"""
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
        connector = get_connector(None, proxy)  # Customize connector as needed (supports proxy)

        async with ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
            async with session.post(url, json=req_body) as resp:
                if not resp.ok:
                    if resp.status == 401:
                        # Possibly token expired: try login retry logic, omitted here for brevity
                        raise MissingAuthError(f"Unauthorized (401) from Gemini API")
                    error_body = await resp.text()
                    raise RuntimeError(f"Gemini API error {resp.status}: {error_body}")

                async for json_data in parse_sse_stream(resp.content):
                    # Process JSON data according to Gemini API structure
                    candidates = json_data.get("response", {}).get("candidates", [])
                    usage_metadata = json_data.get("response", {}).get("usageMetadata", {})

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
                            # Media chunk - yield media asynchronously
                            async for media in save_response_media(part["inlineData"], format_media_prompt(messages)):
                                yield media

                        # File data (e.g. external image)
                        elif "fileData" in part:
                            # Just yield the file URI for now
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

class GeminiCLI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Google Gemini CLI"
    login_url = "https://github.com/GewoonJaap/gemini-cli-openai"

    default_model = "gemini-3-pro-preview"
    models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3-pro-preview"
    ]

    working = True
    supports_message_history = True
    supports_system_message = True
    needs_auth = True
    active_by_default = True

    auth_manager: AuthManager = None

    @classmethod
    def get_models(cls, **kwargs):
        if cls.live == 0:
            if cls.auth_manager is None:
                cls.auth_manager = AuthManager(env=os.environ)
            if cls.auth_manager.get_access_token() is not None:
                cls.live += 1
        return cls.models

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
        if cls.auth_manager is None:
            cls.auth_manager = AuthManager(env=os.environ)

        # Initialize Gemini CLI provider with auth manager and environment
        provider = GeminiCLIProvider(env=os.environ, auth_manager=cls.auth_manager)

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
    def build_authorization_url(cls) -> Tuple[str, str, str]:
        """Build OAuth authorization URL with PKCE."""
        verifier, challenge = generate_pkce_pair()
        state = encode_oauth_state(verifier)
        
        params = {
            "client_id": AuthManager.OAUTH_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": GEMINICLI_REDIRECT_URI,
            "scope": " ".join(GEMINICLI_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        
        url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
        return url, verifier, state

    @classmethod
    async def exchange_code_for_tokens(cls, code: str, state: str) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens."""
        decoded_state = decode_oauth_state(state)
        verifier = decoded_state.get("verifier", "")
        
        if not verifier:
            raise RuntimeError("Missing PKCE verifier in state parameter")
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            token_data = {
                "client_id": AuthManager.OAUTH_CLIENT_ID,
                "client_secret": AuthManager.OAUTH_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": GEMINICLI_REDIRECT_URI,
                "code_verifier": verifier,
            }
            
            async with session.post(
                "https://oauth2.googleapis.com/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
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
        
        expires_at = int((start_time + expires_in) * 1000)  # milliseconds
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expires_at,
            "email": email,
        }

    @classmethod
    async def interactive_login(
        cls,
        no_browser: bool = False,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Perform interactive OAuth login flow."""
        auth_url, verifier, state = cls.build_authorization_url()
        
        print("\n" + "=" * 60)
        print("GeminiCLI OAuth Login")
        print("=" * 60)
        
        callback_server = GeminiCLIOAuthCallbackServer(timeout=timeout)
        server_started = callback_server.start()
        
        if server_started and not no_browser:
            print(f"\nOpening browser for authentication...")
            print(f"If browser doesn't open, visit this URL:\n")
            print(f"{auth_url}\n")
            
            try:
                webbrowser.open(auth_url)
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print("Please open the URL above manually.\n")
        else:
            if not server_started:
                print(f"\nCould not start local callback server on port {GEMINICLI_OAUTH_CALLBACK_PORT}.")
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
                
                tokens = await cls.exchange_code_for_tokens(code, callback_state or state)
                
                print(f"✓ Authentication successful!")
                if tokens.get("email"):
                    print(f"  Logged in as: {tokens['email']}")
                
                return tokens
                
            finally:
                callback_server.stop()
        else:
            print("\nAfter completing authentication, you'll be redirected to a localhost URL.")
            print("Copy and paste the full redirect URL or just the authorization code below:\n")
            
            user_input = input("Paste redirect URL or code: ").strip()
            
            if not user_input:
                raise RuntimeError("No input provided")
            
            if user_input.startswith("http"):
                parsed = urlparse(user_input)
                params = parse_qs(parsed.query)
                code = params.get("code", [None])[0]
                callback_state = params.get("state", [state])[0]
            else:
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
    async def login(
        cls,
        no_browser: bool = False,
        credentials_path: Optional[Path] = None,
    ) -> "AuthManager":
        """
        Perform interactive OAuth login and save credentials.
        
        Args:
            no_browser: If True, don't auto-open browser
            credentials_path: Path to save credentials
            
        Returns:
            AuthManager with active credentials
            
        Example:
            >>> import asyncio
            >>> from g4f.Provider.needs_auth import GeminiCLI
            >>> asyncio.run(GeminiCLI.login())
        """
        tokens = await cls.interactive_login(no_browser=no_browser)
        
        creds = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expiry_date": tokens["expiry_date"],
            "email": tokens.get("email"),
            "client_id": AuthManager.OAUTH_CLIENT_ID,
            "client_secret": AuthManager.OAUTH_CLIENT_SECRET,
        }
        
        if credentials_path:
            path = credentials_path
        else:
            path = AuthManager.get_cache_file()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w") as f:
            json.dump(creds, f, indent=2)
        
        try:
            path.chmod(0o600)
        except Exception:
            pass
        
        print(f"\n✓ Credentials saved to: {path}")
        print("=" * 60 + "\n")
        
        auth_manager = AuthManager(env=os.environ)
        auth_manager._access_token = tokens["access_token"]
        auth_manager._expiry = tokens["expiry_date"] / 1000
        cls.auth_manager = auth_manager
        
        return auth_manager

    @classmethod
    def has_credentials(cls) -> bool:
        """Check if valid credentials exist."""
        cache_path = AuthManager.get_cache_file()
        if cache_path.exists():
            return True
        default_path = get_oauth_creds_path()
        return default_path.exists()

    @classmethod
    def get_credentials_path(cls) -> Optional[Path]:
        """Get path to credentials file if it exists."""
        cache_path = AuthManager.get_cache_file()
        if cache_path.exists():
            return cache_path
        default_path = get_oauth_creds_path()
        if default_path.exists():
            return default_path
        return None


async def main():
    """CLI entry point for GeminiCLI authentication."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GeminiCLI OAuth Authentication for gpt4free",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s login                    # Interactive login with browser
  %(prog)s login --no-browser       # Manual login (paste URL)
  %(prog)s status                   # Check authentication status
  %(prog)s logout                   # Remove saved credentials
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Login command
    login_parser = subparsers.add_parser("login", help="Authenticate with Google")
    login_parser.add_argument(
        "--no-browser", "-n",
        action="store_true",
        help="Don't auto-open browser, print URL instead"
    )
    
    # Status command
    subparsers.add_parser("status", help="Check authentication status")
    
    # Logout command
    subparsers.add_parser("logout", help="Remove saved credentials")
    
    args = parser.parse_args()
    
    if args.command == "login":
        try:
            await GeminiCLI.login(no_browser=args.no_browser)
        except KeyboardInterrupt:
            print("\n\nLogin cancelled.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Login failed: {e}")
            sys.exit(1)
    
    elif args.command == "status":
        print("\nGeminiCLI Authentication Status")
        print("=" * 40)
        
        if GeminiCLI.has_credentials():
            creds_path = GeminiCLI.get_credentials_path()
            print(f"✓ Credentials found at: {creds_path}")
            
            try:
                with creds_path.open() as f:
                    creds = json.load(f)
                
                if creds.get("email"):
                    print(f"  Email: {creds['email']}")
                
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
            print(f"\nRun 'g4f-geminicli login' to authenticate.")
        
        print()
    
    elif args.command == "logout":
        print("\nGeminiCLI Logout")
        print("=" * 40)
        
        removed = False
        
        cache_path = AuthManager.get_cache_file()
        if cache_path.exists():
            cache_path.unlink()
            print(f"✓ Removed: {cache_path}")
            removed = True
        
        default_path = get_oauth_creds_path()
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


def cli_main():
    """Synchronous CLI entry point for setup.py console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()