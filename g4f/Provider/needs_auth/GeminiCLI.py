import os
import json
import base64
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

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
    api_base = "https://cloudcode-pa.googleapis.com/v1internal"

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
                "tools": tools or [],
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": tool_choice.upper(),
                        "allowedFunctionNames": [tool["function"]["name"] for tool in tools]
                    }
                } if tool_choice else None,
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

        url = f"{self.api_base}:streamGenerateContent?alt=sse"

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
                        yield ToolCalls(tool_calls)
                    if usage_metadata:
                        yield Usage(
                            promptTokens=usage_metadata.get("promptTokenCount", 0),
                            completionTokens=usage_metadata.get("candidatesTokenCount", 0),
                        )

class GeminiCLI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Google Gemini CLI"
    login_url = "https://github.com/GewoonJaap/gemini-cli-openai"

    default_model = "gemini-2.5-pro"
    models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]

    working = True
    supports_message_history = True
    supports_system_message = True
    needs_auth = True
    active_by_default = True

    auth_manager: AuthManager = None

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