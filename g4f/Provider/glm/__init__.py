from __future__ import annotations

"""
GLM (z.ai) provider — faithful Python port of the TypeScript reference.

Reference files:
  - glm/spoofing.ts   — fingerprint params, headers, signature, variables
  - glm/session.ts    — user validation, chat session creation
  - glm/pipeline.ts   — full request pipeline with captcha integration
  - glm/stream.ts     — three-phase SSE stream parser
  - glm/captcha-solver.ts — Aliyun Captcha V3 solver

Key design decisions matching the TS reference:
  - x-signature: HMAC-SHA256 with empty key over sorted URL params
    (requestId, timestamp, user_id) — NOT SHA-256 of the body
  - Fingerprint: Linux/Chrome 149/Africa/Cairo timezone
  - x-fe-version: prod-fe-1.1.69
  - Session flow: validate JWT via /api/v1/auths/, then create chat
  - Body: full messages array, signature_prompt = first 500 chars
  - Stream: phase-based parsing (thinking/answer/other/done)
"""

import os
import json
import time
import hmac
import hashlib
import uuid
import requests
import urllib.parse
from datetime import datetime, timezone

from ...typing import AsyncResult, Messages
from ...providers.response import Usage, Reasoning, JsonConversation
from ...requests import StreamSession, raise_for_status
from ...errors import ModelNotFoundError, ProviderException
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin
from .captcha_solver import (
    get_captcha_verify_param,
    invalidate_captcha_token,
    is_available as captcha_solver_available,
)


GLM_BASE_URL = "https://chat.z.ai"
GLM_FE_VERSION = "prod-fe-1.1.69"
GLM_QUERY_VERSION = "0.0.1"
GLM_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)

WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


class _CaptchaRequired(Exception):
    """Raised from the SSE stream when the server rejects the captcha token.

    The error arrives as a ``chat:completion`` chunk with ``done: True`` and an
    ``error`` field containing ``FRONTEND_CAPTCHA_REQUIRED`` — not as an HTTP
    403 — so it must be detected while iterating the stream.
    """


class GLM(AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin):
    url = GLM_BASE_URL
    api_endpoint = f"{GLM_BASE_URL}/api/v2/chat/completions"
    working = True
    active_by_default = True
    default_model = "GLM-4.7"
    # Captcha solving requires zendriver (browser automation).
    use_nodriver = captcha_solver_available()

    api_key = None
    auth_user_id = None
    auth_user_name = None

    # ── Signature (spoofing.ts: computeSignature) ──────────────────────────

    @staticmethod
    def _compute_signature(request_id: str, timestamp: str, user_id: str) -> str:
        """Compute x-signature from sorted URL param key=value pairs.

        HMAC-SHA256 with empty key (browser's key is unknown — best effort,
        matching the TS reference exactly).
        """
        params = {"requestId": request_id, "timestamp": timestamp, "user_id": user_id}
        sorted_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
        return hmac.new(b"", sorted_str.encode("utf-8"), hashlib.sha256).hexdigest()

    # ── Fingerprint params (spoofing.ts: buildFingerprintParams) ────────────

    @classmethod
    def _build_url_params(cls, token: str, user_id: str) -> dict:
        """Build the fingerprint query params for GLM chat completion.

        Matches the browser's URL query params exactly (spoofing.ts).
        Returns a plain dict suitable for urllib.parse.urlencode().
        """
        ts = str(int(time.time() * 1000))
        request_id = str(uuid.uuid4())

        return {
            "timestamp": ts,
            "requestId": request_id,
            "user_id": user_id or "",
            "version": GLM_QUERY_VERSION,
            "platform": "web",
            "token": token,
            "user_agent": GLM_USER_AGENT,
            "language": "en-US",
            "languages": "en-US,en",
            "timezone": "Africa/Cairo",
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "screen_resolution": "1920x1080",
            "viewport_height": "947",
            "viewport_width": "1920",
            "viewport_size": "1920x1080",
            "color_depth": "30",
            "pixel_ratio": "1",
            "current_url": f"{GLM_BASE_URL}/",
            "pathname": "/",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "search": "",
            "hash": "",
            "referrer": "",
            "title": "",
            "timezone_offset": str(-int(time.timezone // 60)),
            "local_time": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
            "utc_time": datetime.now(timezone.utc).isoformat(),
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "0",
            "browser_name": "chrome",
            "os_name": "linux",
            "signature_timestamp": ts,
        }

    # ── Headers (spoofing.ts: buildGlmHeaders) ──────────────────────────────

    @classmethod
    def _build_headers(cls, token: str, body_str: str, request_id: str,
                       timestamp: str, user_id: str) -> dict:
        """Build the full set of headers for a GLM API call.

        Matches the browser's request headers exactly (spoofing.ts).
        """
        signature = cls._compute_signature(request_id, timestamp, user_id)
        return {
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
            "accept-language": "en-US",
            "x-fe-version": GLM_FE_VERSION,
            "referer": "",
            "user-agent": GLM_USER_AGENT,
            "x-region": "overseas",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "connection": "keep-alive",
            "host": "chat.z.ai",
            "origin": GLM_BASE_URL,
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "x-signature": signature,
        }

    # ── Variables (spoofing.ts: buildGlmVariables) ──────────────────────────

    @classmethod
    def _build_variables(cls, user_name: str = "User") -> dict:
        """Build the variables object for the chat completion body.

        Includes {{USER_NAME}}, {{CURRENT_DATETIME}}, etc. (spoofing.ts).
        """
        now = datetime.now()
        pad = lambda n: str(n).zfill(2)
        return {
            "{{USER_NAME}}": user_name or "User",
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": f"{now.year}-{pad(now.month)}-{pad(now.day)} {pad(now.hour)}:{pad(now.minute)}:{pad(now.second)}",
            "{{CURRENT_DATE}}": f"{now.year}-{pad(now.month)}-{pad(now.day)}",
            "{{CURRENT_TIME}}": f"{pad(now.hour)}:{pad(now.minute)}:{pad(now.second)}",
            "{{CURRENT_WEEKDAY}}": WEEKDAYS[now.weekday()],
            "{{CURRENT_TIMEZONE}}": "Africa/Cairo",
            "{{USER_LANGUAGE}}": "en-US",
        }

    # ── Auth cache (AuthFileMixin) ──────────────────────────────────────────

    @classmethod
    def get_auth_from_cache(cls):
        cache_file_path = cls.get_cache_file()
        if cache_file_path.is_file():
            file_mtime = cache_file_path.stat().st_mtime
            if time.time() - file_mtime < 5 * 60:
                try:
                    with open(cache_file_path, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    try:
                        os.remove(cache_file_path)
                    except OSError:
                        pass
        return None

    @classmethod
    def save_auth_to_cache(cls, data):
        cache_file_path = cls.get_cache_file()
        with cache_file_path.open('w') as f:
            json.dump(data, f)

    # ── Models ──────────────────────────────────────────────────────────────

    @classmethod
    def get_models(cls, **kwargs) -> list:
        if not cls.models:
            response = requests.get(f"{cls.url}/api/v1/auths/")
            auth_data = response.json()
            cls.api_key = auth_data.get("token")
            cls.auth_user_id = str(auth_data.get("id", ""))
            cls.auth_user_name = auth_data.get("name") or auth_data.get("nickname") or "User"
            response = requests.get(
                f"{cls.url}/api/models",
                headers={"Authorization": f"Bearer {cls.api_key}"}
            )
            items = response.json().get("data", [])
            cls.model_aliases = {
                item.get("name", "").replace("\u4efb\u52a1\u4e13\u7528", "ChatGLM"): item.get("id")
                for item in items
            }
            cls.models = list(cls.model_aliases.keys())
        return cls.models

    # ── Session (session.ts: getCurrentUser, getOrCreateChatSession) ────────

    @classmethod
    def _get_current_user(cls, session) -> dict:
        """Validate JWT and get user info via /api/v1/auths/ (session.ts).

        Returns dict with id, name, email, or raises ProviderException.
        """
        async def _fetch():
            async with session.get(
                f"{GLM_BASE_URL}/api/v1/auths/",
                headers={
                    "Authorization": f"Bearer {cls.api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status >= 400:
                    raise ProviderException(f"Cannot validate GLM account (status {response.status})")
                data = await response.json()
                user = data.get("user") or data
                if not user or not user.get("id"):
                    raise ProviderException("No user ID in auth response")
                return {
                    "id": str(user["id"]),
                    "name": user.get("name") or user.get("nickname") or "User",
                    "email": user.get("email", ""),
                }
        # Run in the async context — this is called from create_async_generator
        import asyncio
        return asyncio.get_event_loop().run_until_complete(_fetch())

    @classmethod
    async def _get_or_create_chat_session(cls, session, model: str) -> str:
        """Create a new chat session via /api/v1/chats/new (session.ts).

        Returns the chat_id string.
        """
        chat_id = str(uuid.uuid4())
        chat_body = {
            "chat": {
                "id": chat_id,
                "title": "New Chat",
                "models": [model],
                "params": {},
                "history": {"messages": {}, "currentId": None},
                "tags": [],
                "flags": [],
                "features": [],
                "mcp_servers": [],
                "enable_thinking": "glm-5" in model or "glm-4" in model,
                "reasoning_effort": "max" if "glm-5" in model else "",
                "auto_web_search": False,
                "message_version": 1,
                "extra": {},
                "timestamp": int(time.time() * 1000),
                "type": "default",
            }
        }
        async with session.post(
            f"{GLM_BASE_URL}/api/v1/chats/new",
            json=chat_body,
            headers={
                "Authorization": f"Bearer {cls.api_key}",
                "Content-Type": "application/json",
            },
        ) as response:
            if response.status >= 400:
                raise ProviderException(f"Cannot create GLM chat session (status {response.status})")
            chat_data = await response.json()
            return chat_data.get("id") or chat_data.get("chat", {}).get("id") or chat_id

    # ── Captcha ─────────────────────────────────────────────────────────────

    @classmethod
    async def _get_captcha_verify_param(cls) -> str:
        """Resolve a captcha_verify_param, falling back to an empty string when
        the browser-based solver is unavailable (e.g. zendriver not installed)."""
        if not captcha_solver_available():
            return ""
        return await get_captcha_verify_param()

    # ── Main entry point (pipeline.ts: proxyViaGlmWebChat) ──────────────────

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        reasoning_effort: str = None,
        web_search: bool = False,
        conversation: JsonConversation = None,
        **kwargs
    ) -> AsyncResult:
        cls.get_models()
        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            pass
        if conversation is None:
            conversation = JsonConversation(chat_id=None, message_id=None, parent_id=None, completion_id=None)

        if not cls.api_key:
            raise ProviderException("Failed to obtain API key from Z.ai authentication endpoint")

        conversation.parent_id = conversation.completion_id
        conversation.completion_id = str(uuid.uuid4())
        conversation.message_id = str(uuid.uuid4())

        # signature_prompt: first 500 chars of all message contents joined (pipeline.ts)
        signature_prompt = "\n".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in [messages[-1]]
        )[:500] or ""

        # Determine model-specific features (pipeline.ts)
        is_glm5 = "glm-5" in model
        glm_features = {
            "image_generation": False,
            "web_search": web_search,
            "auto_web_search": False,
            "preview_mode": True,
            "flags": [],
            "vlm_tools_enable": False,
            "vlm_web_search_enable": False,
            "vlm_website_mode": False,
            "enable_thinking": reasoning_effort and reasoning_effort != "none",
            "reasoning_effort": "max" if is_glm5 else "",
        }

        async with StreamSession(
            impersonate="chrome",
            proxy=proxy,
        ) as session:
            # 1. Validate JWT and get user info (session.ts: getCurrentUser)
            user_id = cls.auth_user_id or ""
            user_name = cls.auth_user_name or "User"
            try:
                async with session.get(
                    f"{GLM_BASE_URL}/api/v1/auths/",
                    headers={
                        "Authorization": f"Bearer {cls.api_key}",
                        "Content-Type": "application/json",
                    },
                ) as response:
                    if response.status < 400:
                        auth_data = await response.json()
                        user = auth_data.get("user") or auth_data
                        if user and user.get("id"):
                            user_id = str(user["id"])
                            user_name = user.get("name") or user.get("nickname") or "User"
                            cls.auth_user_id = user_id
                            cls.auth_user_name = user_name
            except Exception:
                pass  # Best effort — continue with cached values

            # 2. Create chat session (session.ts: getOrCreateChatSession)
            if conversation.chat_id is None:
                conversation.chat_id = await cls._get_or_create_chat_session(session, model)

            yield conversation

            # 3. Build the request body (pipeline.ts)
            data = {
                "stream": True,
                "model": model,
                "messages": [messages[-1]],
                "signature_prompt": signature_prompt,
                "params": {},
                "extra": {},
                "features": glm_features,
                "variables": cls._build_variables(user_name),
                "chat_id": conversation.chat_id,
                "id": conversation.completion_id,
                "current_user_message_id": conversation.message_id,
                "current_user_message_parent_id": conversation.parent_id,
                "background_tasks": {
                    "title_generation": True,
                    "tags_generation": True,
                },
                "captcha_verify_param": await cls._get_captcha_verify_param(),
            }

            # 4. Build fingerprint query string and URL (spoofing.ts)
            url_params = cls._build_url_params(cls.api_key, user_id)
            query_string = urllib.parse.urlencode(url_params)
            endpoint = f"{GLM_BASE_URL}/api/v2/chat/completions?{query_string}"

            # Extract request_id and timestamp for signature
            request_id = url_params.get("requestId", str(uuid.uuid4()))
            timestamp = url_params.get("timestamp", str(int(time.time() * 1000)))

            # 5. Retry loop for captcha errors (pipeline.ts + handler.ts)
            max_captcha_retries = 2
            for attempt in range(max_captcha_retries + 1):
                body_str = json.dumps(data, separators=(',', ':'))
                headers = cls._build_headers(
                    cls.api_key, body_str, request_id, timestamp, user_id
                )

                async with session.post(
                    endpoint,
                    data=body_str,
                    headers=headers,
                ) as response:
                    # HTTP-level captcha rejection (pipeline.ts)
                    if response.status == 403 and attempt < max_captcha_retries:
                        body = await response.text()
                        if "FRONTEND_CAPTCHA" in body or "captcha" in body.lower():
                            invalidate_captcha_token()
                            data["captcha_verify_param"] = await cls._get_captcha_verify_param()
                            continue
                        await raise_for_status(response)

                    await raise_for_status(response)

                    # 6. Parse SSE stream (stream.ts: parseGlmSseLine)
                    try:
                        async for chunk in cls._iter_completion(response):
                            yield chunk
                        return
                    except _CaptchaRequired:
                        if attempt >= max_captcha_retries:
                            raise
                        invalidate_captcha_token()
                        data["captcha_verify_param"] = await cls._get_captcha_verify_param()
                        continue

    # ── SSE stream parser (stream.ts: parseGlmSseLine) ──────────────────────

    @staticmethod
    async def _iter_completion(response):
        """Yield Usage / Reasoning / content chunks from a chat completion SSE stream.

        Implements the three-phase SSE parser from stream.ts:
          - thinking → Reasoning delta
          - answer   → content delta
          - other    → usage stats
          - done     → completion

        Raises ``_CaptchaRequired`` if the server emits a captcha verification
        error inside the stream (FRONTEND_CAPTCHA_REQUIRED).
        """
        usage = None
        async for chunk in response.sse():
            if chunk.get("type") != "chat:completion":
                continue

            chunk_data = chunk.get("data", {})

            # Detect captcha errors embedded in the SSE stream (pipeline.ts)
            error = chunk_data.get("error")
            if error:
                error_code = error.get("code") or error.get("error_code") or ""
                if "CAPTCHA" in str(error_code):
                    raise _CaptchaRequired(error.get("detail", error_code))
                raise ProviderException(json.dumps(error))

            phase = chunk_data.get("phase", "")
            delta_content = chunk_data.get("delta_content", "")

            if phase == "thinking":
                # Strip summary tags and yield reasoning
                if delta_content:
                    cleaned = delta_content.split("</summary>\n>")[-1]
                    if cleaned:
                        yield Reasoning(cleaned)

            elif phase == "answer":
                # Yield content, stripping details tags
                edit_content = chunk_data.get("edit_content")
                if edit_content:
                    yield edit_content.split("\n</details>\n")[-1]
                elif delta_content:
                    yield delta_content

            elif phase == "other":
                # Extract usage stats
                if not usage and chunk_data.get("usage"):
                    usage = chunk_data.get("usage")
                    yield Usage(**usage)

            elif phase == "done":
                if usage:
                    yield Usage(**usage)
                return

            else:
                # Unrecognized phase — emit delta_content as content if present
                if delta_content:
                    yield delta_content
