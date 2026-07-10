from __future__ import annotations

import os
import json
import time
import hashlib
import uuid
import requests
import urllib.parse

from ..typing import AsyncResult, Messages
from ..providers.response import Usage, Reasoning
from ..requests import StreamSession, raise_for_status
from ..errors import ModelNotFoundError, ProviderException
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin
from .helper import get_last_user_message

class GLM(AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin):
    url = "https://chat.z.ai"
    api_endpoint = "https://chat.z.ai/api/chat/completions"
    working = True
    active_by_default = True
    default_model = "GLM-4.5"

    api_key = None
    auth_user_id = None

    @classmethod
    def _build_url_params(cls, token: str, user_id: str) -> str:
        """Build URL query parameters including browser fingerprint data."""
        current_time = str(int(time.time() * 1000))
        request_id = str(uuid.uuid1())

        params = {
            "timestamp": current_time,
            "requestId": request_id,
            "user_id": user_id or "",
            "version": "0.0.1",
            "platform": "web",
            "token": token,
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/130.0.0.0 Safari/537.36"
            ),
            "language": "en-US",
            "languages": "en-US,en",
            "timezone": "America/New_York",
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "screen_resolution": "1920x1080",
            "viewport_height": "900",
            "viewport_width": "1440",
            "viewport_size": "1440x900",
            "color_depth": "24",
            "pixel_ratio": "1",
            "current_url": "https://chat.z.ai/",
            "pathname": "/",
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "",
            "title": "Z.ai",
            "timezone_offset": str(-(time.timezone if time.daylight == 0 else time.altzone) // 60),
            "local_time": time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime()),
            "utc_time": time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime()),
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "0",
            "browser_name": "Chrome",
            "os_name": "Windows",
        }

        return urllib.parse.urlencode(params)

    @classmethod
    def _compute_signature(cls, body_json: str) -> str:
        """Compute x-signature as SHA-256 hex digest of the serialised request body."""
        return hashlib.sha256(body_json.encode("utf-8")).hexdigest()

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

    @classmethod
    def get_models(cls, **kwargs) -> list:
        if not cls.models:
            response = requests.get(f"{cls.url}/api/v1/auths/")
            auth_data = response.json()
            cls.api_key = auth_data.get("token")
            cls.auth_user_id = auth_data.get("id", "")
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

    @classmethod
    def get_last_user_message_content(cls, messages):
        for message in reversed(messages):
            if message.get('role') == 'user':
                return message.get('content', '')
        return ''

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        reasoning_effort: str = "max",
        enable_thinking: bool = True,
        web_search: bool = False,
        **kwargs
    ) -> AsyncResult:
        cls.get_models()
        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            pass

        if not cls.api_key:
            raise ProviderException("Failed to obtain API key from Z.ai authentication endpoint")

        # Build the request body first so we can sign the exact bytes we send.
        # Shape matches the browser's actual chat completions payload.
        message_id = str(uuid.uuid4())
        prompt = get_last_user_message(messages)
        data = {
            "chat": {
                "id": "",
                "title": "New Chat",
                "models": [
                    "glm-4.7"
                ],
                "params": {},
                "history": {
                    "messages": {
                        message_id: {
                            "id": message_id,
                            "parentId": None,
                            "childrenIds": [],
                            "role": "user",
                            "content": prompt,
                            "timestamp": int(time.time() * 1000),
                            "models": [
                                "glm-4.7"
                            ]
                        }
                    },
                    "currentId": message_id
                },
                "tags": [],
                "flags": [],
                "features": [],
                "mcp_servers": [],
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort,
                "auto_web_search": web_search,
                "message_version": 1,
                "extra": {},
                "timestamp": int(time.time() * 1000),
                "type": "default"
            }
        }
        async with StreamSession(
            impersonate="chrome",
            proxy=proxy,
        ) as session:
            url = "https://chat.z.ai/api/v1/chats/new"
            async with session.post(
                url,
                json=data,
                headers={
                    "Authorization": f"Bearer {cls.api_key}",
                    "Content-Type": "application/json",
                }
            ) as response:
                await raise_for_status(response)
                chat_data = await response.json()
                chat_id = chat_data.get("id")
                if not chat_id:
                    raise ProviderException("Failed to create new chat session")
            # Compact JSON matching browser JSON.stringify() output.
            data = {
                "stream": True,
                "model": "glm-4.7",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "signature_prompt": prompt,
                "params": {},
                "extra": {},
                "features": {
                    "image_generation": False,
                    "web_search": False,
                    "auto_web_search": False,
                    "preview_mode": True,
                    "flags": [],
                    "vlm_tools_enable": False,
                    "vlm_web_search_enable": False,
                    "vlm_website_mode": False,
                    "enable_thinking": True
                },
                "variables": {
                    "{{USER_NAME}}": "Guest-1783644168311",
                    "{{USER_LOCATION}}": "Unknown",
                    "{{CURRENT_DATETIME}}": "2026-07-10 03:54:21",
                    "{{CURRENT_DATE}}": "2026-07-10",
                    "{{CURRENT_TIME}}": "03:54:21",
                    "{{CURRENT_WEEKDAY}}": "Friday",
                    "{{CURRENT_TIMEZONE}}": "Europe/Berlin",
                    "{{USER_LANGUAGE}}": "en-US"
                },
                "chat_id": chat_id,
                "id": str(uuid.uuid4()),
                "current_user_message_id": message_id,
                "current_user_message_parent_id": None,
                "background_tasks": {
                    "title_generation": True,
                    "tags_generation": True
                },
                "captcha_verify_param": "eyJjZXJ0aWZ5SWQiOiJ1eTZSaXVCSkxaIiwic2NlbmVJZCI6ImRpZGszM2UwIiwiaXNTaWduIjp0cnVlLCJzZWN1cml0eVRva2VuIjoiNm9PbzdlNzJuQTYxdVZMaVpWS2lMWXFGMW05ck9ubzN2RUlQSkthTDdLTHhDSnFiMVVCd1JwbDRwN0VjRlRnZFA1OVdiNDA1WVhZRmZkRVlzZjMzZ05qUGNxYWZscWJRTFpRZFgycllkLzhiaG5xaElwQzdTblJsSXhHUHNxdlgifQ=="
            }
            body_json = json.dumps(data, separators=(',', ':'))

            url_params = cls._build_url_params(cls.api_key, cls.auth_user_id or "")
            signature = cls._compute_signature(body_json)
            endpoint = f"https://chat.z.ai/api/v2/chat/completions?{url_params}"
            async with session.get(
                endpoint,
                headers={
                    "Authorization": f"Bearer {cls.api_key}",
                    "Content-Type": "application/json",
                    "x-fe-version": "prod-fe-1.0.95",
                    "x-signature": signature,
                },
            ) as response:
                await raise_for_status(response)
                usage = None
                async for chunk in response.sse():
                    if chunk.get("type") == "chat:completion":
                        if not usage:
                            usage = chunk.get("data", {}).get("usage")
                            if usage:
                                yield Usage(**usage)
                        if chunk.get("data", {}).get("phase") == "thinking":
                            delta_content = chunk.get("data", {}).get("delta_content")
                            delta_content = delta_content.split("</summary>\n>")[-1] if delta_content else ""
                            if delta_content:
                                yield Reasoning(delta_content)
                        else:
                            edit_content = chunk.get("data", {}).get("edit_content")
                            if edit_content:
                                yield edit_content.split("\n</details>\n")[-1]
                            else:
                                delta_content = chunk.get("data", {}).get("delta_content")
                                if delta_content:
                                    yield delta_content