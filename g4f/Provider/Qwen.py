from __future__ import annotations

import asyncio
import datetime
import hashlib
import hmac
import json
import os
import re
import uuid
from time import time
from typing import Literal, Optional, Dict
from urllib.parse import quote

import aiohttp

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message
from .openai.har_file import get_har_files
from .qwen.cookie_generator import generate_cookies
from .. import debug
from ..cookies import get_cookies_dir
from ..errors import MissingAuthError, RateLimitError, ResponseError, CloudflareError
from ..image import to_bytes, detect_file_type
from ..providers.response import (
    JsonConversation,
    Reasoning,
    Usage,
    ImageResponse,
    FinishReason,
    Sources,
    format_link,
)
from ..requests import (
    sse_stream,
    StreamSession,
    raise_for_status,
    get_args_from_nodriver,
    has_cdp,
)
from ..tools.media import merge_media
from ..typing import AsyncResult, Messages, MediaListType

try:
    import curl_cffi

    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False
from ..requests import CDPTab

# Global variables to manage Qwen Image Cache
ImagesCache: Dict[str, dict] = {}


def get_oss_headers(
    method: str, date_str: str, sts_data: dict, content_type: str
) -> dict[str, str]:
    bucket_name = sts_data.get("bucketname", "qwen-webui-prod")
    file_path = sts_data.get("file_path", "")
    access_key_id = sts_data.get("access_key_id")
    access_key_secret = sts_data.get("access_key_secret")
    security_token = sts_data.get("security_token")
    headers = {
        "Content-Type": content_type,
        "x-oss-content-sha256": "UNSIGNED-PAYLOAD",
        "x-oss-date": date_str,
        "x-oss-security-token": security_token,
        "x-oss-user-agent": "aliyun-sdk-js/6.23.0 Chrome 132.0.0.0 on Windows 10 64-bit",
    }
    headers_lower = {k.lower(): v for k, v in headers.items()}

    canonical_headers_list = []
    signed_headers_list = []
    required_headers = [
        "content-md5",
        "content-type",
        "x-oss-content-sha256",
        "x-oss-date",
        "x-oss-security-token",
        "x-oss-user-agent",
    ]
    for header_name in sorted(required_headers):
        if header_name in headers_lower:
            canonical_headers_list.append(f"{header_name}:{headers_lower[header_name]}")
            signed_headers_list.append(header_name)

    canonical_headers = "\n".join(canonical_headers_list) + "\n"
    canonical_uri = f"/{bucket_name}/{quote(file_path, safe='/')}"

    canonical_request = (
        f"{method}\n{canonical_uri}\n\n{canonical_headers}\n\nUNSIGNED-PAYLOAD"
    )

    date_parts = date_str.split("T")
    date_scope = f"{date_parts[0]}/ap-southeast-1/oss/aliyun_v4_request"
    string_to_sign = f"OSS4-HMAC-SHA256\n{date_str}\n{date_scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"

    def sign(key, msg):
        return hmac.new(
            key, msg.encode() if isinstance(msg, str) else msg, hashlib.sha256
        ).digest()

    date_key = sign(f"aliyun_v4{access_key_secret}".encode(), date_parts[0])
    region_key = sign(date_key, "ap-southeast-1")
    service_key = sign(region_key, "oss")
    signing_key = sign(service_key, "aliyun_v4_request")
    signature = hmac.new(
        signing_key, string_to_sign.encode(), hashlib.sha256
    ).hexdigest()

    headers[
        "authorization"
    ] = f"OSS4-HMAC-SHA256 Credential={access_key_id}/{date_scope},Signature={signature}"
    return headers


text_models = [
    "qwen3.7-plus",
    'qwen3.8-max',
    'qwen3.8-omni-flash'
    "qwen3.7-max",
    "qwen3.6-plus",
    "qwen3.6-max-preview",
    "qwen3.6-27b",
    "qwen-latest-series-invite-beta-v24",
    "qwen-latest-series-invite-beta-v16",
    "qwen3.5-plus",
    "qwen3.5-omni-plus",
    "qwen3.6-35b-a3b",
    "qwen3.5-flash",
    "qwen3.5-max-2026-03-08",
    "qwen3.6-plus-preview",
    "qwen3.5-397b-a17b",
    "qwen3.5-122b-a10b",
    "qwen3.5-omni-flash",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3-max-2026-01-23",
    "qwen-plus-2025-07-28",
    "qwen3-coder-plus",
    "qwen3-vl-plus",
    "qwen3-omni-flash-2025-12-01",
]

image_models = [
    "qwen3.7-plus",
    'qwen3.8-max',
    'qwen3.8-omni-flash'
    "qwen3.7-max",
    "qwen3.6-plus",
    "qwen3.6-27b",
    "qwen3.5-plus",
    "qwen3.5-omni-plus",
    "qwen3.6-35b-a3b",
    "qwen3.5-flash",
    "qwen3.5-397b-a17b",
    "qwen3.5-122b-a10b",
    "qwen3.5-omni-flash",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3-max-2026-01-23",
    "qwen-plus-2025-07-28",
    "qwen3-coder-plus",
    "qwen3-vl-plus",
    "qwen3-omni-flash-2025-12-01",
]

vision_models = [
    "qwen3.7-plus",
    'qwen3.8-max',
    'qwen3.8-omni-flash'
    "qwen3.6-plus",
    "qwen3.6-27b",
    "qwen-latest-series-invite-beta-v16",
    "qwen3.5-plus",
    "qwen3.5-omni-plus",
    "qwen3.6-35b-a3b",
    "qwen3.5-flash",
    "qwen3.5-397b-a17b",
    "qwen3.5-122b-a10b",
    "qwen3.5-omni-flash",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3-max-2026-01-23",
    "qwen-plus-2025-07-28",
    "qwen3-coder-plus",
    "qwen3-vl-plus",
    "qwen3-omni-flash-2025-12-01",
]

models = [
    "qwen3.7-plus",
    'qwen3.8-max',
    'qwen3.8-omni-flash'
    "qwen3.7-max",
    "qwen3.6-plus",
    "qwen3.6-max-preview",
    "qwen3.6-27b",
    "qwen-latest-series-invite-beta-v24",
    "qwen-latest-series-invite-beta-v16",
    "qwen3.5-plus",
    "qwen3.5-omni-plus",
    "qwen3.6-35b-a3b",
    "qwen3.5-flash",
    "qwen3.5-max-2026-03-08",
    "qwen3.6-plus-preview",
    "qwen3.5-397b-a17b",
    "qwen3.5-122b-a10b",
    "qwen3.5-omni-flash",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3-max-2026-01-23",
    "qwen-plus-2025-07-28",
    "qwen3-coder-plus",
    "qwen3-vl-plus",
    "qwen3-omni-flash-2025-12-01",
]


class Qwen(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Provider for Qwen's chat service (chat.qwen.ai), with configurable
    parameters (stream, enable_thinking) and print logs.
    """

    url = "https://chat.qwen.ai"
    working = True
    active_by_default = True
    image_cache = True
    _models_loaded = True
    image_models = image_models
    text_models = text_models
    vision_models = vision_models
    models: list[str] = models
    default_model: str = models[0]
    tool_support_prompts = [
        "<tool_response> blocks, or any proprietary function-calling markup. Only the plain "
        "JSON object described below is allowed when a tool is needed.",
        # --- Delegate execution to the caller ---
        "When a tool is needed, do not execute it yourself. Instead, reply with ONLY the JSON "
        "tool-call object described below so the caller can run the tool and return the result "
        "to you. Do not attempt to run, simulate, guess, or imagine the tool's output, and do "
        "not produce a fake tool result.",
        "Do not wrap the JSON in markdown code fences (no ```json or ```), do not add "
        "explanatory text before or after it, and do not prefix it with phrases like 'Here is "
        "the tool call:' or 'I will use a tool.'. Output the raw JSON object only.",
        "Use only the tool names provided in the 'Available tools' list below. Do not invent "
        "tool names, do not call tools that were not listed, and do not rename the listed tools.",
        "The `arguments` value MUST be a JSON object (not a string, not null) matching the "
        "tool's parameter schema. Omit optional parameters you do not need rather than passing "
        "null or empty strings.",
        # --- Plain-text fallback ---
        "If no tool is needed, answer normally with plain text and do not output any JSON "
        "tool-call object. Never mix a normal answer with a tool-call JSON in the same reply.",
        "If the user's request can be answered directly from your own knowledge, do not call a "
        "tool. Only request a tool when the task genuinely requires external data, computation, "
        "or an action that you cannot perform yourself.",
    ]

    _midtoken: str = None
    _midtoken_uses: int = 0

    _har_cookies: Optional[dict] = None
    _har_headers: Optional[dict] = None
    _har_loaded_at: float = 0.0
    _HAR_TTL: float = 600.0
    _HAR_HEADERS_WHITELIST = [
        "user-agent",
        "accept-language",
        "sec-ch-ua",
        "sec-ch-ua-mobile",
        "sec-ch-ua-platform",
        "sec-fetch-dest",
        "sec-fetch-mode",
        "sec-fetch-site",
        "bx-ua",
        "bx-umidtoken",
        "bx-v",
        "version",
        "source",
        "timezone",
    ]

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls._models_loaded and has_curl_cffi:
            _token = kwargs.get("token") or kwargs.get("api_key")
            headers = cls._get_headers(_token) if _token else {}
            response = curl_cffi.get(f"{cls.url}/api/models", headers=headers)
            if response.ok:
                models = response.json().get("data", [])
                cls.text_models = [
                    model["id"]
                    for model in models
                    if "t2t" in model.get("info", {}).get("meta", {}).get("chat_type")
                ]

                cls.image_models = [
                    model["id"]
                    for model in models
                    if "image_edit"
                    in model.get("info", {}).get("meta", {}).get("chat_type")
                    or "t2i" in model.get("info", {}).get("meta", {}).get("chat_type")
                ]

                cls.vision_models = [
                    model["id"]
                    for model in models
                    if model.get("info", {})
                    .get("meta", {})
                    .get("capabilities", {})
                    .get("vision")
                ]

                cls.models = [model["id"] for model in models]
                cls.default_model = cls.models[0]
                cls._models_loaded = True
                cls.live += 1
                debug.log(f"Loaded {len(cls.models)} models from {cls.url}")

            else:
                debug.log(
                    f"Failed to load models from {cls.url}: {response.status_code} {response.reason}"
                )
        return cls.models

    @classmethod
    async def prepare_files(cls, media, session: StreamSession, headers=None) -> list:
        if headers is None:
            headers = {}
        files = []
        for index, (_file, file_name) in enumerate(media):
            data_bytes = to_bytes(_file)
            # Check Cache
            hasher = hashlib.md5()
            hasher.update(data_bytes)
            image_hash = hasher.hexdigest()
            file = ImagesCache.get(image_hash)
            if cls.image_cache and file:
                debug.log("Using cached image")
                files.append(file)
                continue

            extension, file_type = detect_file_type(data_bytes)
            file_name = file_name or f"file-{len(data_bytes)}{extension}"
            file_size = len(data_bytes)

            # Get File Url
            async with session.post(
                f"{cls.url}/api/v2/files/getstsToken",
                json={
                    "filename": file_name,
                    "filesize": file_size,
                    "filetype": file_type,
                },
                headers=headers,
            ) as r:
                await raise_for_status(r, "Create file failed")
                res_data = await r.json()
                data = res_data.get("data")

                if res_data["success"] is False:
                    raise RateLimitError(f"{data['code']}:{data['details']}")
                file_url = data.get("file_url")
                file_id = data.get("file_id")

            # Put File into Url
            str_date = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            headers_put = get_oss_headers("PUT", str_date, data, file_type)
            async with session.put(
                file_url.split("?")[0], data=data_bytes, headers=headers_put
            ) as response:
                await raise_for_status(response)

            file_class: Literal["default", "vision", "video", "audio", "document"]
            _type: Literal["file", "image", "video", "audio"]
            show_type: Literal["file", "image", "video", "audio"]
            if "image" in file_type:
                _type = "image"
                show_type = "image"
                file_class = "vision"
            elif "video" in file_type:
                _type = "video"
                show_type = "video"
                file_class = "video"
            elif "audio" in file_type:
                _type = "audio"
                show_type = "audio"
                file_class = "audio"
            else:
                _type = "file"
                show_type = "file"
                file_class = "document"

            file = {
                "type": _type,
                "file": {
                    "created_at": int(time() * 1000),
                    "data": {},
                    "filename": file_name,
                    "hash": None,
                    "id": file_id,
                    "meta": {
                        "name": file_name,
                        "size": file_size,
                        "content_type": file_type,
                    },
                    "update_at": int(time() * 1000),
                },
                "id": file_id,
                "url": file_url,
                "name": file_name,
                "collection_name": "",
                "progress": 0,
                "status": "uploaded",
                "greenNet": "success",
                "size": file_size,
                "error": "",
                "itemId": str(uuid.uuid4()),
                "file_type": file_type,
                "showType": show_type,
                "file_class": file_class,
                "uploadTaskId": str(uuid.uuid4()),
            }
            debug.log(f"Uploaded file: {file_name}")
            ImagesCache[image_hash] = file
            files.append(file)
        return files

    @classmethod
    async def get_args(cls, proxy, **kwargs):
        grecaptcha = []

        async def callback(page: CDPTab):
            while not await page.evaluate(
                "window.__baxia__ && window.__baxia__.getFYModule"
            ):
                await asyncio.sleep(1)
            captcha = await page.evaluate(
                """window.baxiaCommon.getUA()""", await_promise=True
            )
            if isinstance(captcha, str):
                grecaptcha.append(captcha)
            else:
                raise Exception(captcha)

        args = await get_args_from_nodriver(cls.url, proxy=proxy, callback=callback)

        return args, next(iter(grecaptcha))

    @classmethod
    async def raise_for_status(cls, response, message=None):
        await raise_for_status(response, message)
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("text/html"):
            html = (await response.text()).strip()
            if html.startswith("<!doctypehtml>") and "aliyun_waf_aa" in html:
                raise CloudflareError(message or html)

    @classmethod
    def _read_har(cls) -> tuple[Optional[dict], Optional[dict]]:
        """
        Read cookies and fingerprint headers from a chat.qwen.ai HAR file
        (e.g. `chat.qwen.ai.har` placed in the cookies dir). Returns
        (cookies, headers) or (None, None) if no matching HAR file exists.
        """
        now = time()
        if cls._har_cookies is not None and now - cls._har_loaded_at < cls._HAR_TTL:
            return cls._har_cookies, cls._har_headers
        cls._har_cookies = cls._har_headers = None
        cls._har_loaded_at = now
        try:
            har_files = [
                path
                for path in get_har_files()
                if "qwen" in os.path.basename(path).lower()
            ]
        except Exception as e:
            debug.log(f"[Qwen] No usable HAR file found: {e}")
            return None, None
        for path in reversed(har_files):  # newest first (sorted by mtime)
            try:
                with open(path, "rb") as file:
                    har = json.loads(file.read())
            except (OSError, json.JSONDecodeError) as e:
                debug.log(f"[Qwen] Failed to read HAR file {path}: {e}")
                continue
            best_entry, best_score = None, 0
            for entry in har.get("log", {}).get("entries", []):
                url = entry.get("request", {}).get("url", "")
                if "chat.qwen.ai" not in url:
                    continue
                score = (
                    2
                    if "chat/completions" in url
                    else (1 if "/api/v2/" in url else 0)
                )
                if score > best_score:
                    best_entry, best_score = entry, score
            if best_entry is None:
                continue
            cookies = {}
            headers = {}
            for header in best_entry["request"].get("headers", []):
                name = header.get("name", "")
                value = header.get("value", "")
                lower = name.lower()
                if lower == "cookie":
                    for part in value.split(";"):
                        part = part.strip()
                        if "=" in part:
                            key, val = part.split("=", 1)
                            cookies[key.strip()] = val.strip()
                elif lower in cls._HAR_HEADERS_WHITELIST and not name.startswith(":"):
                    headers[name] = value
            if cookies:
                debug.log(
                    f"[Qwen] Using {len(cookies)} cookies and {len(headers)} fingerprint headers from HAR file: {os.path.basename(path)}"
                )
                cls._har_cookies = cookies
                cls._har_headers = headers
                return cookies, headers
            debug.log(f"[Qwen] No cookies found in HAR file: {os.path.basename(path)}")
        return None, None

    @classmethod
    async def _read_cdp(cls, proxy: str = None) -> tuple[dict, dict]:
        """Capture cookies and fingerprint headers from a live browser via CDP.

        Opens chat.qwen.ai in a visible browser window, sends a seed message
        and intercepts the browser's own "/api/v2/chats/new" or
        "/api/v2/chat/completions" request to copy its baxia fingerprint
        headers (bx-ua, bx-umidtoken) and cookies. Intercepted requests are
        continued, so the chat the user sees completes normally.
        """
        from ..requests.cdp import CDPSession

        async with CDPSession(proxy=proxy, headless=False) as session:
            await session.call("Network.enable")
            await session.call(
                "Fetch.enable",
                patterns=[{"urlPattern": "*chat.qwen.ai/api/v2/*", "requestStage": "Request"}],
            )
            await session.navigate(cls.url)
            # Guest mode shows the composer right away; waiting also covers
            # logins, captchas and slow loads (up to 5 minutes).
            deadline = time() + 300
            while not await session.evaluate_js("!!document.querySelector('#chat-input, textarea')"):
                if time() > deadline:
                    raise MissingAuthError(
                        "[Qwen] chat composer not found in the browser window in time"
                    )
                await asyncio.sleep(2)
            debug.log("[Qwen] Composer detected — sending a seed message")
            await session.evaluate_js("""
                (() => {
                    const editor = document.querySelector('#chat-input, textarea');
                    editor.focus();
                    document.execCommand('insertText', false, 'Hello');
                })()
            """)
            await asyncio.sleep(1)
            sent = await session.evaluate_js("""
                (() => {
                    const button = document.querySelector('#send-message-button');
                    if (button) { button.click(); return true; }
                    return false;
                })()
            """)
            if not sent:
                await session.evaluate_js("""
                    const editor = document.querySelector('#chat-input, textarea');
                    editor.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));
                """)
            # Capture the browser's own chat request (up to 2 minutes).
            # A queue (not wait_for_event) so requests paused while another
            # one is processed are never lost.
            queue: asyncio.Queue = asyncio.Queue()
            session.add_event_handler("Fetch.requestPaused", queue)
            headers = None
            deadline = time() + 120
            while headers is None:
                try:
                    paused = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    if time() > deadline:
                        raise MissingAuthError("[Qwen] no chat request captured in the browser")
                    # The first send may have been blocked by a captcha —
                    # retry it while the user completes the challenge.
                    await session.evaluate_js("""
                        (() => {
                            const button = document.querySelector('#send-message-button');
                            if (button) button.click();
                        })()
                    """)
                    continue
                request = paused.get("request") or {}
                url = request.get("url", "")
                found = {
                    key: value for key, value in (request.get("headers") or {}).items()
                    if not key.startswith(":")
                }
                # Let the browser request pass so the visible chat completes.
                await session.call("Fetch.continueRequest", requestId=paused.get("requestId"))
                if "/api/v2/chats/new" in url or "/api/v2/chat/completions" in url:
                    headers = found
            # Resume requests paused while no one was listening anymore.
            await session.call("Fetch.disable")
            cookies = await session.get_cookies([f"{cls.url}/"])
        return cookies, {
            name: value for name, value in headers.items()
            if name.lower() in cls._HAR_HEADERS_WHITELIST
        }

    @classmethod
    async def _ensure_auth(cls, proxy: str = None, force_cdp: bool = False) -> None:
        """Populate the auth cache from a HAR file, falling back to a live
        browser capture via CDP. Generated cookies are used when both fail.
        Captured data is saved as a HAR file for reuse across restarts."""
        cookies, _ = (None, None) if force_cdp else cls._read_har()
        if cookies:
            return
        if not has_cdp:
            debug.log("[Qwen] No HAR file found and no CDP browser available — using generated cookies")
            return
        try:
            debug.log(
                "[Qwen] Capturing headers and cookies from a CDP browser"
                " — complete the login or captcha in the browser window"
            )
            cookies, headers = await cls._read_cdp(proxy)
        except Exception as e:
            debug.log(f"[Qwen] CDP capture failed: {type(e).__name__}: {e} — using generated cookies")
            return
        cls._har_cookies = cookies
        cls._har_headers = headers
        cls._har_loaded_at = time()
        cls._save_auth_har(cookies, headers)

    @classmethod
    def _save_auth_har(cls, cookies: dict, headers: dict) -> None:
        """Persist captured auth data as a HAR file in the cookies dir so it
        survives restarts (newest HAR file wins in `_read_har`)."""
        try:
            entry_headers = [
                {"name": name, "value": value} for name, value in headers.items()
            ]
            entry_headers.append({
                "name": "Cookie",
                "value": "; ".join(f"{key}={value}" for key, value in cookies.items()),
            })
            har = {
                "log": {
                    "creator": {"name": "g4f", "version": "1.0"},
                    "entries": [
                        {
                            "request": {
                                "method": "POST",
                                "url": f"{cls.url}/api/v2/chat/completions?chat_id=captured",
                                "headers": entry_headers,
                            },
                            "response": {"status": 200},
                        }
                    ],
                }
            }
            path = os.path.join(get_cookies_dir(), "chat.qwen.ai-cdp.har")
            with open(path, "w") as file:
                json.dump(har, file)
            debug.log(f"[Qwen] Auth data saved to {path}")
        except Exception as e:
            debug.log(f"[Qwen] Failed to save auth HAR file: {e}")

    @classmethod
    def _get_headers(cls, token=None):
        har_cookies, har_headers = (None, None) if token else cls._read_har()
        if har_cookies:
            cookie = "; ".join(f"{key}={value}" for key, value in har_cookies.items())
        else:
            data = generate_cookies()
            cookie = f'ssxmod_itna={data["ssxmod_itna"]};ssxmod_itna2={data["ssxmod_itna2"]}'
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
            "Content-Type": "application/json",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Connection": "keep-alive",
            "X-Requested-With": "XMLHttpRequest",
            "Cookie": cookie,
            "source": "web",
            "version": "0.3.12",
            # Fix 'FAIL_SYS_USER_VALIDATE'
            "X-Accel-Buffering": "no",
        }
        if har_headers:
            # Replay browser fingerprint headers (user-agent, sec-ch-ua, bx-*, ...)
            har_lower = {name.lower() for name in har_headers}
            headers = {
                name: value
                for name, value in headers.items()
                if name.lower() not in har_lower
            }
            headers.update(har_headers)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @classmethod
    async def _get_req_headers(cls, session, proxy=None):
        req_headers = session.headers.copy()
        if not any(name.lower() == "bx-umidtoken" for name in req_headers):
            if not cls._midtoken:
                debug.log("[Qwen] INFO: No active midtoken. Fetching a new one...")
                async with session.get(
                    "https://sg-wum.alibaba.com/w/wu.json", proxy=proxy
                ) as r:
                    r.raise_for_status()
                    text = await r.text()
                    match = re.search(r"(?:umx\.wu|__fycb)\('([^']+)'\)", text)
                    if not match:
                        raise RuntimeError("Failed to extract bx-umidtoken.")
                    cls._midtoken = match.group(1)
                    cls._midtoken_uses = 1
                    debug.log(
                        f"[Qwen] INFO: New midtoken obtained. Use count: {cls._midtoken_uses}."
                    )
            else:
                cls._midtoken_uses += 1
                debug.log(f"[Qwen] INFO: Reusing midtoken. Use count: {cls._midtoken_uses}")

            req_headers["bx-umidtoken"] = cls._midtoken
            req_headers["bx-v"] = "2.5.37"
        else:
            debug.log("[Qwen] INFO: Using bx-umidtoken from HAR file.")
        # fix error [g4f.errors.CloudflareError:aliyun_waf_aa]
        req_headers["x-request-id"] = str(uuid.uuid4())
        return req_headers

    @classmethod
    async def get_quota(cls, api_key: Optional[str] = None, **kwargs) -> dict:
        token = kwargs.get("token") or api_key
        if not token:
            await cls._ensure_auth(kwargs.get("proxy"))
        async with StreamSession(
            headers=cls._get_headers(token)
        ) as session:
            chat_payload = {
                "chatId": "",
                "models": [cls.default_model],
                "project_id": "",
                "chat_mode": "normal" if token else "guest",
                "chat_type": "t2t",
                "timestamp": int(time() * 1000),
            }
            async with session.post(
                f"{cls.url}/api/v2/chats/new",
                json=chat_payload,
                headers=await cls._get_req_headers(session, proxy=kwargs.get("proxy")),
                proxy=kwargs.get("proxy"),
            ) as resp:
                await cls.raise_for_status(resp)
                return await resp.json()

    @staticmethod
    def _image_urls(value) -> list[str]:
        """Read the image payloads used by Qwen's image and tool phases."""
        if isinstance(value, str):
            if value.startswith(("https://", "http://", "data:image/")):
                return [value]
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return []
        if isinstance(value, dict):
            for key in ("image", "url", "image_url", "file_path"):
                if isinstance(value.get(key), str) and value[key]:
                    return [value[key]]
            for key in ("images", "results", "data"):
                if key in value:
                    return Qwen._image_urls(value[key])
        if isinstance(value, list):
            return [url for item in value for url in Qwen._image_urls(item)]
        return []

    @classmethod
    def _raise_api_error(cls, error):
        if isinstance(error, dict):
            code = str(error.get("code") or error.get("errorCode") or "QwenError")
            details = error.get("details") or error.get("message") or str(error)
            message = f"{code}: {details}"
        else:
            code = "QwenError"
            message = str(error)
        if code.lower() in ("unauthorized", "401"):
            raise MissingAuthError(message)
        if code in ("RateLimited", "ParallelLimited", "quotaLimited", "429", "402") or re.search(
            r"\b(?:RateLimited|ParallelLimited|quotaLimited)\b", message
        ):
            raise RateLimitError(message)
        if any(marker in message for marker in ("FAIL_SYS_USER_VALIDATE", "RGV587_ERROR", "/punish")):
            # Only validation failures enter the existing browser recovery flow.
            raise RuntimeError(message)
        raise ResponseError(message)

    @classmethod
    async def _response_chunks(cls, response):
        if response.headers.get("content-type", "").startswith("application/json"):
            payload = await response.json()
            if not isinstance(payload, dict):
                raise ResponseError(f"Unexpected Qwen response: {payload}")
            data = payload.get("data")
            if payload.get("success") is False or (
                isinstance(data, dict) and data.get("code")
            ):
                error = payload.get("error") or data or payload
                cls._raise_api_error(error)
            chunk = {**payload, **data} if isinstance(data, dict) else payload
            if not any(key in chunk for key in ("choices", "error", "content")):
                raise ResponseError(f"Unexpected Qwen response: {payload}")
            if "choices" not in chunk and "content" in chunk:
                chunk = {**chunk, "choices": [{"message": chunk}]}
            yield chunk
        else:
            async for chunk in sse_stream(response):
                yield chunk

    @staticmethod
    def _format_citations(text: str, citations: dict) -> str:
        def replace(match):
            url = citations.get(int(match.group(1)))
            return format_link(url, match.group(1)) if url else match.group(0)

        return re.sub(r"\[\[(\d+)\]\]", replace, text)

    @classmethod
    async def _read_response(
        cls, response, conversation: JsonConversation, prompt: str
    ) -> AsyncResult:
        usage = {}
        sources = {}
        citations = {}
        answer_buffer = ""
        images = set()
        summary = ""
        thinking_started = False
        finish_reason = "stop"
        async for chunk in cls._response_chunks(response):
            if not isinstance(chunk, dict):
                continue
            error = chunk.get("error")
            if error:
                cls._raise_api_error(error)
            if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
                usage.update(chunk["usage"])
            created = chunk.get("response.created") or {}
            info = chunk.get("response.info") or {}
            response_id = (
                created.get("response_id")
                or chunk.get("response_id")
                or info.get("response_id")
            )
            if response_id and response_id != conversation.parent_id:
                conversation.parent_id = response_id
                if created.get("chat_id"):
                    conversation.chat_id = created["chat_id"]
                yield conversation
            if info.get("action") == "skip_think":
                thinking_started = False

            source_lists = [chunk.get("sources")]
            choices = chunk.get("choices") or []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                delta = {}
            phase = delta.get("phase")
            status = delta.get("status")
            content = delta.get("content")
            extra = delta.get("extra") or {}
            if not isinstance(extra, dict):
                extra = {}
            source_lists.append(extra.get("web_search_info"))
            source_lists.append(extra.get("web_extract_info"))
            tool_result = extra.get("tool_result")
            if isinstance(tool_result, dict):
                docs = tool_result.get("docs")
                used_id = tool_result.get("used_id")
                if isinstance(docs, list):
                    source_lists.append(docs)
                    if isinstance(used_id, int):
                        # used_id is the next reference number after this batch.
                        for index, doc in enumerate(docs, used_id - len(docs)):
                            if isinstance(doc, dict) and doc.get("url"):
                                citations[index] = doc["url"]
            for source_list in source_lists:
                if isinstance(source_list, list):
                    for source in source_list:
                        if isinstance(source, dict):
                            url = source.get("url") or source.get("link")
                            if url:
                                sources[url] = dict(source)

            if status == "error":
                raise ResponseError(f"Qwen {phase}: {extra.get('error') or content or extra}")
            if phase == "KeepAlive":
                continue
            if phase == "thinking_summary":
                # These fields are cumulative snapshots, repeated in usage-only events.
                thought = extra.get("summary_thought") or extra.get("summary_content") or {}
                value = thought.get("content", []) if isinstance(thought, dict) else thought
                text = "\n".join(value) if isinstance(value, list) else value
                if isinstance(text, str) and text and text != summary:
                    token = text[len(summary):] if text.startswith(summary) else text
                    summary = text
                    yield Reasoning(token)
                if isinstance(content, str) and content:
                    yield Reasoning(content)
                thinking_started = status != "finished"
            else:
                if phase:
                    summary = ""
                    thinking_started = phase in (
                        "think", "DeepThinking", "image_gen_think", "ResearchPlanning"
                    )
                image_phase = phase in (
                    "image_gen", "image_edit", "image", "image_gen_tool",
                    "image_edit_tool", "generate_image"
                )
                if phase == "tool_call":
                    function_call = delta.get("function_call") or {}
                    image_phase = function_call.get("name") in (
                        "image_gen", "image_edit", "generate_image"
                    )
                if image_phase:
                    # Prefer the display URL over another URL for the same image.
                    urls = cls._image_urls(extra.get("image_list"))
                    if not urls:
                        urls = cls._image_urls(extra.get("tool_result")) or cls._image_urls(content)
                    for index, url in enumerate(urls):
                        if not url.startswith(("https://", "http://", "data:image/")):
                            path = quote(url.replace("\\", "/").lstrip("/"), safe="/")
                            urls[index] = f"{cls.url}/api/v2/chat/{conversation.chat_id}/{path}"
                    urls = list(dict.fromkeys(url for url in urls if url not in images))
                    if urls:
                        images.update(urls)
                        yield ImageResponse(urls, prompt, extra)
                elif isinstance(content, str) and content:
                    if thinking_started:
                        yield Reasoning(content)
                    elif (
                        phase in (None, "answer", "ReportGeneration", "slides")
                        and delta.get("role") != "function"
                    ):
                        answer_buffer += content
                        # A reference can be split between SSE chunks (e.g. "[[3" / "7]]").
                        partial = re.search(r"\[(?:\[\d*\]?)?$", answer_buffer)
                        boundary = partial.start() if partial else len(answer_buffer)
                        text, answer_buffer = answer_buffer[:boundary], answer_buffer[boundary:]
                        if text:
                            yield cls._format_citations(text, citations)
                if thinking_started and status == "finished":
                    thinking_started = False

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            # A finished tool/image/thinking phase does not finish the whole response.
            if chunk.get("done") is True or "response.stopped" in chunk:
                break
        if answer_buffer:
            yield cls._format_citations(answer_buffer, citations)
        if sources:
            yield Sources(list(sources.values()))
        if usage:
            yield Usage.from_dict(usage)
        yield FinishReason(finish_reason)

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        media: MediaListType = None,
        conversation: JsonConversation = None,
        proxy: str = None,
        stream: bool = True,
        reasoning_effort: Optional[str] = "none",
        chat_type: Literal[
            "t2t",
            "search",
            "artifacts",
            "web_dev",
            "deep_research",
            "t2i",
            "image_edit",
            "t2v",
        ] = "t2t",
        aspect_ratio: Optional[Literal["1:1", "4:3", "3:4", "16:9", "9:16"]] = None,
        **kwargs,
    ) -> AsyncResult:
        """
        chat_type:
            DeepResearch = "deep_research"
            Artifacts = "artifacts"
            WebSearch = "search"
            ImageGeneration = "t2i"
            ImageEdit = "image_edit"
            VideoGeneration = "t2v"
            Txt2Txt = "t2t"
            WebDev = "web_dev"
        """
        model_name = cls.get_model(model)
        prompt = get_last_user_message(messages)
        enable_thinking = bool(reasoning_effort and reasoning_effort.lower() != "none")
        thinking_mode = kwargs.get("thinking_mode") or (
            "Auto" if enable_thinking else "Fast"
        )
        modes = {"auto": "Auto", "thinking": "Thinking", "fast": "Fast"}
        if not isinstance(thinking_mode, str) or thinking_mode.strip().lower() not in modes:
            raise ValueError("thinking_mode must be Auto, Thinking, or Fast")
        thinking_mode = modes[thinking_mode.strip().lower()]
        enable_thinking = thinking_mode != "Fast"
        auto_thinking = thinking_mode == "Auto"
        timeout = kwargs.get("timeout") or 5 * 60
        token = kwargs.get("token") or kwargs.get("api_key")
        if not token:
            await cls._ensure_auth(proxy)
        async with StreamSession(headers=cls._get_headers(token)) as session:
            if token:
                async with session.get(
                    "https://chat.qwen.ai/api/v1/auths/", proxy=proxy
                ) as user_info_res:
                    await cls.raise_for_status(user_info_res)
                    user_info = await user_info_res.json()
                    if user_info.get("success") is False or user_info.get("error"):
                        cls._raise_api_error(
                            user_info.get("error") or user_info.get("data") or user_info
                        )
                    debug.log("[Qwen] Authenticated session verified.")
            for attempt in range(5):
                try:
                    req_headers = await cls._get_req_headers(session, proxy=proxy)
                    message_id = str(uuid.uuid4())
                    now = int(time() * 1000)
                    chat_mode = "normal" if token else "guest"
                    if conversation is None:
                        chat_payload = {
                            "chatId": "",
                            "models": [model_name],
                            "project_id": "",
                            "chat_type": chat_type,
                            "chat_mode": chat_mode,
                            "timestamp": now,
                        }
                        async with session.post(
                            f"{cls.url}/api/v2/chats/new",
                            json=chat_payload,
                            headers=req_headers,
                            proxy=proxy,
                        ) as resp:
                            await cls.raise_for_status(resp)
                            data = await resp.json()
                            chat_data = data.get("data") or {}
                            if not (data.get("success") and chat_data.get("id")):
                                cls._raise_api_error(data.get("error") or chat_data or data)
                        conversation = JsonConversation(
                            chat_id=data["data"]["id"],
                            cookies={key: value for key, value in resp.cookies.items()},
                            parent_id=None,
                        )
                    files = []
                    media = list(merge_media(media, messages))
                    if media:
                        files = await cls.prepare_files(
                            media, session=session, headers=req_headers
                        )

                    feature_config = (
                        {
                            "auto_thinking": auto_thinking,
                            "thinking_mode": thinking_mode,
                            "thinking_format": "summary",
                            "thinking_enabled": enable_thinking,
                            "output_schema": "phase",
                            "research_mode": "normal",
                            "auto_search": True,
                        }
                        if enable_thinking
                        else {
                            "thinking_enabled": enable_thinking,
                            "output_schema": "phase",
                            "thinking_budget": 81920,
                        }
                    )
                    if "auto_search" in kwargs or chat_type == "search":
                        feature_config["auto_search"] = kwargs.get("auto_search", True)

                    msg_payload = {
                        "stream": stream,
                        "version": "2.1",
                        "incremental_output": stream,
                        "chatId": conversation.chat_id,
                        "parentId": conversation.parent_id,
                        "chat_id": conversation.chat_id,
                        "chat_mode": chat_mode,
                        "model": model_name,
                        "parent_id": conversation.parent_id,
                        "messages": [
                            {
                                "id": None,
                                "fid": message_id,
                                "parentId": conversation.parent_id,
                                "childrenIds": [message_id],
                                "role": "user",
                                "content": prompt,
                                "user_action": "chat",
                                "files": files,
                                "timestamp": now,
                                "models": [model_name],
                                "model": "",
                                "chat_type": chat_type,
                                "feature_config": feature_config,
                                "extra": {"meta": {"subChatType": chat_type}},
                                "sub_chat_type": chat_type,
                                "parent_id": conversation.parent_id,
                            }
                        ],
                        "timestamp": now,
                    }

                    if aspect_ratio:
                        msg_payload["size"] = aspect_ratio

                    async with session.post(
                        f"{cls.url}/api/v2/chat/completions?chat_id={conversation.chat_id}",
                        json=msg_payload,
                        headers=req_headers,
                        proxy=proxy,
                        timeout=timeout,
                        cookies=conversation.cookies,
                    ) as resp:
                        await cls.raise_for_status(resp)
                        async for chunk in cls._read_response(resp, conversation, prompt):
                            yield chunk
                        return

                except (aiohttp.ClientResponseError, RuntimeError) as e:
                    message = str(e)
                    is_rate_limit = (
                        isinstance(e, aiohttp.ClientResponseError) and e.status == 429
                    ) or ("RateLimited" in message)
                    if is_rate_limit:
                        # A quota/concurrency rejection is not a fingerprint error.
                        # Preserve the server's reason instead of replaying uploads/chats.
                        raise RateLimitError(message) from e
                    elif (
                        "FAIL_SYS_USER_VALIDATE" in message
                        or "RGV587_ERROR" in message
                        or "/punish" in message
                    ):
                        # Baxia captcha challenge — wait for the user to solve
                        # it (or log in) in a CDP browser, then capture and
                        # save the fresh auth data before retrying.
                        debug.log(
                            f"[Qwen] Captcha challenge detected (attempt {attempt + 1}/5) — waiting for login/captcha in a CDP browser"
                        )
                        cls._har_cookies = cls._har_headers = None
                        cls._har_loaded_at = 0.0
                        await cls._ensure_auth(proxy, force_cdp=True)
                        conversation = None
                        continue
                    else:
                        raise e
            raise CloudflareError(
                "Qwen browser validation failed after 5 recovery attempts."
            )
