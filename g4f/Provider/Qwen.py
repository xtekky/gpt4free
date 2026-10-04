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
from typing import Any, Dict, Literal, Mapping, Optional, Union
from urllib.parse import quote, urlparse
from http.cookies import SimpleCookie

from aiohttp import ClientSession

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message
from .openai.har_file import get_har_files
from .qwen.cookie_generator import generate_cookies
from .qwen.session import QwenAuth, cached_auth
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
    _har_cookie_records: Optional[list] = None
    _har_from_browser: bool = True
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
        "authorization",
    ]

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls._models_loaded and has_curl_cffi:
            _token = kwargs.get("token") or kwargs.get("api_key")
            auth = kwargs.get("auth_session")
            # The site's public models route does not trigger an auth refresh.
            if isinstance(auth, QwenAuth):
                headers = auth.request_headers(cls._get_headers(use_har=False), f"{cls.url}/api/models")
            else:
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
    async def prepare_files(cls, media, session: StreamSession, headers=None, auth=None, proxy=None) -> list:
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
            res_data = await cls._api_json(
                session, "post", f"{cls.url}/api/v2/files/getstsToken", auth,
                headers=headers, proxy=proxy, json={
                    "filename": file_name,
                    "filesize": file_size,
                    "filetype": file_type,
                },
            )
            data = res_data.get("data") or {}
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
        if (cls._har_cookies is not None or cls._har_headers is not None) and (cls._har_from_browser or now - cls._har_loaded_at < cls._HAR_TTL):
            return cls._har_cookies, cls._har_headers
        cls._har_cookies = cls._har_headers = None
        cls._har_cookie_records = None
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
            best_entry, credential_entry, best_score = None, None, -1
            for entry in reversed(har.get("log", {}).get("entries", [])):
                request = entry.get("request", {})
                url = request.get("url", "")
                target = urlparse(url)
                if target.scheme != "https" or target.hostname not in ("chat.qwen.ai", "auth.qwen.ai"):
                    continue
                # Fingerprint headers can be richest on a completion request,
                # while a later auth/API request holds the current credentials.
                if credential_entry is None:
                    cookie_header = SimpleCookie()
                    authorization = ""
                    for header in request.get("headers", []):
                        name, value = header.get("name", "").lower(), header.get("value", "")
                        if name == "cookie":
                            cookie_header.load(value)
                        elif name == "authorization":
                            authorization = value
                    if (authorization.lower().startswith("bearer ")
                            or "refresh_token" in cookie_header
                            or any(c.get("name") == "refresh_token" for c in request.get("cookies", []))):
                        credential_entry = entry
                score = 2 if "chat/completions" in url else (1 if "/api/v2/" in url else 0)
                if score > best_score:
                    best_entry, best_score = entry, score
            if best_entry is None:
                continue
            # Keep the token and cookies from the same request/account snapshot.
            auth_request = (credential_entry or best_entry)["request"]
            records = auth_request.get("cookies", [])
            cookies = {c["name"]: c["value"] for c in records if c.get("name") and isinstance(c.get("value"), str)}
            headers = {
                h["name"]: h.get("value", "")
                for h in best_entry["request"].get("headers", [])
                if h.get("name", "").lower() in cls._HAR_HEADERS_WHITELIST
                and h["name"].lower() != "authorization"
            }
            for header in auth_request.get("headers", []):
                name, value = header.get("name", ""), header.get("value", "")
                if name.lower() == "cookie":
                    parsed = SimpleCookie()
                    parsed.load(value)
                    cookies.update({k: v.value for k, v in parsed.items()})
                elif name.lower() == "authorization":
                    headers[name] = value
            if cookies or any(name.lower() == "authorization" for name in headers):
                debug.log(
                    f"[Qwen] Using {len(cookies)} cookies and {len(headers)} fingerprint headers from HAR file: {os.path.basename(path)}"
                )
                cls._har_cookies = cookies
                cls._har_headers = headers
                cls._har_cookie_records = [
                    c for c in records if c.get("name") and isinstance(c.get("value"), str)
                ] + [dict(name=k, value=v) for k, v in cookies.items() if k not in {c.get("name") for c in records}]
                return cookies, headers
            debug.log(f"[Qwen] No cookies found in HAR file: {os.path.basename(path)}")
        return None, None

    @classmethod
    async def _read_browser_fingerprint(cls, session, timeout: float = 30) -> dict:
        """Read initialized Baxia/Fireye SDK values without submitting a chat."""
        request_url = json.dumps(f"{cls.url}/api/v2/chat/completions")
        expression = """(async () => {
            const sdk = window.baxiaCommon;
            if (!window.baxiaInitialized || typeof sdk?.getUA !== 'function') return null;
            try {
                const ua = await sdk.getUA(REQUEST_URL);
                const valid = value => typeof value === 'string'
                    && value.length > 0 && !/^default/i.test(value);
                if (!valid(ua)) return null;
                const headers = {'bx-ua': ua};
                if (valid(sdk.version)) headers['bx-v'] = sdk.version;
                return headers;
            } catch { return null; }
        })()""".replace('REQUEST_URL', request_url)
        deadline = time() + timeout
        while time() < deadline:
            headers = await session.evaluate_js(expression)
            ua = headers.get('bx-ua') if isinstance(headers, dict) else None
            if isinstance(ua, str) and ua and not ua.lower().startswith('default'):
                return {k: v for k, v in headers.items()
                        if k in ('bx-ua', 'bx-v') and isinstance(v, str)}
            await asyncio.sleep(0.2)
        raise MissingAuthError("Qwen browser fingerprint SDK was not ready in time.")

    @classmethod
    async def _read_cdp(cls, proxy: str = None) -> tuple[dict, dict]:
        """Read the current account or guest browser session without creating a chat."""
        from ..requests.cdp import CDPSession

        async with CDPSession(proxy=proxy) as session:
            await session.navigate(cls.url)
            deadline = time() + 300
            while time() < deadline:
                # Both guests and signed-in users have a visible composer.
                # Account credentials are read from the refresh cookie below.
                if await session.evaluate_js(
                    """(() => {
                        const editor = document.querySelector('#chat-input, textarea');
                        return !!editor && editor.getClientRects().length > 0
                            && getComputedStyle(editor).visibility !== 'hidden';
                    })()"""
                ):
                    debug.log("[Qwen] Browser interface is ready for session capture.")
                    break
                await asyncio.sleep(2)
            else:
                raise MissingAuthError("Qwen browser session was not ready in time.")
            headers = await cls._read_browser_fingerprint(session)
            # SDK initialization can update cookies. Capture its completed state
            # from the same browser session without subscribing to network events.
            records = await session.get_cookies_list([cls.url + '/', 'https://auth.qwen.ai/'])
            cls._har_cookie_records = records
            cookies = {c['name']: c['value'] for c in records}
        return cookies, headers

    @classmethod
    async def _ensure_auth(cls, proxy: str = None, force_cdp: bool = False, persist: bool = False) -> None:
        """Populate the auth cache from a HAR file, falling back to a live
        browser capture via CDP. Generated cookies are used when both fail.
        Captured data stays in memory unless persistence is explicitly requested."""
        cookies, headers = (None, None) if force_cdp else cls._read_har()
        if cookies or headers:
            return
        if not has_cdp:
            if force_cdp:
                raise MissingAuthError("Qwen browser login requires an available CDP browser.")
            debug.log("[Qwen] No HAR file found and no CDP browser available — using generated cookies")
            return
        try:
            debug.log(
                "[Qwen] Capturing headers and cookies from a CDP browser"
                " — waiting for the account or guest interface to be ready"
            )
            cookies, headers = await cls._read_cdp(proxy)
        except MissingAuthError:
            raise
        except Exception as e:
            if force_cdp:
                raise MissingAuthError("The Qwen browser session could not be read.") from e
            debug.log(f"[Qwen] CDP capture failed: {type(e).__name__}: {e} — using generated cookies")
            return
        cls._har_cookies = cookies
        cls._har_headers = headers
        cls._har_loaded_at = time()
        cls._har_from_browser = True
        if persist:
            cls._save_auth_har(cookies, headers)

    @classmethod
    def _save_auth_har(cls, cookies: dict, headers: dict, cookie_records=None) -> None:
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
                                "cookies": cookie_records if cookie_records is not None else cls._har_cookie_records or [],
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
    def _get_headers(cls, token=None, use_har=True):
        har_cookies, har_headers = (None, None) if token or not use_har else cls._read_har()
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
    async def _auth_context(cls, proxy=None, **kwargs):
        """Resolve one credential source; never combine an explicit key with a HAR account."""
        auth = kwargs.get("auth_session")
        token = kwargs.get("token") or kwargs.get("api_key")
        refresh = kwargs.get("refresh_token")
        explicit = token or refresh is not None or kwargs.get("cookies") is not None
        # Automatic calls reuse HAR/in-memory credentials before opening CDP.
        # Explicit use_browser=True still requests a fresh browser capture.
        use_browser = kwargs.get("use_browser", False)
        persist = kwargs.get("persist_auth", auth is None and not explicit)
        if auth is not None:
            if not isinstance(auth, QwenAuth):
                raise TypeError("auth_session must be a QwenAuth instance")
            if explicit or use_browser:
                raise ValueError("Pass auth_session alone, without other login credentials")
            headers = cls._get_headers(use_har=False)
            headers.update(auth.fingerprint)
        else:
            if use_browser:
                if explicit:
                    raise ValueError("use_browser cannot be combined with explicit login credentials")
                await cls._ensure_auth(proxy, force_cdp=True, persist=persist)
            elif not explicit:
                await cls._ensure_auth(proxy, persist=persist)
            headers = cls._get_headers(token) if token or not explicit else cls._get_headers(use_har=False)
            if explicit:
                cookies = kwargs.get("cookies") or {}
            else:
                parsed = SimpleCookie()
                parsed.load(next((v for k, v in headers.items() if k.lower() == "cookie"), ""))
                cookies = cls._har_cookie_records if parsed and cls._har_cookie_records else {k: v.value for k, v in parsed.items()}
                bearer = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
                if bearer.lower().startswith("bearer "):
                    token = bearer[7:]
            supplied_headers = kwargs.get("headers") or {}
            headers.update({k: v for k, v in supplied_headers.items() if k.lower() in cls._HAR_HEADERS_WHITELIST and k.lower() != "authorization"})
            auth = cached_auth(token, refresh, cookies, headers, proxy)
        if persist:
            def save_updated(updated):
                records = updated.get_cookies()
                saved_headers = dict(updated.fingerprint)
                if updated.access_token:
                    saved_headers["Authorization"] = f"Bearer {updated.access_token}"
                cls._save_auth_har({c["name"]: c["value"] for c in records}, saved_headers, records)
            auth.on_update = save_updated
        else:
            auth.on_update = None
        # Per-request credentials keep Bearer tokens away from upload/anti-bot hosts.
        return auth, {k: v for k, v in headers.items() if k.lower() not in ("authorization", "cookie")}

    @classmethod
    async def _api_json(
        cls, session: Union[StreamSession, ClientSession],
        method: Literal["get", "post"], url: str,
        auth: Optional[QwenAuth] = None,
        headers: Optional[Mapping[str, str]] = None,
        proxy: Optional[str] = None, **kwargs: Any,
    ) -> Dict[str, Any]:
        """Read JSON, refreshing rejected credentials at most once."""
        kwargs.setdefault("allow_redirects", False)
        headers = dict(session.headers if headers is None else headers)
        for attempt in range(2):
            if auth:
                await auth.ensure_valid(session, proxy)
            token = auth.access_token if auth else None
            request_headers = auth.request_headers(headers, url) if auth else headers
            try:
                async with getattr(session, method)(url, headers=request_headers, proxy=proxy, **kwargs) as response:
                    if auth:
                        auth.update_cookies(response, url)
                    await cls.raise_for_status(response)
                    try:
                        payload = await response.json()
                    except ValueError as error:
                        raise ResponseError("Qwen returned invalid JSON") from error
                    if not isinstance(payload, dict):
                        raise ResponseError("Unexpected Qwen JSON response: expected an object")
                    cls._check_baxia_response(payload)
                    if payload.get("success") is False or payload.get("error"):
                        cls._raise_api_error(payload.get("error") or payload.get("data") or payload)
                    return payload
            except MissingAuthError:
                if attempt == 1 or not auth or not auth.can_refresh:
                    raise
                await auth.ensure_valid(session, proxy, rejected_token=token, recover=True)
        raise MissingAuthError("Qwen authentication failed after two attempts")

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
            req_headers.setdefault("bx-v", "2.5.37")
        else:
            debug.log("[Qwen] INFO: Using bx-umidtoken from the current session.")
        # fix error [g4f.errors.CloudflareError:aliyun_waf_aa]
        req_headers["x-request-id"] = str(uuid.uuid4())
        return req_headers

    @classmethod
    async def get_quota(cls, api_key: Optional[str] = None, **kwargs) -> dict:
        proxy = kwargs.pop("proxy", None)
        auth, headers = await cls._auth_context(proxy, api_key=api_key, **kwargs)
        async with StreamSession(headers=headers) as session:
            await auth.ensure_valid(session, proxy)
            chat_payload = {
                "chatId": "",
                "models": [cls.default_model],
                "project_id": "",
                "chat_mode": "normal" if auth.authenticated else "guest",
                "chat_type": "t2t",
                "timestamp": int(time() * 1000),
            }
            return await cls._api_json(
                session, "post", f"{cls.url}/api/v2/chats/new", auth,
                json=chat_payload,
                headers=await cls._get_req_headers(session, proxy=proxy), proxy=proxy,
            )

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
    def _check_baxia_response(cls, payload):
        codes = payload.get("ret")
        if isinstance(codes, list) and any(
            isinstance(code, str) and any(marker in code for marker in ("FAIL_SYS_USER_VALIDATE", "RGV587_ERROR"))
            for code in codes
        ):
            # Baxia uses HTTP 200 with a ret/data envelope. Its challenge URL
            # contains session data and must not be included in an exception.
            raise CloudflareError(
                "Qwen browser verification is required (FAIL_SYS_USER_VALIDATE / RGV587_ERROR); "
                "complete verification in the browser before submitting again."
            )

    @classmethod
    def _raise_api_error(cls, error):
        if isinstance(error, dict):
            code = str(error.get("code") or error.get("errorCode") or "QwenError")
            details = error.get("details") or error.get("message") or str(error)
            message = f"{code}: {details}"
        else:
            code = "QwenError"
            message = str(error)
        if code.lower() in ("unauthorized", "invalid token", "401"):
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
            cls._check_baxia_response(payload)
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

        Login options (keyword arguments):
            token / api_key: an access token; expires without a refresh credential.
            refresh_token: renew access tokens automatically with this account's cookie.
            cookies: cookie dict or browser/HAR cookie records, from the same account.
            auth_session: a reusable QwenAuth instance; do not mix credential sources.
            use_browser: True forces a fresh CDP capture; the default reuses a HAR
                or cached session and opens CDP only when no saved session exists.
                Browser capture never sends a seed chat.
            persist_auth: defaults to True for automatic credential lookup, saving
                browser captures and refreshed credentials for subsequent calls.
                Explicit credentials/auth_session require opting in to persistence.
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
        auth, base_headers = await cls._auth_context(proxy, **kwargs)
        media = list(merge_media(media, messages))
        for guest_attempt in range(2):
            emitted = False
            try:
                async with StreamSession(headers=base_headers) as session:
                    await auth.ensure_valid(session, proxy)
                    if auth.authenticated:
                        await cls._api_json(session, 'get', f'{cls.url}/api/v1/auths/', auth, proxy=proxy, timeout=30)
                        debug.log("[Qwen] Authenticated session verified.")
                    req_headers = await cls._get_req_headers(session, proxy=proxy)
                    message_id = str(uuid.uuid4())
                    now = int(time() * 1000)
                    chat_mode = "normal" if auth.authenticated else "guest"
                    if conversation is not None and not auth.authenticated:
                        for name, value in (getattr(conversation, "cookies", None) or {}).items():
                            if name != "refresh_token":
                                auth._set_cookie(dict(name=name, value=value, domain="chat.qwen.ai"))
                    if conversation is None:
                        data = await cls._api_json(
                            session, 'post', f'{cls.url}/api/v2/chats/new', auth,
                            headers=req_headers, proxy=proxy, json={
                                "chatId": "", "models": [model_name], "project_id": "",
                                "chat_type": chat_type, "chat_mode": chat_mode, "timestamp": now,
                            },
                        )
                        if not (data.get('success') and isinstance(data.get('data'), dict) and data['data'].get('id')):
                            cls._raise_api_error(data.get('error') or data.get('data') or data)
                        conversation = JsonConversation(
                            chat_id=data['data']['id'], parent_id=None,
                            cookies={} if auth.authenticated else {c['name']: c['value'] for c in auth.get_cookies()},
                        )
                    files = []
                    if media:
                        files = await cls.prepare_files(media, session=session, headers=req_headers, auth=auth, proxy=proxy)

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
                    _qwen_image_model: Literal["qwen-image-3.0-pro","qwen-image-2.0-pro", ""] = kwargs.get("qwen_image_model", "")
                    meta_message = {"meta": {"subChatType": chat_type}}
                    if _qwen_image_model:
                        meta_message["meta"]["model"] = _qwen_image_model
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
                                "extra": meta_message,
                                "sub_chat_type": chat_type,
                                "parent_id": conversation.parent_id,
                            }
                        ],
                        "timestamp": now,
                    }

                    if aspect_ratio:
                        msg_payload["size"] = aspect_ratio

                    url = f"{cls.url}/api/v2/chat/completions?chat_id={conversation.chat_id}"
                    for attempt in range(2):
                        await auth.ensure_valid(session, proxy)
                        snapshot = auth.access_token
                        emitted = False
                        try:
                            async with session.post(
                                url, json=msg_payload,
                                headers=auth.request_headers(req_headers, url),
                                proxy=proxy, timeout=timeout,
                                allow_redirects=False,
                            ) as resp:
                                await cls.raise_for_status(resp)
                                auth.update_cookies(resp, url)
                                if not auth.authenticated:
                                    conversation.cookies = {c['name']: c['value'] for c in auth.get_cookies()}
                                async for chunk in cls._read_response(resp, conversation, prompt):
                                    emitted = True
                                    yield chunk
                                return
                        except MissingAuthError:
                            # A known auth rejection before any output is safe to retry once.
                            if attempt or emitted or not auth.can_refresh:
                                raise
                            await auth.ensure_valid(session, proxy, rejected_token=snapshot, recover=True)
                        except RuntimeError as error:
                            if 'RateLimited' in str(error):
                                raise RateLimitError(str(error)) from error
                            if any(marker in str(error) for marker in ('FAIL_SYS_USER_VALIDATE', 'RGV587_ERROR', '/punish')):
                                raise CloudflareError('Qwen browser verification is required; complete it in the browser before submitting again.') from error
                            raise
            except RateLimitError as error:
                if (
                    guest_attempt or emitted or auth.authenticated
                    or "you've reached the guest chat limit" not in str(error).lower()
                ):
                    raise
                debug.log("[Qwen] Guest chat limit reached; retrying once with renewed request headers.")
                cls._midtoken = None
                conversation = None
            except ResponseError as error:
                if guest_attempt or emitted or auth.authenticated or "quota_limit" not in str(error).lower():
                    raise
                debug.error(f"[Qwen] {error}")
                debug.log("[Qwen] Retrying the guest request once.")
