from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import random
import re
import time
import uuid

from pathlib import Path
from urllib.parse import quote, quote_plus, unquote_plus

from aiohttp import BaseConnector, ClientError, ClientSession

try:
    import zendriver as nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ... import debug
from ...typing import Messages, Cookies, MediaListType, AsyncResult, AsyncIterator
from ...providers.response import (
    AudioResponse,
    ImageResponse,
    JsonConversation,
    JsonResponse,
    Reasoning,
    RequestLogin,
    TitleGeneration,
    YouTubeResponse,
)
from ...requests.raise_for_status import raise_for_status
from ...requests.aiohttp import get_connector
from ...requests import get_nodriver
from ...image.copy_images import get_filename, get_media_dir, ensure_media_dir
from ...errors import (
    MissingAuthError,
    RateLimitError,
    ResponseError,
    ResponseStatusError,
)
from ...image import to_bytes
from ...cookies import get_cookies_dir
from ...tools.media import merge_media
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import (
    format_media_prompt,
    format_prompt,
    get_cookies,
    get_last_user_message,
)
from .gemini_utils import (
    ACCOUNT_STATUS_AVAILABLE,
    ACCOUNT_STATUS_UNAUTHENTICATED,
    ANONYMOUS_MODELS,
    BARD_ERROR_PATTERN,
    MODEL_FAMILIES,
    MODEL_HEADER_KEY,
    build_model_headers as _build_model_headers,
    extract_gemini_error_code as _extract_gemini_error_code,
    extract_reasoning as _extract_reasoning,
    get_nested_value as _get_nested_value,
    iter_wrb_payloads as _iter_wrb_payloads,
    parse_account_models as _parse_account_models,
    parse_google_frames as _parse_google_frames,
    raise_gemini_error as _raise_gemini_error,
)

REQUEST_HEADERS = {
    "accept": "*/*",
    "authority": "gemini.google.com",
    "origin": "https://gemini.google.com",
    "referer": "https://gemini.google.com/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
    "x-same-domain": "1",
}
REQUEST_BL_PARAM = "boq_assistant-bard-web-server_20260525.09_p0"
REQUEST_PATH = "/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
UPLOAD_IMAGE_URL = "https://content-push.googleapis.com/upload/"
UPLOAD_IMAGE_HEADERS = {
    "authority": "content-push.googleapis.com",
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.7",
    "authorization": "Basic c2F2ZXM6cyNMdGhlNmxzd2F2b0RsN3J1d1U=",
    "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
    "origin": "https://gemini.google.com",
    "push-id": "feeds/mcudyrk2a4khkz",
    "referer": "https://gemini.google.com/",
    "x-goog-upload-command": "start",
    "x-goog-upload-header-content-length": "",
    "x-goog-upload-protocol": "resumable",
    "x-tenant-id": "bard-storage",
}
GOOGLE_COOKIE_DOMAIN = ".google.com"
ROTATE_COOKIES_URL = "https://accounts.google.com/RotateCookies"
GOOGLE_SID_COOKIE = "__Secure-1PSID"
GOOGLE_SIDTS_COOKIE = "__Secure-1PSIDTS"
BUILD_LABEL_PATTERN = re.compile(r"boq_assistant-bard-web-server_[A-Za-z0-9_.-]+")
XSRF_PATTERN = re.compile(r'SNlM0e(?:\\?"|"):\\?"(.*?)(?:\\?"|")')
SID_PATTERN = re.compile(r'FdrFJe(?:\\?"|"):\\?"([\d-]+)(?:\\?"|")')
PUSH_ID_PATTERN = re.compile(r'qKIAYe(?:\\?"|"):\\?"(.*?)(?:\\?"|")')
INTERNAL_CODE_PATTERN = re.compile(
    r"```(?:python|javascript|text)\?"
    r"code_(?:reference|stdout)&code_event_index=\d+\n"
    r".*?```\n?",
    flags=re.DOTALL,
)
METADATA_CACHE_SECONDS = 60 * 60
MODEL_CACHE_SECONDS = 15 * 60
MAX_PROMPT_CHARACTERS = 1_000_000
MAX_CONCURRENT_UPLOADS = 4
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

models = {
    "gemini-3.5-flash": {"mode": 1, "think": 4},
    "gemini-3.5-flash-thinking": {"mode": 2, "think": 0},
    "gemini-3.1-pro": {"mode": 3, "think": 4},
    "gemini-auto": {"mode": 4, "think": 4},
    "gemini-3.5-flash-thinking-lite": {"mode": 5, "think": 0},
    "gemini-flash-lite": {"mode": 6, "think": 4},
}
MODEL_ALIASES = {
    "gemini-2.0": "gemini-3.5-flash",
    "gemini-2.0-flash": "gemini-3.5-flash",
    "gemini-2.0-flash-thinking": "gemini-3.5-flash-thinking",
    "gemini-2.0-flash-thinking-with-apps": "gemini-3.5-flash-thinking",
    "gemini-2.5-flash": "gemini-3.5-flash",
    "gemini-2.5-pro": "gemini-3.1-pro",
    "gemini-3.1-flash-lite": "gemini-flash-lite",
}


def _account_prefix(auth_user: int | str | None) -> str:
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{quote(str(auth_user), safe='')}"


def _make_sapisid_hash(cookies: Cookies) -> str | None:
    sapisid = cookies.get("SAPISID") or cookies.get("__Secure-1PAPISID")
    if not sapisid:
        return None
    timestamp = int(time.time())
    digest = hashlib.sha1(
        f"{timestamp} {sapisid} https://gemini.google.com".encode()
    ).hexdigest()
    return f"SAPISIDHASH {timestamp}_{digest}"


def _extract_response_content(response_part: list) -> str | None:
    try:
        parts = response_part[4]
    except (IndexError, TypeError):
        return None
    if not isinstance(parts, list):
        return None
    snapshots = []
    for part in parts:
        if not isinstance(part, list) or len(part) <= 1:
            continue
        values = part[1]
        if isinstance(values, str):
            snapshots.append(values)
        elif isinstance(values, list):
            snapshots.extend(value for value in values if isinstance(value, str))
    return snapshots[-1] if snapshots else None


def _extract_response_part(value) -> list | None:
    response_parts = []
    for payload in _iter_wrb_payloads(value):
        try:
            response_part = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(response_part, list):
            response_parts.append(response_part)
    for response_part in reversed(response_parts):
        if _extract_response_content(response_part) is not None:
            return response_part
    return response_parts[-1] if response_parts else None


async def _iter_response_lines(
    content,
    idle_timeout: float | None = None,
) -> AsyncIterator[str]:
    buffer = b""
    iterator = content.iter_any().__aiter__()
    while True:
        try:
            next_chunk = iterator.__anext__()
            chunk = (
                await asyncio.wait_for(next_chunk, timeout=idle_timeout)
                if idle_timeout is not None
                else await next_chunk
            )
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError as exc:
            raise ResponseError(
                f"Gemini stream was idle for {idle_timeout:g} seconds"
            ) from exc
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace")
    if buffer:
        yield buffer.decode("utf-8", errors="replace")


def _resolve_model(model: str, think_override: int = None) -> tuple[str, int]:
    think_mode = think_override
    if "@think=" in model:
        model, think_value = model.rsplit("@think=", 1)
        try:
            think_mode = int(think_value)
        except ValueError as exc:
            raise ValueError(f"Invalid thinking mode: {think_value!r}") from exc
    if think_mode is not None:
        if not isinstance(think_mode, int) or not 0 <= think_mode <= 4:
            raise ValueError("Thinking mode must be an integer between 0 and 4")
    model = MODEL_ALIASES.get(model, model)
    if model not in models:
        raise ValueError(f"Unknown Gemini model: {model}")
    return model, models[model]["think"] if think_mode is None else think_mode


class Gemini(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Google Gemini"
    url = "https://gemini.google.com"
    
    needs_auth = False
    working = True
    active_by_default = True
    use_nodriver = True
    
    default_model = "gemini-3.5-flash"
    default_image_model = default_model
    default_vision_model = default_model
    image_models = [default_image_model]
    models = [*models, *MODEL_ALIASES]
    model_aliases = MODEL_ALIASES

    synthesize_content_type = "audio/vnd.wav"
    
    _cookies: Cookies = None
    _snlm0e: str = None
    _sid: str = None
    _bl: str = REQUEST_BL_PARAM
    _upload_push_id: str = UPLOAD_IMAGE_HEADERS["push-id"]
    _metadata_fetched_at: float = 0
    _metadata_cookie_key: tuple[str | None, str | None] | None = None
    _metadata_auth_user: int | str = None
    _account_status: int | None = None
    _account_models: dict[str, dict] = {}
    _account_models_fetched_at: float = 0

    auto_refresh = True
    refresh_interval = 60 * 15  # 15 minutes
    rotate_tasks = {}

    @classmethod
    async def login_generator(cls, proxy: str = None) -> AsyncIterator[str]:
        if not has_nodriver:
            debug.log("Skip nodriver login in Gemini provider")
            return
        browser, stop_browser = await get_nodriver(proxy=proxy, user_data_dir="gemini")
        try:
            yield RequestLogin(cls.label, os.environ.get("G4F_LOGIN_URL", ""))
            page = await browser.get(f"{cls.url}/app")
            await page.select("div.ql-editor.textarea", 240)
            cookies = {}
            for c in await page.send(nodriver.cdp.network.get_cookies([cls.url])):
                cookies[c.name] = c.value
            await page.close()
            cls._cookies = cookies
        finally:
            await stop_browser()

    @classmethod
    async def login(cls, proxy: str = None) -> AsyncIterator[str]:
        async for _ in cls.login_generator(proxy):
            pass
        return {"success": True, "message": "Login successful"}

    @classmethod
    async def start_auto_refresh(cls, proxy: str = None) -> None:
        """
        Start the background task to automatically refresh cookies.
        """

        while True:
            new_1psidts = None
            try:
                new_1psidts = await rotate_1psidts(cls.url, cls._cookies, proxy)
            except Exception as e:
                debug.error(f"Failed to refresh cookies: {e}")
                debug.error(
                    "Failed to refresh cookies. Background auto refresh task stopped."
                )
                return

            debug.log("Gemini: Cookies refreshed successfully")
            if new_1psidts:
                cls._cookies["__Secure-1PSIDTS"] = new_1psidts
                cls._metadata_fetched_at = 0
                cls._account_models_fetched_at = 0
            await asyncio.sleep(cls.refresh_interval)

    @classmethod
    async def fetch_account_models(
        cls,
        session: ClientSession,
        cookies: Cookies,
        auth_user: int | str = None,
    ) -> None:
        prefix = _account_prefix(auth_user)
        params = {
            "rpcids": "otAQ7b",
            "hl": "en",
            "_reqid": random.randint(100_000, 999_999),
            "rt": "c",
            "source-path": f"{prefix}/app" if prefix else "/app",
            "bl": cls._bl,
        }
        if cls._sid:
            params["f.sid"] = cls._sid
        headers = {
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
            "Referer": f"{cls.url}{prefix}/app",
            MODEL_HEADER_KEY: "[1,null,null,null,null,null,null,null,[4]]",
            "x-goog-ext-73010989-jspb": "[0]",
        }
        if prefix:
            headers["X-Goog-AuthUser"] = str(auth_user)
        authorization = _make_sapisid_hash(cookies)
        if authorization:
            headers["Authorization"] = authorization
        data = {
            "f.req": json.dumps([[["otAQ7b", "[]", None, "generic"]]])
        }
        if cls._snlm0e:
            data["at"] = cls._snlm0e
        async with session.post(
            f"{cls.url}{prefix}/_/BardChatUi/data/batchexecute",
            params=params,
            headers=headers,
            data=data,
            cookies=cookies,
        ) as response:
            await raise_for_status(response)
            status, registry = _parse_account_models(await response.text())
        cls._account_status = status
        cls._account_models = registry
        cls._account_models_fetched_at = time.time()

    @classmethod
    def get_model_headers(cls, model: str) -> dict[str, str]:
        family = MODEL_FAMILIES.get(model)
        # The 80-field request's mode category selects Flash/Thinking/Lite.
        # Pro additionally needs the account-specific model header or Google
        # silently routes it back to Flash.
        if family != "pro":
            return {}
        for model_data in cls._account_models.values():
            if model_data.get("family") == family and model_data.get("available"):
                return model_data.get("headers", {})
        return {}

    @classmethod
    def validate_model_access(
        cls,
        model: str,
        allow_model_fallback: bool = False,
    ) -> None:
        if allow_model_fallback or cls._account_status is None:
            return
        if cls._account_status == ACCOUNT_STATUS_UNAUTHENTICATED:
            if model not in ANONYMOUS_MODELS:
                raise MissingAuthError(
                    f"Gemini session is unauthenticated; model {model!r} would fall back to Flash"
                )
            return
        if cls._account_status != ACCOUNT_STATUS_AVAILABLE:
            raise ResponseError(
                f"Gemini account is unavailable (status {cls._account_status})"
            )
        family = MODEL_FAMILIES.get(model)
        if family != "pro" or not cls._account_models:
            return
        if not any(
            item.get("family") == family and item.get("available")
            for item in cls._account_models.values()
        ):
            raise ResponseError(f"Gemini model {model!r} is unavailable for this account")

    @classmethod
    async def get_quota(cls, **kwargs):
        if not cls._cookies:
            cls._cookies = get_cookies(GOOGLE_COOKIE_DOMAIN, False, True)
        if not cls._cookies:
            raise MissingAuthError('Missing or invalid "__Secure-1PSID" cookie')
        async with ClientSession(
            headers=REQUEST_HEADERS
        ) as session:
            await cls.fetch_snlm0e(session, cls._cookies)
        return cls._snlm0e

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: Cookies = None,
        connector: BaseConnector = None,
        media: MediaListType = None,
        return_conversation: bool = True,
        conversation: Conversation = None,
        language: str = "en",
        prompt: str = None,
        audio: dict = None,
        auth_user: int | str = None,
        think_override: int = None,
        **kwargs
    ) -> AsyncResult:
        model = model or cls.default_model
        if cls.model_aliases and model in cls.model_aliases:
            model = cls.model_aliases[model]
        if audio is not None or model == "gemini-audio":
            prompt = format_media_prompt(messages, prompt)
            filename = get_filename(["gemini"], prompt, ".ogx", prompt)
            ensure_media_dir()
            path = os.path.join(get_media_dir(), filename)
            with open(path, "wb") as f:
                async for chunk in cls.synthesize({"text": prompt}, proxy):
                    f.write(chunk)
            yield AudioResponse(f"/media/{filename}", text=prompt)
            return
        if think_override is None:
            think_override = {
                "none": 4,
                "minimal": 4,
                "low": 3,
                "medium": 2,
                "high": 1,
                "xhigh": 0,
            }.get(kwargs.get("reasoning_effort"))
        model, think_mode = _resolve_model(model, think_override)
        if cookies is not None:
            cls._cookies = cookies
        elif cls._cookies is None:
            cls._cookies = get_cookies(GOOGLE_COOKIE_DOMAIN, False, True)
        if conversation is not None and getattr(conversation, "model", None) != model:
            conversation = None
        prompt = format_prompt(messages) if conversation is None else get_last_user_message(messages)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError(
                f"Prompt exceeds the {MAX_PROMPT_CHARACTERS}-character limit"
            )
        base_connector = get_connector(connector, proxy)

        async with ClientSession(
            headers=REQUEST_HEADERS,
            connector=base_connector
        ) as session:
            cookie_key = (
                (cls._cookies or {}).get(GOOGLE_SID_COOKIE),
                (cls._cookies or {}).get(GOOGLE_SIDTS_COOKIE),
            )
            if (
                cookie_key != cls._metadata_cookie_key
                or auth_user != cls._metadata_auth_user
            ):
                cls._snlm0e = None
                cls._sid = None
                cls._metadata_fetched_at = 0
                cls._metadata_cookie_key = cookie_key
                cls._metadata_auth_user = auth_user
                cls._account_status = None
                cls._account_models = {}
                cls._account_models_fetched_at = 0
            metadata_expired = (
                time.time() - cls._metadata_fetched_at >= METADATA_CACHE_SECONDS
            )
            if not cls._metadata_fetched_at or metadata_expired:
                try:
                    await cls.fetch_snlm0e(session, cls._cookies or {}, auth_user)
                except (ClientError, MissingAuthError, ResponseError) as error:
                    cls._metadata_fetched_at = time.time()
                    debug.log(f"Gemini metadata discovery failed: {error}")
            models_expired = (
                time.time() - cls._account_models_fetched_at >= MODEL_CACHE_SECONDS
            )
            if not cls._account_models_fetched_at or models_expired:
                try:
                    await cls.fetch_account_models(
                        session, cls._cookies or {}, auth_user
                    )
                except (ClientError, ResponseError, ValueError) as error:
                    cls._account_models_fetched_at = time.time()
                    debug.log(f"Gemini model discovery failed: {error}")
            cls.validate_model_access(
                model,
                allow_model_fallback=bool(kwargs.get("allow_model_fallback", False)),
            )
            if cls.auto_refresh and cls._cookies and GOOGLE_SID_COOKIE in cls._cookies:
                task = cls.rotate_tasks.get(cls._cookies[GOOGLE_SID_COOKIE])
                if task is None or task.done():
                    cls.rotate_tasks[cls._cookies[GOOGLE_SID_COOKIE]] = asyncio.create_task(
                        cls.start_auto_refresh(proxy)
                    )

            uploads = await cls.upload_images(session, merge_media(media, messages))
            params = {
                'bl': cls._bl,
                'hl': language,
                '_reqid': random.randint(100_000, 999_999),
                'rt': 'c',
            }
            if cls._sid:
                params["f.sid"] = cls._sid
            request_uuid = str(uuid.uuid4()).upper()
            data = {
                'f.req': json.dumps([None, json.dumps(cls.build_request(
                    prompt,
                    model=model,
                    think_mode=think_mode,
                    language=language,
                    conversation=conversation,
                    uploads=uploads,
                    tools=kwargs.get("tools"),
                    request_uuid=request_uuid,
                ))])
            }
            if cls._snlm0e:
                data["at"] = cls._snlm0e
            request_headers = {}
            prefix = _account_prefix(auth_user)
            request_headers["Referer"] = f"{cls.url}{prefix}/app"
            if prefix:
                request_headers["X-Goog-AuthUser"] = str(auth_user)
            authorization = _make_sapisid_hash(cls._cookies or {})
            if authorization:
                request_headers["Authorization"] = authorization
            model_headers = cls.get_model_headers(model)
            if model_headers:
                request_headers.update(model_headers)
                request_headers["x-goog-ext-525005358-jspb"] = (
                    f'["{request_uuid}",1]'
                )
            max_retries = max(0, int(kwargs.get("max_retries", 2)))
            stream_timeout = kwargs.get("stream_timeout", 120)
            if stream_timeout is not None:
                try:
                    stream_timeout = float(stream_timeout)
                except (TypeError, ValueError) as exc:
                    raise ValueError("stream_timeout must be a positive number or None") from exc
                if stream_timeout <= 0:
                    raise ValueError("stream_timeout must be a positive number or None")
            response = None
            for attempt in range(max_retries + 1):
                try:
                    response = await session.post(
                        f"{cls.url}{prefix}{REQUEST_PATH}",
                        data=data,
                        params=params,
                        headers=request_headers or None,
                        cookies=cls._cookies,
                    )
                    await raise_for_status(response)
                    break
                except (ClientError, RateLimitError, ResponseStatusError) as error:
                    status = response.status if response is not None else None
                    retryable = isinstance(error, ClientError) or status in RETRYABLE_STATUS_CODES
                    if response is not None:
                        response.release()
                    if not retryable or attempt >= max_retries:
                        raise
                    retry_after = None
                    if response is not None:
                        try:
                            retry_after = float(response.headers.get("Retry-After", ""))
                        except ValueError:
                            pass
                    delay = retry_after if retry_after is not None else min(2 ** attempt, 30)
                    await asyncio.sleep(delay + random.uniform(0, 0.25 * max(delay, 0.01)))
            if response is None:
                raise ResponseError("Gemini request failed without a response")
            async with response:
                image_prompt = response_part = None
                last_content = ""
                last_reasoning = ""
                youtube_ids = []
                images_yielded = False
                error_buffer = ""
                routed_model = None
                async for line_text in _iter_response_lines(
                    response.content, stream_timeout
                ):
                    line_text = line_text.strip()
                    error_buffer = (error_buffer + line_text)[-4096:]
                    error_match = BARD_ERROR_PATTERN.search(error_buffer)
                    if error_match:
                        _raise_gemini_error(int(error_match.group(1)), model)
                    try:
                        try:
                            line = json.loads(line_text)
                        except ValueError:
                            continue
                        if not isinstance(line, list):
                            continue
                        error_code = _extract_gemini_error_code(line)
                        if error_code is not None:
                            _raise_gemini_error(error_code, model)
                        yield JsonResponse(data=line, model=model)
                        response_part = _extract_response_part(line)
                        if response_part is None:
                            continue
                        yield JsonResponse(data=response_part, model=model)
                        if len(response_part) > 2 and isinstance(response_part[2], dict) and response_part[2].get("11"):
                            yield TitleGeneration(response_part[2].get("11"))
                        if len(response_part) < 5:
                            continue
                        if return_conversation:
                            try:
                                yield Conversation(
                                    response_part[1][0],
                                    response_part[1][1],
                                    response_part[4][0][0],
                                    model,
                                )
                            except (IndexError, TypeError):
                                pass
                        def find_youtube_ids(content: str):
                            pattern = re.compile(r"https?://www.youtube.com/watch\?v=([\w-]+)")
                            for match in pattern.finditer(content):
                                if match.group(1) not in youtube_ids:
                                    yield match.group(1)
                        content = _extract_response_content(response_part)
                        if content is None:
                            continue
                        if routed_model is None:
                            current_routed_model = _get_nested_value(response_part, [42])
                            if isinstance(current_routed_model, str):
                                routed_model = current_routed_model
                                if model == "gemini-3.1-pro" and "flash" in routed_model.lower():
                                    debug.log(
                                        f"Gemini routed {model!r} to {routed_model!r}"
                                    )
                        reasoning = _extract_reasoning(response_part)
                        if reasoning:
                            reasoning = re.sub(r"<b>|</b>", "**", reasoning)
                            def replace_image(match):
                                return f"![](https:{match.group(0)})"
                            reasoning = re.sub(r"//yt3.(?:ggpht.com|googleusercontent.com/ytc)/[\w=-]+", replace_image, reasoning)
                            reasoning = re.sub(r"\nyoutube\n", "\n\n\n", reasoning)
                            reasoning = re.sub(r"\nyoutube_tool\n", "\n\n", reasoning)
                            reasoning = re.sub(r"\nYouTube\n", "\nYouTube ", reasoning)
                            reasoning = reasoning.replace('\nhttps://www.gstatic.com/images/branding/productlogos/youtube/v9/192px.svg', '<i class="fa-brands fa-youtube"></i>')
                            reasoning_youtube_ids = list(find_youtube_ids(reasoning))
                            youtube_ids.extend(reasoning_youtube_ids)
                            if reasoning.startswith(last_reasoning):
                                reasoning_delta = reasoning[len(last_reasoning):]
                            elif last_reasoning.startswith(reasoning):
                                reasoning_delta = ""
                            else:
                                reasoning_delta = reasoning
                            if reasoning_delta:
                                yield Reasoning(reasoning_delta, status="🤔")
                            last_reasoning = reasoning
                            if reasoning_youtube_ids:
                                yield YouTubeResponse(reasoning_youtube_ids)
                    except (ValueError, KeyError, TypeError, IndexError) as e:
                        if kwargs.get("debug_mode", False):
                            raise e
                        debug.error(f"{cls.__name__} {type(e).__name__}: {e}")
                        continue
                    match = re.search(r'\[Imagen of (.*?)\]', content)
                    if match:
                        image_prompt = match.group(1)
                        content = content.replace(match.group(0), '')
                    pattern = r"http://googleusercontent.com/(?:image_generation|youtube|map)_content/\d+"
                    content = re.sub(pattern, "", content)
                    content = content.replace("<!-- end list -->", "")
                    content = content.replace("<ctrl94>thought", "<think>").replace("<ctrl95>", "</think>")
                    content = INTERNAL_CODE_PATTERN.sub("", content)
                    def replace_link(match):
                        return f"(https://{quote_plus(unquote_plus(match.group(1)), '/?&=#')})"
                    content = re.sub(r"\(https://www.google.com/(?:search\?q=|url\?sa=E&source=gmail&q=)https?://(.+?)\)", replace_link, content)

                    if last_content and content.startswith(last_content):
                        yield content[len(last_content):]
                    else:
                        yield content
                    last_content = content
                    has_images = False
                    try:
                        if not images_yielded and len(response_part[4][0]) >= 13 and response_part[4][0][12] and len(response_part[4][0][12]) >= 8 and response_part[4][0][12][7] and response_part[4][0][12][7][0]:
                            has_images = True
                    except (TypeError, IndexError, KeyError):
                        pass

                    if not images_yielded and (image_prompt or has_images):
                        try:
                            images = []
                            for image in response_part[4][0][12][7][0]:
                                img_data = image[0][3][3]
                                if isinstance(img_data, list):
                                    for item in img_data:
                                        if isinstance(item, str) and item.startswith("http"):
                                            images.append(item + "=s2048")
                                            break
                                elif isinstance(img_data, str):
                                    images.append(img_data + "=s2048")
                            if images:
                                prompt = image_prompt.replace("a fake image", "") if image_prompt else "Generated Image"
                                yield ImageResponse(images, prompt, {"cookies": cls._cookies})
                                image_prompt = None
                                images_yielded = True
                        except (TypeError, IndexError, KeyError):
                            pass
                    new_youtube_ids = list(find_youtube_ids(content))
                    if new_youtube_ids:
                        youtube_ids.extend(new_youtube_ids)
                        yield YouTubeResponse(new_youtube_ids)
                if not last_content and not images_yielded:
                    raise ResponseError("Gemini stream ended without a response")

    @classmethod
    async def synthesize(cls, params: dict, proxy: str = None) -> AsyncIterator[bytes]:
        if "text" not in params:
            raise ValueError("Missing parameter text")
        async with ClientSession(
            cookies=cls._cookies,
            headers=REQUEST_HEADERS,
            connector=get_connector(proxy=proxy),
        ) as session:
            if not cls._snlm0e:
                await cls.fetch_snlm0e(session, cls._cookies) if cls._cookies else None
            inner_data = json.dumps([None, params["text"], "en-US", None, 2])
            async with session.post(
                "https://gemini.google.com/_/BardChatUi/data/batchexecute",
                data={
                      "f.req": json.dumps([[["XqA3Ic", inner_data, None, "generic"]]]),
                      "at": cls._snlm0e,
                },
                params={
                    "rpcids": "XqA3Ic",
                    "source-path": "/app/2704fb4aafcca926",
                    "bl": "boq_assistant-bard-web-server_20241119.00_p1",
                    "f.sid": "" if cls._sid is None else cls._sid,
                    "hl": "de",
                    "_reqid": random.randint(1111, 9999),
                    "rt": "c"
                },
            ) as response:
                await raise_for_status(response)
                iter_base64_response = iter_filter_base64(response.content.iter_chunked(1024))
                async for chunk in iter_base64_decode(iter_base64_response):
                    yield chunk

    def build_request(
        prompt: str,
        language: str,
        model: str,
        think_mode: int,
        conversation: Conversation = None,
        uploads: list[list[str, str]] = None,
        tools: list[list[str]] = None,
        request_uuid: str = None,
    ) -> list:
        image_list = [[[image_url, 1], image_name] for image_url, image_name in uploads] if uploads else []
        request = [None] * 80
        request[0] = [prompt, 0, None, image_list, None, None, 0]
        request[1] = [language]
        request[2] = [
            "" if conversation is None else conversation.conversation_id,
            "" if conversation is None else conversation.response_id,
            "" if conversation is None else conversation.choice_id,
            None,
            None,
            None,
            None,
            None,
            None,
            "",
        ]
        request[6] = [0]
        request[7] = 1
        request[9] = tools or []
        request[10] = 1
        request[11] = 0
        request[17] = [[think_mode]]
        request[18] = 0
        request[27] = 1
        request[30] = [4]
        request[41] = [2]
        request[53] = 0
        request[59] = request_uuid or str(uuid.uuid4())
        request[61] = []
        request[68] = 1
        request[79] = models[model]["mode"]
        return request

    @classmethod
    async def upload_images(cls, session: ClientSession, media: MediaListType) -> list:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

        async def upload_image(image: bytes, image_name: str = None):
            async with semaphore:
                image = to_bytes(image)
                upload_headers = {
                    **UPLOAD_IMAGE_HEADERS,
                    "push-id": cls._upload_push_id,
                    "x-goog-upload-header-content-length": str(len(image)),
                }

                async with session.options(
                    UPLOAD_IMAGE_URL,
                    headers=upload_headers,
                    cookies=cls._cookies,
                ) as response:
                    await raise_for_status(response)

                headers = {
                    **upload_headers,
                    "size": str(len(image)),
                    "x-goog-upload-command": "start",
                }
                data = f"File name: {image_name}" if image_name else None
                async with session.post(
                    UPLOAD_IMAGE_URL,
                    headers=headers,
                    data=data,
                    cookies=cls._cookies,
                ) as response:
                    await raise_for_status(response)
                    upload_url = response.headers.get("X-Goog-Upload-Url")
                    if not upload_url:
                        raise ResponseError("Gemini upload did not return an upload URL")

                async with session.options(
                    upload_url,
                    headers=headers,
                    cookies=cls._cookies,
                ) as response:
                    await raise_for_status(response)

                headers["x-goog-upload-command"] = "upload, finalize"
                headers["X-Goog-Upload-Offset"] = "0"
                async with session.post(
                    upload_url,
                    headers=headers,
                    data=image,
                    cookies=cls._cookies,
                ) as response:
                    await raise_for_status(response)
                    identifier = (await response.text()).strip()
                    if not identifier:
                        raise ResponseError("Gemini upload returned an empty identifier")
                    return [identifier, image_name]
        return await asyncio.gather(*[
            upload_image(image, image_name)
            for image, image_name in media
        ])

    @classmethod
    async def fetch_snlm0e(
        cls,
        session: ClientSession,
        cookies: Cookies,
        auth_user: int | str = None,
    ):
        response_text = ""
        prefix = _account_prefix(auth_user)
        async with session.get(f"{cls.url}{prefix}/app", cookies=cookies) as response:
            await raise_for_status(response)
            response_text = await response.text()
        match = XSRF_PATTERN.search(response_text)
        if match:
            cls._snlm0e = match.group(1)
        build_match = BUILD_LABEL_PATTERN.search(response_text)
        if build_match:
            cls._bl = build_match.group(0)
        push_id_match = PUSH_ID_PATTERN.search(response_text)
        if push_id_match:
            cls._upload_push_id = push_id_match.group(1)
        sid_match = SID_PATTERN.search(response_text)
        if sid_match:
            cls._sid = sid_match.group(1)
            cls.active_by_default = True
        elif cookies:
            cls.active_by_default = False
        cls._metadata_fetched_at = time.time()


class Conversation(JsonConversation):
    def __init__(self,
        conversation_id: str,
        response_id: str,
        choice_id: str,
        model: str
    ) -> None:
        self.conversation_id = conversation_id
        self.response_id = response_id
        self.choice_id = choice_id
        self.model = model


async def iter_filter_base64(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    search_for = b'[["wrb.fr","XqA3Ic","[\\"'
    end_with = b'\\'
    buffer = b""
    is_started = False
    async for chunk in chunks:
        buffer += chunk
        if not is_started:
            marker_index = buffer.find(search_for)
            if marker_index < 0:
                buffer = buffer[-(len(search_for) - 1):]
                continue
            is_started = True
            buffer = buffer[marker_index + len(search_for):]
        end_index = buffer.find(end_with)
        if end_index >= 0:
            if end_index:
                yield buffer[:end_index]
            return
        if buffer:
            yield buffer
            buffer = b""
    if not is_started:
        raise ResponseError("Gemini audio response did not contain audio data")


async def iter_base64_decode(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    buffer = b""
    async for chunk in chunks:
        chunk = buffer + chunk
        rest = len(chunk) % 4
        if rest:
            buffer = chunk[-rest:]
            chunk = chunk[:-rest]
        else:
            buffer = b""
        if chunk:
            yield base64.b64decode(chunk)
    if buffer:
        yield base64.b64decode(buffer + (-len(buffer) % 4) * b"=")


async def rotate_1psidts(url, cookies: dict, proxy: str | None = None) -> str:
    path = Path(get_cookies_dir())
    path.mkdir(parents=True, exist_ok=True)
    filename = "auth_Gemini.json"
    path = path / filename

    # Check if the cache file was modified in the last minute to avoid 429 Too Many Requests
    if not (path.is_file() and time.time() - os.path.getmtime(path) <= 60):
        async with ClientSession(proxy=proxy) as client:
            async with client.post(
                url=ROTATE_COOKIES_URL,
                headers={
                    "Content-Type": "application/json",
                },
                cookies=cookies,
                data='[000,"-0000000000000000000"]',
            ) as response:
                if response.status == 401:
                    raise MissingAuthError("Invalid cookies")
                response.raise_for_status()
                for key, cookie in response.cookies.items():
                    cookies[key] = cookie.value
                new_1psidts = response.cookies.get("__Secure-1PSIDTS")
                path.write_text(json.dumps([{
                    "name": key,
                    "value": value,
                    "domain": GOOGLE_COOKIE_DOMAIN,
                } for key, value in cookies.items()]))
                if new_1psidts:
                    return new_1psidts.value
