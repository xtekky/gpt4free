from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
import uuid
from typing import AsyncIterator
from urllib.parse import quote

from aiohttp import BaseConnector, ClientError, ClientSession

try:
    import zendriver as nodriver

    has_nodriver = True
except ImportError:
    has_nodriver = False

from ... import debug
from ...typing import AsyncResult, Messages, Cookies, AsyncIterator
from ...providers.response import (
    JsonConversation,
    ProviderInfo,
    Reasoning,
    RequestLogin,
    Sources,
    TitleGeneration,
)
from ...requests.raise_for_status import raise_for_status
from ...requests.aiohttp import get_connector
from ...requests import get_nodriver
from ...errors import MissingAuthError, RateLimitError, ResponseError, ResponseStatusError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_cookies, get_last_user_message

REQUEST_HEADERS = {
    "accept": "*/*",
    "authority": "www.google.com",
    "origin": "https://www.google.com",
    "referer": "https://www.google.com/aimode",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
    "x-same-domain": "1",
}
RESPONSE_HEADER_LIMITS = {
    "max_line_size": 64 * 1024,
    "max_field_size": 64 * 1024,
}
REQUEST_PATH = "/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
GOOGLE_COOKIE_DOMAIN = ".google.com"
GOOGLE_SID_COOKIE = "__Secure-1PSID"
GOOGLE_SIDTS_COOKIE = "__Secure-1PSIDTS"
METADATA_CACHE_SECONDS = 60 * 60
MAX_PROMPT_CHARACTERS = 1_000_000
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
BUILD_LABEL_PATTERN = re.compile(r"boq_assistant-bard-web-server_[A-Za-z0-9_.-]+")
XSRF_PATTERN = re.compile(r'SNlM0e(?:\\?"|"):\\?"(.*?)(?:\\?"|")')
SID_PATTERN = re.compile(r'FdrFJe(?:\\?"|"):\\?"([\d-]+)(?:\\?"|")')

models = {
    "gemini-3.6-flash": {"mode": 1},
    "gemini-3.5-flash-lite": {"mode": 6},
    "gemini-3.1-pro": {"mode": 3},
}
MODEL_ALIASES = {
    "gemini-2.0": "gemini-3.6-flash",
    "gemini-2.0-flash": "gemini-3.6-flash",
    "gemini-2.5-flash": "gemini-3.6-flash",
    "gemini-2.5-pro": "gemini-3.1-pro",
    "gemini-3.5-flash": "gemini-3.6-flash",
    "gemini-auto": "gemini-3.6-flash",
    "gemini-flash-lite": "gemini-3.5-flash-lite",
    "google-aimode": "gemini-3.6-flash",
    "aimode": "gemini-3.6-flash",
    **{key: key for key in models.keys()},
}


def _make_sapisid_hash(cookies: Cookies) -> str | None:
    sapisid = cookies.get("SAPISID") or cookies.get("__Secure-1PAPISID")
    if not sapisid:
        return None
    timestamp = int(time.time())
    digest = hashlib.sha1(
        f"{timestamp} {sapisid} https://www.google.com".encode()
    ).hexdigest()
    return f"SAPISIDHASH {timestamp}_{digest}"


def _has_authenticated_session(cookies: Cookies) -> bool:
    return bool(cookies.get(GOOGLE_SID_COOKIE))


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


def _iter_wrb_payloads(value):
    if isinstance(value, list):
        if len(value) >= 3 and value[0] == "wrb.fr" and isinstance(value[2], str):
            yield value[2]
        for item in value:
            yield from _iter_wrb_payloads(item)


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


def _extract_sources(response_part: list) -> list[dict]:
    """Extract web search citation sources from an AI Mode response.

    AI Mode grounds answers in live web results.  The citation metadata is
    nested at ``response_part[4][0]`` in a structure that varies between
    build versions, so we walk it defensively.
    """
    sources: list[dict] = []
    try:
        citations = response_part[4][0]
    except (IndexError, TypeError):
        return sources
    if not isinstance(citations, list):
        return sources
    # Walk the citation list looking for {title, url} pairs.
    for entry in citations:
        if not isinstance(entry, list):
            continue
        url, title = _find_url_and_title(entry)
        if url:
            sources.append({"url": url, "title": title or url})
    return sources


def _find_url_and_title(entry, _depth: int = 0) -> tuple[str | None, str | None]:
    """Recursively search a nested list for the first URL and a title string."""
    if _depth > 10:
        return None, None
    url = None
    title = None
    if isinstance(entry, list):
        for item in entry:
            u, t = _find_url_and_title(item, _depth + 1)
            if url is None and u is not None:
                url = u
            if title is None and t is not None:
                title = t
            if url and title:
                break
    elif isinstance(entry, str):
        if entry.startswith("http"):
            return entry, None
        if len(entry) > 10 and " " in entry and not entry.startswith("["):
            return None, entry
    return url, title


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
                f"Google AI Mode stream was idle for {idle_timeout:g} seconds"
            ) from exc
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace")
    if buffer:
        yield buffer.decode("utf-8", errors="replace")


class GoogleAiMode(AsyncGeneratorProvider, ProviderModelMixin):
    """Provider for Google Search AI Mode (``https://www.google.com/aimode``).

    AI Mode is a Google Search feature powered by Gemini that answers
    complex, multi-part queries with AI-generated, search-grounded
    responses.  It requires a signed-in Google account and returns web
    citation sources alongside the answer.
    """

    label = "Google AI Mode"
    url = "https://www.google.com/aimode"

    needs_auth = True
    working = False
    active_by_default = False
    use_nodriver = True

    default_model = "gemini-3.6-flash"
    models = [*models, *MODEL_ALIASES]
    model_aliases = MODEL_ALIASES

    _cookies: Cookies = None
    _snlm0e: str = None
    _sid: str = None
    _bl: str = "boq_assistant-bard-web-server_20260525.09_p0"
    _metadata_fetched_at: float = 0
    _metadata_cookie_key: tuple[str | None, str | None] | None = None

    @classmethod
    async def login_generator(cls, proxy: str = None) -> AsyncIterator[str]:
        if not has_nodriver:
            debug.log("Skip nodriver login in GoogleAiMode provider")
            return
        browser, stop_browser = await get_nodriver(
            proxy=proxy, user_data_dir="google_aimode"
        )
        try:
            yield RequestLogin(cls.label, os.environ.get("G4F_LOGIN_URL", ""))
            page = await browser.get(cls.url)
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
    async def fetch_snlm0e(
        cls,
        session: ClientSession,
        cookies: Cookies,
    ) -> None:
        """Fetch the XSRF token and build label from the AI Mode page."""
        response_text = ""
        async with session.get(cls.url, cookies=cookies) as response:
            await raise_for_status(response)
            response_text = await response.text()
        match = XSRF_PATTERN.search(response_text)
        if match:
            cls._snlm0e = match.group(1)
        build_match = BUILD_LABEL_PATTERN.search(response_text)
        if build_match:
            cls._bl = build_match.group(0)
        sid_match = SID_PATTERN.search(response_text)
        if sid_match:
            cls._sid = sid_match.group(1)
        cls._metadata_fetched_at = time.time()

    @classmethod
    async def get_quota(cls, **kwargs):
        if not cls._cookies:
            cls._cookies = get_cookies(GOOGLE_COOKIE_DOMAIN, False, True)
        if not cls._cookies:
            raise MissingAuthError('Missing or invalid "__Secure-1PSID" cookie')
        async with ClientSession(
            headers=REQUEST_HEADERS,
            **RESPONSE_HEADER_LIMITS,
        ) as session:
            await cls.fetch_snlm0e(session, cls._cookies)
        return cls._snlm0e

    @classmethod
    def build_request(
        cls,
        prompt: str,
        language: str,
        model: str,
        conversation: Conversation = None,
        request_uuid: str = None,
    ) -> list:
        turn_index = (
            getattr(conversation, "turn_index", 0) if conversation is not None else 0
        )
        request = [None] * 97
        request[0] = [prompt, 0, None, [], None, None, 0]
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
        request[6] = [1]
        request[7] = 1
        request[10] = 1
        request[11] = 0
        request[17] = [[turn_index]]
        request[18] = 0
        request[27] = 1
        request[30] = [4]
        request[41] = [1]
        request[53] = 0
        request[59] = request_uuid or str(uuid.uuid4())
        request[61] = []
        request[68] = 2
        request[79] = models[model]["mode"]
        request[80] = 1
        request[91] = 0
        request[96] = int(conversation is None)
        return request

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str | None = None,
        cookies: Cookies | None = None,
        connector: BaseConnector | None = None,
        return_conversation: bool = True,
        conversation: Conversation = None,
        language: str = "en",
        prompt: str = None,
        **kwargs,
    ) -> AsyncResult:
        model = model or cls.default_model
        model = cls.get_model(model)
        if model not in models:
            raise ResponseError(
                f"Unknown model: {model}. Supported: {', '.join(models)}"
            )
        if cookies is not None:
            cls._cookies = cookies
        elif cls._cookies is None:
            cls._cookies = get_cookies(GOOGLE_COOKIE_DOMAIN, False, True)
        request_cookies = dict(cls._cookies or {})
        if not _has_authenticated_session(request_cookies):
            raise MissingAuthError(
                f"Google AI Mode requires authentication. "
                f"Please login with `g4f login {cls.parent if hasattr(cls, 'parent') else cls.__name__}` "
                f"or provide a valid '__Secure-1PSID' cookie."
            )
        if conversation is not None:
            conversation_model = getattr(conversation, "model", None)
            if MODEL_ALIASES.get(conversation_model, conversation_model) != model:
                conversation = None
        if prompt is None:
            prompt = (
                get_last_user_message(messages)
                if conversation is not None
                else format_prompt(messages)
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError(
                f"Prompt exceeds the {MAX_PROMPT_CHARACTERS}-character limit"
            )
        base_connector = get_connector(connector, proxy)

        async with ClientSession(
            headers=REQUEST_HEADERS,
            connector=base_connector,
            **RESPONSE_HEADER_LIMITS,
        ) as session:
            cookie_key = (
                request_cookies.get(GOOGLE_SID_COOKIE),
                request_cookies.get(GOOGLE_SIDTS_COOKIE),
            )
            if cookie_key != cls._metadata_cookie_key:
                cls._snlm0e = None
                cls._sid = None
                cls._metadata_fetched_at = 0
                cls._metadata_cookie_key = cookie_key
            metadata_expired = (
                time.time() - cls._metadata_fetched_at >= METADATA_CACHE_SECONDS
            )
            if not cls._metadata_fetched_at or metadata_expired:
                try:
                    await cls.fetch_snlm0e(session, request_cookies)
                except (ClientError, MissingAuthError, ResponseError) as error:
                    cls._metadata_fetched_at = time.time()
                    debug.log(f"GoogleAiMode metadata discovery failed: {error}")

            yield ProviderInfo(**cls.get_dict(), model=model)

            params = {
                "bl": cls._bl,
                "hl": language,
                "_reqid": random.randint(100_000, 999_999),
                "rt": "c",
            }
            if cls._sid:
                params["f.sid"] = cls._sid
            request_uuid = str(uuid.uuid4()).upper()
            data = {
                "f.req": json.dumps(
                    [
                        None,
                        json.dumps(
                            cls.build_request(
                                prompt,
                                model=model,
                                language=language,
                                conversation=conversation,
                                request_uuid=request_uuid,
                            )
                        ),
                    ]
                )
            }
            if cls._snlm0e:
                data["at"] = cls._snlm0e
            request_headers = {
                "Referer": f"{cls.url}",
                "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
            }
            authorization = _make_sapisid_hash(request_cookies)
            if authorization:
                request_headers["Authorization"] = authorization

            max_retries = max(0, int(kwargs.get("max_retries", 2)))
            stream_timeout = kwargs.get("stream_timeout", 120)
            response = None
            for attempt in range(max_retries + 1):
                try:
                    response = await session.post(
                        f"https://www.google.com{REQUEST_PATH}",
                        data=data,
                        params=params,
                        headers=request_headers,
                        cookies=request_cookies,
                    )
                    await raise_for_status(response)
                    break
                except (ClientError, RateLimitError, ResponseStatusError) as error:
                    status = response.status if response is not None else None
                    retryable = (
                        isinstance(error, ClientError)
                        or status in RETRYABLE_STATUS_CODES
                    )
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
                    delay = (
                        retry_after
                        if retry_after is not None
                        else min(2**attempt, 30)
                    )
                    await asyncio.sleep(
                        delay + random.uniform(0, 0.25 * max(delay, 0.01))
                    )
            if response is None:
                raise ResponseError("Google AI Mode request failed without a response")

            async with response:
                last_content = ""
                all_sources: dict[str, dict] = {}
                response_part = None
                async for line_text in _iter_response_lines(
                    response.content, stream_timeout
                ):
                    line_text = line_text.strip()
                    if not line_text:
                        continue
                    try:
                        line = json.loads(line_text)
                    except ValueError:
                        continue
                    if not isinstance(line, list):
                        continue
                    response_part = _extract_response_part(line)
                    if response_part is None:
                        continue
                    # Yield title if present
                    if (
                        len(response_part) > 2
                        and isinstance(response_part[2], dict)
                        and response_part[2].get("11")
                    ):
                        yield TitleGeneration(response_part[2].get("11"))
                    if len(response_part) < 5:
                        continue
                    # Yield conversation state for multi-turn support
                    if return_conversation and _has_authenticated_session(
                        request_cookies
                    ):
                        try:
                            yield Conversation(
                                response_part[1][0],
                                response_part[1][1],
                                response_part[4][0][0],
                                model,
                                turn_index=(
                                    getattr(conversation, "turn_index", 0) + 1
                                    if conversation is not None
                                    else 1
                                ),
                            )
                        except (IndexError, TypeError):
                            pass
                    content = _extract_response_content(response_part)
                    if content is None:
                        continue
                    # Extract and collect citation sources
                    try:
                        sources = _extract_sources(response_part)
                        for source in sources:
                            url = source.get("url")
                            if url and url not in all_sources:
                                all_sources[url] = source
                    except (IndexError, TypeError):
                        pass
                    # Clean up content
                    content = content.replace("<!-- end list -->", "")
                    content = re.sub(
                        r"http://googleusercontent.com/(?:image_generation|youtube|map)_content/\d+",
                        "",
                        content,
                    )
                    if last_content and content.startswith(last_content):
                        yield content[len(last_content):]
                    else:
                        yield content
                    last_content = content
                # Yield collected sources at the end
                if all_sources:
                    yield Sources(list(all_sources.values()))
                if not last_content:
                    raise ResponseError(
                        "Google AI Mode stream ended without a response"
                    )


class Conversation(JsonConversation):
    def __init__(
        self,
        conversation_id: str,
        response_id: str,
        choice_id: str,
        model: str,
        turn_index: int = 0,
    ) -> None:
        self.conversation_id = conversation_id
        self.response_id = response_id
        self.choice_id = choice_id
        self.model = model
        self.turn_index = turn_index
