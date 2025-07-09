from __future__ import annotations

import os
import json
import random
import re
import base64
import asyncio
import time

from urllib.parse import quote_plus, unquote_plus
from pathlib import Path
from aiohttp import ClientSession, BaseConnector

try:
    import nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ... import debug
from ...typing import Messages, Cookies, MediaListType, AsyncResult, AsyncIterator
from ...providers.response import JsonConversation, Reasoning, RequestLogin, ImageResponse, YouTube, AudioResponse, TitleGeneration
from ...requests.raise_for_status import raise_for_status
from ...requests.aiohttp import get_connector
from ...requests import get_nodriver
from ...image.copy_images import get_filename, get_media_dir, ensure_media_dir
from ...errors import MissingAuthError, ModelNotFoundError
from ...image import to_bytes
from ...cookies import get_cookies_dir
from ...tools.media import merge_media
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_cookies, get_last_user_message, format_media_prompt
from ... import debug

REQUEST_HEADERS = {
    "authority": "gemini.google.com",
    "origin": "https://gemini.google.com",
    "referer": "https://gemini.google.com/",
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'x-same-domain': '1',
}
REQUEST_BL_PARAM = "boq_assistant-bard-web-server_20240519.16_p0"
REQUEST_URL = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
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

models = {
    "gemini-2.5-pro-exp": {"x-goog-ext-525001261-jspb": '[1,null,null,null,"2525e3954d185b3c"]'},
    "gemini-2.5-flash": {"x-goog-ext-525001261-jspb": '[1,null,null,null,"35609594dbe934d8"]'},
    "gemini-2.0-flash-thinking-exp": {"x-goog-ext-525001261-jspb": '[1,null,null,null,"7ca48d02d802f20a"]'},
    "gemini-deep-research": {"x-goog-ext-525001261-jspb": '[1,null,null,null,"cd472a54d2abba7e"]'},
    "gemini-2.0-flash": {"x-goog-ext-525001261-jspb": '[null,null,null,null,"f299729663a2343f"]'},
    "gemini-2.0-flash-exp": {"x-goog-ext-525001261-jspb": '[null,null,null,null,"f299729663a2343f"]'},
    "gemini-2.0-flash-thinking": {"x-goog-ext-525001261-jspb": '[null,null,null,null,"9c17b1863f581b8a"]'},
    "gemini-2.0-flash-thinking-with-apps": {"x-goog-ext-525001261-jspb": '[null,null,null,null,"f8f8f5ea629f5d37"]'},
    "gemini-audio": {}
}

class Gemini(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Google Gemini"
    url = "https://gemini.google.com"
    
    needs_auth = True
    working = True
    use_nodriver = True
    
    default_model = ""
    default_image_model = default_model
    default_vision_model = default_model
    image_models = [default_image_model]
    models = [
        default_model, *models.keys()
    ]
    model_aliases = {
        "gemini-2.0": "",
        "gemini-2.0-flash": ["gemini-2.0-flash", "gemini-2.0-flash", "gemini-2.0-flash-exp"],
        "gemini-2.5-pro": "gemini-2.5-pro-exp",
    }

    synthesize_content_type = "audio/vnd.wav"
    
    _cookies: Cookies = None
    _snlm0e: str = None
    _sid: str = None

    auto_refresh = True
    refresh_interval = 540
    rotate_tasks = {}

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the internal model name from the user-provided model name."""
        if not model:
            return cls.default_model
        
        # Check if the model exists directly in our models list
        if model in cls.models:
            return model
        
        # Check if there's an alias for this model
        if model in cls.model_aliases:
            alias = cls.model_aliases[model]
            # If the alias is a list, randomly select one of the options
            if isinstance(alias, list):
                selected_model = random.choice(alias)
                debug.log(f"Gemini: Selected model '{selected_model}' from alias '{model}'")
                return selected_model
            debug.log(f"Gemini: Using model '{alias}' for alias '{model}'")
            return alias
        
        raise ModelNotFoundError(f"Model {model} not found")

    @classmethod
    async def nodriver_login(cls, proxy: str = None) -> AsyncIterator[str]:
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
            stop_browser()

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
                task = cls.rotate_tasks.get(cls._cookies[GOOGLE_SID_COOKIE])
                if task:
                    task.cancel()
                debug.error(
                    "Failed to refresh cookies. Background auto refresh task canceled."
                )

            debug.log(f"Gemini: Cookies refreshed. New __Secure-1PSIDTS: {new_1psidts}")
            if new_1psidts:
                cls._cookies["__Secure-1PSIDTS"] = new_1psidts
            await asyncio.sleep(cls.refresh_interval)

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
        **kwargs
    ) -> AsyncResult:
        if model in cls.model_aliases:
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
        cls._cookies = cookies or cls._cookies or get_cookies(GOOGLE_COOKIE_DOMAIN, False, True)
        if conversation is not None and getattr(conversation, "model", None) != model:
            conversation = None
        prompt = format_prompt(messages) if conversation is None else get_last_user_message(messages)
        base_connector = get_connector(connector, proxy)

        async with ClientSession(
            headers=REQUEST_HEADERS,
            connector=base_connector
        ) as session:
            if not cls._snlm0e:
                await cls.fetch_snlm0e(session, cls._cookies) if cls._cookies else None
            if not cls._snlm0e:
                try:
                    async for chunk in cls.nodriver_login(proxy):
                        yield chunk
                except Exception as e:
                    raise MissingAuthError('Missing or invalid "__Secure-1PSID" cookie', e)
            if not cls._snlm0e:
                if cls._cookies is None or "__Secure-1PSID" not in cls._cookies:
                    raise MissingAuthError('Missing "__Secure-1PSID" cookie')
                await cls.fetch_snlm0e(session, cls._cookies)
            if not cls._snlm0e:
                raise RuntimeError("Invalid cookies. SNlM0e not found")
            if GOOGLE_SID_COOKIE in cls._cookies:
                task = cls.rotate_tasks.get(cls._cookies[GOOGLE_SID_COOKIE])
                if not task:
                    cls.rotate_tasks[cls._cookies[GOOGLE_SID_COOKIE]] = asyncio.create_task(
                        cls.start_auto_refresh()
                    )

            uploads = await cls.upload_images(base_connector, merge_media(media, messages))
            async with ClientSession(
                cookies=cls._cookies,
                headers=REQUEST_HEADERS,
                connector=base_connector,
            ) as client:
                params = {
                    'bl': REQUEST_BL_PARAM,
                    'hl': language,
                    '_reqid': random.randint(1111, 9999),
                    'rt': 'c',
                    "f.sid": cls._sid,
                }
                data = {
                    'at': cls._snlm0e,
                    'f.req': json.dumps([None, json.dumps(cls.build_request(
                        prompt,
                        language=language,
                        conversation=conversation,
                        uploads=uploads
                    ))])
                }
                async with client.post(
                    REQUEST_URL,
                    data=data,
                    params=params,
                    headers=models[model] if model in models else None
                ) as response:
                    await raise_for_status(response)
                    image_prompt = response_part = None
                    last_content = ""
                    youtube_ids = []
                    for line in (await response.text()).split("\n"):
                        try:
                            try:
                                line = json.loads(line)
                            except ValueError:
                                continue
                            if not isinstance(line, list):
                                continue
                            if len(line[0]) < 3 or not line[0][2]:
                                continue
                            response_part = json.loads(line[0][2])
                            if response_part[10]:
                                yield TitleGeneration(response_part[10][0].strip())
                            if not response_part[4]:
                                continue
                            if return_conversation:
                                yield Conversation(response_part[1][0], response_part[1][1], response_part[4][0][0], model)
                            def find_youtube_ids(content: str):
                                pattern = re.compile(r"http://www.youtube.com/watch\?v=([\w-]+)")
                                for match in pattern.finditer(content):
                                    if match.group(1) not in youtube_ids:
                                        yield match.group(1)
                            def read_recusive(data):
                                for item in data:
                                    if isinstance(item, list):
                                        yield from read_recusive(item)
                                    elif isinstance(item, str) and not item.startswith("rc_"):
                                        yield item
                            def find_str(data, skip=0):
                                for item in read_recusive(data):
                                    if skip > 0:
                                        skip -= 1
                                        continue
                                    yield item
                            reasoning = "\n\n".join(find_str(response_part[4][0], 3))
                            reasoning = re.sub(r"<b>|</b>", "**", reasoning)
                            def replace_image(match):
                                return f"![](https:{match.group(0)})"
                            reasoning = re.sub(r"//yt3.(?:ggpht.com|googleusercontent.com/ytc)/[\w=-]+", replace_image, reasoning)
                            reasoning = re.sub(r"\nyoutube\n", "\n\n\n", reasoning)
                            reasoning = re.sub(r"\nyoutube_tool\n", "\n\n", reasoning)
                            reasoning = re.sub(r"\nYouTube\n", "\nYouTube ", reasoning)
                            reasoning = reasoning.replace('\nhttps://www.gstatic.com/images/branding/productlogos/youtube/v9/192px.svg', '<i class="fa-brands fa-youtube"></i>')
                            youtube_ids = list(find_youtube_ids(reasoning))
                            content = response_part[4][0][1][0]
                            if reasoning:
                                yield Reasoning(reasoning, status="🤔")
                        except (ValueError, KeyError, TypeError, IndexError) as e:
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
                        def replace_link(match):
                            return f"(https://{quote_plus(unquote_plus(match.group(1)), '/?&=#')})"
                        content = re.sub(r"\(https://www.google.com/(?:search\?q=|url\?sa=E&source=gmail&q=)https?://(.+?)\)", replace_link, content)

                        if last_content and content.startswith(last_content):
                            yield content[len(last_content):]
                        else:
                            yield content
                        last_content = content
                        if image_prompt:
                            try:
                                images = [image[0][3][3] for image in response_part[4][0][12][7][0]]
                                image_prompt = image_prompt.replace("a fake image", "")
                                yield ImageResponse(images, image_prompt, {"cookies": cls._cookies})
                            except (TypeError, IndexError, KeyError):
                                pass
                        youtube_ids = youtube_ids if youtube_ids else find_youtube_ids(content)
                        if youtube_ids:
                            yield YouTube(youtube_ids)

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
        conversation: Conversation = None,
        uploads: list[list[str, str]] = None,
        tools: list[list[str]] = []
    ) -> list:
        image_list = [[[image_url, 1], image_name] for image_url, image_name in uploads] if uploads else []
        return [
            [prompt, 0, None, image_list, None, None, 0],
            [language],
            [
                None if conversation is None else conversation.conversation_id,
                None if conversation is None else conversation.response_id,
                None if conversation is None else conversation.choice_id,
                None,
                None,
                []
            ],
            None,
            None,
            None,
            [1],
            0,
            [],
            tools,
            1,
            0,
        ]

    async def upload_images(connector: BaseConnector, media: MediaListType) -> list:
        async def upload_image(image: bytes, image_name: str = None):
            async with ClientSession(
                headers=UPLOAD_IMAGE_HEADERS,
                connector=connector
            ) as session:
                image = to_bytes(image)

                async with session.options(UPLOAD_IMAGE_URL) as response:
                    await raise_for_status(response)

                headers = {
                    "size": str(len(image)),
                    "x-goog-upload-command": "start"
                }
                data = f"File name: {image_name}" if image_name else None
                async with session.post(
                    UPLOAD_IMAGE_URL, headers=headers, data=data
                ) as response:
                    await raise_for_status(response)
                    upload_url = response.headers["X-Goog-Upload-Url"]

                async with session.options(upload_url, headers=headers) as response:
                    await raise_for_status(response)

                headers["x-goog-upload-command"] = "upload, finalize"
                headers["X-Goog-Upload-Offset"] = "0"
                async with session.post(
                    upload_url, headers=headers, data=image
                ) as response:
                    await raise_for_status(response)
                    return [await response.text(), image_name]
        return await asyncio.gather(*[upload_image(image, image_name) for image, image_name in media])

    @classmethod
    async def fetch_snlm0e(cls, session: ClientSession, cookies: Cookies):
        async with session.get(cls.url, cookies=cookies) as response:
            await raise_for_status(response)
            response_text = await response.text()
        match = re.search(r'SNlM0e\":\"(.*?)\"', response_text)
        if match:
            cls._snlm0e = match.group(1)
        sid_match = re.search(r'"FdrFJe":"([\d-]+)"', response_text)
        if sid_match:
            cls._sid = sid_match.group(1)

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
    is_started = False
    async for chunk in chunks:
        if is_started:
            if end_with in chunk:
                yield chunk.split(end_with, maxsplit=1).pop(0)
                break
            else:
                yield chunk
        elif search_for in chunk:
            is_started = True
            yield chunk.split(search_for, maxsplit=1).pop()
        else:
            raise ValueError(f"Response: {chunk}")

async def iter_base64_decode(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    buffer = b""
    rest = 0
    async for chunk in chunks:
        chunk = buffer + chunk
        rest = len(chunk) % 4
        buffer = chunk[-rest:]
        yield base64.b64decode(chunk[:-rest])
    if rest > 0:
        yield base64.b64decode(buffer+rest*b"=")

async def rotate_1psidts(url, cookies: dict, proxy: str | None = None) -> str:
    path = Path(get_cookies_dir())
    path.mkdir(parents=True, exist_ok=True)
    filename = f"auth_Gemini.json"
    path = path / filename

    # Check if the cache file was modified in the last minute to avoid 429 Too Many Requests
    if not (path.is_file() and time.time() - os.path.getmtime(path) <= 60):
        async with ClientSession(proxy=proxy) as client:
            response = await client.post(
                url=ROTATE_COOKIES_URL,
                headers={
                    "Content-Type": "application/json",
                },
                cookies=cookies,
                data='[000,"-0000000000000000000"]',
            )
            if response.status == 401:
                raise MissingAuthError("Invalid cookies")
            response.raise_for_status()
            for key, c in response.cookies.items():
                cookies[key] = c.value
            new_1psidts = response.cookies.get("__Secure-1PSIDTS")
            path.write_text(json.dumps([{
                "name": k,
                "value": v,
                "domain": GOOGLE_COOKIE_DOMAIN,
            } for k, v in cookies.items()]))
            if new_1psidts:
                return new_1psidts
