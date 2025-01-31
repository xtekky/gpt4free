from __future__ import annotations

import os
import json
import random
import re
import base64
import asyncio

from aiohttp import ClientSession, BaseConnector

try:
    import nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ... import debug
from ...typing import Messages, Cookies, ImagesType, AsyncResult, AsyncIterator
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_cookies
from ...providers.response import JsonConversation, SynthesizeData, RequestLogin
from ...requests.raise_for_status import raise_for_status
from ...requests.aiohttp import get_connector
from ...requests import get_nodriver
from ...errors import MissingAuthError
from ...image import ImageResponse, to_bytes
from ..helper import get_last_user_message
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

class Gemini(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Google Gemini"
    url = "https://gemini.google.com"
    
    needs_auth = True
    working = True
    use_nodriver = True
    
    default_model = 'gemini'
    default_image_model = default_model
    default_vision_model = default_model
    image_models = [default_image_model]
    models = [default_model, "gemini-1.5-flash", "gemini-1.5-pro"]

    synthesize_content_type = "audio/vnd.wav"
    
    _cookies: Cookies = None
    _snlm0e: str = None
    _sid: str = None

    @classmethod
    async def nodriver_login(cls, proxy: str = None) -> AsyncIterator[str]:
        if not has_nodriver:
            if debug.logging:
                print("Skip nodriver login in Gemini provider")
            return
        browser, stop_browser = await get_nodriver(proxy=proxy, user_data_dir="gemini")
        try:
            login_url = os.environ.get("G4F_LOGIN_URL")
            if login_url:
                yield RequestLogin(cls.label, login_url)
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
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: Cookies = None,
        connector: BaseConnector = None,
        images: ImagesType = None,
        return_conversation: bool = False,
        conversation: Conversation = None,
        language: str = "en",
        **kwargs
    ) -> AsyncResult:
        prompt = format_prompt(messages) if conversation is None else get_last_user_message(messages)
        cls._cookies = cookies or cls._cookies or get_cookies(".google.com", False, True)
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

            yield SynthesizeData(cls.__name__, {"text": messages[-1]["content"]})
            images = await cls.upload_images(base_connector, images) if images else None
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
                        images=images
                    ))])
                }
                async with client.post(
                    REQUEST_URL,
                    data=data,
                    params=params,
                ) as response:
                    await raise_for_status(response)
                    image_prompt = response_part = None
                    last_content = ""
                    async for line in response.content:
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
                            if not response_part[4]:
                                continue
                            if return_conversation:
                                yield Conversation(response_part[1][0], response_part[1][1], response_part[4][0][0])
                            content = response_part[4][0][1][0]
                        except (ValueError, KeyError, TypeError, IndexError) as e:
                            debug.log(f"{cls.__name__}:{e.__class__.__name__}:{e}")
                            continue
                        match = re.search(r'\[Imagen of (.*?)\]', content)
                        if match:
                            image_prompt = match.group(1)
                            content = content.replace(match.group(0), '')
                        pattern = r"http://googleusercontent.com/image_generation_content/\d+"
                        content = re.sub(pattern, "", content)
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
        images: list[list[str, str]] = None,
        tools: list[list[str]] = []
    ) -> list:
        image_list = [[[image_url, 1], image_name] for image_url, image_name in images] if images else []
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

    async def upload_images(connector: BaseConnector, images: ImagesType) -> list:
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
        return await asyncio.gather(*[upload_image(image, image_name) for image, image_name in images])

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
        choice_id: str
    ) -> None:
        self.conversation_id = conversation_id
        self.response_id = response_id
        self.choice_id = choice_id

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