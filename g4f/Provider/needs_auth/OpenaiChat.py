from __future__ import annotations

import os
import re
import asyncio
import uuid
import json
import base64
import time
import random
from typing import AsyncIterator, Iterator, Optional, Generator, Dict, List
from copy import copy

try:
    import nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ..base_provider import AsyncAuthedProvider, ProviderModelMixin
from ...typing import AsyncResult, Messages, Cookies, ImagesType
from ...requests.raise_for_status import raise_for_status
from ...requests import StreamSession
from ...requests import get_nodriver
from ...image import ImageResponse, ImageRequest, to_image, to_bytes, is_accepted_format
from ...errors import MissingAuthError, NoValidHarFileError
from ...providers.response import JsonConversation, FinishReason, SynthesizeData, AuthResult
from ...providers.response import Sources, TitleGeneration, RequestLogin, Parameters
from ..helper import format_cookies
from ..openai.har_file import get_request_config
from ..openai.har_file import RequestConfig, arkReq, arkose_url, start_url, conversation_url, backend_url, backend_anon_url
from ..openai.proofofwork import generate_proof_token
from ..openai.new import get_requirements_token, get_config
from ... import debug

DEFAULT_HEADERS = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br, zstd",
    'accept-language': 'en-US,en;q=0.8',
    "referer": "https://chatgpt.com/",
    "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "sec-gpc": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

INIT_HEADERS = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.8',
    'cache-control': 'no-cache',
    'pragma': 'no-cache',
    'priority': 'u=0, i',
    "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
    'sec-ch-ua-arch': '"arm"',
    'sec-ch-ua-bitness': '"64"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '""',
    "sec-ch-ua-platform": "\"Windows\"",
    'sec-ch-ua-platform-version': '"14.4.0"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

UPLOAD_HEADERS = {
    "accept": "application/json, text/plain, */*",
    'accept-language': 'en-US,en;q=0.8',
    "referer": "https://chatgpt.com/",
    "priority": "u=1, i",
    "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?0",
    'sec-ch-ua-platform': '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
    "x-ms-blob-type": "BlockBlob",
    "x-ms-version": "2020-04-08",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

class OpenaiChat(AsyncAuthedProvider, ProviderModelMixin):
    """A class for creating and managing conversations with OpenAI chat service"""

    label = "OpenAI ChatGPT"
    url = "https://chatgpt.com"
    working = True
    use_nodriver = True
    supports_gpt_4 = True
    supports_message_history = True
    supports_system_message = True
    default_model = "auto"
    default_image_model = "dall-e-3"
    image_models = [default_image_model]
    text_models = [default_model, "gpt-4", "gpt-4o", "gpt-4o-mini", "o1", "o1-preview", "o1-mini"]
    vision_models = text_models
    models = text_models + image_models
    synthesize_content_type = "audio/mpeg"

    _api_key: str = None
    _headers: dict = None
    _cookies: Cookies = None
    _expires: int = None

    @classmethod
    async def on_auth_async(cls, **kwargs) -> AsyncIterator:
        async for chunk in cls.login():
            yield chunk
        yield AuthResult(
            api_key=cls._api_key,
            cookies=cls._cookies or RequestConfig.cookies or {},
            headers=cls._headers or RequestConfig.headers or cls.get_default_headers(),
            expires=cls._expires,
            proof_token=RequestConfig.proof_token,
            turnstile_token=RequestConfig.turnstile_token
        )

    @classmethod
    async def upload_images(
        cls,
        session: StreamSession,
        auth_result: AuthResult,
        images: ImagesType,
    ) -> ImageRequest:
        """
        Upload an image to the service and get the download URL
        
        Args:
            session: The StreamSession object to use for requests
            headers: The headers to include in the requests
            images: The images to upload, either a PIL Image object or a bytes object
        
        Returns:
            An ImageRequest object that contains the download URL, file name, and other data
        """
        async def upload_image(image, image_name):
            # Convert the image to a PIL Image object and get the extension
            data_bytes = to_bytes(image)
            image = to_image(data_bytes)
            extension = image.format.lower()
            data = {
                "file_name": "" if image_name is None else image_name,
                "file_size": len(data_bytes),
                "use_case":	"multimodal"
            }
            # Post the image data to the service and get the image data
            headers = auth_result.headers if hasattr(auth_result, "headers") else None
            async with session.post(f"{cls.url}/backend-api/files", json=data, headers=headers) as response:
                cls._update_request_args(auth_result, session)
                await raise_for_status(response, "Create file failed")
                image_data = {
                    **data,
                    **await response.json(),
                    "mime_type": is_accepted_format(data_bytes),
                    "extension": extension,
                    "height": image.height,
                    "width": image.width
                }
            # Put the image bytes to the upload URL and check the status
            await asyncio.sleep(1)
            async with session.put(
                image_data["upload_url"],
                data=data_bytes,
                headers={
                    **UPLOAD_HEADERS,
                    "Content-Type": image_data["mime_type"],
                    "x-ms-blob-type": "BlockBlob",
                    "x-ms-version": "2020-04-08",
                    "Origin": "https://chatgpt.com",
                }
            ) as response:
                await raise_for_status(response)
            # Post the file ID to the service and get the download URL
            async with session.post(
                f"{cls.url}/backend-api/files/{image_data['file_id']}/uploaded",
                json={},
                headers=auth_result.headers
            ) as response:
                cls._update_request_args(auth_result, session)
                await raise_for_status(response, "Get download url failed")
                image_data["download_url"] = (await response.json())["download_url"]
            return ImageRequest(image_data)
        if not images:
            return
        return [await upload_image(image, image_name) for image, image_name in images]

    @classmethod
    def create_messages(cls, messages: Messages, image_requests: ImageRequest = None, system_hints: list = None):
        """
        Create a list of messages for the user input
        
        Args:
            prompt: The user input as a string
            image_response: The image response object, if any
        
        Returns:
            A list of messages with the user input and the image, if any
        """
        # Create a message object with the user role and the content
        messages = [{
            "id": str(uuid.uuid4()),
            "author": {"role": message["role"]},
            "content": {"content_type": "text", "parts": [message["content"]]},
            "metadata": {"serialization_metadata": {"custom_symbol_offsets": []}, **({"system_hints": system_hints} if system_hints else {})},
            "create_time": time.time(),
        } for message in messages]
        # Check if there is an image response
        if image_requests:
            # Change content in last user message
            messages[-1]["content"] = {
                "content_type": "multimodal_text",
                "parts": [*[{
                    "asset_pointer": f"file-service://{image_request.get('file_id')}",
                    "height": image_request.get("height"),
                    "size_bytes": image_request.get("file_size"),
                    "width": image_request.get("width"),
                }
                for image_request in image_requests],
                messages[-1]["content"]["parts"][0]]
            }
            # Add the metadata object with the attachments
            messages[-1]["metadata"] = {
                "attachments": [{
                    "height": image_request.get("height"),
                    "id": image_request.get("file_id"),
                    "mimeType": image_request.get("mime_type"),
                    "name": image_request.get("file_name"),
                    "size": image_request.get("file_size"),
                    "width": image_request.get("width"),
                }
                for image_request in image_requests]
            }
        return messages

    @classmethod
    async def get_generated_image(cls, session: StreamSession, auth_result: AuthResult, element: dict, prompt: str = None) -> ImageResponse:
        try:
            prompt = element["metadata"]["dalle"]["prompt"]
            file_id = element["asset_pointer"].split("file-service://", 1)[1]
        except TypeError:
            return
        except Exception as e:
            raise RuntimeError(f"No Image: {e.__class__.__name__}: {e}")
        try:
            async with session.get(f"{cls.url}/backend-api/files/{file_id}/download", headers=auth_result.headers) as response:
                cls._update_request_args(auth_result, session)
                await raise_for_status(response)
                download_url = (await response.json())["download_url"]
                return ImageResponse(download_url, prompt)
        except Exception as e:
            raise RuntimeError(f"Error in downloading image: {e}")

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        proxy: str = None,
        timeout: int = 180,
        auto_continue: bool = False,
        history_disabled: bool = False,
        action: str = "next",
        conversation_id: str = None,
        conversation: Conversation = None,
        images: ImagesType = None,
        return_conversation: bool = False,
        max_retries: int = 0,
        web_search: bool = False,
        **kwargs
    ) -> AsyncResult:
        """
        Create an asynchronous generator for the conversation.

        Args:
            model (str): The model name.
            messages (Messages): The list of previous messages.
            proxy (str): Proxy to use for requests.
            timeout (int): Timeout for requests.
            api_key (str): Access token for authentication.
            auto_continue (bool): Flag to automatically continue the conversation.
            history_disabled (bool): Flag to disable history and training.
            action (str): Type of action ('next', 'continue', 'variant').
            conversation_id (str): ID of the conversation.
            images (ImagesType): Images to include in the conversation.
            return_conversation (bool): Flag to include response fields in the output.
            **kwargs: Additional keyword arguments.

        Yields:
            AsyncResult: Asynchronous results from the generator.

        Raises:
            RuntimeError: If an error occurs during processing.
        """
        async with StreamSession(
            proxy=proxy,
            impersonate="chrome",
            timeout=timeout
        ) as session:
            image_requests = None
            if not cls.needs_auth:
                if cls._headers is None:
                    cls._create_request_args(cls._cookies)
                    async with session.get(cls.url, headers=INIT_HEADERS) as response:
                        cls._update_request_args(auth_result, session)
                        await raise_for_status(response)
            else:
                if cls._headers is None and getattr(auth_result, "cookies", None):
                    cls._create_request_args(auth_result.cookies, auth_result.headers)
                if not cls._set_api_key(getattr(auth_result, "api_key", None)):
                    raise MissingAuthError("Access token is not valid")
                async with session.get(cls.url, headers=cls._headers) as response:
                    cls._update_request_args(auth_result, session)
                    await raise_for_status(response)
                try:
                    image_requests = await cls.upload_images(session, auth_result, images) if images else None
                except Exception as e:
                    debug.log("OpenaiChat: Upload image failed")
                    debug.log(f"{e.__class__.__name__}: {e}")
            model = cls.get_model(model)
            if conversation is None:
                conversation = Conversation(conversation_id, str(uuid.uuid4()), getattr(auth_result, "cookies", {}).get("oai-did"))
            else:
                conversation = copy(conversation)
            if getattr(auth_result, "cookies", {}).get("oai-did") != getattr(conversation, "user_id", None):
                conversation = Conversation(None, str(uuid.uuid4()))
            if cls._api_key is None:
                auto_continue = False
            conversation.finish_reason = None
            sources = Sources([])
            while conversation.finish_reason is None:
                async with session.post(
                    f"{cls.url}/backend-anon/sentinel/chat-requirements"
                    if cls._api_key is None else
                    f"{cls.url}/backend-api/sentinel/chat-requirements",
                    json={"p": None if not getattr(auth_result, "proof_token", None) else get_requirements_token(getattr(auth_result, "proof_token", None))},
                    headers=cls._headers
                ) as response:
                    if response.status in (401, 403):
                        auth_result.reset()
                    else:
                        cls._update_request_args(auth_result, session)
                    await raise_for_status(response)
                    chat_requirements = await response.json()
                    need_turnstile = chat_requirements.get("turnstile", {}).get("required", False)
                    need_arkose = chat_requirements.get("arkose", {}).get("required", False) 
                    chat_token = chat_requirements.get("token")  

                # if need_arkose and RequestConfig.arkose_token is None:
                #     await get_request_config(proxy)
                #     cls._create_request_args(auth_result.cookies, auth_result.headers)
                #     cls._set_api_key(auth_result.access_token)
                #     if auth_result.arkose_token is None:
                #         raise MissingAuthError("No arkose token found in .har file")
                if "proofofwork" in chat_requirements:
                    if getattr(auth_result, "proof_token", None) is None:
                        auth_result.proof_token = get_config(auth_result.headers.get("user-agent"))
                    proofofwork = generate_proof_token(
                        **chat_requirements["proofofwork"],
                        user_agent=getattr(auth_result, "headers", {}).get("user-agent"),
                        proof_token=getattr(auth_result, "proof_token", None)
                    )
                [debug.log(text) for text in (
                    #f"Arkose: {'False' if not need_arkose else auth_result.arkose_token[:12]+'...'}",
                    f"Proofofwork: {'False' if proofofwork is None else proofofwork[:12]+'...'}",
                    f"AccessToken: {'False' if cls._api_key is None else cls._api_key[:12]+'...'}",
                )]
                data = {
                    "action": action,
                    "parent_message_id": conversation.message_id,
                    "model": model,
                    "timezone_offset_min":-60,
                    "timezone":"Europe/Berlin",
                    "suggestions":[],
                    "history_and_training_disabled": history_disabled and not auto_continue and not return_conversation or not cls.needs_auth,
                    "conversation_mode":{"kind":"primary_assistant","plugin_ids":None},
                    "force_paragen":False,
                    "force_paragen_model_slug":"",
                    "force_rate_limit":False,
                    "reset_rate_limits":False,
                    "websocket_request_id": str(uuid.uuid4()),
                    "system_hints": ["search"] if web_search else None,
                    "supported_encodings":["v1"],
                    "conversation_origin":None,
                    "client_contextual_info":{"is_dark_mode":False,"time_since_loaded":random.randint(20, 500),"page_height":578,"page_width":1850,"pixel_ratio":1,"screen_height":1080,"screen_width":1920},
                    "paragen_stream_type_override":None,
                    "paragen_cot_summary_display_override":"allow",
                    "supports_buffering":True
                }
                if conversation.conversation_id is not None:
                    data["conversation_id"] = conversation.conversation_id
                    debug.log(f"OpenaiChat: Use conversation: {conversation.conversation_id}")
                if action != "continue":
                    data["parent_message_id"] = getattr(conversation, "parent_message_id", conversation.message_id)
                    conversation.parent_message_id = None
                    messages = messages if conversation_id is None else [messages[-1]]
                    data["messages"] = cls.create_messages(messages, image_requests, ["search"] if web_search else None)
                headers = {
                    **cls._headers,
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                    "openai-sentinel-chat-requirements-token": chat_token,
                }
                #if RequestConfig.arkose_token:
                #    headers["openai-sentinel-arkose-token"] = RequestConfig.arkose_token
                if proofofwork is not None:
                    headers["openai-sentinel-proof-token"] = proofofwork
                if need_turnstile and getattr(auth_result, "turnstile_token", None) is not None:
                    headers['openai-sentinel-turnstile-token'] = auth_result.turnstile_token
                async with session.post(
                    f"{cls.url}/backend-anon/conversation"
                    if cls._api_key is None else
                    f"{cls.url}/backend-api/conversation",
                    json=data,
                    headers=headers
                ) as response:
                    cls._update_request_args(auth_result, session)
                    if response.status == 403:
                        auth_result.proof_token = None
                        RequestConfig.proof_token = None
                    await raise_for_status(response)
                    buffer = u""
                    async for line in response.iter_lines():
                        async for chunk in cls.iter_messages_line(session, auth_result, line, conversation, sources):
                            if isinstance(chunk, str):
                                chunk = chunk.replace("\ue203", "").replace("\ue204", "").replace("\ue206", "")
                                buffer += chunk
                                if buffer.find(u"\ue200") != -1:
                                    if buffer.find(u"\ue201") != -1:
                                        buffer = buffer.replace("\ue200", "").replace("\ue202", "\n").replace("\ue201", "")
                                        buffer = buffer.replace("navlist\n", "#### ")
                                        def replacer(match):
                                            link = None
                                            if len(sources.list) > int(match.group(1)):
                                                link = sources.list[int(match.group(1))]["url"]
                                                return f"[[{int(match.group(1))+1}]]({link})"
                                            return f" [{int(match.group(1))+1}]"
                                        buffer = re.sub(r'(?:cite\nturn0search|cite\nturn0news|turn0news)(\d+)', replacer, buffer)
                                    else:
                                        continue
                                yield buffer
                                buffer = ""
                            else:
                                yield chunk
                        if conversation.finish_reason is not None:
                            break
                if sources.list:
                    yield sources
                if return_conversation:
                    yield conversation
                if not history_disabled and auth_result.api_key is not None:
                    yield SynthesizeData(cls.__name__, {
                        "conversation_id": conversation.conversation_id,
                        "message_id": conversation.message_id,
                        "voice": "maple",
                    })
                if auto_continue and conversation.finish_reason == "max_tokens":
                    conversation.finish_reason = None
                    action = "continue"
                    await asyncio.sleep(5)
                else:
                    break
                yield Parameters(**{
                    "action": "continue" if conversation.finish_reason == "max_tokens" else "variant",
                    "conversation": conversation.get_dict(),
                    "proof_token": RequestConfig.proof_token,
                    "cookies": cls._cookies,
                    "headers": cls._headers,
                    "web_search": web_search,
                })
            yield FinishReason(conversation.finish_reason)

    @classmethod
    async def iter_messages_line(cls, session: StreamSession, auth_result: AuthResult, line: bytes, fields: Conversation, sources: Sources) -> AsyncIterator:
        if not line.startswith(b"data: "):
            return
        elif line.startswith(b"data: [DONE]"):
            if fields.finish_reason is None:
                fields.finish_reason = "error"
            return
        try:
            line = json.loads(line[6:])
        except:
            return
        if not isinstance(line, dict):
            return
        if "type" in line:
            if line["type"] == "title_generation":
                yield TitleGeneration(line["title"])
        if "v" in line:
            v = line.get("v")
            if isinstance(v, str) and fields.is_recipient:
                if "p" not in line or line.get("p") == "/message/content/parts/0":
                    yield v
            elif isinstance(v, list):
                for m in v:
                    if m.get("p") == "/message/content/parts/0" and fields.is_recipient:
                        yield m.get("v")
                    elif m.get("p") == "/message/metadata/search_result_groups":
                        for entry in [p.get("entries") for p in m.get("v")]:
                            for link in entry:
                                sources.add_source(link)
                    elif re.match(r"^/message/metadata/content_references/\d+$", m.get("p")):
                        sources.add_source(m.get("v"))
                    elif m.get("p") == "/message/metadata":
                        fields.finish_reason = m.get("v", {}).get("finish_details", {}).get("type")
                        break
            elif isinstance(v, dict):
                if fields.conversation_id is None:
                    fields.conversation_id = v.get("conversation_id")
                    debug.log(f"OpenaiChat: New conversation: {fields.conversation_id}")
                m = v.get("message", {})
                fields.is_recipient = m.get("recipient", "all") == "all"
                if fields.is_recipient:
                    c = m.get("content", {})
                    if c.get("content_type") == "multimodal_text":
                        generated_images = []
                        for element in c.get("parts"):
                            if isinstance(element, dict) and element.get("content_type") == "image_asset_pointer":
                                image = cls.get_generated_image(session, auth_result, element)
                                generated_images.append(image)
                        for image_response in await asyncio.gather(*generated_images):
                            if image_response is not None:
                                yield image_response
                    if m.get("author", {}).get("role") == "assistant":
                        if fields.parent_message_id is None:
                            fields.parent_message_id = v.get("message", {}).get("id")
                        fields.message_id = v.get("message", {}).get("id")
            return
        if "error" in line and line.get("error"):
            raise RuntimeError(line.get("error"))

    @classmethod
    async def synthesize(cls, params: dict) -> AsyncIterator[bytes]:
        async for _ in cls.login():
            pass
        async with StreamSession(
            impersonate="chrome",
            timeout=0
        ) as session:
            async with session.get(
                f"{cls.url}/backend-api/synthesize",
                params=params,
                headers=cls._headers
            ) as response:
                await raise_for_status(response)
                async for chunk in response.iter_content():
                    yield chunk

    @classmethod
    async def login(
        cls,
        proxy: str = None,
        api_key: str = None,
        proof_token: str = None,
        cookies: Cookies = None,
        headers: dict = None,
        **kwargs
    ) -> AsyncIterator:
        if cls._expires is not None and (cls._expires - 60*10) < time.time():
            cls._headers = cls._api_key = None
        if cls._headers is None or headers is not None:
            cls._headers = {} if headers is None else headers
        if proof_token is not None:
            RequestConfig.proof_token = proof_token
        if cookies is not None:
            RequestConfig.cookies = cookies
        if api_key is not None:
            cls._create_request_args(RequestConfig.cookies, RequestConfig.headers)
            cls._set_api_key(api_key)
        else:
            try:
                await get_request_config(proxy)
                cls._create_request_args(RequestConfig.cookies, RequestConfig.headers)
                if RequestConfig.access_token is not None or cls.needs_auth:
                    if not cls._set_api_key(RequestConfig.access_token):
                        raise NoValidHarFileError(f"Access token is not valid: {RequestConfig.access_token}")
            except NoValidHarFileError:
                if has_nodriver:
                    if cls._api_key is None:
                        login_url = os.environ.get("G4F_LOGIN_URL")
                        if login_url:
                            yield RequestLogin(cls.label, login_url)
                        await cls.nodriver_auth(proxy)
                else:
                    raise
        yield Parameters(**{
            "api_key": cls._api_key,
            "proof_token": RequestConfig.proof_token,
            "cookies": RequestConfig.cookies,
        })

    @classmethod
    async def nodriver_auth(cls, proxy: str = None):
        browser, stop_browser = await get_nodriver(proxy=proxy)
        try:
            page = browser.main_tab
            def on_request(event: nodriver.cdp.network.RequestWillBeSent):
                if event.request.url == start_url or event.request.url.startswith(conversation_url):
                    RequestConfig.headers = event.request.headers
                elif event.request.url in (backend_url, backend_anon_url):
                    if "OpenAI-Sentinel-Proof-Token" in event.request.headers:
                            RequestConfig.proof_token = json.loads(base64.b64decode(
                                event.request.headers["OpenAI-Sentinel-Proof-Token"].split("gAAAAAB", 1)[-1].encode()
                            ).decode())
                    if "OpenAI-Sentinel-Turnstile-Token" in event.request.headers:
                        RequestConfig.turnstile_token = event.request.headers["OpenAI-Sentinel-Turnstile-Token"]
                    if "Authorization" in event.request.headers:
                        cls._api_key = event.request.headers["Authorization"].split()[-1]
                elif event.request.url == arkose_url:
                    RequestConfig.arkose_request = arkReq(
                        arkURL=event.request.url,
                        arkBx=None,
                        arkHeader=event.request.headers,
                        arkBody=event.request.post_data,
                        userAgent=event.request.headers.get("User-Agent")
                    )
            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            page = await browser.get(cls.url)
            user_agent = await page.evaluate("window.navigator.userAgent")
            await page.select("#prompt-textarea", 240)
            await page.evaluate("document.getElementById('prompt-textarea').innerText = 'Hello'")
            await page.evaluate("document.querySelector('[data-testid=\"send-button\"]').click()")
            while True:
                if cls._api_key is not None or not cls.needs_auth:
                    break
                body = await page.evaluate("JSON.stringify(window.__remixContext)")
                if body:
                    match = re.search(r'"accessToken":"(.*?)"', body)
                    if match:
                        cls._api_key = match.group(1)
                        break
                await asyncio.sleep(1)
            while True:
                if RequestConfig.proof_token:
                    break
                await asyncio.sleep(1)
            RequestConfig.data_build = await page.evaluate("document.documentElement.getAttribute('data-build')")
            RequestConfig.cookies = await page.send(get_cookies([cls.url]))
            await page.close()
            cls._create_request_args(RequestConfig.cookies, RequestConfig.headers, user_agent=user_agent)
            cls._set_api_key(cls._api_key)
        finally:
            stop_browser()

    @staticmethod
    def get_default_headers() -> Dict[str, str]:
        return {
            **DEFAULT_HEADERS,
            "content-type": "application/json",
        }

    @classmethod
    def _create_request_args(cls, cookies: Cookies = None, headers: dict = None, user_agent: str = None):
        cls._headers = cls.get_default_headers() if headers is None else headers
        if user_agent is not None:
            cls._headers["user-agent"] = user_agent
        cls._cookies = {} if cookies is None else cookies
        cls._update_cookie_header()

    @classmethod
    def _update_request_args(cls, auth_result: AuthResult, session: StreamSession):
        if hasattr(auth_result, "cookies"):
            for c in session.cookie_jar if hasattr(session, "cookie_jar") else session.cookies.jar:
                auth_result.cookies[getattr(c, "key", getattr(c, "name", ""))] = c.value
            cls._cookies = auth_result.cookies
        cls._update_cookie_header()

    @classmethod
    def _set_api_key(cls, api_key: str):
        if api_key:
            exp = api_key.split(".")[1]
            exp = (exp + "=" * (4 - len(exp) % 4)).encode()
            cls._expires = json.loads(base64.b64decode(exp)).get("exp")
            debug.log(f"OpenaiChat: API key expires at\n {cls._expires} we have:\n {time.time()}")
            if time.time() > cls._expires:
                debug.log(f"OpenaiChat: API key is expired")
            else:
                cls._api_key = api_key
                cls._headers["authorization"] = f"Bearer {api_key}"
                return True
        return False

    @classmethod
    def _update_cookie_header(cls):
        if cls._cookies:
            cls._headers["cookie"] = format_cookies(cls._cookies)

class Conversation(JsonConversation):
    """
    Class to encapsulate response fields.
    """
    def __init__(self, conversation_id: str = None, message_id: str = None, user_id: str = None, finish_reason: str = None, parent_message_id: str = None):
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.finish_reason = finish_reason
        self.is_recipient = False
        self.parent_message_id = message_id if parent_message_id is None else parent_message_id
        self.user_id = user_id

def get_cookies(
    urls: Optional[Iterator[str]] = None
) -> Generator[Dict, Dict, Dict[str, str]]:
    params = {}
    if urls is not None:
        params['urls'] = [i for i in urls]
    cmd_dict = {
        'method': 'Network.getCookies',
        'params': params,
    }
    json = yield cmd_dict
    return {c["name"]: c["value"] for c in json['cookies']} if 'cookies' in json else {}
