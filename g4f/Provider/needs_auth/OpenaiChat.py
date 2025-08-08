from __future__ import annotations

import os
import re
import asyncio
import uuid
import json
import base64
import time
import random
from typing import AsyncIterator, Iterator, Optional, Generator, Dict, Union, List, Any
from copy import copy

try:
    import nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ..base_provider import AsyncAuthedProvider, ProviderModelMixin
from ...typing import AsyncResult, Messages, Cookies, MediaListType
from ...requests.raise_for_status import raise_for_status
from ...requests import StreamSession
from ...requests import get_nodriver
from ...image import ImageRequest, to_image, to_bytes, is_accepted_format
from ...errors import MissingAuthError, NoValidHarFileError, ModelNotFoundError
from ...providers.response import JsonConversation, FinishReason, SynthesizeData, AuthResult, ImageResponse, ImagePreview, ResponseType, format_link
from ...providers.response import TitleGeneration, RequestLogin, Reasoning
from ...tools.media import merge_media
from ..helper import format_cookies, format_media_prompt, to_string
from ..openai.models import default_model, default_image_model, models, image_models, text_models, model_aliases
from ..openai.har_file import get_request_config
from ..openai.har_file import RequestConfig, arkReq, arkose_url, start_url, conversation_url, backend_url, prepare_url, backend_anon_url
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
    active_by_default = True
    use_nodriver = True
    supports_gpt_4 = True
    supports_message_history = True
    supports_system_message = True
    default_model = default_model
    default_image_model = default_image_model
    image_models = image_models
    vision_models = text_models
    models = models
    model_aliases = model_aliases
    synthesize_content_type = "audio/aac"
    request_config = RequestConfig()

    _api_key: str = None
    _headers: dict = None
    _cookies: Cookies = None
    _expires: int = None

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        async for chunk in cls.login(proxy=proxy):
            yield chunk
        yield AuthResult(
            api_key=cls._api_key,
            cookies=cls._cookies or cls.request_config.cookies or {},
            headers=cls._headers or cls.request_config.headers or cls.get_default_headers(),
            expires=cls._expires,
            proof_token=cls.request_config.proof_token,
            turnstile_token=cls.request_config.turnstile_token
        )

    @classmethod
    async def upload_images(
        cls,
        session: StreamSession,
        auth_result: AuthResult,
        media: MediaListType,
    ) -> ImageRequest:
        """
        Upload an image to the service and get the download URL
        
        Args:
            session: The StreamSession object to use for requests
            headers: The headers to include in the requests
            media: The images to upload, either a PIL Image object or a bytes object
        
        Returns:
            An ImageRequest object that contains the download URL, file name, and other data
        """
        async def upload_image(image, image_name):
            debug.log(f"Uploading image: {image_name}")
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
            async with session.post(f"{cls.url}/backend-api/files", json=data, headers=cls._headers) as response:
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
        return [await upload_image(image, image_name) for image, image_name in media]

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
        # merged_messages = []
        # last_message = None
        # for message in messages:
        #     current_message = last_message
        #     if current_message is not None:
        #         if current_message["role"] == message["role"]:
        #             current_message["content"] += "\n" + message["content"]
        #         else:
        #             merged_messages.append(current_message)
        #             last_message = message.copy()
        #     else:
        #         last_message = message.copy()
        # if last_message is not None:
        #     merged_messages.append(last_message)

        messages = [{
            "id": str(uuid.uuid4()),
            "author": {"role": message["role"]},
            "content": {"content_type": "text", "parts": [to_string(message["content"])]},
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
    async def get_generated_image(cls, session: StreamSession, auth_result: AuthResult, element: Union[dict, str], prompt: str = None, conversation_id: str = None) -> AsyncIterator:
        download_urls = []
        is_sediment = False
        if prompt is None:
            try:
                prompt = element["metadata"]["dalle"]["prompt"]
            except KeyError:
                pass
        if "asset_pointer" in element:
            element = element["asset_pointer"] 
        if isinstance(element, str) and element.startswith("file-service://"):
            element = element.split("file-service://", 1)[-1]
        elif isinstance(element, str) and element.startswith("sediment://"):
            is_sediment = True
            element = element.split("sediment://")[-1]
        else:
            raise RuntimeError(f"Invalid image element: {element}")
        if is_sediment:
            url = f"{cls.url}/backend-api/conversation/{conversation_id}/attachment/{element}/download"
        else:
            url =f"{cls.url}/backend-api/files/{element}/download"
        try:
            async with session.get(url, headers=auth_result.headers) as response:
                cls._update_request_args(auth_result, session)
                await raise_for_status(response)
                data = await response.json()
                download_url = data.get("download_url")
                if download_url is not None:
                    download_urls.append(download_url)
                    debug.log(f"OpenaiChat: Found image: {download_url}")
                else:
                    debug.log("OpenaiChat: No download URL found in response: ", data)
        except Exception as e:
            debug.error("OpenaiChat: Download image failed")
            debug.error(e)
        if download_urls:
            return ImagePreview(download_urls, prompt) if is_sediment else ImageResponse(download_urls, prompt)

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        proxy: str = None,
        timeout: int = 360,
        auto_continue: bool = False,
        action: str = "next",
        conversation: Conversation = None,
        media: MediaListType = None,
        return_conversation: bool = True,
        web_search: bool = False,
        prompt: str = None,
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
            action (str): Type of action ('next', 'continue', 'variant').
            conversation_id (str): ID of the conversation.
            media (MediaListType): Images to include in the conversation.
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
            media  = merge_media(media, messages)
            if not cls.needs_auth and not media:
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
                    image_requests = await cls.upload_images(session, auth_result, media)
                except Exception as e:
                    debug.error("OpenaiChat: Upload image failed")
                    debug.error(e)
            try:
                model = cls.get_model(model)
            except ModelNotFoundError:
                pass
            image_model = False
            if model in cls.image_models:
                image_model = True
                model = cls.default_model
            if conversation is None:
                conversation = Conversation(None, str(uuid.uuid4()), getattr(auth_result, "cookies", {}).get("oai-did"))
            else:
                conversation = copy(conversation)
            if getattr(auth_result, "cookies", {}).get("oai-did") != getattr(conversation, "user_id", None):
                conversation = Conversation(None, str(uuid.uuid4()))
            if cls._api_key is None:
                auto_continue = False
            conversation.finish_reason = None
            sources = OpenAISources([])
            references = ContentReferences()
            while conversation.finish_reason is None:
                conduit_token = None
                if cls._api_key is not None:
                    data = {
                        "action": "next",
                        "fork_from_shared_post": False,
                        "parent_message_id": conversation.message_id,
                        "model": model,
                        "timezone_offset_min": -120,
                        "timezone": "Europe/Berlin",
                        "conversation_mode": {"kind": "primary_assistant"},
                        "system_hints": [
                            "picture_v2"
                        ] if image_model else [],
                        "supports_buffering": True,
                        "supported_encodings": ["v1"]
                    }
                    async with session.post(
                        prepare_url,
                        json=data,
                        headers=cls._headers
                    ) as response:
                        await raise_for_status(response)
                        conduit_token = (await response.json())["conduit_token"]
                async with session.post(
                    f"{cls.url}/backend-anon/sentinel/chat-requirements"
                    if cls._api_key is None else
                    f"{cls.url}/backend-api/sentinel/chat-requirements",
                    json={"p": None if not getattr(auth_result, "proof_token", None) else get_requirements_token(getattr(auth_result, "proof_token", None))},
                    headers=cls._headers
                ) as response:
                    if response.status in (401, 403):
                        raise MissingAuthError(f"Response status: {response.status}")
                    else:
                        cls._update_request_args(auth_result, session)
                    await raise_for_status(response)
                    chat_requirements = await response.json()
                    need_turnstile = chat_requirements.get("turnstile", {}).get("required", False)
                    need_arkose = chat_requirements.get("arkose", {}).get("required", False) 
                    chat_token = chat_requirements.get("token")  

                # if need_arkose and cls.request_config.arkose_token is None:
                #     await get_request_config(proxy)
                #     cls._create_request_args(auth_result.cookies, auth_result.headers)
                #     cls._set_api_key(auth_result.access_token)
                #     if auth_result.arkose_token is None:
                #         raise MissingAuthError("No arkose token found in .har file")
                if "proofofwork" in chat_requirements:
                    user_agent = getattr(auth_result, "headers", {}).get("user-agent")
                    proof_token = getattr(auth_result, "proof_token", None)
                    if proof_token is None:
                        auth_result.proof_token = get_config(user_agent)
                    proofofwork = generate_proof_token(
                        **chat_requirements["proofofwork"],
                        user_agent=user_agent,
                        proof_token=proof_token
                    )
                [debug.log(text) for text in (
                    #f"Arkose: {'False' if not need_arkose else auth_result.arkose_token[:12]+'...'}",
                    #f"Proofofwork: {'False' if proofofwork is None else proofofwork[:12]+'...'}",
                    #f"AccessToken: {'False' if cls._api_key is None else cls._api_key[:12]+'...'}",
                )]
                data = {
                    "action": "next",
                    "parent_message_id": conversation.message_id,
                    "model": model,
                    "timezone_offset_min":-120,
                    "timezone":"Europe/Berlin",
                    "conversation_mode":{"kind":"primary_assistant"},
                    "enable_message_followups":True,
                    "system_hints": ["search"] if web_search else None,
                    "supports_buffering":True,
                    "supported_encodings":["v1"],
                    "client_contextual_info":{"is_dark_mode":False,"time_since_loaded":random.randint(20, 500),"page_height":578,"page_width":1850,"pixel_ratio":1,"screen_height":1080,"screen_width":1920},
                    "paragen_cot_summary_display_override":"allow"
                }
                if conversation.conversation_id is not None:
                    data["conversation_id"] = conversation.conversation_id
                    debug.log(f"OpenaiChat: Use conversation: {conversation.conversation_id}")
                prompt = conversation.prompt = format_media_prompt(messages, prompt)
                if action != "continue":
                    data["parent_message_id"] = getattr(conversation, "parent_message_id", conversation.message_id)
                    conversation.parent_message_id = None
                    new_messages = messages
                    if conversation.conversation_id is not None:
                        new_messages = []
                        for message in messages:
                            if message.get("role") == "assistant":
                                new_messages = []
                            else:
                                new_messages.append(message)
                    data["messages"] = cls.create_messages(new_messages, image_requests, ["search"] if web_search else None)
                headers = {
                    **cls._headers,
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                    "openai-sentinel-chat-requirements-token": chat_token,
                    **({} if conduit_token is None else {"x-conduit-token": conduit_token})
                }
                #if cls.request_config.arkose_token:
                #    headers["openai-sentinel-arkose-token"] = cls.request_config.arkose_token
                if proofofwork is not None:
                    headers["openai-sentinel-proof-token"] = proofofwork
                if need_turnstile and getattr(auth_result, "turnstile_token", None) is not None:
                    headers['openai-sentinel-turnstile-token'] = auth_result.turnstile_token
                async with session.post(
                    backend_anon_url
                    if cls._api_key is None else
                    backend_url,
                    json=data,
                    headers=headers
                ) as response:
                    cls._update_request_args(auth_result, session)
                    if response.status in (401, 403, 429):
                        raise MissingAuthError("Access token is not valid")
                    elif response.status == 422:
                        raise RuntimeError((await response.json()), data)
                    await raise_for_status(response)
                    buffer = u""
                    matches = []
                    async for line in response.iter_lines():
                        pattern = re.compile(r"file-service://[\w-]+")
                        for match in pattern.finditer(line.decode(errors="ignore")):
                            if match.group(0) in matches:
                                continue
                            matches.append(match.group(0))
                            generated_image = await cls.get_generated_image(session, auth_result, match.group(0), prompt)
                            if generated_image is not None:
                                yield generated_image
                        async for chunk in cls.iter_messages_line(session, auth_result, line, conversation, sources, references):
                            if isinstance(chunk, str):
                                chunk = chunk.replace("\ue203", "").replace("\ue204", "").replace("\ue206", "")
                                buffer += chunk
                                if buffer.find(u"\ue200") != -1:
                                    if buffer.find(u"\ue201") != -1:
                                        def sequence_replacer(match):
                                            def citation_replacer(match: re.Match[str]):
                                                ref_type = match.group(1)
                                                ref_index = int(match.group(2))
                                                if ((ref_type == "image" and is_image_embedding) or 
                                                    is_video_embedding or 
                                                    ref_type == "forecast"):

                                                    reference = references.get_reference({
                                                        "ref_index": ref_index,
                                                        "ref_type": ref_type
                                                    })
                                                    if not reference:
                                                        return ""
                                                    
                                                    if ref_type == "forecast":
                                                        if reference.get("alt"):
                                                            return reference.get("alt")
                                                        if reference.get("prompt_text"):
                                                            return reference.get("prompt_text")

                                                    if is_image_embedding and reference.get("content_url", ""):
                                                        return f"![{reference.get('title', '')}]({reference.get('content_url')})"
                                                    
                                                    if is_video_embedding:
                                                        if reference.get("url", "") and reference.get("thumbnail_url", ""):
                                                            return f"[![{reference.get('title', '')}]({reference['thumbnail_url']})]({reference['url']})"
                                                        video_match = re.match(r"video\n(.*?)\nturn[0-9]+", match.group(0))
                                                        if video_match:
                                                            return video_match.group(1)
                                                    return ""

                                                source_index = sources.get_index({
                                                    "ref_index": ref_index,
                                                    "ref_type": ref_type
                                                })
                                                if source_index is not None and len(sources.list) > source_index:
                                                    link = sources.list[source_index]["url"]
                                                    return f"[[{source_index+1}]]({link})"
                                                return f""
                                            
                                            def products_replacer(match: re.Match[str]):
                                                try:
                                                    products_data = json.loads(match.group(1))
                                                    products_str = ""
                                                    for idx, _ in enumerate(products_data.get("selections", []) or []):
                                                        name = products_data.get('selections', [])[idx][1]
                                                        tags = products_data.get('tags', [])[idx]
                                                        products_str += f"{name} - {tags}\n\n"

                                                    return products_str
                                                except:
                                                    return ""

                                            sequence_content = match.group(1)
                                            sequence_content = sequence_content.replace("\ue200", "").replace("\ue202", "\n").replace("\ue201", "")
                                            sequence_content = sequence_content.replace("navlist\n", "#### ")
                                            
                                            # Handle search, news, view and image citations
                                            is_image_embedding = sequence_content.startswith("i\nturn")
                                            is_video_embedding = sequence_content.startswith("video\n")
                                            sequence_content = re.sub(
                                                r'(?:cite\nturn[0-9]+|forecast\nturn[0-9]+|video\n.*?\nturn[0-9]+|i?\n?turn[0-9]+)(search|news|view|image|forecast)(\d+)', 
                                                citation_replacer, 
                                                sequence_content
                                            )
                                            sequence_content = re.sub(r'products\n(.*)', products_replacer, sequence_content)
                                            sequence_content = re.sub(r'product_entity\n\[".*","(.*)"\]', lambda x: x.group(1), sequence_content)
                                            return sequence_content
                                        
                                        # process only completed sequences and do not touch start of next not completed sequence
                                        buffer = re.sub(r'\ue200(.*?)\ue201', sequence_replacer, buffer, flags=re.DOTALL)
                                        
                                        if buffer.find(u"\ue200") != -1: # still have uncompleted sequence
                                            continue
                                    else:
                                        # do not yield to consume rest part of special sequence
                                        continue

                                yield buffer
                                buffer = ""
                            else:
                                yield chunk
                        if conversation.finish_reason is not None:
                            break
                    if buffer:
                        yield buffer
                if sources.list:
                    yield sources
                if conversation.generated_images:
                    yield ImageResponse(conversation.generated_images.urls, conversation.prompt)
                    conversation.generated_images = None
                conversation.prompt = None
                if return_conversation:
                    yield conversation
                if auth_result.api_key is not None:
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
            yield FinishReason(conversation.finish_reason)

    @classmethod
    async def iter_messages_line(cls, session: StreamSession, auth_result: AuthResult, line: bytes, fields: Conversation, sources: OpenAISources, references: ContentReferences) -> AsyncIterator:
        if not line.startswith(b"data: "):
            return
        elif line.startswith(b"data: [DONE]"):
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
        fields.p = line.get("p", fields.p)
        if fields.p is not None and fields.p.startswith("/message/content/thoughts"):
            if fields.p.endswith("/content"):
                if fields.thoughts_summary:
                    yield Reasoning(token="", status=fields.thoughts_summary)
                    fields.thoughts_summary = ""
                yield Reasoning(token=line.get("v"))
            elif fields.p.endswith("/summary"):
                fields.thoughts_summary += line.get("v")
            return
        if "v" in line:
            v = line.get("v")
            if isinstance(v, str) and fields.recipient == "all":
                if fields.p == "/message/metadata/refresh_key_info":
                    yield ""
                elif "p" not in line or line.get("p") == "/message/content/parts/0":
                    yield Reasoning(token=v) if fields.is_thinking else v
            elif isinstance(v, list):
                buffer = ""
                for m in v:
                    if m.get("p") == "/message/content/parts/0" and fields.recipient == "all":
                        buffer += m.get("v")
                    elif m.get("p") == "/message/metadata/image_gen_title":
                        fields.prompt = m.get("v")
                    elif m.get("p") == "/message/content/parts/0/asset_pointer":
                        generated_images = fields.generated_images = await cls.get_generated_image(session, auth_result, m.get("v"), fields.prompt, fields.conversation_id)
                        if generated_images is not None:
                            if buffer: 
                                yield buffer
                            yield generated_images
                    elif m.get("p") == "/message/metadata/search_result_groups":
                        for entry in [p.get("entries") for p in m.get("v")]:
                            for link in entry:
                                sources.add_source(link)
                    elif m.get("p") == "/message/metadata/content_references" and not isinstance(m.get("v"), int):
                        for entry in m.get("v"):
                            for link in entry.get("sources", []):
                                sources.add_source(link)
                            for link in entry.get("items", []):
                                sources.add_source(link)
                            for link in entry.get("fallback_items", []) or []:
                                sources.add_source(link)
                            if m.get("o", None) == "append":
                                references.add_reference(entry)
                    elif m.get("p") and re.match(r"^/message/metadata/content_references/\d+$", m.get("p")):
                        if "url" in m.get("v") or "link" in m.get("v"):
                            sources.add_source(m.get("v"))
                        for link in m.get("v").get("fallback_items", []) or []:
                            sources.add_source(link)

                        match = re.match(r"^/message/metadata/content_references/(\d+)$", m.get("p"))
                        if match and m.get("o") == "append" and isinstance(m.get("v"), dict):
                            idx = int(match.group(1))
                            references.merge_reference(idx, m.get("v"))
                    elif m.get("p") and re.match(r"^/message/metadata/content_references/\d+/fallback_items$", m.get("p")) and isinstance(m.get("v"), list):
                        for link in m.get("v", []) or []:
                            sources.add_source(link)
                    elif m.get("p") and re.match(r"^/message/metadata/content_references/\d+/items$", m.get("p")) and isinstance(m.get("v"), list):
                        for link in m.get("v", []) or []:
                            sources.add_source(link)
                    elif m.get("p") and re.match(r"^/message/metadata/content_references/\d+/refs$", m.get("p")) and isinstance(m.get("v"), list):
                        match = re.match(r"^/message/metadata/content_references/(\d+)/refs$", m.get("p"))
                        if match:
                            idx = int(match.group(1))
                            references.update_reference(idx, m.get("o"), "refs", m.get("v"))
                    elif m.get("p") and re.match(r"^/message/metadata/content_references/\d+/alt$", m.get("p")) and isinstance(m.get("v"), list):
                        match = re.match(r"^/message/metadata/content_references/(\d+)/alt$", m.get("p"))
                        if match:
                            idx = int(match.group(1))
                            references.update_reference(idx, m.get("o"), "alt", m.get("v"))
                    elif m.get("p") and re.match(r"^/message/metadata/content_references/\d+/prompt_text$", m.get("p")) and isinstance(m.get("v"), list):
                        match = re.match(r"^/message/metadata/content_references/(\d+)/prompt_text$", m.get("p"))
                        if match:
                            idx = int(match.group(1))
                            references.update_reference(idx, m.get("o"), "prompt_text", m.get("v"))
                    elif m.get("p") and re.match(r"^/message/metadata/content_references/\d+/refs/\d+$", m.get("p")) and isinstance(m.get("v"), dict):
                        match = re.match(r"^/message/metadata/content_references/(\d+)/refs/(\d+)$", m.get("p"))
                        if match:
                            reference_idx = int(match.group(1))
                            ref_idx = int(match.group(2))
                            references.update_reference(reference_idx, m.get("o"), "refs", m.get("v"), ref_idx)
                    elif m.get("p") and re.match(r"^/message/metadata/content_references/\d+/images$", m.get("p")) and isinstance(m.get("v"), list):
                        match = re.match(r"^/message/metadata/content_references/(\d+)/images$", m.get("p"))
                        if match:
                            idx = int(match.group(1))
                            references.update_reference(idx, m.get("o"), "images", m.get("v"))
                    elif m.get("p") == "/message/metadata/finished_text":
                        fields.is_thinking = False
                        if buffer: 
                            yield buffer
                        yield Reasoning(status=m.get("v"))
                    elif m.get("p") == "/message/metadata" and fields.recipient == "all":
                        fields.finish_reason = m.get("v", {}).get("finish_details", {}).get("type")
                        break

                yield buffer
            elif isinstance(v, dict):
                if fields.conversation_id is None:
                    fields.conversation_id = v.get("conversation_id")
                    debug.log(f"OpenaiChat: New conversation: {fields.conversation_id}")
                m = v.get("message", {})
                fields.recipient = m.get("recipient", fields.recipient)
                if fields.recipient == "all":
                    c = m.get("content", {})
                    if c.get("content_type") == "text" and m.get("author", {}).get("role") == "tool" and "initial_text" in m.get("metadata", {}):
                        fields.is_thinking = True
                        yield Reasoning(status=m.get("metadata", {}).get("initial_text"))
                    #if c.get("content_type") == "multimodal_text":
                    #    for part in c.get("parts"):
                    #        if isinstance(part, dict) and part.get("content_type") == "image_asset_pointer":
                    #            yield await cls.get_generated_image(session, auth_result, part, fields.prompt, fields.conversation_id)
                    if m.get("author", {}).get("role") == "assistant":
                        if fields.parent_message_id is None:
                            fields.parent_message_id = v.get("message", {}).get("id")
                        fields.message_id = v.get("message", {}).get("id")
            return
        if "error" in line and line.get("error"):
            raise RuntimeError(line.get("error"))

    @classmethod
    async def synthesize(cls, params: dict) -> AsyncIterator[bytes]:
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
            cls.request_config.proof_token = proof_token
        if cookies is not None:
            cls.request_config.cookies = cookies
        if api_key is not None:
            cls._create_request_args(cls.request_config.cookies, cls.request_config.headers)
            cls._set_api_key(api_key)
        else:
            try:
                cls.request_config = await get_request_config(cls.request_config, proxy)
                if cls.request_config is None:
                    cls.request_config = RequestConfig()
                cls._create_request_args(cls.request_config.cookies, cls.request_config.headers)
                if cls.needs_auth and cls.request_config.access_token is None:
                    raise NoValidHarFileError(f"Missing access token")
                if not cls._set_api_key(cls.request_config.access_token):
                    raise NoValidHarFileError(f"Access token is not valid: {cls.request_config.access_token}")
            except NoValidHarFileError:
                if has_nodriver:
                    if cls.request_config.access_token is None:
                        yield RequestLogin(cls.label, os.environ.get("G4F_LOGIN_URL", ""))
                        await cls.nodriver_auth(proxy)
                else:
                    raise

    @classmethod
    async def nodriver_auth(cls, proxy: str = None):
        browser, stop_browser = await get_nodriver(proxy=proxy)
        try:
            page = await browser.get(cls.url)
            def on_request(event: nodriver.cdp.network.RequestWillBeSent, page=None):
                if event.request.url == start_url or event.request.url.startswith(conversation_url):
                    if cls.request_config.headers is None:
                        cls.request_config.headers = {}
                    for key, value in event.request.headers.items():
                        cls.request_config.headers[key.lower()] = value
                elif event.request.url in (backend_url, backend_anon_url):
                    if "OpenAI-Sentinel-Proof-Token" in event.request.headers:
                            cls.request_config.proof_token = json.loads(base64.b64decode(
                                event.request.headers["OpenAI-Sentinel-Proof-Token"].split("gAAAAAB", 1)[-1].split("~")[0].encode()
                            ).decode())
                    if "OpenAI-Sentinel-Turnstile-Token" in event.request.headers:
                        cls.request_config.turnstile_token = event.request.headers["OpenAI-Sentinel-Turnstile-Token"]
                    if "Authorization" in event.request.headers:
                        cls._api_key = event.request.headers["Authorization"].split()[-1]
                elif event.request.url == arkose_url:
                    cls.request_config.arkose_request = arkReq(
                        arkURL=event.request.url,
                        arkBx=None,
                        arkHeader=event.request.headers,
                        arkBody=event.request.post_data,
                        userAgent=event.request.headers.get("User-Agent")
                    )
            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            await page.reload()
            user_agent = await page.evaluate("window.navigator.userAgent", return_by_value=True)
            if cls.needs_auth:
                await page.select('[data-testid="accounts-profile-button"]', 300)
            textarea = await page.select("#prompt-textarea", 300)
            await textarea.send_keys("Hello")
            await asyncio.sleep(1)
            button = await page.select("[data-testid=\"send-button\"]")
            if button:
                await button.click()
            while True:
                body = await page.evaluate("JSON.stringify(window.__remixContext)", return_by_value=True)
                if hasattr(body, "value"):
                    body = body.value
                if body:
                    match = re.search(r'"accessToken":"(.+?)"', body)
                    if match:
                        cls._api_key = match.group(1)
                        break
                if cls._api_key is not None or not cls.needs_auth:
                    break
                await asyncio.sleep(1)
            while True:
                if cls.request_config.proof_token:
                    break
                await asyncio.sleep(1)
            cls.request_config.data_build = await page.evaluate("document.documentElement.getAttribute('data-build')")
            cls.request_config.cookies = await page.send(get_cookies([cls.url]))
            await page.close()
            cls._create_request_args(cls.request_config.cookies, cls.request_config.headers, user_agent=user_agent)
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
        cls._api_key = api_key
        if api_key:
            exp = api_key.split(".")[1]
            exp = (exp + "=" * (4 - len(exp) % 4)).encode()
            cls._expires = json.loads(base64.b64decode(exp)).get("exp")
            debug.log(f"OpenaiChat: API key expires at\n {cls._expires} we have:\n {time.time()}")
            if time.time() > cls._expires:
                debug.log(f"OpenaiChat: API key is expired")
                return False
            else:
                cls._headers["authorization"] = f"Bearer {api_key}"
                return True
        return True

    @classmethod
    def _update_cookie_header(cls):
        if cls._cookies:
            cls._headers["cookie"] = format_cookies(cls._cookies)

class Conversation(JsonConversation):
    """
    Class to encapsulate response fields.
    """
    def __init__(self, conversation_id: str = None, message_id: str = None, user_id: str = None, finish_reason: str = None, parent_message_id: str = None, is_thinking: bool = False):
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.finish_reason = finish_reason
        self.recipient = "all"
        self.parent_message_id = message_id if parent_message_id is None else parent_message_id
        self.user_id = user_id
        self.is_thinking = is_thinking
        self.p = None
        self.thoughts_summary = ""
        self.prompt = None
        self.generated_images: ImagePreview = None

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

class OpenAISources(ResponseType):
    list: List[Dict[str, str]]

    def __init__(self, sources: List[Dict[str, str]]) -> None:
        """Initialize with a list of source dictionaries."""
        self.list = []
        for source in sources:
            self.add_source(source)

    def add_source(self, source: Union[Dict[str, str], str]) -> None:
        """Add a source to the list, cleaning the URL if necessary."""
        source = source if isinstance(source, dict) else {"url": source}
        url = source.get("url", source.get("link", None))
        if not url:
            return

        url = re.sub(r"[&?]utm_source=.+", "", url)
        source["url"] = url

        ref_info = self.get_ref_info(source)
        if ref_info:
            existing_source, idx = self.find_by_ref_info(ref_info)
            if existing_source and idx is not None:
                self.list[idx] = source
                return
        
        existing_source, idx = self.find_by_url(source["url"])
        if existing_source and idx is not None:
            self.list[idx] = source
            return

        self.list.append(source)

    def __str__(self) -> str:
        """Return formatted sources as a string."""
        if not self.list:
            return ""
        return "\n\n\n\n" + ("\n>\n".join([
            f"> [{idx+1}] {format_link(link['url'], link.get('title', ''))}"
            for idx, link in enumerate(self.list)
        ]))
    
    def get_ref_info(self, source: Dict[str, str]) -> dict[str, str|int] | None:
        ref_index = source.get("ref_id", {}).get("ref_index", None)
        ref_type = source.get("ref_id", {}).get("ref_type", None)
        if isinstance(ref_index, int):
            return {
                "ref_index": ref_index, 
                "ref_type": ref_type,
            }
        
        for ref_info in source.get('refs') or []:
            ref_index = ref_info.get("ref_index", None)
            ref_type = ref_info.get("ref_type", None)
            if isinstance(ref_index, int):
                return {
                    "ref_index": ref_index, 
                    "ref_type": ref_type,
                }
        
        return None

    def find_by_ref_info(self, ref_info: dict[str, str|int]):
        for idx, source in enumerate(self.list):
            source_ref_info = self.get_ref_info(source)
            if (source_ref_info and 
                source_ref_info["ref_index"] == ref_info["ref_index"] and 
                source_ref_info["ref_type"] == ref_info["ref_type"]):
                return source, idx 

        return None, None
    
    def find_by_url(self, url: str):
        for idx, source in enumerate(self.list):
            if source["url"] == url:
                return source, idx
        return None, None 

    def get_index(self, ref_info: dict[str, str|int]) -> int | None: 
        _, index = self.find_by_ref_info(ref_info)
        if index is not None:
            return index 

        return None

class ContentReferences:
    def __init__(self) -> None:
        self.list: List[Dict[str, Any]] = []

    def add_reference(self, reference_part: dict) -> None:
        self.list.append(reference_part)

    def merge_reference(self, idx: int, reference_part: dict):
        while len(self.list) <= idx:
            self.list.append({})

        self.list[idx] = {**self.list[idx], **reference_part}

    def update_reference(self, idx: int, operation: str, field: str, value: Any, ref_idx = None) -> None:
        while len(self.list) <= idx:
            self.list.append({})
        
        if operation == "append" or operation == "add":
            if not isinstance(self.list[idx].get(field, None), list):
                self.list[idx][field] = []
            if isinstance(value, list):
                self.list[idx][field].extend(value)
            else:    
                self.list[idx][field].append(value)

        if operation == "replace" and ref_idx is not None:
            if field == "refs" and not isinstance(self.list[idx].get(field, None), list):
                self.list[idx][field] = []

            if isinstance(self.list[idx][field], list):
                if len(self.list[idx][field]) <= ref_idx:
                    self.list[idx][field].append(value)
                else:
                    self.list[idx][field][ref_idx] = value
            else:
                self.list[idx][field] = value

    def get_ref_info(
        self, 
        source: Dict[str, str], 
        target_ref_info: Dict[str, Union[str, int]]
    ) -> dict[str, str|int] | None:
        for idx, ref_info in enumerate(source.get("refs", [])) or []:
            if not isinstance(ref_info, dict):
                continue

            ref_index = ref_info.get("ref_index", None)
            ref_type = ref_info.get("ref_type", None)
            if isinstance(ref_index, int) and isinstance(ref_type, str):
                if (not target_ref_info or 
                    (target_ref_info["ref_index"] == ref_index and 
                     target_ref_info["ref_type"] == ref_type)):
                    return {
                        "ref_index": ref_index, 
                        "ref_type": ref_type,
                        "idx": idx
                    }

        return None

    def get_reference(self, ref_info: Dict[str, Union[str, int]]) -> Any:
        for reference in self.list:
            reference_ref_info = self.get_ref_info(reference, ref_info)

            if (not reference_ref_info or 
                reference_ref_info["ref_index"] != ref_info["ref_index"] or 
                reference_ref_info["ref_type"] != ref_info["ref_type"]):
                continue

            if ref_info["ref_type"] != "image":
                return reference

            images = reference.get("images", [])
            if isinstance(images, list) and len(images) > reference_ref_info["idx"]:
                return images[reference_ref_info["idx"]]

        return None
