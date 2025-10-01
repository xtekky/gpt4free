from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import uuid
from time import time
from typing import Literal, Optional

import aiohttp
from ..errors import RateLimitError, ResponseError
from ..typing import AsyncResult, Messages, MediaListType
from ..providers.response import JsonConversation, Reasoning, Usage, ImageResponse, FinishReason
from ..requests import sse_stream
from ..tools.media import merge_media
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message
from .. import debug

try:
    import curl_cffi

    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

text_models = [
    'qwen3-max-preview', 'qwen-plus-2025-09-11', 'qwen3-235b-a22b', 'qwen3-coder-plus', 'qwen3-30b-a3b',
    'qwen3-coder-30b-a3b-instruct', 'qwen-max-latest', 'qwen-plus-2025-01-25', 'qwq-32b', 'qwen-turbo-2025-02-11',
    'qwen2.5-omni-7b', 'qvq-72b-preview-0310', 'qwen2.5-vl-32b-instruct', 'qwen2.5-14b-instruct-1m',
    'qwen2.5-coder-32b-instruct', 'qwen2.5-72b-instruct']

image_models = [
    'qwen3-max-preview', 'qwen-plus-2025-09-11', 'qwen3-235b-a22b', 'qwen3-coder-plus', 'qwen3-30b-a3b',
    'qwen3-coder-30b-a3b-instruct', 'qwen-max-latest', 'qwen-plus-2025-01-25', 'qwen-turbo-2025-02-11',
    'qwen2.5-omni-7b', 'qwen2.5-vl-32b-instruct', 'qwen2.5-14b-instruct-1m', 'qwen2.5-coder-32b-instruct',
    'qwen2.5-72b-instruct']

vision_models = [
    'qwen3-max-preview', 'qwen-plus-2025-09-11', 'qwen3-235b-a22b', 'qwen3-coder-plus', 'qwen3-30b-a3b',
    'qwen3-coder-30b-a3b-instruct', 'qwen-max-latest', 'qwen-plus-2025-01-25', 'qwen-turbo-2025-02-11',
    'qwen2.5-omni-7b', 'qvq-72b-preview-0310', 'qwen2.5-vl-32b-instruct', 'qwen2.5-14b-instruct-1m',
    'qwen2.5-coder-32b-instruct', 'qwen2.5-72b-instruct']

models = [
    'qwen3-max-preview', 'qwen-plus-2025-09-11', 'qwen3-235b-a22b', 'qwen3-coder-plus', 'qwen3-30b-a3b',
    'qwen3-coder-30b-a3b-instruct', 'qwen-max-latest', 'qwen-plus-2025-01-25', 'qwq-32b', 'qwen-turbo-2025-02-11',
    'qwen2.5-omni-7b', 'qvq-72b-preview-0310', 'qwen2.5-vl-32b-instruct', 'qwen2.5-14b-instruct-1m',
    'qwen2.5-coder-32b-instruct', 'qwen2.5-72b-instruct']


class Qwen(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Provider for Qwen's chat service (chat.qwen.ai), with configurable
    parameters (stream, enable_thinking) and print logs.
    """
    url = "https://chat.qwen.ai"
    working = True
    active_by_default = True
    supports_stream = True
    supports_message_history = False

    _models_loaded = True
    image_models = image_models
    text_models = text_models
    vision_models = vision_models
    models = models
    default_model = "qwen3-235b-a22b"

    _midtoken: str = None
    _midtoken_uses: int = 0

    @classmethod
    def get_models(cls) -> list[str]:
        if not cls._models_loaded and has_curl_cffi:
            response = curl_cffi.get(f"{cls.url}/api/models")
            if response.ok:
                models = response.json().get("data", [])
                cls.text_models = [model["id"] for model in models if "t2t" in model["info"]["meta"]["chat_type"]]

                cls.image_models = [
                    model["id"] for model in models if
                    "image_edit" in model["info"]["meta"]["chat_type"] or "t2i" in model["info"]["meta"]["chat_type"]
                ]

                cls.vision_models = [model["id"] for model in models if model["info"]["meta"]["capabilities"]["vision"]]

                cls.models = [model["id"] for model in models]
                cls.default_model = cls.models[0]
                cls._models_loaded = True
                cls.live += 1
                debug.log(f"Loaded {len(cls.models)} models from {cls.url}")

            else:
                debug.log(f"Failed to load models from {cls.url}: {response.status_code} {response.reason}")
        return cls.models

    @classmethod
    async def prepare_files(cls, media, chat_type="")->list:
        files = []
        for _file, file_name in media:
            file_type, _ = mimetypes.guess_type(file_name)
            file_class: Literal["default", "vision", "video", "audio", "document"] = "default"
            _type: Literal["file", "image", "video", "audio"] = "file"
            showType: Literal["file", "image", "video", "audio"] = "file"

            if isinstance(_file, str) and _file.startswith('http'):
                if chat_type == "image_edit" or (file_type and file_type.startswith("image")):
                    file_class = "vision"
                    _type = "image"
                    if not file_type:
                        # Try to infer from file extension, fallback to generic
                        ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
                        file_type = mimetypes.types_map.get(f'.{ext}', 'application/octet-stream')
                    showType = "image"

                files.append(
                    {
                        "type": _type,
                        "name": file_name,
                        "file_type": file_type,
                        "showType": showType,
                        "file_class": file_class,
                        "url": _file
                    }
                )
        return files

    @classmethod
    async def create_async_generator(
            cls,
            model: str,
            messages: Messages,
            media: MediaListType = None,
            conversation: JsonConversation = None,
            proxy: str = None,
            timeout: int = 120,
            stream: bool = True,
            enable_thinking: bool = True,
            chat_type: Literal[
                "t2t", "search", "artifacts", "web_dev", "deep_research", "t2i", "image_edit", "t2v"
            ] = "t2t",
            aspect_ratio: Optional[Literal["1:1", "4:3", "3:4", "16:9", "9:16"]] = None,
            **kwargs
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

        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Origin': cls.url,
            'Referer': f'{cls.url}/',
            'Content-Type': 'application/json',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Connection': 'keep-alive',
            'Authorization': 'Bearer',
            'Source': 'web'
        }

        prompt = get_last_user_message(messages)

        async with aiohttp.ClientSession(headers=headers) as session:
            for attempt in range(5):
                try:
                    if not cls._midtoken:
                        debug.log("[Qwen] INFO: No active midtoken. Fetching a new one...")
                        async with session.get('https://sg-wum.alibaba.com/w/wu.json', proxy=proxy) as r:
                            r.raise_for_status()
                            text = await r.text()
                            match = re.search(r"(?:umx\.wu|__fycb)\('([^']+)'\)", text)
                            if not match:
                                raise RuntimeError("Failed to extract bx-umidtoken.")
                            cls._midtoken = match.group(1)
                            cls._midtoken_uses = 1
                            debug.log(
                                f"[Qwen] INFO: New midtoken obtained. Use count: {cls._midtoken_uses}. Midtoken: {cls._midtoken}")
                    else:
                        cls._midtoken_uses += 1
                        debug.log(f"[Qwen] INFO: Reusing midtoken. Use count: {cls._midtoken_uses}")

                    req_headers = session.headers.copy()
                    req_headers['bx-umidtoken'] = cls._midtoken
                    req_headers['bx-v'] = '2.5.31'
                    message_id = str(uuid.uuid4())
                    if conversation is None:
                        chat_payload = {
                            "title": "New Chat",
                            "models": [model_name],
                            "chat_mode": "normal",
                            "chat_type": chat_type,
                            "timestamp": int(time() * 1000)
                        }
                        async with session.post(
                                f'{cls.url}/api/v2/chats/new', json=chat_payload, headers=req_headers, proxy=proxy
                        ) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                            if not (data.get('success') and data['data'].get('id')):
                                raise RuntimeError(f"Failed to create chat: {data}")
                        conversation = JsonConversation(
                            chat_id=data['data']['id'],
                            cookies={key: value for key, value in resp.cookies.items()},
                            parent_id=None
                        )
                    files = []
                    media = list(merge_media(media, messages))
                    if media:
                        files = await cls.prepare_files(media, chat_type=chat_type)

                    msg_payload = {
                        "stream": stream,
                        "incremental_output": stream,
                        "chat_id": conversation.chat_id,
                        "chat_mode": "normal",
                        "model": model_name,
                        "parent_id": conversation.parent_id,
                        "messages": [
                            {
                                "fid": message_id,
                                "parentId": conversation.parent_id,
                                "childrenIds": [],
                                "role": "user",
                                "content": prompt,
                                "user_action": "chat",
                                "files": files,
                                "models": [model_name],
                                "chat_type": chat_type,
                                "feature_config": {
                                    "thinking_enabled": enable_thinking,
                                    "output_schema": "phase",
                                    "thinking_budget": 81920
                                },
                                "extra": {
                                    "meta": {
                                        "subChatType": chat_type
                                    }
                                },
                                "sub_chat_type": chat_type,
                                "parent_id": None
                            }
                        ]
                    }
                    if aspect_ratio:
                        msg_payload["size"] = aspect_ratio

                    async with session.post(
                            f'{cls.url}/api/v2/chat/completions?chat_id={conversation.chat_id}', json=msg_payload,
                            headers=req_headers, proxy=proxy, timeout=timeout, cookies=conversation.cookies
                    ) as resp:
                        first_line = await resp.content.readline()
                        line_str = first_line.decode().strip()
                        if line_str.startswith('{'):
                            data = json.loads(line_str)
                            if data.get("data", {}).get("code"):
                                raise RuntimeError(f"Response: {data}")
                            conversation.parent_id = data.get("response.created", {}).get("response_id")
                            yield conversation

                        thinking_started = False
                        usage = None
                        async for chunk in sse_stream(resp):
                            try:
                                error = chunk.get("error", {})
                                if error:
                                    raise ResponseError(f'{error["code"]}: {error["details"]}')
                                usage = chunk.get("usage", usage)
                                choices = chunk.get("choices", [])
                                if not choices: continue
                                delta = choices[0].get("delta", {})
                                phase = delta.get("phase")
                                content = delta.get("content")
                                status = delta.get("status")
                                extra = delta.get("extra", {})
                                if phase == "think" and not thinking_started:
                                    thinking_started = True
                                elif phase == "answer" and thinking_started:
                                    thinking_started = False
                                elif phase == "image_gen" and status == "typing":
                                    yield ImageResponse(content, prompt, extra)
                                    continue
                                elif phase == "image_gen" and status == "finished":
                                    yield FinishReason("stop")
                                if content:
                                    yield Reasoning(content) if thinking_started else content
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
                        if usage:
                            yield Usage(**usage)
                        return

                except (aiohttp.ClientResponseError, RuntimeError) as e:
                    is_rate_limit = (isinstance(e, aiohttp.ClientResponseError) and e.status == 429) or \
                                    ("RateLimited" in str(e))
                    if is_rate_limit:
                        debug.log(
                            f"[Qwen] WARNING: Rate limit detected (attempt {attempt + 1}/5). Invalidating current midtoken.")
                        cls._midtoken = None
                        cls._midtoken_uses = 0
                        conversation = None
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise e

            raise RateLimitError("The Qwen provider reached the request limit after 5 attempts.")
