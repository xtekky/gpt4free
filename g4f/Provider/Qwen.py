from __future__ import annotations

import asyncio
import datetime
import hashlib
import hmac
import json
import re
import uuid
from time import time
from typing import Literal, Optional, Dict
from urllib.parse import quote

import aiohttp

from g4f.image import to_bytes, detect_file_type
from g4f.requests import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message
from .. import debug
from ..errors import RateLimitError, ResponseError
from ..providers.response import JsonConversation, Reasoning, Usage, ImageResponse, FinishReason
from ..requests import sse_stream
from ..requests.aiohttp import StreamSession
from ..tools.media import merge_media
from ..typing import AsyncResult, Messages, MediaListType

try:
    import curl_cffi

    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

# Global variables to manage Yupp Image Cache
ImagesCache: Dict[str, dict] = {}


def get_oss_headers(method: str, date_str: str, sts_data: dict, content_type: str) -> dict[str, str]:
    bucket_name = sts_data.get('bucketname', 'qwen-webui-prod')
    file_path = sts_data.get('file_path', '')
    access_key_id = sts_data.get('access_key_id')
    access_key_secret = sts_data.get('access_key_secret')
    security_token = sts_data.get('security_token')
    headers = {
        'Content-Type': content_type,
        'x-oss-content-sha256': 'UNSIGNED-PAYLOAD',
        'x-oss-date': date_str,
        'x-oss-security-token': security_token,
        'x-oss-user-agent': 'aliyun-sdk-js/6.23.0 Chrome 132.0.0.0 on Windows 10 64-bit'
    }
    headers_lower = {k.lower(): v for k, v in headers.items()}

    canonical_headers_list = []
    signed_headers_list = []
    required_headers = ['content-md5', 'content-type', 'x-oss-content-sha256', 'x-oss-date', 'x-oss-security-token',
                        'x-oss-user-agent']
    for header_name in sorted(required_headers):
        if header_name in headers_lower:
            canonical_headers_list.append(f"{header_name}:{headers_lower[header_name]}")
            signed_headers_list.append(header_name)

    canonical_headers = '\n'.join(canonical_headers_list) + '\n'
    canonical_uri = f"/{bucket_name}/{quote(file_path, safe='/')}"

    canonical_request = f"{method}\n{canonical_uri}\n\n{canonical_headers}\n\nUNSIGNED-PAYLOAD"

    date_parts = date_str.split('T')
    date_scope = f"{date_parts[0]}/ap-southeast-1/oss/aliyun_v4_request"
    string_to_sign = f"OSS4-HMAC-SHA256\n{date_str}\n{date_scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"

    def sign(key, msg):
        return hmac.new(key, msg.encode() if isinstance(msg, str) else msg, hashlib.sha256).digest()

    date_key = sign(f"aliyun_v4{access_key_secret}".encode(), date_parts[0])
    region_key = sign(date_key, "ap-southeast-1")
    service_key = sign(region_key, "oss")
    signing_key = sign(service_key, "aliyun_v4_request")
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    headers['authorization'] = f"OSS4-HMAC-SHA256 Credential={access_key_id}/{date_scope},Signature={signature}"
    return headers


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
    image_cache = True
    _models_loaded = True
    image_models = image_models
    text_models = text_models
    vision_models = vision_models
    models: list = models
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
    async def prepare_files(cls, media, session: aiohttp.ClientSession, headers=None) -> list:
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
                    f'{cls.url}/api/v2/files/getstsToken',
                    json={"filename": file_name,
                          "filesize": file_size, "filetype": file_type},
                    headers=headers

            ) as r:
                await raise_for_status(r, "Create file failed")
                res_data = await r.json()
                data = res_data.get("data")

                if res_data["success"] is False:
                    raise RateLimitError(f"{data['code']}:{data['details']}")
                file_url = data.get("file_url")
                file_id = data.get("file_id")

            # Put File into Url
            str_date = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%SZ')
            headers = get_oss_headers('PUT', str_date, data, file_type)
            async with session.put(
                    file_url.split("?")[0],
                    data=data_bytes,
                    headers=headers
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
                        "content_type": file_type
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
                "uploadTaskId": str(uuid.uuid4())
            }
            ImagesCache[image_hash] = file
            files.append(file)
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
        token = kwargs.get("token")
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
            'Authorization': f'Bearer {token}' if token else "Bearer",
            'Source': 'web'
        }

        prompt = get_last_user_message(messages)

        async with StreamSession(headers=headers) as session:
            try:
                async with session.get('https://chat.qwen.ai/api/v1/auths/', proxy=proxy) as user_info_res:
                    user_info_res.raise_for_status()
                    debug.log(await user_info_res.json())
            except:
                ...
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
                        files = await cls.prepare_files(media, session=session,
                                                        headers=req_headers)

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
