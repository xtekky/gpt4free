from __future__ import annotations

import os
import json
import uuid
import random
from urllib.parse import urlparse

from ...typing import AsyncResult, Messages, MediaListType
from ...requests import DEFAULT_HEADERS, StreamSession, StreamResponse, FormData, raise_for_status
from ...providers.response import JsonConversation
from ...tools.media import merge_media
from ...image import to_bytes, is_accepted_format
from ...errors import ResponseError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_last_user_message
from ..openai.har_file import get_headers
from ..LegacyLMArena import LegacyLMArena

class HarProvider(AsyncGeneratorProvider, ProviderModelMixin):
    label = "LMArena (Har)"
    url = "https://legacy.lmarena.ai"
    api_endpoint = "/queue/join?"
    working = True
    default_model = "chatgpt-4o-latest-20250326"
    model_aliases = LegacyLMArena.model_aliases
    vision_models = [
		"o3-2025-04-16",
		"o4-mini-2025-04-16",
		"gpt-4.1-2025-04-14",
		"gemini-2.5-pro-exp-03-25",
		"claude-3-7-sonnet-20250219",
		"claude-3-7-sonnet-20250219-thinking-32k",
		"llama-4-maverick-17b-128e-instruct",
		"gpt-4.1-mini-2025-04-14",
		"gpt-4.1-nano-2025-04-14",
		"gemini-2.0-flash-thinking-exp-01-21",
		"gemini-2.0-flash-001",
		"gemini-2.0-flash-lite-preview-02-05",
		"claude-3-5-sonnet-20241022",
		"gpt-4o-mini-2024-07-18",
		"gpt-4o-2024-11-20",
		"gpt-4o-2024-08-06",
		"gpt-4o-2024-05-13",
		"claude-3-5-sonnet-20240620",
		"doubao-1.5-vision-pro-32k-250115",
		"amazon-nova-pro-v1.0",
		"amazon-nova-lite-v1.0",
		"qwen2.5-vl-32b-instruct",
		"qwen2.5-vl-72b-instruct",
		"gemini-1.5-pro-002",
		"gemini-1.5-flash-002",
		"gemini-1.5-flash-8b-001",
		"gemini-1.5-pro-001",
		"gemini-1.5-flash-001",
		"hunyuan-standard-vision-2024-12-31",
		"pixtral-large-2411",
		"step-1o-vision-32k-highres",
		"claude-3-haiku-20240307",
		"claude-3-sonnet-20240229",
		"claude-3-opus-20240229",
		"qwen-vl-max-1119",
		"qwen-vl-max-0809",
		"reka-core-20240904",
		"reka-flash-20240904",
		"c4ai-aya-vision-32b",
		"pixtral-12b-2409"
	]

    @classmethod
    def get_models(cls):
        for domain, harFile in read_har_files():
            for v in harFile['log']['entries']:
                request_url = v['request']['url']
                if domain not in request_url or "." in urlparse(request_url).path or "heartbeat" in request_url:
                    continue
                if "\n\ndata: " not in v['response']['content']['text']:
                    continue
                chunk = v['response']['content']['text'].split("\n\ndata: ")[2]
                cls.models = list(dict.fromkeys(get_str_list(find_list(json.loads(chunk), 'choices'))).keys())
                cls.models[0] = cls.default_model
                if cls.models:
                    break
        return cls.models

    @classmethod
    def _build_second_payloads(cls, model_id: str, session_hash: str, text: str, max_tokens: int, temperature: float, top_p: float):
        first_payload = {
            "data":[None,model_id,text,{
                "text_models":[model_id],
                "all_text_models":[model_id],
                "vision_models":[],
                "image_gen_models":[],
                "all_image_gen_models":[],
                "search_models":[],
                "all_search_models":[],
                "models":[model_id],
                "all_models":[model_id],
                "arena_type":"text-arena"}],
            "event_data": None,
            "fn_index": 122,
            "trigger_id": 157,
            "session_hash": session_hash
        }

        second_payload = {
            "data": [],
            "event_data": None,
            "fn_index": 123,
            "trigger_id": 157,
            "session_hash": session_hash
        }

        third_payload = {
            "data": [None, temperature, top_p, max_tokens],
            "event_data": None,
            "fn_index": 124,
            "trigger_id": 157,
            "session_hash": session_hash
        }

        return first_payload, second_payload, third_payload

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        media: MediaListType = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1,
        conversation: JsonConversation = None,
        **kwargs
    ) -> AsyncResult:
        async def read_response(response: StreamResponse):
            returned_data = ""
            async for line in response.iter_lines():
                if not line.startswith(b"data: "):
                    continue
                for content in find_str(json.loads(line[6:]), 3):
                    if "**NETWORK ERROR DUE TO HIGH TRAFFIC." in content:
                        raise ResponseError(content)
                    if content == '<span class="cursor"></span> ' or content == 'update':
                        continue
                    if content.endswith("▌"):
                        content = content[:-2]
                    new_content = content
                    if content.startswith(returned_data):
                        new_content = content[len(returned_data):]
                    if not new_content:
                        continue
                    returned_data += new_content
                    yield new_content
        if model in cls.model_aliases:
            model = cls.model_aliases[model]
        if isinstance(model, list):
            model = random.choice(model)
        prompt = get_last_user_message(messages)
        async with StreamSession(impersonate="chrome") as session:
            if conversation is None:
                conversation = JsonConversation(session_hash=str(uuid.uuid4()).replace("-", ""))
                media = list(merge_media(media, messages))
                if media:
                    data = FormData()
                    for i in range(len(media)):
                        media[i] = (to_bytes(media[i][0]), media[i][1])
                    for image, image_name in media:
                        data.add_field(f"files", image, filename=image_name)
                    async with session.post(f"{cls.url}/upload", params={"upload_id": conversation.session_hash}, data=data) as response:
                        await raise_for_status(response)
                        image_files = await response.json()
                    media = [{
                        "path": image_file,
                        "url": f"{cls.url}/file={image_file}",
                        "orig_name": media[i][1],
                        "size": len(media[i][0]),
                        "mime_type": is_accepted_format(media[i][0]),
                        "meta": {
                            "_type": "gradio.FileData"
                        }
                    } for i, image_file in enumerate(image_files)]
                for domain, harFile in read_har_files():
                    for v in harFile['log']['entries']:
                        request_url = v['request']['url']
                        if domain not in request_url or "." in urlparse(request_url).path or "heartbeat" in request_url:
                            continue
                        postData = None
                        if "postData" in v['request']:
                            postData = v['request']['postData']['text']
                            postData = postData.replace('"hello"', json.dumps(prompt))
                            postData = postData.replace('[null,0.7,1,2048]', json.dumps([None, temperature, top_p, max_tokens]))
                            postData = postData.replace('"files":[]', f'"files":{json.dumps(media)}')
                            postData = postData.replace("__SESSION__", conversation.session_hash)
                            if model:
                                postData = postData.replace("__MODEL__", model)
                        request_url = request_url.replace("__SESSION__", conversation.session_hash)
                        method = v['request']['method'].lower()
                        async with getattr(session, method)(request_url, data=postData, headers={**get_headers(v), **DEFAULT_HEADERS}, proxy=proxy) as response:
                            await raise_for_status(response)
                            async for chunk in read_response(response):
                                yield chunk
                yield conversation
            else:
                first_payload, second_payload, third_payload = cls._build_second_payloads(model, conversation.session_hash, prompt, max_tokens, temperature, top_p)
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                # POST 1
                async with session.post(f"{cls.url}{cls.api_endpoint}", json=first_payload, proxy=proxy, headers=headers) as response:
                    await raise_for_status(response)
                # POST 2
                async with session.post(f"{cls.url}{cls.api_endpoint}", json=second_payload, proxy=proxy, headers=headers) as response:
                    await raise_for_status(response)
                # POST 3
                async with session.post(f"{cls.url}{cls.api_endpoint}", json=third_payload, proxy=proxy, headers=headers) as response:
                    await raise_for_status(response)
                stream_url = f"{cls.url}/queue/data?session_hash={conversation.session_hash}"
                async with session.get(stream_url, headers={"Accept": "text/event-stream"}, proxy=proxy) as response:
                    await raise_for_status(response)
                    async for chunk in read_response(response):
                        yield chunk

def read_har_files():
    for root, _, files in os.walk(os.path.dirname(__file__)):
        for file in files:
            if not file.endswith(".har"):
                continue
            with open(os.path.join(root, file), 'rb') as f:
                try:
                    yield os.path.splitext(file)[0], json.load(f)
                except json.JSONDecodeError:
                    raise RuntimeError(f"Failed to read HAR file: {file}")

def read_str_recusive(data):
    if isinstance(data, dict):
        data = data.values()
    for item in data:
        if isinstance(item, (list, dict)):
            yield from read_str_recusive(item)
        elif isinstance(item, str):
            yield item

def find_str(data, skip: int = 0):
    for item in read_str_recusive(data):
        if skip > 0:
            skip -= 1
            continue
        yield item
        break

def read_list_recusive(data, key):
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                yield v
            else:
                yield from read_list_recusive(v, key)
    elif isinstance(data, list):
        for item in data:
            yield from read_list_recusive(item, key)

def find_list(data, key):
    for item in read_list_recusive(data, key):
        if isinstance(item, str):
            yield item
        elif isinstance(item, list):
            yield from item

def get_str_list(data):
    for item in data:
        if isinstance(item, list):
            yield from get_str_list(item)
        else:
            yield item

# with open("g4f/Provider/har/lmarena.ai.har", "r") as f:
#     try:
#         harFile = json.loads(f.read())
#     except json.JSONDecodeError:
#         raise RuntimeError(f"Failed to read HAR file")

# new_entries = []
# for v in harFile['log']['entries']:
#     request_url = v['request']['url']
#     if not request_url.startswith("https://lmarena.ai") or "." in urlparse(request_url).path or "heartbeat" in request_url:
#         continue
#     v['request']['cookies'] = []
#     v['request']['headers'] = [header for header in v['request']['headers'] if header['name'].lower() != "cookie"]
#     v['response']['headers'] = []
#     new_entries.append(v)
#     print(f"Request URL: {request_url}"
