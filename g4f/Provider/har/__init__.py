from __future__ import annotations

import os
import json
import uuid
from urllib.parse import urlparse

from ...typing import AsyncResult, Messages
from ...requests import StreamSession, raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_last_user_message
from ..openai.har_file import get_headers

class HarProvider(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://lmarena.ai"
    working = True

    @classmethod
    def get_models(cls):
        for harFile in read_har_files():
            for v in harFile['log']['entries']:
                request_url = v['request']['url']
                if not request_url.startswith(cls.url) or "." in urlparse(request_url).path or "heartbeat" in request_url:
                    continue
                if "\n\ndata: " not in v['response']['content']['text']:
                    continue
                chunk = v['response']['content']['text'].split("\n\ndata: ")[2]
                cls.models = list(dict.fromkeys(get_str_list(find_list(json.loads(chunk), 'choices'))).keys())
                if cls.models:
                    break
        return cls.models

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if model in cls.model_aliases:
            model = cls.model_aliases[model]
        session_hash = str(uuid.uuid4()).replace("-", "")
        prompt = get_last_user_message(messages)

        for harFile in read_har_files():
            async with StreamSession(impersonate="chrome") as session:
                for v in harFile['log']['entries']:
                    request_url = v['request']['url']
                    if not request_url.startswith(cls.url) or "." in urlparse(request_url).path or "heartbeat" in request_url:
                        continue
                    postData = None
                    if "postData" in v['request']:
                        postData = v['request']['postData']['text']
                        postData = postData.replace('"hello"', json.dumps(prompt))
                        postData = postData.replace("__SESSION__", session_hash)
                        if model:
                            postData = postData.replace("__MODEL__", model)
                    request_url = request_url.replace("__SESSION__", session_hash)
                    method = v['request']['method'].lower()

                    async with getattr(session, method)(request_url, data=postData, headers=get_headers(v), proxy=proxy) as response:
                        await raise_for_status(response)
                        if "heartbeat" in request_url:
                            continue
                        returned_data = ""
                        async for line in response.iter_lines():
                            if not line.startswith(b"data: "):
                                continue
                            for content in find_str(json.loads(line[6:]), 3):
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

def read_har_files():
    for root, _, files in os.walk(os.path.dirname(__file__)):
        for file in files:
            if not file.endswith(".har"):
                continue
            with open(os.path.join(root, file), 'rb') as file:
                try:
                    yield json.loads(file.read())
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

def find_str(data, skip=0):
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
                print(k, v)
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
            for sub_item in item:
                yield sub_item

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