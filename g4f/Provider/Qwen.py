from __future__ import annotations

import asyncio
import json
import re
import uuid
from time import time

import aiohttp
from ..errors import RateLimitError
from ..typing import AsyncResult, Messages
from ..providers.response import JsonConversation, Reasoning, Usage
from ..requests import sse_stream
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message
from .. import debug

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

    # Complete list of models, extracted from the API
    models = [
        "qwen3-235b-a22b",
        "qwen3-coder-plus",
        "qwen3-30b-a3b",
        "qwen3-coder-30b-a3b-instruct",
        "qwen-max-latest",
        "qwen-plus-2025-01-25",
        "qwq-32b",
        "qwen-turbo-2025-02-11",
        "qwen2.5-omni-7b",
        "qvq-72b-preview-0310",
        "qwen2.5-vl-32b-instruct",
        "qwen2.5-14b-instruct-1m",
        "qwen2.5-coder-32b-instruct",
        "qwen2.5-72b-instruct",
    ]
    default_model = "qwen3-235b-a22b"

    _midtoken: str = None
    _midtoken_uses: int = 0

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        conversation: JsonConversation = None,
        proxy: str = None,
        timeout: int = 120,
        stream: bool = True,
        enable_thinking: bool = True,
        **kwargs
    ) -> AsyncResult:
        
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
                            debug.log(f"[Qwen] INFO: New midtoken obtained. Use count: {cls._midtoken_uses}. Midtoken: {cls._midtoken}")
                    else:
                        cls._midtoken_uses += 1
                        debug.log(f"[Qwen] INFO: Reusing midtoken. Use count: {cls._midtoken_uses}")

                    req_headers = session.headers.copy()
                    req_headers['bx-umidtoken'] = cls._midtoken
                    req_headers['bx-v'] = '2.5.31'
                    message_id = str(uuid.uuid4())
                    parent_id = None
                    if conversation is None:
                        chat_payload = {
                            "title": "New Chat",
                            "models": [model_name],
                            "chat_mode": "normal",
                            "chat_type": "t2t",
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
                                "files": [],
                                "models": [model_name],
                                "chat_type": "t2t",
                                "feature_config": {
                                    "thinking_enabled": enable_thinking,
                                    "output_schema": "phase",
                                    "thinking_budget": 81920
                                },
                                "extra": {
                                    "meta": {
                                        "subChatType": "t2t"
                                    }
                                },
                                "sub_chat_type": "t2t",
                                "parent_id": None
                            }
                        ]
                    }

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
                                usage = chunk.get("usage", usage)
                                choices = chunk.get("choices", [])
                                if not choices: continue
                                delta = choices[0].get("delta", {})
                                phase = delta.get("phase")
                                content = delta.get("content")
                                if phase == "think" and not thinking_started:
                                    thinking_started = True
                                elif phase == "answer" and thinking_started:
                                    thinking_started = False
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
                        debug.log(f"[Qwen] WARNING: Rate limit detected (attempt {attempt + 1}/5). Invalidating current midtoken.")
                        cls._midtoken = None
                        cls._midtoken_uses = 0
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise e

            raise RateLimitError("The Qwen provider reached the request limit after 5 attempts.")
