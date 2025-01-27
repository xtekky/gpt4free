from __future__ import annotations

import time
from aiohttp import ClientSession, ClientTimeout
import json
import asyncio
import random

from ..typing import AsyncResult, Messages, Cookies
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..providers.response import FinishReason, JsonConversation
from ..errors import ModelNotSupportedError, ResponseStatusError, RateLimitError, TimeoutError, ConversationLimitError

class DuckDuckGoSearchException(Exception):
    """Base exception class for duckduckgo_search."""

class Conversation(JsonConversation):
    vqd: str = None
    message_history: Messages = []
    cookies: dict = {}

    def __init__(self, model: str):
        self.model = model

class DDG(AsyncGeneratorProvider, ProviderModelMixin):
    label = "DuckDuckGo AI Chat"
    url = "https://duckduckgo.com/aichat"
    api_endpoint = "https://duckduckgo.com/duckchat/v1/chat"
    status_url = "https://duckduckgo.com/duckchat/v1/status"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "gpt-4o-mini"
    models = [default_model, "claude-3-haiku-20240307", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

    model_aliases = {
        "gpt-4": "gpt-4o-mini",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }

    last_request_time = 0

    @classmethod 
    def validate_model(cls, model: str) -> str:
        """Validates and returns the correct model name"""
        if not model:
            return cls.default_model
        if model in cls.model_aliases:
            model = cls.model_aliases[model]
        if model not in cls.models:
            raise ModelNotSupportedError(f"Model {model} not supported. Available models: {cls.models}")
        return model

    @classmethod
    async def sleep(cls):
        """Implements rate limiting between requests"""
        now = time.time()
        if cls.last_request_time > 0:
            delay = max(0.0, 0.75 - (now - cls.last_request_time))
            if delay > 0:
                await asyncio.sleep(delay)
        cls.last_request_time = now

    @classmethod
    async def fetch_vqd(cls, session: ClientSession, max_retries: int = 3) -> str:
        """Fetches the required VQD token for the chat session with retries."""
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json", 
            "x-vqd-accept": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        for attempt in range(max_retries):
            try:
                await cls.sleep()
                async with session.get(cls.status_url, headers=headers) as response:
                    await raise_for_status(response)
                    vqd = response.headers.get("x-vqd-4", "")
                    if vqd:
                        return vqd
                    response_text = await response.text()
                    raise RuntimeError(f"Failed to fetch VQD token: {response.status} {response_text}")
            except ResponseStatusError as e:
                if attempt < max_retries - 1:
                    wait_time = random.uniform(1, 3) * (attempt + 1)
                    await asyncio.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to fetch VQD token after {max_retries} attempts: {str(e)}")

        raise RuntimeError("Failed to fetch VQD token: Maximum retries exceeded")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 30,
        cookies: Cookies = None,
        conversation: Conversation = None,
        return_conversation: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.validate_model(model)

        if cookies is None and conversation is not None:
            cookies = conversation.cookies

        try:
            async with ClientSession(timeout=ClientTimeout(total=timeout), cookies=cookies) as session:
                if conversation is None:
                    conversation = Conversation(model)
                    conversation.vqd = await cls.fetch_vqd(session)
                    conversation.message_history = [{"role": "user", "content": format_prompt(messages)}]
                else:
                    conversation.message_history.append(messages[-1])

                headers = {
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "x-vqd-4": conversation.vqd,
                }

                data = {
                    "model": model,
                    "messages": conversation.message_history,
                }

                await cls.sleep()
                async with session.post(cls.api_endpoint, json=data, headers=headers, proxy=proxy) as response:
                    await raise_for_status(response)
                    reason = None
                    full_message = ""

                    async for line in response.content:
                        line = line.decode("utf-8").strip()

                        if line.startswith("data:"):
                            try:
                                message = json.loads(line[5:].strip())
                            except json.JSONDecodeError:
                                continue

                            if "action" in message and message["action"] == "error":
                                error_type = message.get("type", "")
                                if message.get("status") == 429:
                                    if error_type == "ERR_CONVERSATION_LIMIT":
                                        raise ConversationLimitError(error_type)
                                    raise RateLimitError(error_type)
                                raise DuckDuckGoSearchException(error_type)

                            if "message" in message:
                                if message["message"]:
                                    yield message["message"]
                                    full_message += message["message"]
                                    reason = "length"
                                else:
                                    reason = "stop"

                    if return_conversation:
                        conversation.message_history.append({"role": "assistant", "content": full_message})
                        conversation.vqd = response.headers.get("x-vqd-4", conversation.vqd)
                        conversation.cookies = {
                            n: c.value 
                            for n, c in session.cookie_jar.filter_cookies(cls.url).items()
                        }

                    if reason is not None:
                        yield FinishReason(reason)
    
                    if return_conversation:
                        yield conversation

        except asyncio.TimeoutError as e:
            raise TimeoutError(f"Request timed out: {str(e)}")