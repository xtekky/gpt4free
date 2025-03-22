from __future__ import annotations

import time
from aiohttp import ClientSession, ClientTimeout
import json
import asyncio
import random
from yarl import URL

from ..typing import AsyncResult, Messages, Cookies
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt, get_last_user_message
from ..providers.response import FinishReason, JsonConversation
from ..errors import ModelNotSupportedError, ResponseStatusError, RateLimitError, TimeoutError, ConversationLimitError


class DuckDuckGoSearchException(Exception):
    """Base exception class for duckduckgo_search."""

class Conversation(JsonConversation):
    vqd: str = None
    vqd_hash_1: str = None
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
    models = [default_model, "meta-llama/Llama-3.3-70B-Instruct-Turbo", "claude-3-haiku-20240307", "o3-mini", "mistralai/Mistral-Small-24B-Instruct-2501"]

    model_aliases = {
        "gpt-4": "gpt-4o-mini",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "mixtral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    }

    last_request_time = 0
    max_retries = 3
    base_delay = 2

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
    async def sleep(cls, multiplier=1.0):
        """Implements rate limiting between requests"""
        now = time.time()
        if cls.last_request_time > 0:
            delay = max(0.0, 1.5 - (now - cls.last_request_time)) * multiplier
            if delay > 0:
                await asyncio.sleep(delay)
        cls.last_request_time = time.time()

    @classmethod
    async def get_default_cookies(cls, session: ClientSession) -> dict:
        """Obtains default cookies needed for API requests"""
        try:
            await cls.sleep()
            # Make initial request to get cookies
            async with session.get(cls.url) as response:
                # We also manually set required cookies
                cookies = {}
                cookies_dict = {'dcs': '1', 'dcm': '3'}
                
                for name, value in cookies_dict.items():
                    cookies[name] = value
                    url_obj = URL(cls.url)
                    session.cookie_jar.update_cookies({name: value}, url_obj)
                
                return cookies
        except Exception as e:
            return {}

    @classmethod
    async def fetch_vqd_and_hash(cls, session: ClientSession, retry_count: int = 0) -> tuple[str, str]:
        """Fetches the required VQD token and hash for the chat session with retries."""
        headers = {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json", 
            "pragma": "no-cache",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "origin": "https://duckduckgo.com",
            "referer": "https://duckduckgo.com/",
            "x-vqd-accept": "1",
        }

        # Make sure we have cookies first
        if len(session.cookie_jar) == 0:
            await cls.get_default_cookies(session)

        try:
            await cls.sleep(multiplier=1.0 + retry_count * 0.5)
            async with session.get(cls.status_url, headers=headers) as response:
                await raise_for_status(response)
                
                vqd = response.headers.get("x-vqd-4", "")
                vqd_hash_1 = response.headers.get("x-vqd-hash-1", "")
                
                if vqd and vqd_hash_1:
                    return vqd, vqd_hash_1
                
                if vqd and not vqd_hash_1:
                    return vqd, ""
                
                response_text = await response.text()
                raise RuntimeError(f"Failed to fetch VQD token and hash: {response.status} {response_text}")
                
        except Exception as e:
            if retry_count < cls.max_retries:
                wait_time = cls.base_delay * (2 ** retry_count) * (1 + random.random())
                await asyncio.sleep(wait_time)
                return await cls.fetch_vqd_and_hash(session, retry_count + 1)
            else:
                raise RuntimeError(f"Failed to fetch VQD token and hash after {cls.max_retries} attempts: {str(e)}")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 60,
        cookies: Cookies = None,
        conversation: Conversation = None,
        return_conversation: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.validate_model(model)
        retry_count = 0

        while retry_count <= cls.max_retries:
            try:
                session_timeout = ClientTimeout(total=timeout)
                async with ClientSession(timeout=session_timeout, cookies=cookies) as session:
                    if conversation is None:
                        # Get initial cookies if not provided
                        if not cookies:
                            await cls.get_default_cookies(session)
                        
                        conversation = Conversation(model)
                        vqd, vqd_hash_1 = await cls.fetch_vqd_and_hash(session)
                        conversation.vqd = vqd
                        conversation.vqd_hash_1 = vqd_hash_1
                        conversation.message_history = [{"role": "user", "content": format_prompt(messages)}]
                    else:
                        last_message = get_last_user_message(messages.copy())
                        conversation.message_history.append({"role": "user", "content": last_message})

                    headers = {
                        "accept": "text/event-stream",
                        "accept-language": "en-US,en;q=0.9",
                        "content-type": "application/json",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
                        "origin": "https://duckduckgo.com",
                        "referer": "https://duckduckgo.com/",
                        "x-vqd-4": conversation.vqd,
                    }
                    
                    # Add the x-vqd-hash-1 header if available
                    if conversation.vqd_hash_1:
                        headers["x-vqd-hash-1"] = conversation.vqd_hash_1

                    data = {
                        "model": model,
                        "messages": conversation.message_history,
                    }

                    await cls.sleep(multiplier=1.0 + retry_count * 0.5)
                    async with session.post(cls.api_endpoint, json=data, headers=headers, proxy=proxy) as response:
                        # Handle 429 errors specifically
                        if response.status == 429:
                            response_text = await response.text()
                            
                            if retry_count < cls.max_retries:
                                retry_count += 1
                                wait_time = cls.base_delay * (2 ** retry_count) * (1 + random.random())
                                await asyncio.sleep(wait_time)
                                
                                # Get fresh tokens and cookies
                                cookies = await cls.get_default_cookies(session)
                                continue
                            else:
                                raise RateLimitError(f"Rate limited after {cls.max_retries} retries")
                        
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
                            conversation.vqd_hash_1 = response.headers.get("x-vqd-hash-1", conversation.vqd_hash_1)
                            conversation.cookies = {
                                n: c.value 
                                for n, c in session.cookie_jar.filter_cookies(URL(cls.url)).items()
                            }
                            yield conversation

                        if reason is not None:
                            yield FinishReason(reason)
                        
                        # If we got here, the request was successful
                        break

            except (RateLimitError, ResponseStatusError) as e:
                if "429" in str(e) and retry_count < cls.max_retries:
                    retry_count += 1
                    wait_time = cls.base_delay * (2 ** retry_count) * (1 + random.random())
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except asyncio.TimeoutError as e:
                raise TimeoutError(f"Request timed out: {str(e)}")
            except Exception as e:
                raise
