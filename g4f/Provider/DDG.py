from __future__ import annotations

import asyncio
from aiohttp import ClientSession, ClientTimeout, ClientError, ClientResponseError
import json

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin, BaseConversation
from ..providers.response import FinishReason
from .. import debug

class Conversation(BaseConversation):
    vqd: str = None
    message_history: Messages = []
    cookies: dict = {}

    def __init__(self, model: str):
        self.model = model

class DDG(AsyncGeneratorProvider, ProviderModelMixin):
    label = "DuckDuckGo AI Chat"
    url = "https://duckduckgo.com/aichat"
    api_endpoint = "https://duckduckgo.com/duckchat/v1/chat"

    working = True
    needs_auth = False
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

    @classmethod
    async def fetch_vqd(cls, session: ClientSession) -> str:
        """
        Fetches the required VQD token for the chat session.

        Args:
            session (ClientSession): The active HTTP session.

        Returns:
            str: The VQD token.

        Raises:
            Exception: If the token cannot be fetched.
        """
        async with session.get("https://duckduckgo.com/duckchat/v1/status", headers={"x-vqd-accept": "1"}) as response:
            if response.status == 200:
                vqd = response.headers.get("x-vqd-4", "")
                if not vqd:
                    raise Exception("Failed to fetch VQD token: Empty token.")
                return vqd
            else:
                raise Exception(f"Failed to fetch VQD token: {response.status} {await response.text()}")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        conversation: Conversation = None,
        return_conversation: bool = False,
        proxy: str = None,
        headers: dict = {
            "Content-Type": "application/json",
        },
        cookies: dict = None,
        max_retries: int = 0,
        **kwargs
    ) -> AsyncResult:
        if cookies is None and conversation is not None:
            cookies = conversation.cookies
        async with ClientSession(headers=headers, cookies=cookies, timeout=ClientTimeout(total=30)) as session:
            # Fetch VQD token
            if conversation is None:
                conversation = Conversation(model)
                conversation.cookies = session.cookie_jar
                conversation.vqd = await cls.fetch_vqd(session)

            if conversation.vqd is not None:
                headers["x-vqd-4"] = conversation.vqd

            if return_conversation:
                yield conversation

            if len(messages) >= 2:
                conversation.message_history.extend([messages[-2], messages[-1]])
            elif len(messages) == 1:
                conversation.message_history.append(messages[-1])

            payload = {
                "model": conversation.model,
                "messages": conversation.message_history,
            }

            try:
                async with session.post(cls.api_endpoint, headers=headers, json=payload, proxy=proxy) as response:
                    conversation.vqd = response.headers.get("x-vqd-4")
                    response.raise_for_status()
                    reason = None
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data:"):
                            try:
                                message = json.loads(line[5:].strip())
                                if "message" in message and message["message"]:
                                    yield message["message"] 
                                    reason = "max_tokens"
                                elif message.get("message") == '':
                                    reason = "stop"
                            except json.JSONDecodeError:
                                continue
                    if reason is not None:
                        yield FinishReason(reason)
            except ClientResponseError as e:
                if e.code in (400, 429) and max_retries > 0:
                    debug.log(f"Retry: max_retries={max_retries}, wait={512 - max_retries * 48}: {e}")
                    await asyncio.sleep(512 - max_retries * 48)
                    is_started = False
                    async for chunk in cls.create_async_generator(model, messages, conversation, return_conversation, max_retries=max_retries-1, **kwargs):
                        if chunk:
                            yield chunk
                            is_started = True
                    if is_started:
                        return
                raise e
            except ClientError as e:
                raise Exception(f"HTTP ClientError occurred: {e}")
            except asyncio.TimeoutError:
                raise Exception("Request timed out.")
