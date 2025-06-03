from __future__ import annotations

import json
import base64
import time
import random
import hashlib
import asyncio
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ...errors import ResponseError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_last_user_message
from ...providers.response import FinishReason, JsonConversation

class Conversation(JsonConversation):
    message_history: Messages = []

    def __init__(self, model: str):
        self.model = model
        self.message_history = []

class DDG(AsyncGeneratorProvider, ProviderModelMixin):
    label = "DuckDuckGo AI Chat"
    url = "https://duckduckgo.com"
    api_endpoint = "https://duckduckgo.com/duckchat/v1/chat"
    status_url = "https://duckduckgo.com/duckchat/v1/status"
    working = False
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "gpt-4o-mini"
    model_aliases = {
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "mistral-small": "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    }
    models = [default_model, "o3-mini"] + list(model_aliases.keys())

    @staticmethod
    def generate_user_agent() -> str:
        return f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.{random.randint(1000,9999)}.0 Safari/537.36"

    @staticmethod
    def generate_fe_signals() -> str:
        current_time = int(time.time() * 1000)
        signals_data = {
            "start": current_time - 35000,
            "events": [
                {"name": "onboarding_impression_1", "delta": 383},
                {"name": "onboarding_impression_2", "delta": 6004},
                {"name": "onboarding_finish", "delta": 9690},
                {"name": "startNewChat", "delta": 10082},
                {"name": "initSwitchModel", "delta": 16586}
            ],
            "end": 35163
        }
        return base64.b64encode(json.dumps(signals_data).encode()).decode()

    @staticmethod
    def generate_fe_version(page_content: str = "") -> str:
        try:
            fe_hash = page_content.split('__DDG_FE_CHAT_HASH__="', 1)[1].split('"', 1)[0]
            return f"serp_20250510_052906_ET-{fe_hash}"
        except Exception:
            return "serp_20250510_052906_ET-ed4f51dc2e106020bc4b"

    @staticmethod
    def generate_x_vqd_hash_1(vqd: str, fe_version: str) -> str:
        # Placeholder logic; in reality DuckDuckGo uses dynamic JS challenge
        concat = f"{vqd}#{fe_version}"
        hash_digest = hashlib.sha256(concat.encode()).digest()
        b64 = base64.b64encode(hash_digest).decode()
        return base64.b64encode(json.dumps({
            "server_hashes": [],
            "client_hashes": [b64],
            "signals": {},
            "meta": {
                "v": "1",
                "challenge_id": hashlib.md5(concat.encode()).hexdigest(),
                "origin": "https://duckduckgo.com",
                "stack": "Generated in Python"
            }
        }).encode()).decode()

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        conversation: Conversation = None,
        return_conversation: bool = True,
        retry_count: int = 0,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        if conversation is None:
            conversation = Conversation(model)
            conversation.message_history = messages.copy()
        else:
            last_message = next((m for m in reversed(messages) if m["role"] == "user"), None)
            if last_message and last_message not in conversation.message_history:
                conversation.message_history.append(last_message)

        base_headers = {
            "accept-language": "en-US,en;q=0.9",
            "dnt": "1",
            "origin": "https://duckduckgo.com",
            "referer": "https://duckduckgo.com/",
            "sec-ch-ua": '"Chromium";v="135", "Not-A.Brand";v="8"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": cls.generate_user_agent(),
        }
        cookies = {'dcs': '1', 'dcm': '3'}

        formatted_prompt = format_prompt(conversation.message_history) if len(conversation.message_history) > 1 else get_last_user_message(messages)
        data = {"model": model, "messages": [{"role": "user", "content": formatted_prompt}], "canUseTools": False}

        async with ClientSession(cookies=cookies) as session:
            try:
                # Step 1: Initial page load
                async with session.get(f"{cls.url}/?q=DuckDuckGo+AI+Chat&ia=chat&duckai=1",
                                       headers={**base_headers, "accept": "text/html"}, proxy=proxy) as r:
                    r.raise_for_status()
                    page = await r.text()
                    fe_version = cls.generate_fe_version(page)

                # Step 2: Get VQD
                status_headers = {**base_headers, "accept": "*/*", "cache-control": "no-store", "x-vqd-accept": "1"}
                async with session.get(cls.status_url, headers=status_headers, proxy=proxy) as r:
                    r.raise_for_status()
                    vqd = r.headers.get("x-vqd-4", "") or f"4-{random.randint(10**29, 10**30 - 1)}"

                x_vqd_hash_1 = cls.generate_x_vqd_hash_1(vqd, fe_version)

                # Step 3: Actual chat request
                chat_headers = {
                    **base_headers,
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                    "x-fe-signals": cls.generate_fe_signals(),
                    "x-fe-version": fe_version,
                    "x-vqd-4": vqd,
                    "x-vqd-hash-1": x_vqd_hash_1,
                }

                async with session.post(cls.api_endpoint, json=data, headers=chat_headers, proxy=proxy) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        if "ERR_BN_LIMIT" in error_text:
                            raise ResponseError("Blocked by DuckDuckGo: Bot limit exceeded (ERR_BN_LIMIT).")
                        if "ERR_INVALID_VQD" in error_text and retry_count < 3:
                            await asyncio.sleep(random.uniform(2.5, 5.5))
                            async for chunk in cls.create_async_generator(
                                model, messages, proxy, conversation, return_conversation, retry_count + 1, **kwargs
                            ):
                                yield chunk
                            return
                        raise ResponseError(f"HTTP {response.status} - {error_text}")
                    full_message = ""
                    async for line in response.content:
                        line_text = line.decode("utf-8").strip()
                        if line_text.startswith("data:"):
                            payload = line_text[5:].strip()
                            if payload == "[DONE]":
                                if full_message:
                                    conversation.message_history.append({"role": "assistant", "content": full_message})
                                if return_conversation:
                                    yield conversation
                                yield FinishReason("stop")
                                break
                            try:
                                msg = json.loads(payload)
                                if msg.get("action") == "error":
                                    raise ResponseError(f"Error: {msg.get('type', 'unknown')}")
                                if "message" in msg:
                                    content = msg["message"]
                                    yield content
                                    full_message += content
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                if retry_count < 3:
                    await asyncio.sleep(random.uniform(2.5, 5.5))
                    async for chunk in cls.create_async_generator(
                        model, messages, proxy, conversation, return_conversation, retry_count + 1, **kwargs
                    ):
                        yield chunk
                else:
                    raise ResponseError(f"Error: {str(e)}")
