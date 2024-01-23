from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin


class Llama2(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.llama2.ai"
    working = True
    supports_message_history = True
    default_model = "meta/llama-2-70b-chat"
    models = [
        "meta/llama-2-7b-chat",
        "meta/llama-2-13b-chat",
        "meta/llama-2-70b-chat",
    ]
    model_aliases = {
        "meta-llama/Llama-2-7b-chat-hf": "meta/llama-2-7b-chat",
        "meta-llama/Llama-2-13b-chat-hf": "meta/llama-2-13b-chat",
        "meta-llama/Llama-2-70b-chat-hf": "meta/llama-2-70b-chat",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/",
            "Content-Type": "text/plain;charset=UTF-8",
            "Origin": cls.url,
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "TE": "trailers"
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "prompt": prompt,
                "model": cls.get_model(model),
                "systemPrompt": kwargs.get("system_message", "You are a helpful assistant."),
                "temperature": kwargs.get("temperature", 0.75),
                "topP": kwargs.get("top_p", 0.9),
                "maxTokens": kwargs.get("max_tokens", 8000),
                "image": None
            }
            started = False
            async with session.post(f"{cls.url}/api", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    if not started:
                        chunk = chunk.lstrip()
                        started = True
                    yield chunk.decode(errors="ignore")
            
def format_prompt(messages: Messages):
    messages = [
        f"[INST] {message['content']} [/INST]"
        if message["role"] == "user"
        else message["content"]
        for message in messages
    ]
    return "\n".join(messages) + "\n"
