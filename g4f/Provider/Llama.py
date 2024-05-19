from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin


class Llama(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.llama2.ai"
    working = False
    supports_message_history = True
    default_model = "meta/meta-llama-3-70b-instruct"
    models = [
        "meta/llama-2-7b-chat",
        "meta/llama-2-13b-chat",
        "meta/llama-2-70b-chat",
        "meta/meta-llama-3-8b-instruct",
        "meta/meta-llama-3-70b-instruct",
    ]
    model_aliases = {
        "meta-llama/Meta-Llama-3-8B-Instruct": "meta/meta-llama-3-8b-instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct": "meta/meta-llama-3-70b-instruct",
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
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.75,
        top_p: float = 0.9,
        max_tokens: int = 8000,
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
            system_messages = [message["content"] for message in messages if message["role"] == "system"]
            if system_messages:
                system_message = "\n".join(system_messages)
                messages = [message for message in messages if message["role"] != "system"] 
            prompt = format_prompt(messages)
            data = {
                "prompt": prompt,
                "model": cls.get_model(model),
                "systemPrompt": system_message,
                "temperature": temperature,
                "topP": top_p,
                "maxTokens": max_tokens,
                "image": None
            }
            started = False
            async with session.post(f"{cls.url}/api", json=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
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
