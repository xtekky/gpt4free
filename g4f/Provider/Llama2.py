from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

models = {
    "meta-llama/Llama-2-7b-chat-hf": {"name": "Llama 2 7B", "version": "d24902e3fa9b698cc208b5e63136c4e26e828659a9f09827ca6ec5bb83014381", "shortened":"7B"},
    "meta-llama/Llama-2-13b-chat-hf": {"name": "Llama 2 13B", "version": "9dff94b1bed5af738655d4a7cbcdcde2bd503aa85c94334fe1f42af7f3dd5ee3", "shortened":"13B"},
    "meta-llama/Llama-2-70b-chat-hf": {"name": "Llama 2 70B", "version": "2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf", "shortened":"70B"},
    "Llava": {"name": "Llava 13B", "version": "6bc1c7bb0d2a34e413301fee8f7cc728d2d4e75bfab186aa995f63292bda92fc", "shortened":"Llava"}
}

class Llama2(AsyncGeneratorProvider):
    url = "https://www.llama2.ai"
    working = True
    supports_message_history = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "meta-llama/Llama-2-70b-chat-hf"
        elif model not in models:
            raise ValueError(f"Model are not supported: {model}")
        version = models[model]["version"]
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
                "version": version,
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
                    yield chunk.decode()
            
def format_prompt(messages: Messages):
    messages = [
        f"[INST] {message['content']} [/INST]"
        if message["role"] == "user"
        else message["content"]
        for message in messages
    ]
    return "\n".join(messages) + "\n"