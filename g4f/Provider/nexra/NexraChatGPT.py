from __future__ import annotations
from aiohttp import ClientSession
from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
import json

class NexraChatGPT(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra ChatGPT"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/gpt"

    models = [
        'gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-0314', 'gpt-4-32k-0314',
        'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 
        'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-0301',
        'gpt-3', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002',
        'text-curie-001', 'text-babbage-001', 'text-ada-001',
        'davinci', 'curie', 'babbage', 'ada', 'babbage-002', 'davinci-002',
    ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Referer": f"{cls.url}/chat",
        }

        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "prompt": prompt,
                "model": model,
                "markdown": False,
                "messages": messages or [],
            }

            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()

                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    result = await response.json()
                    if result.get("status"):
                        yield result.get("gpt", "")
                    else:
                        raise Exception(f"Error in response: {result.get('message', 'Unknown error')}")
                elif 'text/plain' in content_type:
                    text = await response.text()
                    try:
                        result = json.loads(text)
                        if result.get("status"):
                            yield result.get("gpt", "")
                        else:
                            raise Exception(f"Error in response: {result.get('message', 'Unknown error')}")
                    except json.JSONDecodeError:
                        yield text  # If not JSON, return text
                else:
                    raise Exception(f"Unexpected response type: {content_type}. Response text: {await response.text()}")

