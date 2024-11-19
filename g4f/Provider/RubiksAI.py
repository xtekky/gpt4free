from __future__ import annotations

import asyncio
import aiohttp
import random
import string
import json
from urllib.parse import urlencode

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class RubiksAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Rubiks AI"
    url = "https://rubiks.ai"
    api_endpoint = "https://rubiks.ai/search/api.php"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = 'llama-3.1-70b-versatile'
    models = [default_model, 'gpt-4o-mini']

    model_aliases = {
        "llama-3.1-70b": "llama-3.1-70b-versatile",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @staticmethod
    def generate_mid() -> str:
        """
        Generates a 'mid' string following the pattern:
        6 characters - 4 characters - 4 characters - 4 characters - 12 characters
        Example: 0r7v7b-quw4-kdy3-rvdu-ekief6xbuuq4
        """
        parts = [
            ''.join(random.choices(string.ascii_lowercase + string.digits, k=6)),
            ''.join(random.choices(string.ascii_lowercase + string.digits, k=4)),
            ''.join(random.choices(string.ascii_lowercase + string.digits, k=4)),
            ''.join(random.choices(string.ascii_lowercase + string.digits, k=4)),
            ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        ]
        return '-'.join(parts)

    @staticmethod
    def create_referer(q: str, mid: str, model: str = '') -> str:
        """
        Creates a Referer URL with dynamic q and mid values, using urlencode for safe parameter encoding.
        """
        params = {'q': q, 'model': model, 'mid': mid}
        encoded_params = urlencode(params)
        return f'https://rubiks.ai/search/?{encoded_params}'

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        websearch: bool = False,
        **kwargs
    ) -> AsyncResult:
        """
        Creates an asynchronous generator that sends requests to the Rubiks AI API and yields the response.

        Parameters:
        - model (str): The model to use in the request.
        - messages (Messages): The messages to send as a prompt.
        - proxy (str, optional): Proxy URL, if needed.
        - websearch (bool, optional): Indicates whether to include search sources in the response. Defaults to False.
        """
        model = cls.get_model(model)
        prompt = format_prompt(messages)
        q_value = prompt
        mid_value = cls.generate_mid()
        referer = cls.create_referer(q=q_value, mid=mid_value, model=model)

        url = cls.api_endpoint
        params = {
            'q': q_value,
            'model': model,
            'id': '',
            'mid': mid_value
        }

        headers = {
            'Accept': 'text/event-stream',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Pragma': 'no-cache',
            'Referer': referer,
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Chromium";v="129", "Not=A?Brand";v="8"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"'
        }

        try:
            timeout = aiohttp.ClientTimeout(total=None)
            async with ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, params=params, proxy=proxy) as response:
                    if response.status != 200:
                        yield f"Request ended with status code {response.status}"
                        return

                    assistant_text = ''
                    sources = []

                    async for line in response.content:
                        decoded_line = line.decode('utf-8').strip()
                        if not decoded_line.startswith('data: '):
                            continue
                        data = decoded_line[6:]
                        if data in ('[DONE]', '{"done": ""}'):
                            break
                        try:
                            json_data = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if 'url' in json_data and 'title' in json_data:
                            if websearch:
                                sources.append({'title': json_data['title'], 'url': json_data['url']})

                        elif 'choices' in json_data:
                            for choice in json_data['choices']:
                                delta = choice.get('delta', {})
                                content = delta.get('content', '')
                                role = delta.get('role', '')
                                if role == 'assistant':
                                    continue
                                assistant_text += content

                    if websearch and sources:
                        sources_text = '\n'.join([f"{i+1}. [{s['title']}]: {s['url']}" for i, s in enumerate(sources)])
                        assistant_text += f"\n\n**Source:**\n{sources_text}"

                    yield assistant_text

        except asyncio.CancelledError:
            yield "The request was cancelled."
        except aiohttp.ClientError as e:
            yield f"An error occurred during the request: {e}"
        except Exception as e:
            yield f"An unexpected error occurred: {e}"
