from __future__ import annotations

import asyncio
import json
import requests
from typing import Any, Dict

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt


class NexraChatGPT(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra ChatGPT"
    url = "https://nexra.aryahcr.cc/documentation/chatgpt/en"
    api_endpoint_nexra_chatgpt = "https://nexra.aryahcr.cc/api/chat/gpt"
    api_endpoint_nexra_chatgpt4o = "https://nexra.aryahcr.cc/api/chat/complements"
    api_endpoint_nexra_chatgpt_v2 = "https://nexra.aryahcr.cc/api/chat/complements"
    api_endpoint_nexra_gptweb = "https://nexra.aryahcr.cc/api/chat/gptweb"
    working = True
    supports_system_message = True
    supports_message_history = True
    supports_stream = True
    
    default_model = 'gpt-3.5-turbo'
    nexra_chatgpt = [
        'gpt-4', 'gpt-4-0613', 'gpt-4-0314', 'gpt-4-32k-0314',
        default_model, 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-0301', 
        'text-davinci-003', 'text-davinci-002', 'code-davinci-002', 'gpt-3', 'text-curie-001', 'text-babbage-001', 'text-ada-001', 'davinci', 'curie', 'babbage', 'ada', 'babbage-002', 'davinci-002'
    ]
    nexra_chatgpt4o = ['gpt-4o']
    nexra_chatgptv2 = ['chatgpt']
    nexra_gptweb = ['gptweb']
    models = nexra_chatgpt + nexra_chatgpt4o + nexra_chatgptv2 + nexra_gptweb
    
    model_aliases = {
        "gpt-4": "gpt-4-0613",
        "gpt-4-32k": "gpt-4-32k-0314",
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613": "gpt-3.5-turbo-16k-0613",
        "gpt-3": "text-davinci-003",
        "text-davinci-002": "code-davinci-002",
        "text-curie-001": "text-babbage-001",
        "text-ada-001": "davinci",
        "curie": "babbage",
        "ada": "babbage-002",
        "davinci-002": "davinci-002",
        "chatgpt": "chatgpt",
        "gptweb": "gptweb"
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        markdown: bool = False,
        **kwargs
    ) -> AsyncResult:
        if model in cls.nexra_chatgpt:
            async for chunk in cls._create_async_generator_nexra_chatgpt(model, messages, proxy, **kwargs):
                yield chunk
        elif model in cls.nexra_chatgpt4o:
            async for chunk in cls._create_async_generator_nexra_chatgpt4o(model, messages, stream, proxy, markdown, **kwargs):
                yield chunk
        elif model in cls.nexra_chatgptv2:
            async for chunk in cls._create_async_generator_nexra_chatgpt_v2(model, messages, stream, proxy, markdown, **kwargs):
                yield chunk
        elif model in cls.nexra_gptweb:
            async for chunk in cls._create_async_generator_nexra_gptweb(model, messages, proxy, **kwargs):
                yield chunk

    @classmethod
    async def _create_async_generator_nexra_chatgpt(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        markdown: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        prompt = format_prompt(messages)
        data = {
            "messages": messages,
            "prompt": prompt,
            "model": model,
            "markdown": markdown
        }

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, cls._sync_post_request, cls.api_endpoint_nexra_chatgpt, data, headers, proxy)
            filtered_response = cls._filter_response(response)
            
            for chunk in filtered_response:
                yield chunk
        except Exception as e:
            print(f"Error during API request (nexra_chatgpt): {e}")

    @classmethod
    async def _create_async_generator_nexra_chatgpt4o(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        markdown: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        prompt = format_prompt(messages)
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": stream,
            "markdown": markdown,
            "model": model
        }

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, cls._sync_post_request, cls.api_endpoint_nexra_chatgpt4o, data, headers, proxy, stream)
            
            if stream:
                async for chunk in cls._process_streaming_response(response):
                    yield chunk
            else:
                for chunk in cls._process_non_streaming_response(response):
                    yield chunk
        except Exception as e:
            print(f"Error during API request (nexra_chatgpt4o): {e}")

    @classmethod
    async def _create_async_generator_nexra_chatgpt_v2(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        markdown: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        prompt = format_prompt(messages)
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": stream,
            "markdown": markdown,
            "model": model
        }

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, cls._sync_post_request, cls.api_endpoint_nexra_chatgpt_v2, data, headers, proxy, stream)
            
            if stream:
                async for chunk in cls._process_streaming_response(response):
                    yield chunk
            else:
                for chunk in cls._process_non_streaming_response(response):
                    yield chunk
        except Exception as e:
            print(f"Error during API request (nexra_chatgpt_v2): {e}")

    @classmethod
    async def _create_async_generator_nexra_gptweb(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        markdown: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        prompt = format_prompt(messages)
        data = {
            "prompt": prompt,
            "markdown": markdown,
        }

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, cls._sync_post_request, cls.api_endpoint_nexra_gptweb, data, headers, proxy)
            
            for chunk in response.iter_content(1024):
                if chunk:
                    decoded_chunk = chunk.decode().lstrip('_')
                    try:
                        response_json = json.loads(decoded_chunk)
                        if response_json.get("status"):
                            yield response_json.get("gpt", "")
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error during API request (nexra_gptweb): {e}")

    @staticmethod
    def _sync_post_request(url: str, data: Dict[str, Any], headers: Dict[str, str], proxy: str = None, stream: bool = False) -> requests.Response:
        proxies = {
            "http": proxy,
            "https": proxy,
        } if proxy else None
        
        try:
            response = requests.post(url, json=data, headers=headers, proxies=proxies, stream=stream)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            raise

    @staticmethod
    def _process_non_streaming_response(response: requests.Response) -> str:
        if response.status_code == 200:
            try:
                content = response.text.lstrip('')
                data = json.loads(content)
                return data.get('message', '')
            except json.JSONDecodeError:
                return "Error: Unable to decode JSON response"
        else:
            return f"Error: {response.status_code}"

    @staticmethod
    async def _process_streaming_response(response: requests.Response):
        full_message = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    line = line.lstrip('')
                    data = json.loads(line)
                    if data.get('finish'):
                        break
                    message = data.get('message', '')
                    if message:
                        yield message[len(full_message):]
                        full_message = message
                except json.JSONDecodeError:
                    pass

    @staticmethod
    def _filter_response(response: requests.Response) -> str:
        response_json = response.json()
        return response_json.get("gpt", "")
