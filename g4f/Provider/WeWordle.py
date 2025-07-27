from __future__ import annotations

import json
import asyncio
import re
from typing import Union
from aiohttp import ClientSession, ClientResponse, ClientResponseError, ClientConnectorError

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class WeWordle(AsyncGeneratorProvider, ProviderModelMixin):
    label = "WeWordle"
    url = "https://chat-gpt.com"
    api_endpoint = "https://wewordle.org/gptapi/v1/web/turbo" 
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4'
    models = [default_model]

    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY_SECONDS = 5
    MAX_RETRY_DELAY_SECONDS = 60
    POST_REQUEST_DELAY_SECONDS = 1

    @staticmethod
    async def iter_any(response: ClientResponse):
        if response.headers.get("Transfer-Encoding") == "chunked" or \
           response.headers.get("Content-Type") == "text/event-stream":
            async for chunk in response.content:
                if chunk:
                    yield chunk.decode()
        else:
            content = await response.text()
            yield content

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        raw_url = cls.api_endpoint
        request_url = raw_url

        markdown_link_match = re.search(r'\]\((https?://[^\)]+)\)', raw_url)
        if markdown_link_match:
            actual_url = markdown_link_match.group(1)
            request_url = actual_url
        elif not (raw_url.startswith("http://") or raw_url.startswith("https://")):
            if "%5B" in raw_url and "%5D" in raw_url and "%28" in raw_url and "%29" in raw_url:
                 try:
                    import urllib.parse
                    decoded_url_outer = urllib.parse.unquote(raw_url)
                    markdown_link_match_decoded = re.search(r'\]\((https?://[^\)]+)\)', decoded_url_outer)
                    if markdown_link_match_decoded:
                        actual_url = markdown_link_match_decoded.group(1)
                        request_url = actual_url
                    else:
                        raise ValueError(f"Invalid API endpoint URL format: {raw_url}")
                 except Exception as e:
                    raise ValueError(f"Invalid API endpoint URL format: {raw_url}")
            elif not (raw_url.startswith("http://") or raw_url.startswith("https://")):
                raise ValueError(f"Invalid API endpoint URL format: {raw_url}")

        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://chat-gpt.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://chat-gpt.com/",
            "sec-ch-ua": "\"Not.A/Brand\";v=\"99\", \"Chromium\";v=\"136\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Linux\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
        }
        
        if isinstance(messages, list) and all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
            data_payload = {"messages": messages, "model": model}
        else:
            data_payload = {
                "messages": messages,
                "model": model
            }

        retries = 0
        current_delay = cls.INITIAL_RETRY_DELAY_SECONDS

        async with ClientSession(headers=headers) as session:
            while retries <= cls.MAX_RETRIES:
                try:
                    async with session.post(request_url, json=data_payload, proxy=proxy) as response:
                        if response.status == 429:
                            pass
                        
                        response.raise_for_status()
                        
                        async for chunk in cls.iter_any(response):
                            try:
                                json_data = json.loads(chunk)
                                if isinstance(json_data, dict):
                                    if "message" in json_data and isinstance(json_data["message"], dict) and "content" in json_data["message"]:
                                        yield json_data["message"]["content"]
                                    elif "choices" in json_data and isinstance(json_data["choices"], list) and \
                                         json_data["choices"] and isinstance(json_data["choices"][0], dict) and \
                                         "message" in json_data["choices"][0] and isinstance(json_data["choices"][0]["message"], dict) and \
                                         "content" in json_data["choices"][0]["message"]:
                                        yield json_data["choices"][0]["message"]["content"]
                                    elif "limit" in json_data and json_data["limit"] == 0:
                                        if "error" in json_data and isinstance(json_data["error"], dict) and "message" in json_data["error"]:
                                           raise ValueError(f"API error: {json_data['error']['message']}")
                                    else:
                                        yield chunk 
                                else:
                                    yield chunk
                            except json.JSONDecodeError:
                                yield chunk
                        
                        await asyncio.sleep(cls.POST_REQUEST_DELAY_SECONDS)
                        return

                except ClientResponseError as e:
                    if e.status == 429:
                        await asyncio.sleep(current_delay)
                        retries += 1
                        current_delay = min(current_delay * 2, cls.MAX_RETRY_DELAY_SECONDS) 
                        if retries > cls.MAX_RETRIES:
                            raise
                    else:
                        raise
                except ClientConnectorError as e:
                    await asyncio.sleep(current_delay)
                    retries += 1
                    current_delay = min(current_delay * 2, cls.MAX_RETRY_DELAY_SECONDS)
                    if retries > cls.MAX_RETRIES:
                        raise
                except Exception as e:
                    raise

            raise Exception(f"Failed to get response from {request_url} after multiple retries")
