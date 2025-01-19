from __future__ import annotations

import json
from aiohttp import ClientSession

from ..image import to_data_uri
from ..typing import AsyncResult, Messages, ImagesType
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..providers.response import FinishReason


class OIVSCode(AsyncGeneratorProvider, ProviderModelMixin):
    label = "OI VSCode Server"
    url = "https://oi-vscode-server.onrender.com"
    api_endpoint = "https://oi-vscode-server.onrender.com/v1/chat/completions"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "gpt-4o-mini-2024-07-18"
    default_vision_model = default_model
    vision_models = [default_model, "gpt-4o-mini"]
    models = vision_models
    
    model_aliases = {"gpt-4o-mini": "gpt-4o-mini-2024-07-18"}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        images: ImagesType = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:      
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        
        async with ClientSession(headers=headers) as session:
            
            if images is not None:
                messages[-1]['content'] = [
                    {
                        "type": "text",
                        "text": messages[-1]['content']
                    },
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": to_data_uri(image)
                            }
                        }
                        for image, _ in images
                    ]
                ]
                
            data = {
                "model": model,
                "stream": stream,
                "messages": messages
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                
                full_response = ""
                
                if stream:
                    async for line in response.content:
                        if line:
                            line = line.decode()
                            if line.startswith("data: "):
                                if line.strip() == "data: [DONE]":
                                    break
                                try:
                                    data = json.loads(line[6:])
                                    if content := data["choices"][0]["delta"].get("content"):
                                        yield content
                                        full_response += content
                                except:
                                    continue
                    
                    reason = "length" if len(full_response) > 0 else "stop"
                    yield FinishReason(reason)
                else:
                    response_data = await response.json()
                    full_response = response_data["choices"][0]["message"]["content"]
                    yield full_response
                    
                    reason = "length" if len(full_response) > 0 else "stop"
                    yield FinishReason(reason)
