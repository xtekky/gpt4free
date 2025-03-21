from __future__ import annotations

from ..typing import AsyncResult, Messages, MediaListType
from .template import OpenaiTemplate
from ..image import to_data_uri

class TypeGPT(OpenaiTemplate):
    url = "https://chat.typegpt.net"
    api_base = "https://chat.typegpt.net/api/openai/typegpt/v1"
    working = True
    
    supports_stream = True 
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o-mini-2024-07-18'
    default_vision_model = default_model
    vision_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-202201', default_vision_model, "o3-mini"]
    models = vision_models + ["deepseek-r1", "deepseek-v3", "evil", "o1"]
    model_aliases = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-202201",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    }
    
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        temperature: float = 0.5,
        top_p: float = 1.0,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        proxy: str = None,
        media: MediaListType = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        }
        
        if media is not None and messages:
            last_message = messages[-1].copy()
            last_message["content"] = [
                {
                    "type": "text",
                    "text": last_message["content"]
                },
                *[{
                    "type": "image_url",
                    "image_url": {"url": to_data_uri(image)}
                } for image, _ in media]
            ]
            messages[-1] = last_message

        async for chunk in super().create_async_generator(
            model=model,
            messages=messages,
            proxy=proxy,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            **kwargs
        ):
            yield chunk
