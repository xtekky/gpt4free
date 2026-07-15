from __future__ import annotations

from typing import Any

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..requests import StreamSession

class Miklium(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Miklium"
    url = "https://miklium.vercel.app"
    api_endpoint = "/api/chatbot"
    
    working = True
    needs_auth = False
    supports_stream = False
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'miklium'
    models = ['miklium', 'personalityless', 'male', 'female', 'all']

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str | None = None,
        **kwargs: Any
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/"
        }
        
        prompt = format_prompt(messages)
        
        data_payload = {
            "message": prompt,
            "response_stacking": kwargs.get("response_stacking", 4),
            "personality": model
        }

        async with StreamSession(headers=headers, impersonate="chrome") as session:
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=data_payload, proxy=proxy) as response:
                response.raise_for_status()
                json_data = await response.json()
                
                if str(json_data.get("success")).lower() == "true":
                    yield json_data.get("response", "")
                else:
                    error_msg = json_data.get("error", "Unknown error")
                    raise RuntimeError(f"Miklium error: {error_msg}")
