from __future__ import annotations

from aiohttp import ClientSession
from asyncio import sleep

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class CodeNews(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://codenews.cc"
    api_endpoint = "https://codenews.cc/chatxyz13"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = False
    supports_stream = True
    supports_system_message = False
    supports_message_history = False
    
    default_model = 'free_gpt'
    models = ['free_gpt', 'gpt-4o-mini', 'deepseek-coder', 'chatpdf']
    
    model_aliases = {
        "glm-4": "free_gpt",
        "gpt-3.5-turbo": "chatpdf",
        "deepseek": "deepseek-coder",
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
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "origin": cls.url,
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": f"{cls.url}/chatgpt",
            "sec-ch-ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "x-requested-with": "XMLHttpRequest",
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "chatgpt_input": prompt,
                "qa_type2": model,
                "chatgpt_version_value": "20240804",
                "enable_web_search": "0",
                "enable_agent": "0",
                "dy_video_text_extract": "0",
                "enable_summary": "0",
            }
            async with session.post(cls.api_endpoint, data=data, proxy=proxy) as response:
                response.raise_for_status()
                json_data = await response.json()
                chat_id = json_data["data"]["id"]

            headers["content-type"] = "application/x-www-form-urlencoded; charset=UTF-8"
            data = {"current_req_count": "2"}
            
            while True:
                async with session.post(f"{cls.url}/chat_stream", headers=headers, data=data, proxy=proxy) as response:
                    response.raise_for_status()
                    json_data = await response.json()
                    if json_data["data"]:
                        yield json_data["data"]
                        break
                    else:
                        await sleep(1)  # Затримка перед наступним запитом
