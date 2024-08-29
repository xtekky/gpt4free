from __future__ import annotations
import json
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class FreeChatgpt(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chat.chatgpt.org.uk"
    api_endpoint = "/api/openai/v1/chat/completions"
    working = True
    supports_gpt_35_turbo = True
    default_model = 'gpt-3.5-turbo'
    models = [
        'gpt-3.5-turbo',
        'SparkDesk-v1.1',
        'deepseek-coder',
        '@cf/qwen/qwen1.5-14b-chat-awq',
        'deepseek-chat',
        'Qwen2-7B-Instruct',
        'glm4-9B-chat',
        'chatglm3-6B',
        'Yi-1.5-9B-Chat',
    ]
    model_aliases = {
        "qwen-1.5-14b": "@cf/qwen/qwen1.5-14b-chat-awq",
        "sparkdesk-v1.1": "SparkDesk-v1.1",
        "qwen2-7b": "Qwen2-7B-Instruct",
        "glm4-9b": "glm4-9B-chat",
        "chatglm3-6b": "chatglm3-6B",
        "yi-1.5-9b": "Yi-1.5-9B-Chat",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model.lower() in cls.model_aliases:
            return cls.model_aliases[model.lower()]
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
        headers = {
            "accept": "application/json, text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "dnt": "1",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        }
        model = cls.get_model(model)
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "messages": [
                    {"role": "system", "content": "\nYou are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent model: gpt-3.5-turbo\nCurrent time: Thu Jul 04 2024 21:35:59 GMT+0300 (Eastern European Summer Time)\nLatex inline: \\(x^2\\) \nLatex block: $$e=mc^2$$\n\n"},
                    {"role": "user", "content": prompt}
                ],
                "stream": True,
                "model": model,
                "temperature": 0.5,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "top_p": 1
            }
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()
                accumulated_text = ""
                async for line in response.content:
                    if line:
                        line_str = line.decode().strip()
                        if line_str == "data: [DONE]":
                            yield accumulated_text
                            break
                        elif line_str.startswith("data: "):
                            try:
                                chunk = json.loads(line_str[6:])
                                delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                accumulated_text += delta_content
                                yield delta_content
                            except json.JSONDecodeError:
                                pass
