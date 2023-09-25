from __future__ import annotations

import random, json

from g4f.requests import AsyncSession, StreamRequest
from .base_provider import AsyncGeneratorProvider, format_prompt

domains = {
    "gpt-3.5-turbo": ".aitianhu.space",
    "gpt-4": ".aitianhu.website",
}

class AItianhuSpace(AsyncGeneratorProvider):
    url = "https://chat3.aiyunos.top/"
    working = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = True,
        **kwargs
    ) -> str:
        if not model:
            model = "gpt-3.5-turbo"
        elif not model in domains:
            raise ValueError(f"Model are not supported: {model}")
        
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        rand = ''.join(random.choice(chars) for _ in range(6))
        domain = domains[model]
        url = f'https://{rand}{domain}/api/chat-process'

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        }
        async with AsyncSession(headers=headers, impersonate="chrome107", verify=False) as session:
            data = {
                "prompt": format_prompt(messages),
                "options": {},
                "systemMessage": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully.",
                "temperature": 0.8,
                "top_p": 1,
                **kwargs
            }
            async with StreamRequest(session, "POST", url, json=data) as response:
                response.raise_for_status()
                async for line in response.content:
                    line = json.loads(line.rstrip())
                    if "detail" in line:
                        content = line["detail"]["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                    elif "message" in line and "AI-4接口非常昂贵" in line["message"]:
                        raise RuntimeError("Rate limit for GPT 4 reached")
                    else:
                        raise RuntimeError("Response: {line}")
        

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
            ("top_p", "int"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
