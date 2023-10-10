from __future__ import annotations

import random, json

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider, format_prompt, get_cookies

domains = {
    "gpt-3.5-turbo": "aitianhu.space",
    "gpt-4": "aitianhu.website",
}

class AItianhuSpace(AsyncGeneratorProvider):
    url = "https://chat3.aiyunos.top/"
    working = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        domain: str = None,
        cookies: dict = None,
        timeout: int = 120,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "gpt-3.5-turbo"
        elif not model in domains:
            raise ValueError(f"Model are not supported: {model}")
        if not domain:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            rand = ''.join(random.choice(chars) for _ in range(6))
            domain = f"{rand}.{domains[model]}"
        if not cookies:
            cookies = get_cookies(domain)
        
        url = f'https://{domain}'
        async with StreamSession(
            proxies={"https": proxy},
            cookies=cookies,
            timeout=timeout,
            impersonate="chrome110",
            verify=False
        ) as session:
            data = {
                "prompt": format_prompt(messages),
                "options": {},
                "systemMessage": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully.",
                "temperature": 0.8,
                "top_p": 1,
                **kwargs
            }
            headers = {
                "Authority": url,
                "Accept": "application/json, text/plain, */*",
                "Origin": url,
                "Referer": f"{url}/"
            }
            async with session.post(f"{url}/api/chat-process", json=data, headers=headers) as response:
                response.raise_for_status()
                async for line in response.iter_lines():
                    if line == b"<script>":
                        raise RuntimeError("Solve challenge and pass cookies and a fixed domain")
                    if b"platform's risk control" in line:
                        raise RuntimeError("Platform's Risk Control")
                    line = json.loads(line)
                    if "detail" in line:
                        content = line["detail"]["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                    elif "message" in line and "AI-4接口非常昂贵" in line["message"]:
                        raise RuntimeError("Rate limit for GPT 4 reached")
                    else:
                        raise RuntimeError(f"Response: {line}")
        

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
