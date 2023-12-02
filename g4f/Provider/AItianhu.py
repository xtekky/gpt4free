from __future__ import annotations

import json

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider, format_prompt, get_cookies


class AItianhu(AsyncGeneratorProvider):
    url = "https://www.aitianhu.com"
    working = False
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: dict = None,
        timeout: int = 120, **kwargs) -> AsyncResult:
        
        if not cookies:
            cookies = get_cookies(domain_name='www.aitianhu.com')
        if not cookies:
            raise RuntimeError(f"g4f.provider.{cls.__name__} requires cookies [refresh https://www.aitianhu.com on chrome]")

        data = {
            "prompt": format_prompt(messages),
            "options": {},
            "systemMessage": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully.",
            "temperature": 0.8,
            "top_p": 1,
            **kwargs
        }

        headers = {
            'authority': 'www.aitianhu.com',
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'content-type': 'application/json',
            'origin': 'https://www.aitianhu.com',
            'referer': 'https://www.aitianhu.com/',
            'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        }

        async with StreamSession(headers=headers,
                                        cookies=cookies,
                                        timeout=timeout,
                                        proxies={"https": proxy},
                                        impersonate="chrome107", verify=False) as session:
            
            async with session.post(f"{cls.url}/api/chat-process", json=data) as response:
                response.raise_for_status()

                async for line in response.iter_lines():
                    if line == b"<script>":
                        raise RuntimeError("Solve challenge and pass cookies")

                    if b"platform's risk control" in line:
                        raise RuntimeError("Platform's Risk Control")

                    line = json.loads(line)

                    if "detail" not in line:
                        raise RuntimeError(f"Response: {line}")

                    content = line["detail"]["choices"][0]["delta"].get(
                        "content"
                    )
                    if content:
                        yield content
