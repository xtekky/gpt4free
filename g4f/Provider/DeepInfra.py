from __future__ import annotations

import json
from ..typing           import AsyncResult, Messages
from .base_provider     import AsyncGeneratorProvider
from ..requests         import StreamSession

class DeepInfra(AsyncGeneratorProvider):
    url = "https://deepinfra.com"
    working = True
    supports_stream = True
    supports_message_history = True

    @staticmethod
    async def create_async_generator(
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        timeout: int = 120,
        auth: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = 'meta-llama/Llama-2-70b-chat-hf'
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-embed',
            'accept': 'text/event-stream',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        }
        if auth:
            headers['Authorization'] = f"bearer {auth}" 
            
        async with StreamSession(headers=headers,
            timeout=timeout,
            proxies={"https": proxy},
            impersonate="chrome110"
        ) as session:
            json_data = {
                'model'   : model,
                'messages': messages,
                'stream'  : True
            }
            async with session.post('https://api.deepinfra.com/v1/openai/chat/completions',
                                    json=json_data) as response:
                response.raise_for_status()
                first = True
                async for line in response.iter_lines():
                    try:
                        if line.startswith(b"data: [DONE]"):
                            break
                        elif line.startswith(b"data: "):
                            chunk = json.loads(line[6:])["choices"][0]["delta"].get("content")
                        if chunk:
                            if first:
                                chunk = chunk.lstrip()
                                if chunk:
                                    first = False
                            yield chunk
                    except Exception:
                        raise RuntimeError(f"Response: {line}")