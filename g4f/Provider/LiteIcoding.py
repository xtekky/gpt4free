from __future__ import annotations

from aiohttp import ClientSession, ClientResponseError
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class LiteIcoding(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://lite.icoding.ink"
    api_endpoint = "/api/v1/gpt/message"
    working = True
    supports_gpt_4 = True
    default_model = "gpt-4o"
    models = [
        'gpt-4o',
        'gpt-4-turbo',
        'claude-3',
        'claude-3.5',
        'gemini-1.5',
    ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Authorization": "Bearer b3b2712cf83640a5acfdc01e78369930",
            "Connection": "keep-alive",
            "Content-Type": "application/json;charset=utf-8",
            "DNT": "1",
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
        }

        data = {
            "model": model,
            "chatId": "-1",
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "time": msg.get("time", ""),
                    "attachments": msg.get("attachments", []),
                }
                for msg in messages
            ],
            "plugins": [],
            "systemPrompt": "",
            "temperature": 0.5,
        }

        async with ClientSession(headers=headers) as session:
            try:
                async with session.post(
                    f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy
                ) as response:
                    response.raise_for_status()
                    buffer = ""
                    full_response = ""
                    def decode_content(data):
                        bytes_array = bytes([int(b, 16) ^ 255 for b in data.split()])
                        return bytes_array.decode('utf-8')
                    async for chunk in response.content.iter_any():
                        if chunk:
                            buffer += chunk.decode()
                            while "\n\n" in buffer:
                                part, buffer = buffer.split("\n\n", 1)
                                if part.startswith("data: "):
                                    content = part[6:].strip()
                                    if content and content != "[DONE]":
                                        content = content.strip('"')
                                        # Decoding each content block
                                        decoded_content = decode_content(content)
                                        full_response += decoded_content
                    full_response = (
                    full_response.replace('""', '')  # Handle double quotes
                                  .replace('" "', ' ')  # Handle space within quotes
                                  .replace("\\n\\n", "\n\n")
                                  .replace("\\n", "\n")
                                  .replace('\\"', '"')
                                  .strip()
                    )
                    yield full_response.strip()

            except ClientResponseError as e:
                raise RuntimeError(
                    f"ClientResponseError {e.status}: {e.message}, url={e.request_info.url}, data={data}"
                ) from e

            except Exception as e:
                raise RuntimeError(f"Unexpected error: {str(e)}") from e
