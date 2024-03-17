from __future__ import annotations

import json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider

class Gpt6(AsyncGeneratorProvider):
    url = "https://gpt6.ai"
    working = False
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "application/json",
            "Origin": "https://gpt6.ai",
            "Connection": "keep-alive",
            "Referer": "https://gpt6.ai/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "TE": "trailers",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "prompts":messages,
                "geoInfo":{"ip":"100.90.100.222","hostname":"ip-100-090-100-222.um36.pools.vodafone-ip.de","city":"Muenchen","region":"North Rhine-Westphalia","country":"DE","loc":"44.0910,5.5827","org":"AS3209 Vodafone GmbH","postal":"41507","timezone":"Europe/Berlin"},
                "paid":False,
                "character":{"textContent":"","id":"52690ad6-22e4-4674-93d4-1784721e9944","name":"GPT6","htmlContent":""}
            }
            async with session.post(f"https://seahorse-app-d29hu.ondigitalocean.app/api/v1/query", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    print(line)
                    if line.startswith(b"data: [DONE]"):
                        break
                    elif line.startswith(b"data: "):
                        line = json.loads(line[6:-1])

                        chunk = line["choices"][0]["delta"].get("content")
                        if chunk:
                            yield chunk