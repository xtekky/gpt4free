from __future__ import annotations

import json
import time
import base64
from curl_cffi.requests import AsyncSession

from ..base_provider import AsyncProvider, format_prompt, get_cookies


class PerplexityAi(AsyncProvider):
    url                   = "https://www.perplexity.ai"
    supports_gpt_35_turbo = True
    _sources              = []

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> str:
        url = cls.url + "/socket.io/?EIO=4&transport=polling"
        headers = {
            "Referer": f"{cls.url}/"
        }
        async with AsyncSession(headers=headers, proxies={"https": proxy}, impersonate="chrome107") as session:
            url_session = "https://www.perplexity.ai/api/auth/session"
            response = await session.get(url_session)
            response.raise_for_status()

            url_session = "https://www.perplexity.ai/api/auth/session"
            response = await session.get(url_session)
            response.raise_for_status()

            response = await session.get(url, params={"t": timestamp()})
            response.raise_for_status()
            sid = json.loads(response.text[1:])["sid"]

            response = await session.get(url, params={"t": timestamp(), "sid": sid})
            response.raise_for_status()

            data = '40{"jwt":"anonymous-ask-user"}'
            response = await session.post(url, params={"t": timestamp(), "sid": sid}, data=data)
            response.raise_for_status()

            response = await session.get(url, params={"t": timestamp(), "sid": sid})
            response.raise_for_status()

            data = "424" + json.dumps([
                "perplexity_ask",
                format_prompt(messages),
                {
                    "version":"2.1",
                    "source":"default",
                    "language":"en",
                    "timezone": time.tzname[0],
                    "search_focus":"internet",
                    "mode":"concise"
                }
            ])
            response = await session.post(url, params={"t": timestamp(), "sid": sid}, data=data)
            response.raise_for_status()

            while True:
                response = await session.get(url, params={"t": timestamp(), "sid": sid})
                response.raise_for_status()
                for line in response.text.splitlines():
                    if line.startswith("434"):
                        result = json.loads(json.loads(line[3:])[0]["text"])

                        cls._sources = [{
                            "title": source["name"],
                            "url": source["url"],
                            "snippet": source["snippet"]
                        } for source in result["web_results"]]

                        return result["answer"]

    @classmethod
    def get_sources(cls):
        return cls._sources


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"


def timestamp() -> str:
    return base64.urlsafe_b64encode(int(time.time()-1407782612).to_bytes(4, 'big')).decode()