from __future__ import annotations

from aiohttp import ClientSession

from ..typing       import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider


class ChatBase(AsyncGeneratorProvider):
    url                   = "https://www.chatbase.co"
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    working               = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if model == "gpt-4":
            chat_id = "quran---tafseer-saadi-pdf-wbgknt7zn"
        elif model == "gpt-3.5-turbo" or not model:
            chat_id = "chatbase--1--pdf-p680fxvnm"
        else:
            raise ValueError(f"Model are not supported: {model}")
        headers = {
            "User-Agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Accept"             : "*/*",
            "Accept-language"    : "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "Origin"             : cls.url,
            "Referer"            : cls.url + "/",
            "Sec-Fetch-Dest"     : "empty",
            "Sec-Fetch-Mode"     : "cors",
            "Sec-Fetch-Site"     : "same-origin",
        }
        async with ClientSession(
            headers=headers
        ) as session:
            data = {
                "messages": messages,
                "captchaCode": "hadsa",
                "chatId": chat_id,
                "conversationId": f"kcXpqEnqUie3dnJlsRi_O-{chat_id}"
            }
            async with session.post("https://www.chatbase.co/api/fe/chat", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for stream in response.content.iter_any():
                    yield stream.decode()


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"