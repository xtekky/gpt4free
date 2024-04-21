from __future__ import annotations

from ..typing import AsyncResult, Messages, Cookies
from .helper import format_prompt, get_cookies
from .MetaAI import MetaAI

class MetaAIAccount(MetaAI):
    needs_auth = True
    parent = "MetaAI"
    image_models = ["meta"]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: Cookies = None,
        **kwargs
    ) -> AsyncResult:
        cookies = get_cookies(".meta.ai", True, True) if cookies is None else cookies
        async for chunk in cls(proxy).prompt(format_prompt(messages), cookies):
            yield chunk