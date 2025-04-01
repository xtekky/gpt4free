from __future__ import annotations

import asyncio

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException, ConversationLimitException
    has_requirements = True
except ImportError:
    has_requirements = False
try:
    import nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ..typing import AsyncResult, Messages
from ..requests import get_nodriver
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message

class DuckDuckGo(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Duck.ai (duckduckgo_search)"
    url = "https://duckduckgo.com/aichat"
    api_base = "https://duckduckgo.com/duckchat/v1/"
    
    working = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "gpt-4o-mini"
    models = [default_model, "meta-llama/Llama-3.3-70B-Instruct-Turbo", "claude-3-haiku-20240307", "o3-mini", "mistralai/Mistral-Small-24B-Instruct-2501"]

    ddgs: DDGS = None

    model_aliases = {
        "gpt-4": "gpt-4o-mini",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "mixtral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 60,
        **kwargs
    ) -> AsyncResult:
        if not has_requirements:
            raise ImportError("duckduckgo_search is not installed. Install it with `pip install duckduckgo-search`.")
        if cls.ddgs is None:
            cls.ddgs = DDGS(proxy=proxy, timeout=timeout)
            if has_nodriver:
                await cls.nodriver_auth(proxy=proxy)
        model = cls.get_model(model)
        for chunk in cls.ddgs.chat_yield(get_last_user_message(messages), model, timeout):
            yield chunk

    @classmethod
    async def nodriver_auth(cls, proxy: str = None):
        browser, stop_browser = await get_nodriver(proxy=proxy)
        try:
            page = browser.main_tab
            def on_request(event: nodriver.cdp.network.RequestWillBeSent, page=None):
                if cls.api_base in event.request.url:
                    if "X-Vqd-4" in event.request.headers:
                        cls.ddgs._chat_vqd = event.request.headers["X-Vqd-4"]
                    if "X-Vqd-Hash-1" in event.request.headers:
                        cls.ddgs._chat_vqd_hash = event.request.headers["X-Vqd-Hash-1"]
                    if "F-Fe-Version" in event.request.headers:
                        cls.ddgs._chat_xfe = event.request.headers["F-Fe-Version" ]
            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            page = await browser.get(cls.url)
            while True:
                if cls.ddgs._chat_vqd:
                    break
                await asyncio.sleep(1)
            await page.close()
        finally:
            stop_browser()