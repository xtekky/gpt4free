from __future__ import annotations

import asyncio
import json

from ...typing import AsyncResult, Messages, Cookies
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin, get_running_loop
from ...requests import Browser, get_nodriver, has_nodriver
from ...errors import MissingRequirementsError, ModelNotFoundError
from ... import debug
from ..helper import get_last_user_message

class GoogleSearch(AsyncGeneratorProvider, AuthFileMixin):
    label = "Google Search"
    url = "https://google.com"
    working = has_nodriver
    use_nodriver = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        browser: Browser = None,
        proxy: str = None,
        timeout: int = 300,
        **kwargs
    ) -> AsyncResult:
        if not has_nodriver:
            raise MissingRequirementsError("Google requires a browser to be installed.")
        if not cls.working:
            raise ModelNotFoundError(f"Model {model} not found.")
        try:
            stop_browser = None
            if browser is None:
               browser, stop_browser = await get_nodriver(proxy=proxy, timeout=timeout)
            tab = await browser.get(cls.url)
            await asyncio.sleep(3)
            while True:
                try:
                    await tab.wait_for('[aria-modal="true"]', timeout=10)
                    await tab.wait_for('[aria-modal="true"][style*="display: none"]', timeout=timeout)
                except Exception as e:
                    break
                break
            element = await tab.wait_for('textarea')
            await element.send_keys(get_last_user_message(messages))
            button = await tab.find("Google Suche")
            await button.click()
            await asyncio.sleep(1000)
        finally:
            if stop_browser is not None:
                stop_browser()