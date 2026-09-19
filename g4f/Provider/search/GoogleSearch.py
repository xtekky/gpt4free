from __future__ import annotations

import asyncio
import urllib.parse

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...providers.response import SearchResults, format_link
from ...requests.cdp import CDPSession
from ... import debug
from ..helper import get_last_user_message


class GoogleSearch(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Google Search"
    url = "https://google.com"
    screenshot_url = f"{url}/search?q=Hello"
    working = True
    active_by_default = True
    supports_native_tools = True
    default_model = "search"
    models = [default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        **kwargs,
    ) -> AsyncResult:
        query = get_last_user_message(messages)
        search_url = f"{cls.url}/search?q={urllib.parse.quote(query)}"

        debug.log(f"Google Search: Starting CDPSession for query: {query}")
        session = CDPSession()
        await session.start()

        try:
            debug.log(f"Google Search: Navigating to search URL: {search_url}")
            await session.navigate(search_url)
            # debug.log(f"Google Search: Waiting for the page to load...")
            await session.click_button_by_text()
            # debug.log(f"Google Search: Clicking accept button if present...")

            # Wait for Google search results page to load
            for _ in range(10):
                has_results = None
                try:
                    has_results = await session.evaluate_js(
                        "document.querySelectorAll('h3').length > 0"
                    )
                except Exception as e:
                    debug.log(f"Google Search: Error checking for results: {e}")
                if not has_results:
                    debug.log(f"Google Search: No results yet, retrying...")
                    await asyncio.sleep(1)
                    continue

                # Extract search results from the DOM
                yield SearchResults(await cls._read_search_results(session))
                yield f'\n\nSource: {format_link(search_url, "Google Search")}'
                break
        finally:
            await session.close()

    async def _read_search_results(session: CDPSession) -> list:
        return await session.evaluate_js("""
            (() => {
                const results = [];
                const boxes = document.querySelectorAll('[data-snhf="0"]')
                boxes.forEach(box => {
                    const linkEl = box.querySelector('a');
                    const title = linkEl.querySelector('h3') ? linkEl.querySelector('h3').innerText : '';
                    const link = linkEl.href  ? new URL(linkEl.href || '/') : null;
                    if (link) link.searchParams.delete("srsltid")
                    const snippetEl = box.nextElementSibling;
                    const snippet = snippetEl ? snippetEl.innerText.replace('...Read more', '...') : undefined;
                    if (title && link) {
                        results.push({ title, link: link.toString(), snippet });
                    }
                });
                return results;
            })()
            """)