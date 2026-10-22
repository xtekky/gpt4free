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

            # Wait for Google search results page to load and extract them.
            # A present <h3> alone is not enough: the page can contain result
            # titles that are not organic results (e.g. "People also ask"), so
            # only stop once the extraction actually returned entries.
            results = None
            for _ in range(10):
                try:
                    results = await cls._read_search_results(session)
                except Exception as e:
                    debug.log(f"Google Search: Error reading results: {e}")
                    results = None
                if results:
                    break
                debug.log("Google Search: No results yet, retrying...")
                await asyncio.sleep(1)

            if not results:
                raise RuntimeError("No search results found.")

            yield SearchResults(results)
            yield f'\n\nSource: {format_link(search_url, "Google Search")}'
        finally:
            await session.close()

    async def _read_search_results(session: CDPSession) -> list:
        return await session.evaluate_js("""
            (() => {
                const results = [];
                const seen = new Set();
                const pushResult = (linkEl, titleEl, snippetEl) => {
                    if (!linkEl || !linkEl.href) return;
                    const title = ((titleEl && titleEl.innerText) || linkEl.innerText || '').trim();
                    if (!title) return;
                    let link;
                    try {
                        link = new URL(linkEl.href, location.origin);
                    } catch (e) {
                        return;
                    }
                    link.searchParams.delete("srsltid");
                    const url = link.toString();
                    if (seen.has(url)) return;
                    seen.add(url);
                    const snippet = snippetEl
                        ? snippetEl.innerText.replace('...Read more', '...').trim()
                        : undefined;
                    results.push({ title, link: url, snippet });
                };

                // Current Google SERP markup: organic result boxes carry
                // data-snhf="0" and the snippet follows in the next sibling.
                const boxes = document.querySelectorAll('[data-snhf="0"]');
                boxes.forEach(box => {
                    const linkEl = box.querySelector('a');
                    const titleEl = box.querySelector('h3') || (linkEl && linkEl.querySelector('h3'));
                    const snippetEl = box.nextElementSibling || box.querySelector('[data-sncf]');
                    pushResult(linkEl, titleEl, snippetEl);
                });

                // Fallback for markup changes (and consent/alternate layouts):
                // derive results from the result titles themselves.
                if (!results.length) {
                    document.querySelectorAll('h3').forEach(titleEl => {
                        const linkEl = titleEl.closest('a')
                            || (titleEl.parentElement && titleEl.parentElement.closest('a'));
                        if (!linkEl) return;
                        const container = titleEl.closest('div.g')
                            || titleEl.closest('[data-snhf="0"]');
                        const snippetEl = container
                            ? (container.querySelector('[data-sncf]')
                                || container.querySelector('div.VwiC3b'))
                            : (titleEl.parentElement && titleEl.parentElement.nextElementSibling);
                        pushResult(linkEl, titleEl, snippetEl);
                    });
                }
                return results;
            })()
            """)
