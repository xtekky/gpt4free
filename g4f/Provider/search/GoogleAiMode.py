from __future__ import annotations

import asyncio
import urllib.parse

from ...typing import AsyncResult, Messages
from ...requests.cdp import CDPSession
from ...providers.response import SearchResults
from ... import debug
from ..helper import get_last_user_message
from .GoogleSearch import GoogleSearch


class GoogleAiMode(GoogleSearch):
    label = "Google AI Mode"
    url = "https://google.com"
    screenshot_url = f"{url}/search?q=Hello&ai-mode=true"
    working = True
    active_by_default = True
    supports_native_tools = True
    default_model = "ai-mode"
    models = [default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        **kwargs,
    ) -> AsyncResult:
        query = get_last_user_message(messages)
        search_url = f"{cls.url}/search?q={urllib.parse.quote_plus(query)}&ai-mode=true"

        debug.log(f"Google Search: Starting CDPSession for query: {query}")
        session = CDPSession()
        await session.start()

        try:
            await session.navigate(search_url)
            await session.click_accept_button()
        except Exception as e:
            await session.close()
            raise e

        try:
            await session.wait_for_network_idle(idle_time=1, timeout=10.0)
            search_results = await cls._read_search_results(session)
            if search_results:
                yield SearchResults(search_results)
                yield "\n\n---\n\n"
        except Exception as e:
            debug.log(f"Google Search: Error reading search results: {e}")
            await session.close()
            raise e

        try:
            for _ in range(5):
                result = await session.evaluate_js("""const b = Array.from(document.querySelectorAll("a, button")).filter(a => {
                    return a.textContent.endsWith("KI‑Modus") || a.textContent.endsWith("AI Mode");
                }).pop(); b ? b.click() : null; !!b""")
                await asyncio.sleep(1)
                if not result:
                    continue
                await session.wait_for_network_idle(idle_time=1, timeout=10.0)
                results = await session.call("Runtime.evaluate", expression=r"""
const rootElement = document.querySelector('[decode-data-ved="1"]');
const allElements = rootElement.querySelectorAll('*');
const textNodes = [];
Array.from(allElements).forEach(el => {
    for (const child of el.childNodes) {
        if (child.nodeType === Node.TEXT_NODE && child.textContent) {
            const trimedText = child.textContent.trim();
            if ([
                "KI-Antworten können Fehler enthalten.",
                "AI responses may include mistakes."].includes(trimedText)) {
                break;
            }
            textNodes.push(child);
        }   
    }
});
textNodes.map(n => n.textContent).filter(Boolean);
""", returnByValue=True);
                debug.log(f"Google Search: AI mode results: {results}")
                if results:
                    for text in results.get("result", {}).get("value", []):
                        yield f"{text}\n"
                    return
            raise RuntimeError("No AI mode results found.")
        finally:
            await session.close()
