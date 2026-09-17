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
            await session.click_accept_button(False)
        except Exception as e:
            await session.close()
            raise e

        try:
            await session.wait_for_network_idle()
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
                await session.evaluate_js("""b = Array.from(document.querySelectorAll("a, button")).filter(a => {
                    return a.textContent.endsWith("KI‑Modus") || a.textContent.endsWith("AI Mode");
                }).pop(); b ? b.click() : null; !!b""")
                await session.wait_for_network_idle(idle_time=1, timeout=10.0)
                result = await session.evaluate_js("""
const rootElement = document.querySelector('[decode-data-ved="1"]');
function getTextNodes(element) {
    const textNodes = [];
    for (const child of element.childNodes) {
        if (child.nodeType === Node.ELEMENT_NODE && child.tagName === "STRONG") {
            textNodes.push("**");
        }
        if (child.nodeType === Node.TEXT_NODE && child.textContent) {
            textNodes.push(child.textContent);
        } else {
            textNodes.push(...getTextNodes(child));
        }
        if (child.nodeType === Node.ELEMENT_NODE && child.tagName === "STRONG") {
            textNodes.push("**");
        }
        if (child.nodeType === Node.ELEMENT_NODE && child.tagName === "DIV") {
            if (child.parentElement && child.parentElement.tagName !== "DIV") {
                textNodes.push("\\n");
            }
        }
    }
    return textNodes;
}
const allTexts = [];
for (const text of getTextNodes(rootElement)) {
    if (text.includes("KI-Antworten können Fehler enthalten.") || text.includes("AI responses may include mistakes.")) {
        break;
    }
    allTexts.push(text);
}
allTexts;""");
                if result:
                    first = True
                    for text in result:
                        if first:
                            first = False
                            debug.log(f"Google AI Mode: {text}")
                        else:
                            yield text
                    return
            raise RuntimeError("No AI mode results found.")
        finally:
            await session.close()
