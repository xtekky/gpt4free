import os
import aiohttp
import asyncio
from ..typing import Messages, AsyncResult
from ..providers.base_provider import AsyncGeneratorProvider
from ..providers.response import FinishReason
from ..tools.web_search import fetch_and_scrape 

class SearXNG(AsyncGeneratorProvider):
    default_url = os.environ.get("SEARXNG_URL", "http://searxng:8080")
    label = "SearXNG"
  
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 30,
        max_results: int = 5,
        max_words: int = 2500,
        add_text: bool = True,
        **kwargs
    ) -> AsyncResult:
        url = cls.default_url
        query = messages[-1]["content"] if isinstance(messages[-1], dict) else getattr(messages[-1], "content", "")


        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            params = {
                "q": query,
                "format": "json",
                "language": "it",
                "safesearch": 0,
                "categories": "general",
            }

            async with session.get(f"{url}/search", params=params) as resp:
                print(f"Request URL on SearXNG: {resp.url}")
                data = await resp.json()
                results = data.get("results", [])

                if not results:
                    yield "Nessun risultato trovato."
                    yield FinishReason("stop")
                    return

                if add_text:
                    requests = []
                    for r in results[:max_results]:
                        requests.append(fetch_and_scrape(session, r["url"], int(max_words / max_results), False))
                    texts = await asyncio.gather(*requests)
                    for i, r in enumerate(results[:max_results]):
                        r["text"] = texts[i]

                formatted = ""
                used_words = 0
                for i, r in enumerate(results[:max_results]):
                    title = r.get("title", "Senza titolo")
                    url = r.get("url", "#")
                    content = r.get("text") or r.get("snippet") or ""
                    formatted += f"Title: {title}\n\n{content}\n\nSource: [[{i}]]({url})\n\n"
                    used_words += content.count(" ")
                    if max_words and used_words >= max_words:
                        break

                yield formatted.strip()
                yield FinishReason("stop")
