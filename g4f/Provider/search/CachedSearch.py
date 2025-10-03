from __future__ import annotations

import json
import hashlib
from pathlib import Path
from urllib.parse import quote_plus
from datetime import date

from ...typing import AsyncResult, Messages, Optional
from ..base_provider import AsyncGeneratorProvider, AuthFileMixin
from ...cookies import get_cookies_dir
from ..helper import format_media_prompt
from .DDGS import DDGS, SearchResults, SearchResultEntry
from .SearXNG import SearXNG
from ... import debug

async def search(
    query: str,
    max_results: int = 5,
    max_words: int = 2500,
    backend: str = "auto",
    add_text: bool = True,
    timeout: int = 5,
    region: str = "us-en",
    provider: str = "DDG"
) -> SearchResults:
    """
    Performs a web search and returns search results.
    """
    if provider == "SearXNG":
        debug.log(f"[SearXNG] Using local container for query: {query}")
        results_texts = []
        async for chunk in SearXNG.create_async_generator(
            "SearXNG",
            [{"role": "user", "content": query}],
            max_results=max_results,
            max_words=max_words,
            add_text=add_text
        ):
            if isinstance(chunk, str):
                results_texts.append(chunk)
        used_words = sum(text.count(" ") for text in results_texts)
        return SearchResults([
            SearchResultEntry(
                title=f"Result {i + 1}",
                url="",
                snippet=text,
                text=text
            ) for i, text in enumerate(results_texts)
        ], used_words=used_words)

    return await anext(DDGS.create_async_generator(
        provider,
        [],
        prompt=query,
        max_results=max_results,
        max_words=max_words,
        add_text=add_text,
        timeout=timeout,
        region=region,
        backend=backend
    ))

class CachedSearch(AsyncGeneratorProvider, AuthFileMixin):
    working = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        **kwargs
    ) -> AsyncResult:
        """
        Combines search results with the user prompt, using caching for improved efficiency.
        """
        prompt = format_media_prompt(messages, prompt)
        search_parameters = ["max_results", "max_words", "add_text", "timeout", "region"]
        search_parameters = {k: v for k, v in kwargs.items() if k in search_parameters}
        json_bytes = json.dumps({"model": model, "query": prompt, **search_parameters}, sort_keys=True).encode(errors="ignore")
        md5_hash = hashlib.md5(json_bytes).hexdigest()
        cache_dir: Path = Path(get_cookies_dir()) / ".scrape_cache" / "web_search" / f"{date.today()}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{quote_plus(prompt[:20])}.{md5_hash}.cache"

        search_results: Optional[SearchResults] = None
        if cache_file.exists():
            with cache_file.open("r") as f:
                try:
                    search_results = SearchResults.from_dict(json.loads(f.read()))
                except json.JSONDecodeError:
                    search_results = None

        if search_results is None:
            if model:
                search_parameters["provider"] = model
            search_results = await search(prompt, **search_parameters)
            if search_results.results:
                with cache_file.open("w") as f:
                    f.write(json.dumps(search_results.get_dict()))

        yield search_results