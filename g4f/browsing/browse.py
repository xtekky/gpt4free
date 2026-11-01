"""Fast browsing and search across free/paid web APIs.

The module keeps network access in one place so providers, tools and HTTP
routes can share it.  Every function accepts a timeout and never raises for a
single failed source in multi-source operations; failures are reported as data
instead of aborting the whole browse.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote_plus, urlencode

from .catalog import get_api, resolve_apis, serialize_api
from .feeds import _validate_url, fetch_feed, parse_feed

_MAX_PAGE_BYTES = 3 * 1024 * 1024
_WS_RE = re.compile(r"\s+")


def _clean(value: Optional[str]) -> str:
    if not value:
        return ""
    from .feeds import clean_text

    return clean_text(str(value))


def _headers(api_id: str = "") -> Dict[str, str]:
    headers = {
        "User-Agent": os.environ.get(
            "G4F_BROWSING_USER_AGENT",
            "g4f-browsing/1.0 (+https://g4f.dev)",
        ),
        "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    }
    if api_id == "github" and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        headers["Accept"] = "application/vnd.github+json"
    return headers


async def _get_json(session, url: str, params: Optional[Dict] = None, headers=None):
    async with session.get(url, params=params, headers=headers or _headers()) as response:
        if response.status >= 400:
            raise ValueError(f"HTTP {response.status} from {url}")
        return await response.json(content_type=None)


async def _get_text(session, url: str, params: Optional[Dict] = None, headers=None):
    async with session.get(url, params=params, headers=headers or _headers()) as response:
        if response.status >= 400:
            raise ValueError(f"HTTP {response.status} from {url}")
        raw = await response.content.read(_MAX_PAGE_BYTES)
        return raw.decode(response.charset or "utf-8", errors="replace")


def extract_text(html_text: str, max_words: Optional[int] = 600) -> str:
    """Extract readable text; use BeautifulSoup when available."""
    if not html_text:
        return ""
    try:
        from ..tools.fetch_and_scrape import scrape_text

        return "".join(scrape_text(html_text, max_words=max_words))
    except Exception:
        pass
    text = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html_text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _WS_RE.sub(" ", text).strip()
    if max_words:
        text = " ".join(text.split()[:max_words])
    return text


async def fetch_page(
    session,
    url: str,
    max_words: int = 600,
    timeout: float = 15.0,
    allow_local: bool = False,
) -> Dict:
    _validate_url(url, allow_local=allow_local)
    started = time.time()
    client_timeout = getattr(session, "_timeout", None)
    try:
        async with session.get(
            url,
            headers=_headers(),
            timeout=client_timeout or timeout,
            allow_redirects=True,
        ) as response:
            raw = await response.content.read(_MAX_PAGE_BYTES)
            charset = response.charset or "utf-8"
            content_type = response.headers.get("content-type", "")
            text = raw.decode(charset, errors="replace")
            if "json" in content_type:
                content = text[:max_words * 8] if max_words else text
            elif "html" in content_type or text.lstrip().startswith("<"):
                content = extract_text(text, max_words=max_words)
            else:
                content = _WS_RE.sub(" ", text).strip()
                if max_words:
                    content = " ".join(content.split()[:max_words])
            return {
                "url": str(response.url),
                "status": response.status,
                "content_type": content_type,
                "title": _html_title(text) if "html" in content_type else "",
                "content": content,
                "elapsed_ms": int((time.time() - started) * 1000),
            }
    except Exception as exc:
        return {
            "url": url,
            "status": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": int((time.time() - started) * 1000),
        }


def _html_title(html_text: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html_text)
    return _clean(match.group(1)) if match else ""


async def fast_browse(
    urls: Iterable[str],
    max_words: int = 600,
    timeout: float = 15.0,
    concurrency: int = 6,
    allow_local: bool = False,
) -> List[Dict]:
    """Fetch several URLs concurrently and return readable text."""
    import aiohttp

    urls = [url for url in urls if isinstance(url, str) and url.strip()]
    if not urls:
        return []
    semaphore = asyncio.Semaphore(max(1, concurrency))
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async def run(url: str) -> Dict:
            async with semaphore:
                return await fetch_page(
                    session,
                    url.strip(),
                    max_words=max_words,
                    timeout=timeout,
                    allow_local=allow_local,
                )

        return list(await asyncio.gather(*(run(url) for url in urls)))


def browse(urls: Iterable[str], **kwargs) -> List[Dict]:
    return asyncio.run(fast_browse(urls, **kwargs))


def _result(api: str, title: str, url: str, snippet: str = "", published: str = "") -> Dict:
    return {
        "api": api,
        "title": _clean(title),
        "url": url or "",
        "snippet": _clean(snippet)[:1200],
        "published": published or "",
    }


async def _search_wikipedia(session, query: str, limit: int) -> List[Dict]:
    data = await _get_json(
        session,
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": limit,
        },
    )
    results = []
    for row in (data.get("query") or {}).get("search") or []:
        title = row.get("title", "")
        results.append(
            _result(
                "wikipedia",
                title,
                f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
                row.get("snippet", ""),
            )
        )
    return results


async def _search_hackernews(session, query: str, limit: int) -> List[Dict]:
    data = await _get_json(
        session,
        "https://hn.algolia.com/api/v1/search",
        {"query": query, "tags": "story", "hitsPerPage": limit},
    )
    return [
        _result(
            "hackernews",
            hit.get("title") or hit.get("story_title") or "",
            hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            hit.get("story_text") or hit.get("comment_text") or "",
            hit.get("created_at") or "",
        )
        for hit in data.get("hits") or []
    ]


async def _search_reddit(session, query: str, limit: int) -> List[Dict]:
    data = await _get_json(
        session,
        "https://www.reddit.com/search.json",
        {"q": query, "limit": limit, "sort": "relevance"},
        headers=_headers(),
    )
    results = []
    for child in (data.get("data") or {}).get("children") or []:
        row = child.get("data") or {}
        results.append(
            _result(
                "reddit",
                row.get("title", ""),
                "https://www.reddit.com" + (row.get("permalink") or ""),
                row.get("selftext", "") or row.get("url_overridden_by_dest", ""),
                row.get("created_utc") and time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(row["created_utc"])
                ),
            )
        )
    return results


async def _search_arxiv(session, query: str, limit: int) -> List[Dict]:
    text = await _get_text(
        session,
        "https://export.arxiv.org/api/query",
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
        },
    )
    feed = parse_feed(text, source_url="https://arxiv.org", limit=limit)
    return [
        _result(
            "arxiv",
            item.get("title", ""),
            item.get("link", ""),
            item.get("summary", ""),
            item.get("published", ""),
        )
        for item in feed.get("items") or []
    ]


async def _search_github(session, query: str, limit: int) -> List[Dict]:
    data = await _get_json(
        session,
        "https://api.github.com/search/repositories",
        {"q": query, "per_page": limit},
        headers=_headers("github"),
    )
    return [
        _result(
            "github",
            row.get("full_name", ""),
            row.get("html_url", ""),
            row.get("description") or "",
            row.get("updated_at") or "",
        )
        for row in data.get("items") or []
    ]


async def _search_searxng(session, query: str, limit: int) -> List[Dict]:
    base = (os.environ.get("G4F_SEARXNG_URL") or "").rstrip("/")
    if not base:
        raise ValueError("G4F_SEARXNG_URL is not configured")
    data = await _get_json(
        session,
        f"{base}/search",
        {"q": query, "format": "json", "safesearch": 1},
    )
    return [
        _result("searxng", row.get("title", ""), row.get("url", ""), row.get("content", ""))
        for row in (data.get("results") or [])[:limit]
    ]


async def _search_brave(session, query: str, limit: int) -> List[Dict]:
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        raise ValueError("BRAVE_API_KEY is not configured")
    data = await _get_json(
        session,
        "https://api.search.brave.com/res/v1/web/search",
        {"q": query, "count": limit},
        headers={**_headers(), "X-Subscription-Token": key},
    )
    return [
        _result("brave", row.get("title", ""), row.get("url", ""), row.get("description", ""))
        for row in ((data.get("web") or {}).get("results") or [])[:limit]
    ]


async def _search_tavily(session, query: str, limit: int) -> List[Dict]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY is not configured")
    async with session.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": limit},
        headers=_headers(),
    ) as response:
        if response.status >= 400:
            raise ValueError(f"HTTP {response.status} from Tavily")
        data = await response.json(content_type=None)
    return [
        _result("tavily", row.get("title", ""), row.get("url", ""), row.get("content", ""))
        for row in (data.get("results") or [])[:limit]
    ]


async def _search_serper(session, query: str, limit: int) -> List[Dict]:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise ValueError("SERPER_API_KEY is not configured")
    async with session.post(
        "https://google.serper.dev/search",
        json={"q": query, "num": limit},
        headers={**_headers(), "X-API-KEY": key},
    ) as response:
        if response.status >= 400:
            raise ValueError(f"HTTP {response.status} from Serper")
        data = await response.json(content_type=None)
    rows = (data.get("organic") or [])[:limit]
    return [
        _result("serper", row.get("title", ""), row.get("link", ""), row.get("snippet", ""))
        for row in rows
    ]


async def _search_exa(session, query: str, limit: int) -> List[Dict]:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise ValueError("EXA_API_KEY is not configured")
    async with session.post(
        "https://api.exa.ai/search",
        json={"query": query, "numResults": limit, "contents": {"text": {"maxCharacters": 600}}},
        headers={**_headers(), "x-api-key": key},
    ) as response:
        if response.status >= 400:
            raise ValueError(f"HTTP {response.status} from Exa")
        data = await response.json(content_type=None)
    return [
        _result("exa", row.get("title", ""), row.get("url", ""), row.get("text", ""), row.get("publishedDate", ""))
        for row in (data.get("results") or [])[:limit]
    ]


async def _search_newsapi(session, query: str, limit: int) -> List[Dict]:
    key = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWS_API_KEY")
    if not key:
        raise ValueError("NEWSAPI_KEY is not configured")
    data = await _get_json(
        session,
        "https://newsapi.org/v2/everything",
        {"q": query, "pageSize": limit, "sortBy": "publishedAt", "apiKey": key},
    )
    return [
        _result("newsapi", row.get("title", ""), row.get("url", ""), row.get("description", ""), row.get("publishedAt", ""))
        for row in (data.get("articles") or [])[:limit]
    ]


async def _search_duckduckgo(session, query: str, limit: int) -> List[Dict]:
    def run():
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=limit))

    rows = await asyncio.to_thread(run)
    return [
        _result("duckduckgo", row.get("title", ""), row.get("href", ""), row.get("body", ""))
        for row in rows
    ]


_SEARCHERS = {
    "wikipedia": _search_wikipedia,
    "hackernews": _search_hackernews,
    "reddit": _search_reddit,
    "arxiv": _search_arxiv,
    "github": _search_github,
    "searxng": _search_searxng,
    "brave": _search_brave,
    "tavily": _search_tavily,
    "serper": _search_serper,
    "exa": _search_exa,
    "newsapi": _search_newsapi,
    "duckduckgo": _search_duckduckgo,
}


async def search(
    query: str,
    apis: Optional[Iterable[str]] = None,
    limit: int = 10,
    timeout: float = 20.0,
    dedupe: bool = True,
) -> Dict:
    """Search one or more APIs concurrently.

    Returns ``{"query", "results", "errors", "apis"}``.  Results are
    de-duplicated by URL and interleaved by source so a single slow/empty
    provider cannot hide the others.
    """
    import aiohttp

    query = (query or "").strip()
    if not query:
        return {"query": "", "results": [], "errors": {}, "apis": []}
    specs = resolve_apis(apis)
    selected = [spec for spec in specs if spec["id"] in _SEARCHERS]
    if apis:
        # Explicit ids that are unavailable should be reported, not silently
        # replaced by the defaults.
        requested = [str(api).strip().lower() for api in apis]
        selected = [spec for spec in specs if spec["id"] in _SEARCHERS]
        for api_id in requested:
            if api_id in _SEARCHERS and not any(spec["id"] == api_id for spec in selected):
                spec = get_api(api_id)
                if spec:
                    selected.append(spec)

    limit = max(1, min(int(limit or 10), 50))
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    errors: Dict[str, str] = {}
    per_api: List[List[Dict]] = []

    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async def run(spec: Dict):
            api_id = spec["id"]
            try:
                results = await asyncio.wait_for(
                    _SEARCHERS[api_id](session, query, limit), timeout=timeout
                )
                return api_id, results, None
            except Exception as exc:
                return api_id, [], f"{type(exc).__name__}: {exc}"

        outcomes = await asyncio.gather(*(run(spec) for spec in selected))
        for api_id, results, error in outcomes:
            if error:
                errors[api_id] = error
            if results:
                per_api.append(results)

    results: List[Dict] = []
    seen_urls = set()
    index = 0
    while any(index < len(group) for group in per_api):
        for group in per_api:
            if index < len(group):
                row = group[index]
                url = row.get("url") or ""
                if dedupe and url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                results.append(row)
                if len(results) >= limit:
                    break
        if len(results) >= limit:
            break
        index += 1

    return {
        "query": query,
        "results": results[:limit],
        "errors": errors,
        "apis": [spec["id"] for spec in selected],
    }


async def execute_api(
    api_id: str,
    params: Optional[Dict] = None,
    timeout: float = 20.0,
) -> Dict:
    """Execute an allow-listed catalog API operation.

    Supported operations are derived from each catalog entry's capabilities.
    Credentials are read from the environment only.
    """
    import aiohttp

    params = dict(params or {})
    spec = get_api(api_id)
    if spec is None:
        raise ValueError(f"Unknown browsing API '{api_id}'")
    serialized = serialize_api(spec)
    if not serialized["available"]:
        env_var = spec.get("env_var") or "the required credential"
        raise ValueError(f"'{api_id}' is not configured (set {env_var})")
    operation = (params.pop("operation", None) or "").lower()
    if not operation:
        if params.get("url") and "feed" in spec["capabilities"]:
            operation = "feed"
        elif params.get("url"):
            operation = "fetch"
        else:
            operation = "search"

    if operation in ("search", "query"):
        return await search(
            params.get("query") or params.get("q") or "",
            apis=[api_id],
            limit=int(params.get("limit") or 10),
            timeout=timeout,
        )

    if operation in ("feed", "rss"):
        url = params.get("url") or params.get("feed_url")
        if not url:
            raise ValueError("A feed url is required")
        feed = await fetch_feed(url, limit=int(params.get("limit") or 50), timeout=timeout)
        return {"api": api_id, "feed": feed}

    if operation in ("fetch", "scrape"):
        url = params.get("url")
        if not url:
            raise ValueError("A url is required")
        if api_id == "jina-reader":
            reader_url = f"https://r.jina.ai/{url}"
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                text = await _get_text(session, reader_url, headers=_headers())
            return {"api": api_id, "url": url, "content": text[:20000]}
        if api_id == "firecrawl":
            key = os.environ.get("FIRECRAWL_API_KEY")
            if not key:
                raise ValueError("FIRECRAWL_API_KEY is not configured")
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    json={"url": url, "formats": ["markdown"]},
                    headers={**_headers(), "Authorization": f"Bearer {key}"},
                ) as response:
                    if response.status >= 400:
                        raise ValueError(f"HTTP {response.status} from Firecrawl")
                    data = await response.json(content_type=None)
            return {"api": api_id, "url": url, "data": data}
        results = await fast_browse(
            [url],
            max_words=params.get("max_words", 1200),
            timeout=timeout,
            allow_local=params.get("allow_local", False),
        )
        return {"api": api_id, "result": results[0] if results else None}

    raise ValueError(f"Unsupported operation '{operation}' for '{api_id}'")
