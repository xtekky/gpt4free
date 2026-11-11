"""Catalog of free and paid browsing / web APIs.

The catalog is intentionally declarative: entries describe how an API is
reached and which environment variable holds its credential.  No credential is
ever hard-coded here and entries without a configured key are simply reported as
``available: False`` by :func:`list_apis`.

A spec looks like::

    {
        "id": "rss",
        "name": "RSS / Atom feeds",
        "category": "feeds",
        "pricing": "free",
        "requires_key": False,
        "env_var": None,
        "capabilities": ["feed", "fetch"],
        "docs_url": "https://www.rssboard.org/rss-specification",
    }
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

# ``pricing`` is one of: free, free-tier, paid.
API_CATALOG: List[Dict] = [
    {
        "id": "rss",
        "name": "RSS / Atom feeds",
        "category": "feeds",
        "pricing": "free",
        "requires_key": False,
        "env_var": None,
        "base_url": "",
        "capabilities": ["feed", "fetch"],
        "docs_url": "https://www.rssboard.org/rss-specification",
        "description": "Read any public RSS/Atom feed and render it as a live conversation.",
    },
    {
        "id": "wikipedia",
        "name": "Wikipedia",
        "category": "reference",
        "pricing": "free",
        "requires_key": False,
        "env_var": None,
        "base_url": "https://en.wikipedia.org/w/api.php",
        "capabilities": ["search", "fetch"],
        "docs_url": "https://www.mediawiki.org/wiki/API:Search",
        "description": "Full-text search and article summaries without an account.",
    },
    {
        "id": "hackernews",
        "name": "Hacker News (Algolia)",
        "category": "news",
        "pricing": "free",
        "requires_key": False,
        "env_var": None,
        "base_url": "https://hn.algolia.com/api/v1",
        "capabilities": ["search", "feed"],
        "docs_url": "https://hn.algolia.com/api",
        "description": "Search stories and comments from Hacker News.",
    },
    {
        "id": "duckduckgo",
        "name": "DuckDuckGo",
        "category": "search",
        "pricing": "free",
        "requires_key": False,
        "env_var": None,
        "base_url": "https://duckduckgo.com",
        "capabilities": ["search"],
        "docs_url": "https://duckduckgo.com",
        "description": "Keyless web search through the optional ddgs package.",
    },
    {
        "id": "searxng",
        "name": "SearXNG",
        "category": "search",
        "pricing": "free",
        "requires_key": False,
        "env_var": "G4F_SEARXNG_URL",
        "base_url": "",
        "capabilities": ["search"],
        "docs_url": "https://docs.searxng.org/dev/search_api.html",
        "description": "Self-hosted metasearch; set G4F_SEARXNG_URL to your instance.",
    },
    {
        "id": "reddit",
        "name": "Reddit",
        "category": "social",
        "pricing": "free",
        "requires_key": False,
        "env_var": None,
        "base_url": "https://www.reddit.com",
        "capabilities": ["search", "feed"],
        "docs_url": "https://www.reddit.com/dev/api/",
        "description": "Public JSON search and subreddit feeds.",
    },
    {
        "id": "github",
        "name": "GitHub",
        "category": "code",
        "pricing": "free-tier",
        "requires_key": False,
        "env_var": "GITHUB_TOKEN",
        "base_url": "https://api.github.com",
        "capabilities": ["search"],
        "docs_url": "https://docs.github.com/en/rest/search",
        "description": "Repository search; an optional token raises the rate limit.",
    },
    {
        "id": "arxiv",
        "name": "arXiv",
        "category": "research",
        "pricing": "free",
        "requires_key": False,
        "env_var": None,
        "base_url": "https://export.arxiv.org/api",
        "capabilities": ["search", "feed"],
        "docs_url": "https://info.arxiv.org/help/api/index.html",
        "description": "Search scientific preprints and read them as an Atom feed.",
    },
    {
        "id": "jina-reader",
        "name": "Jina Reader",
        "category": "reader",
        "pricing": "free-tier",
        "requires_key": False,
        "env_var": "JINA_API_KEY",
        "base_url": "https://r.jina.ai",
        "capabilities": ["fetch"],
        "docs_url": "https://jina.ai/reader/",
        "description": "Fast URL to clean markdown reader; anonymous use is rate limited.",
    },
    {
        "id": "brave",
        "name": "Brave Search",
        "category": "search",
        "pricing": "free-tier",
        "requires_key": True,
        "env_var": "BRAVE_API_KEY",
        "base_url": "https://api.search.brave.com/res/v1",
        "capabilities": ["search"],
        "docs_url": "https://brave.com/search/api/",
        "description": "Independent web index with a free developer tier.",
    },
    {
        "id": "tavily",
        "name": "Tavily",
        "category": "search",
        "pricing": "free-tier",
        "requires_key": False,
        "env_var": "TAVILY_API_KEY",
        "base_url": "https://api.tavily.com",
        "capabilities": ["search", "fetch"],
        "docs_url": "https://docs.tavily.com/",
        "description": "Search API with a keyless access mode; set TAVILY_API_KEY for higher limits.",
    },
    {
        "id": "serper",
        "name": "Serper",
        "category": "search",
        "pricing": "free-tier",
        "requires_key": True,
        "env_var": "SERPER_API_KEY",
        "base_url": "https://google.serper.dev",
        "capabilities": ["search"],
        "docs_url": "https://serper.dev/",
        "description": "Google search results as JSON.",
    },
    {
        "id": "exa",
        "name": "Exa",
        "category": "search",
        "pricing": "free-tier",
        "requires_key": False,
        "env_var": "EXA_API_KEY",
        "base_url": "https://mcp.exa.ai/mcp",
        "capabilities": ["search"],
        "docs_url": "https://docs.exa.ai/",
        "description": "Neural web search via Exa's keyless MCP endpoint; set EXA_API_KEY for higher limits.",
    },
    {
        "id": "parallel",
        "name": "Parallel",
        "category": "search",
        "pricing": "free",
        "requires_key": False,
        "env_var": "PARALLEL_API_KEY",
        "base_url": "https://search.parallel.ai/mcp",
        "capabilities": ["search"],
        "docs_url": "https://parallel.ai/",
        "description": "Keyless Parallel Search MCP; set PARALLEL_API_KEY for authenticated usage.",
    },
    {
        "id": "firecrawl",
        "name": "Firecrawl",
        "category": "search",
        "pricing": "free-tier",
        "requires_key": False,
        "env_var": "FIRECRAWL_API_KEY",
        "base_url": "https://mcp.firecrawl.dev/v2/mcp",
        "capabilities": ["search", "fetch"],
        "docs_url": "https://docs.firecrawl.dev/",
        "description": "Keyless web search via the Firecrawl MCP; a key enables crawling/scraping.",
    },
    {
        "id": "tinyfish",
        "name": "TinyFish",
        "category": "search",
        "pricing": "free",
        "requires_key": False,
        "env_var": "TINYFISH_API_KEY",
        "base_url": "https://agent.tinyfish.ai/mcp",
        "capabilities": ["search"],
        "docs_url": "https://tinyfish.ai/",
        "description": "Keyless TinyFish MCP search; set TINYFISH_API_KEY for authenticated usage.",
    },
    {
        "id": "keenable",
        "name": "Keenable",
        "category": "search",
        "pricing": "free-tier",
        "requires_key": False,
        "env_var": "KEENABLE_API_KEY",
        "base_url": "https://api.keenable.ai/v1",
        "capabilities": ["search", "fetch"],
        "docs_url": "https://docs.keenable.ai/",
        "description": "Independent web index for agents with keyless search and markdown fetch; set KEENABLE_API_KEY for higher limits.",
    },
    {
        "id": "newsapi",
        "name": "NewsAPI",
        "category": "news",
        "pricing": "free-tier",
        "requires_key": True,
        "env_var": "NEWSAPI_KEY",
        "base_url": "https://newsapi.org/v2",
        "capabilities": ["search", "feed"],
        "docs_url": "https://newsapi.org/docs",
        "description": "Headlines and article search across news sources.",
    },
    {
        "id": "firecrawl",
        "name": "Firecrawl",
        "category": "reader",
        "pricing": "paid",
        "requires_key": True,
        "env_var": "FIRECRAWL_API_KEY",
        "base_url": "https://api.firecrawl.dev/v1",
        "capabilities": ["fetch"],
        "docs_url": "https://docs.firecrawl.dev/",
        "description": "Crawl and convert pages to markdown.",
    },
]


def _get_env(env_var: Optional[str]) -> Optional[str]:
    if not env_var:
        return None
    value = os.environ.get(env_var)
    return value.strip() if value else None


def api_available(spec: Dict) -> bool:
    """Return True when a spec can be used with the current environment."""
    if spec.get("id") == "searxng":
        return bool(_get_env(spec.get("env_var")))
    if not spec.get("requires_key"):
        return True
    return bool(_get_env(spec.get("env_var")))


def serialize_api(spec: Dict) -> Dict:
    """Return a copy of *spec* with runtime availability information."""
    data = dict(spec)
    data["available"] = api_available(spec)
    # Never expose the credential itself, only whether it is set.
    data["credential_set"] = bool(_get_env(spec.get("env_var"))) if spec.get("env_var") else False
    return data


def get_api(api_id: str) -> Optional[Dict]:
    if not api_id:
        return None
    api_id = api_id.strip().lower()
    for spec in API_CATALOG:
        if spec["id"] == api_id:
            return spec
    return None


def list_apis(
    category: Optional[str] = None,
    capability: Optional[str] = None,
    include_unavailable: bool = True,
) -> List[Dict]:
    specs: Iterable[Dict] = API_CATALOG
    if category:
        category = category.lower()
        specs = [spec for spec in specs if spec["category"] == category]
    if capability:
        capability = capability.lower()
        specs = [spec for spec in specs if capability in spec["capabilities"]]
    result = [serialize_api(spec) for spec in specs]
    if not include_unavailable:
        result = [spec for spec in result if spec["available"]]
    return result


DEFAULT_SEARCH_APIS = (
    "exa",
    "parallel",
    "tavily",
    "firecrawl",
    "tinyfish",
    "wikipedia",
    "hackernews",
    "duckduckgo",
)


def resolve_apis(api_ids: Optional[Iterable[str]] = None) -> List[Dict]:
    """Resolve ids to available specs.

    Without explicit ids the keyless OpenCode-style providers are preferred,
    followed by Wikipedia, Hacker News and DuckDuckGo.  Override with
    ``G4F_SEARCH_APIS=exa,wikipedia``.
    """
    specs: List[Dict] = []
    for api_id in api_ids or []:
        spec = get_api(api_id)
        if spec and api_available(spec):
            specs.append(spec)
    if specs:
        return specs

    configured = os.environ.get("G4F_SEARCH_APIS", "")
    default_ids = (
        [item.strip() for item in configured.split(",") if item.strip()]
        if configured
        else list(DEFAULT_SEARCH_APIS)
    )
    for api_id in default_ids:
        spec = get_api(api_id)
        if spec and api_available(spec) and "search" in spec["capabilities"]:
            specs.append(spec)
    if specs:
        return specs
    return [
        spec
        for spec in API_CATALOG
        if api_available(spec) and "search" in spec["capabilities"]
    ]
