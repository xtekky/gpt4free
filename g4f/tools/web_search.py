from __future__ import annotations

from aiohttp import ClientSession, ClientTimeout, ClientError
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse, quote_plus
from datetime import date
import asyncio

# Optional dependencies using the new 'ddgs' package name
try:
    from ddgs import DDGS, DDGSError
    from bs4 import BeautifulSoup
    has_requirements = True
except ImportError:
    has_requirements = False

try:
    import spacy
    has_spacy = True
except ImportError:
    has_spacy = False

from typing import Iterator, List, Optional
from ..cookies import get_cookies_dir
from ..providers.response import format_link, JsonMixin, Sources
from ..errors import MissingRequirementsError
from .. import debug

DEFAULT_INSTRUCTIONS = """
Using the provided web search results, to write a comprehensive reply to the user request.
Make sure to add the sources of cites using [[Number]](Url) notation after the reference. Example: [[0]](http://google.com)
"""

class SearchResults(JsonMixin):
    """
    Represents a collection of search result entries along with the count of used words.
    """
    def __init__(self, results: List[SearchResultEntry], used_words: int):
        self.results = results
        self.used_words = used_words

    @classmethod
    def from_dict(cls, data: dict) -> SearchResults:
        return cls(
            [SearchResultEntry(**item) for item in data["results"]],
            data["used_words"]
        )

    def __iter__(self) -> Iterator[SearchResultEntry]:
        yield from self.results

    def __str__(self) -> str:
        # Build a string representation of the search results with markdown formatting.
        output = []
        for idx, result in enumerate(self.results):
            parts = [
                f"Title: {result.title}",
                "",
                result.text if result.text else result.snippet,
                "",
                f"Source: [[{idx}]]({result.url})"
            ]
            output.append("\n".join(parts))
        return "\n\n\n".join(output)

    def __len__(self) -> int:
        return len(self.results)

    def get_sources(self) -> Sources:
        return Sources([{"url": result.url, "title": result.title} for result in self.results])

    def get_dict(self) -> dict:
        return {
            "results": [result.get_dict() for result in self.results],
            "used_words": self.used_words
        }

class SearchResultEntry(JsonMixin):
    """
    Represents a single search result entry.
    """
    def __init__(self, title: str, url: str, snippet: str, text: Optional[str] = None):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.text = text

    def set_text(self, text: str) -> None:
        self.text = text

def scrape_text(html: str, max_words: Optional[int] = None, add_source: bool = True, count_images: int = 2) -> Iterator[str]:
    """
    Parses the provided HTML and yields text fragments.
    """
    soup = BeautifulSoup(html, "html.parser")
    for selector in [
        "main", ".main-content-wrapper", ".main-content", ".emt-container-inner",
        ".content-wrapper", "#content", "#mainContent",
    ]:
        selected = soup.select_one(selector)
        if selected:
            soup = selected
            break

    for remove_selector in [".c-globalDisclosure"]:
        unwanted = soup.select_one(remove_selector)
        if unwanted:
            unwanted.extract()

    image_selector = "img[alt][src^=http]:not([alt='']):not(.avatar):not([width])"
    image_link_selector = f"a:has({image_selector})"
    seen_texts = []
    
    for element in soup.select(f"h1, h2, h3, h4, h5, h6, p, pre, table:not(:has(p)), ul:not(:has(p)), {image_link_selector}"):
        if count_images > 0:
            image = element.select_one(image_selector)
            if image:
                title = str(element.get("title", element.text))
                if title:
                    yield f"!{format_link(image['src'], title)}\n"
                    if max_words is not None:
                        max_words -= 10
                    count_images -= 1
                continue

        for line in element.get_text(" ").splitlines():
            words = [word for word in line.split() if word]
            if not words:
                continue
            joined_line = " ".join(words)
            if joined_line in seen_texts:
                continue
            if max_words is not None:
                max_words -= len(words)
                if max_words <= 0:
                    break
            yield joined_line + "\n"
            seen_texts.append(joined_line)

    if add_source:
        canonical_link = soup.find("link", rel="canonical")
        if canonical_link and "href" in canonical_link.attrs:
            link = canonical_link["href"]
            domain = urlparse(link).netloc
            yield f"\nSource: [{domain}]({link})"

async def fetch_and_scrape(session: ClientSession, url: str, max_words: Optional[int] = None, add_source: bool = False) -> str:
    """
    Fetches a URL and returns the scraped text, using caching to avoid redundant downloads.
    """
    try:
        cache_dir: Path = Path(get_cookies_dir()) / ".scrape_cache" / "fetch_and_scrape"
        cache_dir.mkdir(parents=True, exist_ok=True)
        md5_hash = hashlib.md5(url.encode(errors="ignore")).hexdigest()
        cache_file = cache_dir / f"{quote_plus(url.split('?')[0].split('//')[1].replace('/', ' ')[:48])}.{date.today()}.{md5_hash[:16]}.cache"
        if cache_file.exists():
            return cache_file.read_text()

        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text(errors="replace")
                scraped_text = "".join(scrape_text(html, max_words, add_source))
                with open(cache_file, "wb") as f:
                    f.write(scraped_text.encode(errors="replace"))
                return scraped_text
    except (ClientError, asyncio.TimeoutError):
        return ""
    return ""

async def search(
    query: str,
    max_results: int = 5,
    max_words: int = 2500,
    backend: str = "auto",
    add_text: bool = True,
    timeout: int = 5,
    region: str = "wt-wt",
    provider: str = "DDG"
) -> SearchResults:
    """
    Performs a web search and returns search results.
    """
    if provider == "SearXNG":
        from ..Provider.SearXNG import SearXNG
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

    debug.log(f"[DuckDuckGo] Using local container for query: {query}")

    if not has_requirements:
        raise MissingRequirementsError('Install "ddgs" and "beautifulsoup4" | pip install -U g4f[search]')

    results: List[SearchResultEntry] = []
    # Use the new DDGS() context manager style
    async with DDGS() as ddgs:
        async for result in ddgs.text(
            query,
            region=region,
            safesearch="moderate",
            timelimit="y",
            max_results=max_results,
            backend=backend,
        ):
            if ".google." in result["href"]:
                continue
            results.append(SearchResultEntry(
                title=result["title"],
                url=result["href"],
                snippet=result["body"]
            ))

    if add_text:
        tasks = []
        async with ClientSession(timeout=ClientTimeout(timeout)) as session:
            for entry in results:
                tasks.append(fetch_and_scrape(session, entry.url, int(max_words / (max_results - 1)), False))
            texts = await asyncio.gather(*tasks)

    formatted_results: List[SearchResultEntry] = []
    used_words = 0
    left_words = max_words
    for i, entry in enumerate(results):
        if add_text:
            entry.text = texts[i]
        left_words -= entry.title.count(" ") + 5
        if entry.text:
            left_words -= entry.text.count(" ")
        else:
            left_words -= entry.snippet.count(" ")
        if left_words < 0:
            break
        used_words = max_words - left_words
        formatted_results.append(entry)

    return SearchResults(formatted_results, used_words)

async def do_search(
    prompt: str,
    query: Optional[str] = None,
    instructions: str = DEFAULT_INSTRUCTIONS,
    **kwargs
) -> tuple[str, Optional[Sources]]:
    """
    Combines search results with the user prompt, using caching for improved efficiency.
    """
    if not isinstance(prompt, str):
        return prompt, None

    if instructions and instructions in prompt:
        return prompt, None

    if prompt.startswith("##") and query is None:
        return prompt, None

    if query is None:
        query = prompt.strip().splitlines()[0]

    json_bytes = json.dumps({"query": query, **kwargs}, sort_keys=True).encode(errors="ignore")
    md5_hash = hashlib.md5(json_bytes).hexdigest()
    cache_dir: Path = Path(get_cookies_dir()) / ".scrape_cache" / "web_search" / f"{date.today()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{quote_plus(query[:20])}.{md5_hash}.cache"

    search_results: Optional[SearchResults] = None
    if cache_file.exists():
        with cache_file.open("r") as f:
            try:
                search_results = SearchResults.from_dict(json.loads(f.read()))
            except json.JSONDecodeError:
                search_results = None

    if search_results is None:
        search_results = await search(query, **kwargs)
        if search_results.results:
            with cache_file.open("w") as f:
                f.write(json.dumps(search_results.get_dict()))

    if instructions:
        new_prompt = f"{search_results}\n\nInstruction: {instructions}\n\nUser request:\n{prompt}"
    else:
        new_prompt = f"{search_results}\n\n{prompt}"

    debug.log(f"Web search: '{query.strip()[:50]}...'")
    debug.log(f"with {len(search_results.results)} Results {search_results.used_words} Words")
    return new_prompt.strip(), search_results.get_sources()

def get_search_message(prompt: str, raise_search_exceptions: bool = False, **kwargs) -> str:
    """
    Synchronously obtains the search message by wrapping the async search call.
    """
    try:
        result, _ = asyncio.run(do_search(prompt, **kwargs))
        return result
    # Use the new DDGSError exception
    except (DDGSError, MissingRequirementsError) as e:
        if raise_search_exceptions:
            raise e
        debug.error(f"Couldn't do web search: {e.__class__.__name__}: {e}")
        return prompt
