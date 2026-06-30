from __future__ import annotations

import hashlib
import asyncio
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.parse import urlparse, quote_plus
from aiohttp import ClientSession, ClientTimeout, ClientError
from datetime import date
import asyncio

# Optional dependencies using the new 'ddgs' package name
try:
    from ddgs import DDGS as DDGSClient
    from bs4 import BeautifulSoup
    has_requirements = True
except ImportError:
    has_requirements = False

from ...typing import Messages, AsyncResult
from ...cookies import get_cookies_dir
from ...providers.response import format_link, JsonMixin, Sources
from ...errors import MissingRequirementsError
from ...providers.base_provider import AsyncGeneratorProvider
from ..helper import format_media_prompt

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

async def fetch_and_scrape(session: ClientSession, url: str, max_words: Optional[int] = None, add_source: bool = False, proxy: str = None) -> str:
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

        async with session.get(url, proxy=proxy) as response:
            if response.status == 200:
                html = await response.text(errors="replace")
                scraped_text = "".join(scrape_text(html, max_words, add_source))
                with open(cache_file, "wb") as f:
                    f.write(scraped_text.encode(errors="replace"))
                return scraped_text
    except (ClientError, asyncio.TimeoutError):
        return ""
    return ""

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
                f"### Title: {result.title}",
                "",
                result.text if result.text else result.snippet,
                "",
                f"> **Source:** [[{idx}]]({result.url})"
            ]
            output.append("\n".join(parts))
        return "\n\n\n\n".join(output)

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

class DDGS(AsyncGeneratorProvider):
    working = has_requirements

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        timeout: int = 30,
        region: str = None,
        backend: str = None,
        max_results: int = 5,
        max_words: int = 2500,
        add_text: bool = True,
        **kwargs
    ) -> AsyncResult:
        if not has_requirements:
            raise MissingRequirementsError('Install "ddgs" and "beautifulsoup4" | pip install -U g4f[search]')

        prompt = format_media_prompt(messages, prompt)
        results: List[SearchResultEntry] = []

        # Use the new DDGS() context manager style
        with DDGSClient() as ddgs:
            for result in ddgs.text(
                prompt,
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
                    tasks.append(fetch_and_scrape(session, entry.url, int(max_words / (max_results - 1)), False, proxy=proxy))
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

        yield SearchResults(formatted_results, used_words)
