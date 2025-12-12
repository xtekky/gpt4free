from __future__ import annotations

import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Iterator, Optional
from urllib.parse import urlparse, quote_plus
from aiohttp import ClientSession, ClientError
from datetime import date
import asyncio

try:
    from bs4 import BeautifulSoup, Tag
    has_requirements = True
except ImportError:
    has_requirements = False

from ..cookies import get_cookies_dir
from ..providers.response import format_link

def scrape_text(html: str, max_words: Optional[int] = None, add_source: bool = True, count_images: int = 2, add_metadata: bool = False) -> Iterator[str]:
    """
    Parses the provided HTML and yields text fragments.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Read the meta tags
    seen_texts = []  # Initialize seen_texts list to track already processed text
    if add_metadata:
        metadata: Dict[str, str] = {}

        if soup.title and soup.title.string:
            yield  f"## {soup.title.string}\n"
            seen_texts.append(soup.title.string)
            max_words = None if max_words is None else max_words - len(soup.title.string.split())

        for meta in soup(["meta"]):
            if not isinstance(meta, Tag):
                continue

            for a in meta.attrs:
                if a in ["itemprop", "property", "name"]:
                    key = str(meta.get(a, ""))
                    content = str(meta.get("content", ""))
                    if key and content:  # Only add non-empty content
                        metadata[key] = content
                    break

            description = metadata.get('description', metadata.get('og:description', '')).strip()
            if description:
                yield f"### Description\n{description}\n"
                seen_texts.append(description)
                max_words = None if max_words is None else max_words - len(description.split())

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

async def fetch_and_scrape(session: ClientSession, url: str, max_words: Optional[int] = None, add_source: bool = False, add_metadata: bool = False, proxy: str = None) -> str:
    """
    Fetches a URL and returns the scraped text, using caching to avoid redundant downloads.
    """
    try:
        cache_dir: Path = Path(get_cookies_dir()) / ".scrape_cache" / "fetch_and_scrape"
        cache_dir.mkdir(parents=True, exist_ok=True)
        md5_hash = hashlib.md5(url.encode(errors="ignore")+str([max_words, add_source, add_metadata]).encode(errors="ignore")).hexdigest()
        cache_file = cache_dir / f"{quote_plus(url.split('?')[0].split('//')[1].replace('/', ' ')[:48])}.{date.today()}.{md5_hash[:16]}.cache"
        if cache_file.exists():
            return cache_file.read_text()

        async with session.get(url, proxy=proxy) as response:
            if response.status == 200:
                html = await response.text(errors="replace")
                scraped_text = "".join(scrape_text(html, max_words, add_source, add_metadata=add_metadata))
                with open(cache_file, "wb") as f:
                    f.write(scraped_text.encode(errors="replace"))
                return scraped_text
    except (ClientError, asyncio.TimeoutError):
        return ""
    return ""