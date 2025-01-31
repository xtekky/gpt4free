from __future__ import annotations

from aiohttp import ClientSession, ClientTimeout, ClientError
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse, quote_plus
from datetime import datetime
import datetime
import asyncio

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException
    from bs4 import BeautifulSoup
    has_requirements = True
except ImportError:
    has_requirements = False
try:
    import spacy
    has_spacy = True
except:
    has_spacy = False

from typing import Iterator
from ..cookies import get_cookies_dir
from ..providers.response import format_link
from ..errors import MissingRequirementsError
from .. import debug

DEFAULT_INSTRUCTIONS = """
Using the provided web search results, to write a comprehensive reply to the user request.
Make sure to add the sources of cites using [[Number]](Url) notation after the reference. Example: [[0]](http://google.com)
"""

class SearchResults():
    def __init__(self, results: list, used_words: int):
        self.results = results
        self.used_words = used_words

    def __iter__(self):
        yield from self.results

    def __str__(self):
        search = ""
        for idx, result in enumerate(self.results):
            if search:
                search += "\n\n\n"
            search += f"Title: {result.title}\n\n"
            if result.text:
                search += result.text
            else:
                search += result.snippet
            search += f"\n\nSource: [[{idx}]]({result.url})"
        return search

    def __len__(self) -> int:
        return len(self.results)

class SearchResultEntry():
    def __init__(self, title: str, url: str, snippet: str, text: str = None):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.text = text

    def set_text(self, text: str):
        self.text = text

def scrape_text(html: str, max_words: int = None, add_source=True, count_images: int = 2) -> Iterator[str]:
    source = BeautifulSoup(html, "html.parser")
    soup = source
    for selector in [
            "main",
            ".main-content-wrapper",
            ".main-content",
            ".emt-container-inner",
            ".content-wrapper",
            "#content",
            "#mainContent",
        ]:
        select = soup.select_one(selector)
        if select:
            soup = select
            break
    # Zdnet
    for remove in [".c-globalDisclosure"]:
        select = soup.select_one(remove)
        if select:
            select.extract()

    image_select = "img[alt][src^=http]:not([alt=''])"
    image_link_select = f"a:has({image_select})"
    yield_words = []
    for paragraph in soup.select(f"h1, h2, h3, h4, h5, h6, p, table:not(:has(p)), ul:not(:has(p)), {image_link_select}"):
        if count_images > 0:
            image = paragraph.select_one(image_select)
            if image:
                title = str(paragraph.get("title", paragraph.text))
                if title:
                    yield f"!{format_link(image['src'], title)}\n"
                    if max_words is not None:
                        max_words -= 10
                    count_images -= 1
                continue

        for line in paragraph.get_text(" ").splitlines():
            words = [word for word in line.split() if word]
            count = len(words)
            if not count:
                continue
            words = " ".join(words)
            if words in yield_words:
                continue
            if max_words:
                max_words -= count
                if max_words <= 0:
                    break
            yield words + "\n"
            yield_words.append(words)

    if add_source:
        canonical_link = source.find("link", rel="canonical")
        if canonical_link and "href" in canonical_link.attrs:
            link = canonical_link["href"]
            domain = urlparse(link).netloc
            yield f"\nSource: [{domain}]({link})"

async def fetch_and_scrape(session: ClientSession, url: str, max_words: int = None, add_source: bool = False) -> str:
    try:
        bucket_dir: Path = Path(get_cookies_dir()) / ".scrape_cache" / "fetch_and_scrape"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        md5_hash = hashlib.md5(url.encode(errors="ignore")).hexdigest()
        cache_file = bucket_dir / f"{quote_plus(url.split('?')[0].split('//')[1].replace('/', ' ')[:48])}.{datetime.date.today()}.{md5_hash[:16]}.cache"
        if cache_file.exists():
            return cache_file.read_text()
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text(errors="replace")
                text = "".join(scrape_text(html, max_words, add_source))
                with open(cache_file, "wb") as f:
                    f.write(text.encode(errors="replace"))
                return text
    except (ClientError, asyncio.TimeoutError):
        return

async def search(query: str, max_results: int = 5, max_words: int = 2500, backend: str = "auto", add_text: bool = True, timeout: int = 5, region: str = "wt-wt") -> SearchResults:
    if not has_requirements:
        raise MissingRequirementsError('Install "duckduckgo-search" and "beautifulsoup4" package | pip install -U g4f[search]')
    with DDGS() as ddgs:
        results = []
        for result in ddgs.text(
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
                result["title"],
                result["href"],
                result["body"]
            ))

        if add_text:
            requests = []
            async with ClientSession(timeout=ClientTimeout(timeout)) as session:
                for entry in results:
                    requests.append(fetch_and_scrape(session, entry.url, int(max_words / (max_results - 1)), False))
                texts = await asyncio.gather(*requests)

        formatted_results = []
        used_words = 0
        left_words = max_words
        for i, entry in enumerate(results):
            if add_text:
                entry.text = texts[i]
            if max_words:
                left_words -= entry.title.count(" ") + 5
                if entry.text:
                    left_words -= entry.text.count(" ")
                else:
                    left_words -= entry.snippet.count(" ")
                if 0 > left_words:
                    break
            used_words = max_words - left_words
            formatted_results.append(entry)

        return SearchResults(formatted_results, used_words)

async def do_search(prompt: str, query: str = None, instructions: str = DEFAULT_INSTRUCTIONS, **kwargs) -> str:
    if instructions and instructions in prompt:
        return prompt # We have already added search results
    if query is None:
        query = spacy_get_keywords(prompt)
    json_bytes = json.dumps({"query": query, **kwargs}, sort_keys=True).encode(errors="ignore")
    md5_hash = hashlib.md5(json_bytes).hexdigest()
    bucket_dir: Path = Path(get_cookies_dir()) / ".scrape_cache" / f"web_search" / f"{datetime.date.today()}"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    cache_file = bucket_dir / f"{quote_plus(query[:20])}.{md5_hash}.cache"
    if cache_file.exists():
        with cache_file.open("r") as f:
            search_results = f.read()
    else:
        search_results = await search(query, **kwargs)
        if search_results.results:
            with cache_file.open("wb") as f:
                f.write(str(search_results).encode(errors="replace"))
    if instructions:
        new_prompt = f"""
{search_results}

Instruction: {instructions}

User request:
{prompt}
"""
    else:
        new_prompt = f"""
{search_results}

{prompt}
"""
    debug.log(f"Web search: '{query.strip()[:50]}...'")
    if isinstance(search_results, SearchResults):
        debug.log(f"with {len(search_results.results)} Results {search_results.used_words} Words")
    return new_prompt

def get_search_message(prompt: str, raise_search_exceptions=False, **kwargs) -> str:
    try:
        return asyncio.run(do_search(prompt, **kwargs))
    except (DuckDuckGoSearchException, MissingRequirementsError) as e:
        if raise_search_exceptions:
            raise e
        debug.log(f"Couldn't do web search: {e.__class__.__name__}: {e}")
        return prompt

def spacy_get_keywords(text: str):
    if not has_spacy:
        return text

    # Load the spaCy language model
    nlp = spacy.load("en_core_web_sm")

    # Process the query
    doc = nlp(text)

    # Extract keywords based on POS and named entities
    keywords = []
    for token in doc:
        # Filter for nouns, proper nouns, and adjectives
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and not token.is_stop:
            keywords.append(token.lemma_)

    # Add named entities as keywords
    for ent in doc.ents:
        keywords.append(ent.text)

    # Remove duplicates and print keywords
    keywords = list(set(keywords))
    #print("Keyword:", keywords)

    #keyword_freq = Counter(keywords)
    #keywords = keyword_freq.most_common()
    #print("Keyword Frequencies:", keywords)

    keywords = [chunk.text for chunk in doc.noun_chunks if not chunk.root.is_stop]
    #print("Phrases:", keywords)

    return keywords