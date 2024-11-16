from __future__ import annotations

from aiohttp import ClientSession, ClientTimeout
try:
    from duckduckgo_search import DDGS
    from bs4 import BeautifulSoup
    has_requirements = True
except ImportError:
    has_requirements = False
from ...errors import MissingRequirementsError
    
import asyncio

class SearchResults():
    def __init__(self, results: list):
        self.results = results

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

def scrape_text(html: str, max_words: int = None) -> str:
    soup = BeautifulSoup(html, "html.parser")
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
    clean_text = ""
    for paragraph in soup.select("p, h1, h2, h3, h4, h5, h6"):
        text = paragraph.get_text()
        for line in text.splitlines():
            words = []
            for word in line.replace("\t", " ").split(" "):
                if word:
                    words.append(word)
            count = len(words)
            if not count:
                continue
            if max_words:
                max_words -= count
                if max_words <= 0:
                    break
            if clean_text:
                clean_text += "\n"
            clean_text += " ".join(words)

    return clean_text

async def fetch_and_scrape(session: ClientSession, url: str, max_words: int = None) -> str:
    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                return scrape_text(html, max_words)
    except:
        return

async def search(query: str, n_results: int = 5, max_words: int = 2500, add_text: bool = True) -> SearchResults:
    if not has_requirements:
        raise MissingRequirementsError('Install "duckduckgo-search" and "beautifulsoup4" package | pip install -U g4f[search]')
    with DDGS() as ddgs:
        results = []
        for result in ddgs.text(
                query,
                region="wt-wt",
                safesearch="moderate",
                timelimit="y",
                max_results=n_results
            ):
            results.append(SearchResultEntry(
                result["title"],
                result["href"],
                result["body"]
            ))

        if add_text:
            requests = []
            async with ClientSession(timeout=ClientTimeout(5)) as session:
                for entry in results:
                    requests.append(fetch_and_scrape(session, entry.url, int(max_words / (n_results - 1))))
                texts = await asyncio.gather(*requests)

        formatted_results = []
        left_words = max_words
        for i, entry in enumerate(results):
            if add_text:
                entry.text = texts[i]
            if left_words:
                left_words -= entry.title.count(" ") + 5
                if entry.text:
                    left_words -= entry.text.count(" ")
                else:
                    left_words -= entry.snippet.count(" ")
                if 0 > left_words:
                    break
            formatted_results.append(entry)

        return SearchResults(formatted_results)

def get_search_message(prompt) -> str:
    try:
        search_results = asyncio.run(search(prompt))
        message = f"""
{search_results}


Instruction: Using the provided web search results, to write a comprehensive reply to the user request.
Make sure to add the sources of cites using [[Number]](Url) notation after the reference. Example: [[0]](http://google.com)

User request:
{prompt}
"""
        return message
    except Exception as e:
        print("Couldn't do web search:", e)
        return prompt