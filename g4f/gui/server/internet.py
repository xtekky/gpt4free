from __future__ import annotations

from aiohttp import ClientSession, ClientTimeout
try:
    from duckduckgo_search import DDGS
    from bs4 import BeautifulSoup
    has_requirements = True
except ImportError:
    has_requirements = False
from ...errors import MissingRequirementsError
from ... import debug

import asyncio
import random

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

async def fetch_and_scrape(session: ClientSession, url: str, max_words: int = None) -> str:
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    def get_headers(user_agent: str) -> dict:
        return {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5,uk;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
        }

    async def try_fetch(attempt: int = 0) -> str:
        try:
            current_user_agent = random.choice(user_agents)
            headers = get_headers(current_user_agent)
            
            if attempt > 0:
                await asyncio.sleep(1 + attempt)

            async with session.get(
                url,
                headers=headers,
                allow_redirects=True,
                timeout=ClientTimeout(total=10 + (attempt * 2)),
                ssl=False
            ) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    encoding = None
                    
                    if 'charset=' in content_type:
                        encoding = content_type.split('charset=')[-1].strip()
                    
                    try:
                        html = await response.text(encoding=encoding if encoding else 'utf-8')
                        return scrape_text(html, max_words)
                    except UnicodeDecodeError:
                        for enc in ['utf-8', 'cp1251', 'iso-8859-1']:
                            try:
                                html = await response.text(encoding=enc)
                                return scrape_text(html, max_words)
                            except UnicodeDecodeError:
                                continue
                        return None
                
                elif response.status == 403 and attempt < 3:
                    return await try_fetch(attempt + 1)
                else:
                    debug.log(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            if attempt < 3:
                return await try_fetch(attempt + 1)
            debug.log(f"Timeout fetching {url}")
            return None
        except Exception as e:
            debug.log(f"Failed to fetch {url}: {str(e)}")
            return None

    return await try_fetch()

def scrape_text(html: str, max_words: int = None) -> str:
    soup = BeautifulSoup(html, "html.parser")
    
    for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
        element.decompose()
        
    content_selectors = [
        "main", "article", ".main-content", "#content", ".content", "body"
    ]
    
    content = None
    for selector in content_selectors:
        content = soup.select_one(selector)
        if content:
            break
            
    if not content:
        content = soup
        
    text_elements = content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    clean_text = []
    word_count = 0
    
    for element in text_elements:
        text = element.get_text().strip()
        if len(text) < 10:
            continue
            
        text = ' '.join(text.split())
        words = text.split()
        
        if max_words and (word_count + len(words)) > max_words:
            remaining = max_words - word_count
            if remaining > 0:
                clean_text.append(' '.join(words[:remaining]))
            break
            
        clean_text.append(text)
        word_count += len(words)
        
        if max_words and word_count >= max_words:
            break
            
    return '\n'.join(clean_text)

async def search(query: str, n_results: int = 5, max_words: int = 2500, add_text: bool = True) -> SearchResults:
    if not has_requirements:
        raise MissingRequirementsError('Install "duckduckgo-search" and "beautifulsoup4" package | pip install -U g4f[search]')
    
    def perform_search(ddgs, backend: str) -> list:
        try:
            # For HTML backend, we need different parameters
            if backend == "html":
                return list(ddgs.text(
                    query,  # keywords as first positional argument
                    region="wt-wt",
                    safesearch="moderate",
                    backend=backend,
                    max_results=max(n_results * 2, 10)
                ))
            # For API backend
            return list(ddgs.text(
                query,  # keywords as first positional argument
                region="wt-wt",
                safesearch="moderate",
                timelimit="y",
                backend=backend,
                max_results=max(n_results * 2, 10)
            ))
        except Exception as e:
            debug.log(f"Search failed with backend {backend}: {str(e)}")
            return []

    try:
        with DDGS() as ddgs:
            results = []
            
            # Try HTML backend first since API is failing
            for backend in ["html", "lite", "api"]:
                search_results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: perform_search(ddgs, backend)
                )
                
                # Process valid results
                for result in search_results:
                    if isinstance(result, dict) and "title" in result and "href" in result:
                        # Skip results with empty or invalid URLs
                        if not result.get("href") or not result["href"].startswith(("http://", "https://")):
                            continue
                        
                        results.append(SearchResultEntry(
                            result.get("title", "").strip(),
                            result.get("href", "").strip(),
                            result.get("body", "").strip() or result.get("snippet", "").strip()
                        ))
                
                # If we have enough results, break
                if len(results) >= n_results:
                    break

            if not results:
                debug.log("No search results found")
                return SearchResults([], 0)

            # Limit to requested number of results
            results = results[:n_results]

            if add_text:
                async with ClientSession(timeout=ClientTimeout(10)) as session:
                    texts = await asyncio.gather(*[
                        fetch_and_scrape(session, entry.url, int(max_words / len(results)))
                        for entry in results
                    ])

            formatted_results = []
            used_words = 0
            left_words = max_words
            
            for i, entry in enumerate(results):
                if add_text and texts[i]:
                    entry.text = texts[i]
                if left_words > 0:
                    title_words = len(entry.title.split()) + 5
                    content_words = len(entry.text.split()) if entry.text else len(entry.snippet.split())
                    words_to_subtract = title_words + content_words
                    
                    left_words -= words_to_subtract
                    used_words += words_to_subtract
                    formatted_results.append(entry)

            return SearchResults(formatted_results, used_words)

    except Exception as e:
        debug.log(f"Search error: {e.__class__.__name__}: {e}")
        return SearchResults([], 0)

def get_search_message(prompt, n_results: int = 5, max_words: int = 2500) -> str:
    try:
        search_results = asyncio.run(search(prompt, n_results, max_words))
        if not search_results or len(search_results) == 0:
            return prompt
            
        message = f"""
{search_results}

Instruction: Using the provided web search results, to write a comprehensive reply to the user request.
Make sure to add the sources of cites using [[Number]](Url) notation after the reference. Example: [[0]](http://google.com)

User request:
{prompt}
"""
        debug.log(f"Web search: '{prompt.strip()[:50]}...' {search_results.used_words} Words")
        return message
    except Exception as e:
        debug.log(f"Couldn't do web search: {e.__class__.__name__}: {e}")
        return prompt
