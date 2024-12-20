from __future__ import annotations

from aiohttp import ClientSession
from bs4 import BeautifulSoup
import asyncio
import re
from duckduckgo_search import AsyncDDGS

from ... import debug

class SearchResultEntry:
    def __init__(self, title: str, url: str, snippet: str, text: str = None):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.text = text

class SearchResults:
    def __init__(self, results: list, used_words: int):
        self.results = results
        self.used_words = used_words

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

async def fetch_html(session: ClientSession, url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        async with session.get(url, headers=headers, allow_redirects=True) as response:
            if response.status == 200:
                return await response.text()
    except Exception as e:
        debug.log(f"Error fetching {url}: {e}")
    return None

def clean_text(text: str, max_words: int = None) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    if max_words:
        words = text.split()
        text = ' '.join(words[:max_words])
    return text

async def search(query: str, n_results: int = 5, max_words: int = 2500) -> SearchResults:
    try:
        async with AsyncDDGS() as ddgs:
            results = await ddgs.atext(
                keywords=query,
                max_results=n_results,
                backend='lite'
            )
            
            if not results:
                results = await ddgs.atext(
                    keywords=query,
                    max_results=n_results,
                    backend='api'
                )
                
            if not results:
                debug.log("No search results found")
                return SearchResults([], 0)

            search_entries = []
            total_words = 0

            async with ClientSession() as session:
                for result in results:
                    url = result.get('link') or result.get('url') or result.get('href', '')
                    
                    entry = SearchResultEntry(
                        title=result.get('title', ''),
                        url=url,
                        snippet=result.get('body', '') or result.get('snippet', '') or result.get('text', '')
                    )
                    
                    if entry.url:
                        html = await fetch_html(session, entry.url)
                        if html:
                            soup = BeautifulSoup(html, 'html.parser')
                            text_elements = soup.select('p, article, .content, .main-content')
                            content = ' '.join(elem.get_text() for elem in text_elements)
                            content = clean_text(content, max_words // n_results)
                            entry.text = content
                            total_words += len(content.split())
                    
                    search_entries.append(entry)

            return SearchResults(search_entries, total_words)

    except Exception as e:
        debug.log(f"Search error: {str(e)}")
        try:
            async with AsyncDDGS() as ddgs:
                results = await ddgs.atext(
                    keywords=query,
                    max_results=n_results,
                    backend='api'
                )
                if results:
                    search_entries = [
                        SearchResultEntry(
                            title=r.get('title', ''),
                            url=r.get('link', ''),
                            snippet=r.get('body', ''),
                            text=None
                        ) for r in results
                    ]
                    return SearchResults(search_entries, 0)
        except Exception as e2:
            debug.log(f"Alternative search error: {str(e2)}")
        return SearchResults([], 0)

def get_search_message(prompt: str, n_results: int = 5, max_words: int = 2500) -> str:
    try:
        debug.log(f"Web search: '{prompt.strip()[:50]}...'")
        
        search_results = asyncio.run(search(prompt, n_results, max_words))
        
        if not search_results or len(search_results) == 0:
            debug.log("No search results found")
            return prompt

        message = f"""
{search_results}

Instruction: Using the provided web search results, write a comprehensive reply to the user request.
Make sure to add the sources of cites using [[Number]](Url) notation after the reference.

User request:
{prompt}
"""
        debug.log(f"Search completed: {len(search_results)} results, {search_results.used_words} words")
        return message

    except Exception as e:
        debug.log(f"Error in get_search_message: {e}")
        return prompt
