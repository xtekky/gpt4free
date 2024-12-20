from __future__ import annotations

from aiohttp import ClientSession, ClientTimeout
from bs4 import BeautifulSoup
import asyncio
import logging
import urllib.parse
import re

# Logging
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
    
    # Fix DuckDuckGo redirect URLs
    if url.startswith('//'):
        url = 'https:' + url
    
    try:
        async with session.get(url, headers=headers, allow_redirects=True) as response:
            if response.status == 200:
                return await response.text()
    except Exception as e:
        debug.log(f"Error fetching {url}: {e}")
    return None

def extract_search_results(html: str) -> list:
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    
    # Search for results on the page
    for result in soup.select('.result'):
        try:
            title = result.select_one('.result__title').get_text(strip=True)
            url = result.select_one('.result__url')['href']
            snippet = result.select_one('.result__snippet').get_text(strip=True)
            
            results.append({
                'title': title,
                'url': url,
                'snippet': snippet
            })
        except Exception as e:
            debug.log(f"Error parsing result: {e}")
            continue
            
    return results

async def search_duckduckgo(query: str, max_results: int = 5) -> list:
    encoded_query = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    
    async with ClientSession() as session:
        html = await fetch_html(session, url)
        if html:
            results = extract_search_results(html)
            return results[:max_results]
    return []

def clean_text(text: str, max_words: int = None) -> str:
    # Remove extra spaces and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    
    if max_words:
        words = text.split()
        text = ' '.join(words[:max_words])
    
    return text

async def search(query: str, n_results: int = 5, max_words: int = 2500) -> SearchResults:
    try:
        # Get the search results
        results = await search_duckduckgo(query, n_results)
        
        if not results:
            debug.log("No search results found")
            return SearchResults([], 0)

        # Create result objects
        search_entries = []
        total_words = 0

        async with ClientSession() as session:
            for result in results:
                entry = SearchResultEntry(
                    title=result['title'],
                    url=result['url'],
                    snippet=result['snippet']
                )
                
                # Get the text of the page
                html = await fetch_html(session, result['url'])
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    # Getting the main content
                    content = ' '.join(p.get_text() for p in soup.select('p'))
                    content = clean_text(content, max_words // n_results)
                    entry.text = content
                    total_words += len(content.split())
                
                search_entries.append(entry)

        return SearchResults(search_entries, total_words)

    except Exception as e:
        debug.log(f"Search error: {e}")
        return SearchResults([], 0)

def get_search_message(prompt: str, n_results: int = 5, max_words: int = 2500) -> str:
    try:
        # Add a request log
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
