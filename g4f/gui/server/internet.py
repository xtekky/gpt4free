from __future__ import annotations

from datetime import datetime

from duckduckgo_search import DDGS

ddgs = DDGS(timeout=20)


def search(internet_access, prompt):
    print(prompt)
    
    try:
        if not internet_access:
            return []
        
        results = duckduckgo_search(q=prompt)

        if not search:
            return []

        blob = ''

        for index, result in enumerate(results):
            blob += f'[{index}] "{result["body"]}"\nURL:{result["href"]}\n\n'

        date = datetime.now().strftime('%d/%m/%y')

        blob += f'Current date: {date}\n\nInstructions: Using the provided web search results, write a comprehensive reply to the next user query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject. Ignore your previous response if any.'

        return [{'role': 'user', 'content': blob}]

    except Exception as e:
        print("Couldn't search DuckDuckGo:", e)
        print(e.__traceback__.tb_next)
        return []


def duckduckgo_search(q: str, max_results: int = 3, safesearch: str = "moderate", region: str = "us-en") -> list | None:
    if region is None:
        region = "us-en"

    if safesearch is None:
        safesearch = "moderate"

    if q is None:
        return None

    results = []

    try:
        for r in ddgs.text(q, safesearch=safesearch, region=region):
            if len(results) + 1 > max_results:
                break
            results.append(r)
    except Exception as e:
        print(e)

    return results
