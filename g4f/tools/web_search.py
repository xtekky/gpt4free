from __future__ import annotations

import asyncio
from typing import Optional

try:
    from ddgs.exceptions import DDGSException
except ImportError:
    from typing import Type as DDGSException

from ..providers.response import Sources
from ..errors import MissingRequirementsError
from ..Provider.search.CachedSearch import CachedSearch
from .. import debug

DEFAULT_INSTRUCTIONS = """
Using the provided web search results, to write a comprehensive reply to the user request.
Make sure to add the sources of cites using [[Number]](Url) notation after the reference. Example: [[0]](http://google.com)
"""

async def do_search(
    prompt: str,
    query: Optional[str] = None,
    instructions: str = DEFAULT_INSTRUCTIONS,
    **kwargs
) -> tuple[str, Optional[Sources]]:
    if not prompt or not isinstance(prompt, str):
        return

    if instructions and instructions in prompt:
        return

    if prompt.startswith("##") and query is None:
        return

    if query is None:
        query = prompt.strip().splitlines()[0]

    search_results = await anext(CachedSearch.create_async_generator(
        "",
        [],
        prompt=query,
        **kwargs
    ))

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
    except (DDGSException, MissingRequirementsError) as e:
        if raise_search_exceptions:
            raise e
        debug.error(f"Couldn't do web search:", e)
        return prompt
