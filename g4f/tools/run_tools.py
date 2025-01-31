from __future__ import annotations

import re
import json
import asyncio
import time
from pathlib import Path
from typing import Optional, Callable, AsyncIterator

from ..typing import Messages
from ..providers.helper import filter_none
from ..providers.asyncio import to_async_iterator
from ..providers.response import Reasoning
from ..providers.types import ProviderType
from ..cookies import get_cookies_dir
from .web_search import do_search, get_search_message
from .files import read_bucket, get_bucket_dir
from .. import debug

BUCKET_INSTRUCTIONS = """
Instruction: Make sure to add the sources of cites using [[domain]](Url) notation after the reference. Example: [[a-z0-9.]](http://example.com)
"""

def validate_arguments(data: dict) -> dict:
    if "arguments" in data:
        if isinstance(data["arguments"], str):
            data["arguments"] = json.loads(data["arguments"])
        if not isinstance(data["arguments"], dict):
            raise ValueError("Tool function arguments must be a dictionary or a json string")
        else:
            return filter_none(**data["arguments"])
    else:
        return {}

def get_api_key_file(cls) -> Path:
    return Path(get_cookies_dir()) / f"api_key_{cls.parent if hasattr(cls, 'parent') else cls.__name__}.json"

async def async_iter_run_tools(provider: ProviderType, model: str, messages, tool_calls: Optional[list] = None, **kwargs):
    # Handle web_search from kwargs
    web_search = kwargs.get('web_search')
    if web_search:
        try:
            messages = messages.copy()
            web_search = web_search if isinstance(web_search, str) and web_search != "true" else None
            messages[-1]["content"] = await do_search(messages[-1]["content"], web_search)
        except Exception as e:
            debug.log(f"Couldn't do web search: {e.__class__.__name__}: {e}")
            # Keep web_search in kwargs for provider native support
            pass

    # Read api_key from config file
    if getattr(provider, "needs_auth", False) and "api_key" not in kwargs:
        auth_file = get_api_key_file(provider)
        if auth_file.exists():
            with auth_file.open("r") as f:
                auth_result = json.load(f)
            if "api_key" in auth_result:
                kwargs["api_key"] = auth_result["api_key"]

    if tool_calls is not None:
        for tool in tool_calls:
            if tool.get("type") == "function":
                if tool.get("function", {}).get("name") == "search_tool":
                    tool["function"]["arguments"] = validate_arguments(tool["function"])
                    messages = messages.copy()
                    messages[-1]["content"] = await do_search(
                        messages[-1]["content"],
                        **tool["function"]["arguments"]
                    )
                elif tool.get("function", {}).get("name") == "continue":
                    last_line = messages[-1]["content"].strip().splitlines()[-1]
                    content = f"Carry on from this point:\n{last_line}"
                    messages.append({"role": "user", "content": content})
                elif tool.get("function", {}).get("name") == "bucket_tool":
                    def on_bucket(match):
                        return "".join(read_bucket(get_bucket_dir(match.group(1))))
                    has_bucket = False
                    for message in messages:
                        if "content" in message and isinstance(message["content"], str):
                            new_message_content = re.sub(r'{"bucket_id":"([^"]*)"}', on_bucket, message["content"])
                            if new_message_content != message["content"]:
                                has_bucket = True
                                message["content"] = new_message_content
                    if has_bucket and isinstance(messages[-1]["content"], str):
                        messages[-1]["content"] += BUCKET_INSTRUCTIONS
    create_function = provider.get_async_create_function()
    response = to_async_iterator(create_function(model=model, messages=messages, **kwargs))
    async for chunk in response:
        yield chunk

def iter_run_tools(
    iter_callback: Callable,
    model: str,
    messages: Messages,
    provider: Optional[str] = None,
    tool_calls: Optional[list] = None,
    **kwargs
) -> AsyncIterator:
    # Handle web_search from kwargs
    web_search = kwargs.get('web_search')
    if web_search:
        try:
            messages = messages.copy()
            web_search = web_search if isinstance(web_search, str) and web_search != "true" else None
            messages[-1]["content"] = asyncio.run(do_search(messages[-1]["content"], web_search))
        except Exception as e:
            debug.log(f"Couldn't do web search: {e.__class__.__name__}: {e}")
            # Keep web_search in kwargs for provider native support
            pass

    # Read api_key from config file
    if provider is not None and provider.needs_auth and "api_key" not in kwargs:
        auth_file = get_api_key_file(provider)
        if auth_file.exists():
            with auth_file.open("r") as f:
                auth_result = json.load(f)
            if "api_key" in auth_result:
                kwargs["api_key"] = auth_result["api_key"]

    if tool_calls is not None:
        for tool in tool_calls:
            if tool.get("type") == "function":
                if tool.get("function", {}).get("name") == "search_tool":
                    tool["function"]["arguments"] = validate_arguments(tool["function"])
                    messages[-1]["content"] = get_search_message(
                        messages[-1]["content"],
                        raise_search_exceptions=True,
                        **tool["function"]["arguments"]
                    )
                elif tool.get("function", {}).get("name") == "continue_tool":
                    if provider not in ("OpenaiAccount", "HuggingFace"):
                        last_line = messages[-1]["content"].strip().splitlines()[-1]
                        content = f"Carry on from this point:\n{last_line}"
                        messages.append({"role": "user", "content": content})
                    else:
                        # Enable provider native continue
                        if "action" not in kwargs:
                            kwargs["action"] = "continue"
                elif tool.get("function", {}).get("name") == "bucket_tool":
                    def on_bucket(match):
                        return "".join(read_bucket(get_bucket_dir(match.group(1))))
                    has_bucket = False
                    for message in messages:
                        if "content" in message and isinstance(message["content"], str):
                            new_message_content = re.sub(r'{"bucket_id":"([^"]*)"}', on_bucket, message["content"])
                            if new_message_content != message["content"]:
                                has_bucket = True
                                message["content"] = new_message_content
                    if has_bucket and isinstance(messages[-1]["content"], str):
                        messages[-1]["content"] += BUCKET_INSTRUCTIONS

    is_thinking = 0
    for chunk in iter_callback(model=model, messages=messages, provider=provider, **kwargs):
        if not isinstance(chunk, str):
            yield chunk
            continue
        if "<think>" in chunk:
            chunk = chunk.split("<think>", 1)
            yield chunk[0]
            yield Reasoning(is_thinking="<think>")
            yield Reasoning(chunk[1])
            yield Reasoning(None, "Is thinking...")
            is_thinking = time.time()
        if "</think>" in chunk:
            chunk = chunk.split("</think>", 1)
            yield Reasoning(chunk[0])
            yield Reasoning(is_thinking="</think>")
            yield Reasoning(None, f"Finished in {round(time.time()-is_thinking, 2)} seconds")
            yield chunk[1]
            is_thinking = 0
        elif is_thinking:
            yield Reasoning(chunk)
        else:
            yield chunk
