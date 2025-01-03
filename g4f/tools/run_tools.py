from __future__ import annotations

import re
import json
import asyncio
from typing import Optional, Callable, AsyncIterator

from ..typing import Messages
from ..providers.helper import filter_none
from ..providers.asyncio import to_async_iterator
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

async def async_iter_run_tools(async_iter_callback, model, messages, tool_calls: Optional[list] = None, **kwargs):
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

    response = to_async_iterator(async_iter_callback(model=model, messages=messages, **kwargs))
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
                elif tool.get("function", {}).get("name") == "safe_search_tool":
                    tool["function"]["arguments"] = validate_arguments(tool["function"])
                    try:
                        messages[-1]["content"] = asyncio.run(do_search(messages[-1]["content"], **tool["function"]["arguments"]))
                    except Exception as e:
                        debug.log(f"Couldn't do web search: {e.__class__.__name__}: {e}")
                        # Enable provider native web search
                        kwargs["web_search"] = True
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

    return iter_callback(model=model, messages=messages, provider=provider, **kwargs)