from __future__ import annotations

import re
import json
import os
import math
import asyncio
import time
import datetime
from pathlib import Path
from typing import Optional, AsyncIterator, Iterator, Dict, Any, Tuple, List, Union

try:
    from aiofile import async_open

    has_aiofile = True
except ImportError:
    has_aiofile = False

from ..typing import Messages
from ..providers.helper import filter_none
from ..providers.asyncio import to_async_iterator, to_sync_generator
from ..providers.response import Reasoning, FinishReason, Sources, Usage, ProviderInfo
from ..providers.types import ProviderType
from ..cookies import get_cookies_dir
from ..config import AppConfig
from .web_search import do_search, get_search_message
from .auth import AuthManager
from .files import read_bucket, get_bucket_dir
from .. import debug

# Provider tool_choice capabilities
# Some providers don't support all tool_choice options
PROVIDER_TOOL_CHOICE_SUPPORT = {
    "Antigravity": ["auto"],  # Only supports "auto", fails on "required"
    "default": ["auto", "none", "required"]
}


def get_provider_tool_choice(provider_name: str, requested_tool_choice) -> str:
    """
    Get provider-compatible tool_choice value.
    
    Args:
        provider_name: Name of the provider
        requested_tool_choice: Tool choice requested by user
    
    Returns:
        Provider-compatible tool_choice value
    """
    supported = PROVIDER_TOOL_CHOICE_SUPPORT.get(provider_name, PROVIDER_TOOL_CHOICE_SUPPORT["default"])
    
    # If tool_choice is a dict, extract function name
    if isinstance(requested_tool_choice, dict):
        # Dict format is generally supported
        return requested_tool_choice
    
    # Check if requested choice is supported
    if requested_tool_choice in supported:
        return requested_tool_choice
    
    # Fall back to "auto" for unsupported choices
    debug.log(f"Provider {provider_name} doesn't support tool_choice='{requested_tool_choice}', using 'auto'")
    return "auto"

TOOL_NAMES = {
    "SEARCH": "search_tool",
    "CONTINUE": "continue_tool",
    "BUCKET": "bucket_tool",
    # Filesystem tools
    "READ_FILE": "read_file",
    "WRITE_FILE": "write_file",
    "LIST_DIRECTORY": "list_directory",
    "CREATE_DIRECTORY": "create_directory",
    "DELETE_FILE": "delete_file",
    "DELETE_DIRECTORY": "delete_directory",
    "MOVE_FILE": "move_file",
    "COPY_FILE": "copy_file",
    "SEARCH_FILES": "search_files",
    "GET_FILE_INFO": "get_file_info",
    "FILE_EXISTS": "file_exists",
    "SEARCH_IN_FILES": "search_in_files",
}


class ToolHandler:
    """Handles processing of different tool types"""

    @staticmethod
    def validate_arguments(data: dict) -> dict:
        """Validate and parse tool arguments"""
        if "arguments" in data:
            if isinstance(data["arguments"], str):
                data["arguments"] = json.loads(data["arguments"])
            if not isinstance(data["arguments"], dict):
                raise ValueError(
                    "Tool function arguments must be a dictionary or a json string"
                )
            else:
                return filter_none(**data["arguments"])
        else:
            return {}

    @staticmethod
    async def process_search_tool(messages: Messages, tool: dict) -> Messages:
        """Process search tool requests"""
        messages = messages.copy()
        args = ToolHandler.validate_arguments(tool["function"])
        messages[-1]["content"], sources = await do_search(
            messages[-1]["content"], **args
        )
        return messages, sources

    @staticmethod
    def process_continue_tool(
        messages: Messages, tool: dict, provider: Any
    ) -> Tuple[Messages, Dict[str, Any]]:
        """Process continue tool requests"""
        kwargs = {}
        if provider not in ("OpenaiAccount", "HuggingFaceAPI"):
            messages = messages.copy()
            last_line = messages[-1]["content"].strip().splitlines()[-1]
            content = f"Carry on from this point:\n{last_line}"
            messages.append({"role": "user", "content": content})
        else:
            # Enable provider native continue
            kwargs["action"] = "continue"
        return messages, kwargs

    @staticmethod
    def process_bucket_tool(messages: Messages, tool: dict) -> Messages:
        """Process bucket tool requests"""
        messages = messages.copy()

        def on_bucket(match):
            return "".join(read_bucket(get_bucket_dir(match.group(1))))

        has_bucket = False
        for message in messages:
            if "content" in message and isinstance(message["content"], str):
                new_message_content = re.sub(
                    r'{"bucket_id":\s*"([^"]*)"}', on_bucket, message["content"]
                )
                if new_message_content != message["content"]:
                    has_bucket = True
                    message["content"] = new_message_content

        last_message_content = messages[-1]["content"]
        if has_bucket and isinstance(last_message_content, str):
            if "\nSource: " in last_message_content:
                messages[-1]["content"] = last_message_content + BUCKET_INSTRUCTIONS

        return messages

    @staticmethod
    async def process_filesystem_tool(
        messages: Messages,
        tool: dict,
        provider: Any
    ) -> Tuple[Messages, Dict[str, Any]]:
        """
        Process filesystem tool requests.

        Executes filesystem operations and adds results to messages.
        """
        from .filesystem import execute_filesystem_tool

        messages = messages.copy()
        function_name = tool.get("function", {}).get("name")
        arguments = ToolHandler.validate_arguments(tool["function"])

        debug.log(f"Executing filesystem tool: {function_name}")
        debug.log(f"Arguments: {arguments}")

        # Execute the filesystem tool
        result = execute_filesystem_tool(function_name, arguments)

        # Add tool response to messages
        if result.get("success"):
            # Format result as tool response
            response_content = json.dumps(result, indent=2, default=str)
            messages.append({
                "role": "tool",
                "tool_call_id": tool.get("id", f"fs_{function_name}"),
                "content": response_content
            })
            debug.log(f"Filesystem tool result: {result}")
        else:
            # Add error response
            error_content = f"Error executing {function_name}: {result.get('error', 'Unknown error')}"
            messages.append({
                "role": "tool",
                "tool_call_id": tool.get("id", f"fs_{function_name}"),
                "content": error_content
            })
            debug.log(f"Filesystem tool error: {result}")

        return messages, {}

    @staticmethod
    async def process_tools(
        messages: Messages, tool_calls: List[dict], provider: Any
    ) -> Tuple[Messages, Dict[str, Any]]:
        """Process all tool calls and return updated messages and kwargs"""
        if not tool_calls:
            return messages, {}

        extra_kwargs = {}
        messages = messages.copy()
        sources = None

        # Filesystem tool names
        fs_tools = {
            "read_file", "write_file", "list_directory", "create_directory",
            "delete_file", "delete_directory", "move_file", "copy_file",
            "search_files", "get_file_info", "file_exists", "search_in_files"
        }

        for tool in tool_calls:
            if tool.get("type") != "function":
                continue

            function_name = tool.get("function", {}).get("name")

            debug.log(f"Processing tool call: {function_name}")

            # Check if it's a filesystem tool
            if function_name in fs_tools:
                messages, kwargs = await ToolHandler.process_filesystem_tool(
                    messages, tool, provider
                )
                extra_kwargs.update(kwargs)

            elif function_name == TOOL_NAMES["SEARCH"]:
                messages, sources = await ToolHandler.process_search_tool(
                    messages, tool
                )

            elif function_name == TOOL_NAMES["CONTINUE"]:
                messages, kwargs = ToolHandler.process_continue_tool(
                    messages, tool, provider
                )
                extra_kwargs.update(kwargs)

            elif function_name == TOOL_NAMES["BUCKET"]:
                messages = ToolHandler.process_bucket_tool(messages, tool)

        return messages, sources, extra_kwargs


class ThinkingProcessor:
    """Processes thinking chunks"""

    @staticmethod
    def process_thinking_chunk(
        chunk: str, start_time: float = 0
    ) -> Tuple[float, List[Union[str, Reasoning]]]:
        """Process a thinking chunk and return timing and results."""
        results = []

        # Handle non-thinking chunk
        if not start_time and "<think>" not in chunk and "</think>" not in chunk:
            return 0, [chunk]

        # Handle thinking start
        if "<think>" in chunk and "`<think>`" not in chunk:
            before_think, *after = chunk.split("<think>", 1)

            if before_think:
                results.append(before_think)

            results.append(Reasoning(status="🤔 Is thinking...", is_thinking="<think>"))

            if after:
                if "</think>" in after[0]:
                    after, *after_end = after[0].split("</think>", 1)
                    results.append(Reasoning(after))
                    results.append(Reasoning(status="", is_thinking="</think>"))
                    if after_end:
                        results.append(after_end[0])
                    return 0, results
                else:
                    results.append(Reasoning(after[0]))

            return time.time(), results

        # Handle thinking end
        if "</think>" in chunk:
            before_end, *after = chunk.split("</think>", 1)

            if before_end:
                results.append(Reasoning(before_end))

            thinking_duration = time.time() - start_time if start_time > 0 else 0

            status = (
                f"Thought for {thinking_duration:.2f}s" if thinking_duration > 1 else ""
            )
            results.append(Reasoning(status=status, is_thinking="</think>"))

            # Make sure to handle text after the closing tag
            if after and after[0].strip():
                results.append(after[0])

            return 0, results

        # Handle ongoing thinking
        if start_time:
            return start_time, [Reasoning(chunk)]

        return start_time, [chunk]


async def perform_web_search(
    messages: Messages, web_search_param: Any
) -> Tuple[Messages, Optional[Sources]]:
    """Perform web search and return updated messages and sources"""
    messages = messages.copy()
    sources = None

    if not web_search_param:
        return messages, sources

    try:
        search_query = (
            web_search_param
            if isinstance(web_search_param, str) and web_search_param != "true"
            else None
        )
        messages[-1]["content"], sources = await do_search(
            messages[-1]["content"], search_query
        )
    except Exception as e:
        debug.error(f"Couldn't do web search:", e)

    return messages, sources


async def async_iter_run_tools(
    provider: ProviderType,
    model: str,
    messages: Messages,
    tool_calls: Optional[List[dict]] = None,
    **kwargs,
) -> AsyncIterator:
    """Asynchronously run tools and yield results"""

    tool_emulation = kwargs.pop("tool_emulation", None)
    if tool_emulation is None:
        tool_emulation = os.environ.get("G4F_TOOL_EMULATION", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    stream = bool(kwargs.get("stream"))
    tools = kwargs.get("tools")
    if tool_emulation and tools and not tool_calls:
        from ..providers.tool_support import ToolSupportProvider

        emu_kwargs = dict(kwargs)
        emu_kwargs.pop("tools", None)
        tool_choice = emu_kwargs.pop("tool_choice", None)
        emu_kwargs.pop("parallel_tool_calls", None)
        emu_kwargs.pop("stream", None)
        emu_kwargs.pop("stream_timeout", None)
        
        # Get provider-compatible tool_choice
        provider_name = getattr(provider, '__name__', type(provider).__name__)
        tool_choice = get_provider_tool_choice(provider_name, tool_choice)
        
        async for chunk in ToolSupportProvider.create_async_generator(
            model=model,
            messages=messages,
            stream=stream,
            media=kwargs.get("media"),
            tools=tools,
            tool_choice=tool_choice,
            provider=provider,
            **emu_kwargs,
        ):
            yield chunk
        return

    # Process web search
    sources = None
    web_search = kwargs.get("web_search")
    if web_search:
        debug.log(f"Performing web search with value: {web_search}")
        messages, sources = await perform_web_search(messages, web_search)

    # Get API key
    if not kwargs.get("api_key") or AppConfig.disable_custom_api_key:
        api_key = AuthManager.load_api_key(provider)
        if api_key:
            kwargs["api_key"] = api_key

    # Process tool calls
    if tool_calls:
        messages, sources, extra_kwargs = await ToolHandler.process_tools(
            messages, tool_calls, provider
        )
        kwargs.update(extra_kwargs)

    # Generate response
    response = to_async_iterator(
        provider.async_create_function(model=model, messages=messages, **kwargs)
    )

    try:
        usage_model = model
        usage_provider = provider.__name__
        completion_tokens = 0
        usage = None
        async for chunk in response:
            if isinstance(chunk, FinishReason):
                if sources is not None:
                    yield sources
                    sources = None
                yield chunk
                continue
            elif isinstance(chunk, Sources):
                sources = None
            elif isinstance(chunk, str):
                completion_tokens += round(len(chunk.encode("utf-8")) / 4)
            elif isinstance(chunk, ProviderInfo):
                usage_model = getattr(chunk, "model", usage_model)
                usage_provider = getattr(chunk, "name", usage_provider)
            elif isinstance(chunk, Usage):
                usage = chunk
            yield chunk
        if usage is None:
            usage = get_usage(messages, completion_tokens)
            yield usage
        usage = {
            "user": kwargs.get("user"),
            "model": usage_model,
            "provider": usage_provider,
            **usage.get_dict(),
        }
        usage_dir = Path(get_cookies_dir()) / ".usage"
        usage_file = usage_dir / f"{datetime.date.today()}.jsonl"
        usage_dir.mkdir(parents=True, exist_ok=True)
        if has_aiofile:
            async with async_open(usage_file, "a") as f:
                asyncio.create_task(f.write(f"{json.dumps(usage)}\n"))
        else:
            with usage_file.open("a") as f:
                f.write(f"{json.dumps(usage)}\n")
        if completion_tokens > 0:
            provider.live += 1
    except:
        provider.live -= 1
        raise

    # Yield sources if available
    if sources is not None:
        yield sources


def iter_run_tools(
    provider: ProviderType,
    model: str,
    messages: Messages,
    tool_calls: Optional[List[dict]] = None,
    **kwargs,
) -> Iterator:
    """Run tools synchronously and yield results"""

    tool_emulation = kwargs.pop("tool_emulation", None)
    if tool_emulation is None:
        tool_emulation = os.environ.get("G4F_TOOL_EMULATION", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    stream = bool(kwargs.get("stream"))
    tools = kwargs.get("tools")
    if tool_emulation and tools and not tool_calls:
        from ..providers.tool_support import ToolSupportProvider

        emu_kwargs = dict(kwargs)
        emu_kwargs.pop("tools", None)
        tool_choice = emu_kwargs.pop("tool_choice", None)
        emu_kwargs.pop("parallel_tool_calls", None)
        emu_kwargs.pop("stream", None)
        emu_kwargs.pop("stream_timeout", None)
        
        # Get provider-compatible tool_choice
        provider_name = getattr(provider, '__name__', type(provider).__name__)
        tool_choice = get_provider_tool_choice(provider_name, tool_choice)
        
        yield from to_sync_generator(
            ToolSupportProvider.create_async_generator(
                model=model,
                messages=messages,
                stream=stream,
                media=kwargs.get("media"),
                tools=tools,
                tool_choice=tool_choice,
                provider=provider,
                **emu_kwargs,
            ),
            stream=stream,
        )
        return
    # Process web search
    web_search = kwargs.get("web_search")
    sources = None

    if web_search:
        debug.log(f"Performing web search with value: {web_search}")
        try:
            messages = messages.copy()
            search_query = (
                web_search
                if isinstance(web_search, str) and web_search != "true"
                else None
            )
            # Note: Using asyncio.run inside sync function is not ideal, but maintaining original pattern
            messages[-1]["content"], sources = asyncio.run(
                do_search(messages[-1]["content"], search_query)
            )
        except Exception as e:
            debug.error(f"Couldn't do web search:", e)

    # Get API key if needed
    if not kwargs.get("api_key") or AppConfig.disable_custom_api_key:
        api_key = AuthManager.load_api_key(provider)
        if api_key:
            kwargs["api_key"] = api_key

    # Process tool calls
    if tool_calls:
        for tool in tool_calls:
            if tool.get("type") == "function":
                function_name = tool.get("function", {}).get("name")
                debug.log(f"Processing tool call: {function_name}")
                if function_name == TOOL_NAMES["SEARCH"]:
                    tool["function"]["arguments"] = ToolHandler.validate_arguments(
                        tool["function"]
                    )
                    messages[-1]["content"] = get_search_message(
                        messages[-1]["content"],
                        raise_search_exceptions=True,
                        **tool["function"]["arguments"],
                    )
                elif function_name == TOOL_NAMES["CONTINUE"]:
                    if provider.__name__ not in ("OpenaiAccount", "HuggingFace"):
                        last_line = messages[-1]["content"].strip().splitlines()[-1]
                        content = f"Carry on from this point:\n{last_line}"
                        messages.append({"role": "user", "content": content})
                    else:
                        # Enable provider native continue
                        kwargs["action"] = "continue"
                elif function_name == TOOL_NAMES["BUCKET"]:

                    def on_bucket(match):
                        return "".join(read_bucket(get_bucket_dir(match.group(1))))

                    has_bucket = False
                    for message in messages:
                        if "content" in message and isinstance(message["content"], str):
                            new_message_content = re.sub(
                                r'{"bucket_id":"([^"]*)"}',
                                on_bucket,
                                message["content"],
                            )
                            if new_message_content != message["content"]:
                                has_bucket = True
                                message["content"] = new_message_content
                    last_message = messages[-1]["content"]
                    if has_bucket and isinstance(last_message, str):
                        if "\nSource: " in last_message:
                            messages[-1]["content"] = last_message + BUCKET_INSTRUCTIONS

    # Process response chunks
    try:
        thinking_start_time = 0
        processor = ThinkingProcessor()
        usage_model = model
        usage_provider = provider.__name__
        completion_tokens = 0
        usage = None
        for chunk in provider.create_function(
            model=model, messages=messages, provider=provider, **kwargs
        ):
            if isinstance(chunk, FinishReason):
                if sources is not None:
                    yield sources
                    sources = None
                yield chunk
                continue
            elif isinstance(chunk, Sources):
                sources = None
            elif isinstance(chunk, str):
                completion_tokens += round(len(chunk.encode("utf-8")) / 4)
            elif isinstance(chunk, ProviderInfo):
                usage_model = getattr(chunk, "model", usage_model)
                usage_provider = getattr(chunk, "name", usage_provider)
            elif isinstance(chunk, Usage):
                usage = chunk
            if not isinstance(chunk, str):
                yield chunk
                continue

            thinking_start_time, results = processor.process_thinking_chunk(
                chunk, thinking_start_time
            )
            for result in results:
                yield result
        if usage is None:
            usage = get_usage(messages, completion_tokens)
            yield usage
        usage = {
            "user": kwargs.get("user"),
            "model": usage_model,
            "provider": usage_provider,
            **usage.get_dict(),
        }
        usage_dir = Path(get_cookies_dir()) / ".usage"
        usage_file = usage_dir / f"{datetime.date.today()}.jsonl"
        usage_dir.mkdir(parents=True, exist_ok=True)
        with usage_file.open("a") as f:
            f.write(f"{json.dumps(usage)}\n")
        if completion_tokens > 0:
            provider.live += 1
    except:
        provider.live -= 1
        raise

    if sources is not None:
        yield sources


def caculate_prompt_tokens(messages: Messages) -> int:
    """Calculate the total number of tokens in messages"""
    token_count = 1  # Bos Token
    for message in messages:
        if isinstance(message.get("content"), str):
            token_count += math.floor(len(message["content"].encode("utf-8")) / 4)
            token_count += 4  # Role and start/end message token
        elif isinstance(message.get("content"), list):
            for item in message["content"]:
                if isinstance(item, str):
                    token_count += math.floor(len(item.encode("utf-8")) / 4)
                elif (
                    isinstance(item, dict)
                    and "text" in item
                    and isinstance(item["text"], str)
                ):
                    token_count += math.floor(len(item["text"].encode("utf-8")) / 4)
                token_count += 4  # Role and start/end message token
    return token_count


def get_usage(messages: Messages, completion_tokens: int) -> Usage:
    prompt_tokens = caculate_prompt_tokens(messages)
    return Usage(
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
