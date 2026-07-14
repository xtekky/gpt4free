from __future__ import annotations

import re
import json
import os
import math
import asyncio
import time
import datetime
import hashlib
from pathlib import Path
from typing import Optional, AsyncIterator, Iterator, Dict, Any, Tuple, List, Union

try:
    from aiofile import async_open

    has_aiofile = True
except ImportError:
    has_aiofile = False

from ..typing import Messages
from ..providers.helper import filter_none
from ..providers.asyncio import to_sync_generator
from ..providers.response import Reasoning, FinishReason, Sources, Usage, ProviderInfo, HeadersResponse, JsonConversation
from .optimize_request import optimize_request
from .token_optimizer import optimize_messages
from ..providers.types import ProviderType
from ..providers.base_provider import get_async_provider_method, get_provider_method, wait_for
from ..cookies import get_cookies_dir
from ..config import AppConfig
from .web_search import do_search, get_search_message
from .auth import AuthManager
from .files import read_bucket, get_bucket_dir
from .. import debug


# ---- In-memory conversation cache -------------------------------------------
# Stores the ``JsonConversation`` session state yielded by the underlying
# provider, keyed by a hash of all messages except the last user message and the
# last assistant/bot response (combined with the model name).  When the same
# conversation prefix is seen again the cached ``JsonConversation`` is passed to
# the provider so it can continue the session without starting fresh.
#
# This cache is applied for ALL providers (not only tool-emulated ones) so that
# web-API providers that rely on a server-side conversation handle can resume
# across multi-turn tool interactions.

_conversation_cache: dict[str, dict] = {}
_CACHE_MAX_SIZE = 128
_CACHE_TTL = 3600.0  # seconds


def _messages_cache_key(messages: Messages, model: str) -> Optional[str]:
    """Build a cache key from all messages except the last user message and the
    last assistant/bot response, combined with the model name.

    Returns ``None`` when there is no conversation history to cache on (e.g.
    only a single user message with no prior turns).
    """
    if not messages:
        return None
    last_user_idx = None
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user" and last_user_idx is None:
            last_user_idx = i
        elif role == "assistant" and last_assistant_idx is None:
            last_assistant_idx = i
        if last_user_idx is not None and last_assistant_idx is not None:
            break
    exclude = {idx for idx in (last_user_idx, last_assistant_idx) if idx is not None}
    if len(exclude) >= len(messages):
        return None
    parts = [model]
    for i, msg in enumerate(messages):
        if i in exclude:
            continue
        try:
            parts.append(json.dumps(msg, sort_keys=True, ensure_ascii=True, default=str))
        except Exception:
            return None
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _cache_get(key: Optional[str]) -> Optional[JsonConversation]:
    """Return cached ``JsonConversation`` for *key* or ``None`` on miss / expiry."""
    if not key:
        return None
    entry = _conversation_cache.get(key)
    if entry is None:
        return None
    if time.time() - entry["time"] > _CACHE_TTL:
        _conversation_cache.pop(key, None)
        return None
    return entry["conversation"]


def _cache_put(key: Optional[str], conversation: JsonConversation) -> None:
    """Store *conversation* under *key*, evicting oldest entries when full."""
    if not key or conversation is None:
        return
    if len(_conversation_cache) >= _CACHE_MAX_SIZE:
        oldest = sorted(_conversation_cache.items(), key=lambda kv: kv[1]["time"])
        for k, _ in oldest[: max(1, len(_conversation_cache) - _CACHE_MAX_SIZE + 1)]:
            _conversation_cache.pop(k, None)
    _conversation_cache[key] = {"conversation": conversation, "time": time.time()}

# Constants
BUCKET_INSTRUCTIONS = """
Instruction: Make sure to add the sources of cites using [[domain]](Url) notation after the reference. Example: [[a-z0-9.]](http://example.com)
"""

TOOL_NAMES = {
    "SEARCH": "search_tool",
    "CONTINUE": "continue_tool",
    "BUCKET": "bucket_tool",
}

def is_provider_api_key(api_key: str) -> bool:
    return isinstance(api_key, str) and api_key and not api_key.startswith("g4f_") and not api_key.startswith("gfs_")


def provider_supports_native_tools(provider: ProviderType) -> bool:
    """Return True if the provider supports native OpenAI-style tool calls.

    Providers that extend ``OpenaiTemplate`` (or set ``supports_native_tools = True``)
    are assumed to forward ``tools``/``tool_choice`` to an OpenAI-compatible endpoint
    and therefore do not need prompt-injection emulation.
    """
    return bool(getattr(provider, "supports_native_tools", False))


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
    async def process_tools(
        messages: Messages, tool_calls: List[dict], provider: Any
    ) -> Tuple[Messages, Dict[str, Any]]:
        """Process all tool calls and return updated messages and kwargs"""
        if not tool_calls:
            return messages, {}

        extra_kwargs = {}
        messages = messages.copy()
        sources = None

        for tool in tool_calls:
            if tool.get("type") != "function":
                continue

            function_name = tool.get("function", {}).get("name")

            debug.log(f"Processing tool call: {function_name}")
            if function_name == TOOL_NAMES["SEARCH"]:
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

    # Optimize the system prompt and tool descriptions to reduce token usage.
    # This is applied for all providers and the saved tokens are tracked.
    tools_ref = kwargs.get("tools")
    saved_tokens, _optimize_logs = optimize_request(messages, tools_ref)
    if saved_tokens:
        debug.log(f"Optimized request: saved ~{saved_tokens} tokens")

    # Optional token-optimizer plugin: compress the prompt messages before
    # they reach the provider. Only active when the `token_optimizer` package
    # is installed in the environment.
    to_saved, _to_logs = optimize_messages(messages, tools_ref)
    if to_saved:
        saved_tokens += to_saved
        debug.log(f"Token Optimizer plugin: saved ~{to_saved} tokens")

    tool_emulation = kwargs.pop("tool_emulation", None)
    if tool_emulation is None:
        tool_emulation = os.environ.get("G4F_TOOL_EMULATION", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    stream = bool(kwargs.get("stream"))
    tools = kwargs.get("tools")
    # Auto-enable tool emulation for providers without native tool support
    # (i.e. web-API providers that are not OpenaiTemplate subclasses).
    if tools and not tool_calls and not tool_emulation:
        if not provider_supports_native_tools(provider):
            tool_emulation = True
    if tool_emulation and tools and not tool_calls:
        from ..providers.tool_support import ToolSupportProvider

        emu_kwargs = dict(kwargs)
        emu_kwargs.pop("tools", None)
        tool_choice = emu_kwargs.pop("tool_choice", None)
        emu_kwargs.pop("parallel_tool_calls", None)
        emu_kwargs.pop("stream", None)
        emu_kwargs.pop("stream_timeout", None)
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
    if not kwargs.get("api_key") or AppConfig.disable_custom_api_key or not is_provider_api_key(kwargs.get("api_key")):
        api_key = AuthManager.load_api_key(provider) or kwargs.get("api_key")
        if api_key:
            kwargs["api_key"] = api_key

    # Process tool calls
    if tool_calls:
        messages, sources, extra_kwargs = await ToolHandler.process_tools(
            messages, tool_calls, provider
        )
        kwargs.update(extra_kwargs)

    # Build a cache key from all messages except the last user message and the
    # last assistant/bot response.  A cache hit supplies the cached
    # ``JsonConversation`` to the provider so it can continue the session.
    cache_key = _messages_cache_key(messages, model)
    cached_conversation = _cache_get(cache_key)
    if cached_conversation is not None:
        kwargs["conversation"] = cached_conversation
    conversation: JsonConversation = kwargs.get("conversation")

    # Generate response
    method = get_async_provider_method(provider)
    response = method(model=model, messages=messages, **kwargs)
    timeout = kwargs.get("stream_timeout") if provider.use_stream_timeout else kwargs.get("timeout")
    response = wait_for(response, timeout=timeout) if stream else response

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
            elif isinstance(chunk, JsonConversation):
                conversation = chunk
            yield chunk

        # Store the JsonConversation session state in the cache for reuse on
        # subsequent requests that share the same conversation prefix.
        if cached_conversation is None and conversation is not None:
            _cache_put(cache_key, conversation)
        if usage is None:
            usage = get_usage(messages, completion_tokens)
            yield usage
        usage_dict = {
            "user": kwargs.get("user"),
            "model": usage_model,
            "provider": usage_provider,
            **usage.get_dict(),
        }
        if saved_tokens:
            usage_dict["saved_tokens"] = saved_tokens
        usage = usage_dict
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
    except Exception:
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

    # Optimize the system prompt and tool descriptions to reduce token usage.
    # This is applied for all providers and the saved tokens are tracked.
    tools_ref = kwargs.get("tools")
    saved_tokens, _optimize_logs = optimize_request(messages, tools_ref)
    if saved_tokens:
        debug.log(f"VCS - Optimized request: saved ~{saved_tokens} tokens")

    # Optional token-optimizer plugin: compress the prompt messages before
    # they reach the provider. Only active when the `token_optimizer` package
    # is installed in the environment.
    to_saved, _to_logs = optimize_messages(messages, tools_ref)
    if to_saved:
        saved_tokens += to_saved
        debug.log(f"Token Optimizer plugin: saved ~{to_saved} tokens")

    tool_emulation = kwargs.pop("tool_emulation", None)
    if tool_emulation is None:
        tool_emulation = os.environ.get("G4F_TOOL_EMULATION", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    stream = bool(kwargs.get("stream"))
    tools = kwargs.get("tools")
    # Auto-enable tool emulation for providers without native tool support
    # (i.e. web-API providers that are not OpenaiTemplate subclasses).
    if tools and not tool_calls and not tool_emulation:
        if not provider_supports_native_tools(provider):
            tool_emulation = True
    if tool_emulation and tools and not tool_calls:
        from ..providers.tool_support import ToolSupportProvider

        emu_kwargs = dict(kwargs)
        emu_kwargs.pop("tools", None)
        tool_choice = emu_kwargs.pop("tool_choice", None)
        emu_kwargs.pop("parallel_tool_calls", None)
        emu_kwargs.pop("stream", None)
        emu_kwargs.pop("stream_timeout", None)
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
    if not kwargs.get("api_key") or AppConfig.disable_custom_api_key or not is_provider_api_key(kwargs.get("api_key")):
        api_key = AuthManager.load_api_key(provider) or kwargs.get("api_key")
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

    # Build a cache key from all messages except the last user message and the
    # last assistant/bot response.  A cache hit supplies the cached
    # ``JsonConversation`` to the provider so it can continue the session.
    cache_key = _messages_cache_key(messages, model)
    cached_conversation = _cache_get(cache_key)
    if cached_conversation is not None:
        kwargs["conversation"] = cached_conversation
    conversation: JsonConversation = kwargs.get("conversation")

    # Process response chunks
    try:
        thinking_start_time = 0
        processor = ThinkingProcessor()
        usage_model = model
        usage_provider = provider.__name__
        completion_tokens = 0
        usage = None
        method = get_provider_method(provider)
        for chunk in method(
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
            elif isinstance(chunk, JsonConversation):
                conversation = chunk
            if not isinstance(chunk, str):
                yield chunk
                continue

            thinking_start_time, results = processor.process_thinking_chunk(
                chunk, thinking_start_time
            )
            for result in results:
                yield result

        # Store the JsonConversation session state in the cache for reuse on
        # subsequent requests that share the same conversation prefix.
        if cached_conversation is None and conversation is not None:
            _cache_put(cache_key, conversation)
        if usage is None:
            usage = get_usage(messages, completion_tokens)
            yield usage
        usage_dict = {
            "user": kwargs.get("user"),
            "model": usage_model,
            "provider": usage_provider,
            **usage.get_dict(),
        }
        if saved_tokens:
            usage_dict["saved_tokens"] = saved_tokens
        usage = usage_dict
        usage_dir = Path(get_cookies_dir()) / ".usage"
        usage_file = usage_dir / f"{datetime.date.today()}.jsonl"
        usage_dir.mkdir(parents=True, exist_ok=True)
        with usage_file.open("a") as f:
            f.write(f"{json.dumps(usage)}\n")
        if completion_tokens > 0:
            provider.live += 1
    except Exception:
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
