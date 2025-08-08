from __future__ import annotations

import os
import re
import json
import asyncio
import time
import datetime
from pathlib import Path
from typing import Optional, AsyncIterator, Iterator, Dict, Any, Tuple, List, Union

from ..typing import Messages
from ..providers.helper import filter_none
from ..providers.asyncio import to_async_iterator
from ..providers.response import Reasoning, FinishReason, Sources, Usage, ProviderInfo
from ..providers.types import ProviderType
from ..cookies import get_cookies_dir
from .web_search import do_search, get_search_message
from .files import read_bucket, get_bucket_dir
from .. import debug

# Constants
BUCKET_INSTRUCTIONS = """
Instruction: Make sure to add the sources of cites using [[domain]](Url) notation after the reference. Example: [[a-z0-9.]](http://example.com)
"""

TOOL_NAMES = {
    "SEARCH": "search_tool",
    "CONTINUE": "continue_tool",
    "BUCKET": "bucket_tool"
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
                raise ValueError("Tool function arguments must be a dictionary or a json string")
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
            messages[-1]["content"],
            **args
        )
        return messages, sources
    
    @staticmethod
    def process_continue_tool(messages: Messages, tool: dict, provider: Any) -> Tuple[Messages, Dict[str, Any]]:
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
                new_message_content = re.sub(r'{"bucket_id":\s*"([^"]*)"}', on_bucket, message["content"])
                if new_message_content != message["content"]:
                    has_bucket = True
                    message["content"] = new_message_content

        last_message_content = messages[-1]["content"]      
        if has_bucket and isinstance(last_message_content, str):
            if "\nSource: " in last_message_content:
                messages[-1]["content"] = last_message_content + BUCKET_INSTRUCTIONS
                    
        return messages

    @staticmethod
    async def process_tools(messages: Messages, tool_calls: List[dict], provider: Any) -> Tuple[Messages, Dict[str, Any]]:
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
                messages, sources = await ToolHandler.process_search_tool(messages, tool)

            elif function_name == TOOL_NAMES["CONTINUE"]:
                messages, kwargs = ToolHandler.process_continue_tool(messages, tool, provider)
                extra_kwargs.update(kwargs)

            elif function_name == TOOL_NAMES["BUCKET"]:
                messages = ToolHandler.process_bucket_tool(messages, tool)

        return messages, sources, extra_kwargs

class AuthManager:
    """Handles API key management"""
    aliases = {
        "GeminiPro": "Gemini",
        "PollinationsAI": "Pollinations",
        "OpenaiAPI": "Openai",
        "PuterJS": "Puter",
    }

    @classmethod
    def load_api_key(cls, provider: ProviderType) -> Optional[str]:
        """Load API key from config file"""
        if not provider.needs_auth and not hasattr(provider, "login_url"):
            return None
        provider_name = provider.get_parent()
        env_var = f"{provider_name.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key and provider_name in cls.aliases:
            env_var = f"{cls.aliases[provider_name].upper()}_API_KEY"
            api_key = os.environ.get(env_var)
        if api_key:
            debug.log(f"Loading API key for {provider_name} from environment variable {env_var}")
            return api_key
        return None

class ThinkingProcessor:
    """Processes thinking chunks"""
    
    @staticmethod
    def process_thinking_chunk(chunk: str, start_time: float = 0) -> Tuple[float, List[Union[str, Reasoning]]]:
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

            status = f"Thought for {thinking_duration:.2f}s" if thinking_duration > 1 else ""
            results.append(Reasoning(status=status, is_thinking="</think>"))

            # Make sure to handle text after the closing tag
            if after and after[0].strip():
                results.append(after[0])
                
            return 0, results
            
        # Handle ongoing thinking
        if start_time:
            return start_time, [Reasoning(chunk)]
            
        return start_time, [chunk]


async def perform_web_search(messages: Messages, web_search_param: Any) -> Tuple[Messages, Optional[Sources]]:
    """Perform web search and return updated messages and sources"""
    messages = messages.copy()
    sources = None
    
    if not web_search_param:
        return messages, sources
        
    try:
        search_query = web_search_param if isinstance(web_search_param, str) and web_search_param != "true" else None
        messages[-1]["content"], sources = await do_search(messages[-1]["content"], search_query)
    except Exception as e:
        debug.error(f"Couldn't do web search:", e)

    return messages, sources


async def async_iter_run_tools(
    provider: ProviderType, 
    model: str, 
    messages: Messages, 
    tool_calls: Optional[List[dict]] = None, 
    **kwargs
) -> AsyncIterator:
    """Asynchronously run tools and yield results"""
    # Process web search
    sources = None
    web_search = kwargs.get('web_search')
    if web_search:
        debug.log(f"Performing web search with value: {web_search}")
        messages, sources = await perform_web_search(messages, web_search)

    # Get API key
    if not kwargs.get("api_key"):
        api_key = AuthManager.load_api_key(provider)
        if api_key:
            kwargs["api_key"] = api_key
    
    # Process tool calls
    if tool_calls:
        messages, sources, extra_kwargs = await ToolHandler.process_tools(messages, tool_calls, provider)
        kwargs.update(extra_kwargs)
    
    # Generate response
    response = to_async_iterator(provider.async_create_function(model=model, messages=messages, **kwargs))
    
    model_info = model
    async for chunk in response:
        if isinstance(chunk, ProviderInfo):
            model_info = getattr(chunk, 'model', model_info)
        elif isinstance(chunk, Usage):
            usage = {"user": kwargs.get("user"), "model": model_info, "provider": provider.get_parent(), **chunk.get_dict()}
            usage_dir = Path(get_cookies_dir()) / ".usage"
            usage_file = usage_dir / f"{datetime.date.today()}.jsonl"
            usage_dir.mkdir(parents=True, exist_ok=True)
            with usage_file.open("a" if usage_file.exists() else "w") as f:
                f.write(f"{json.dumps(usage)}\n")
        yield chunk

    # Yield sources if available
    if sources:
        yield sources

def iter_run_tools(
    provider: ProviderType,
    model: str,
    messages: Messages,
    tool_calls: Optional[List[dict]] = None,
    **kwargs
) -> Iterator:
    """Run tools synchronously and yield results"""
    # Process web search
    web_search = kwargs.get('web_search')
    sources = None
    
    if web_search:
        debug.log(f"Performing web search with value: {web_search}")
        try:
            messages = messages.copy()
            search_query = web_search if isinstance(web_search, str) and web_search != "true" else None
            # Note: Using asyncio.run inside sync function is not ideal, but maintaining original pattern
            messages[-1]["content"], sources = asyncio.run(do_search(messages[-1]["content"], search_query))
        except Exception as e:
            debug.error(f"Couldn't do web search:", e)
    
    # Get API key if needed
    if not kwargs.get("api_key"):
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
                    tool["function"]["arguments"] = ToolHandler.validate_arguments(tool["function"])
                    messages[-1]["content"] = get_search_message(
                        messages[-1]["content"],
                        raise_search_exceptions=True,
                        **tool["function"]["arguments"]
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
                            new_message_content = re.sub(r'{"bucket_id":"([^"]*)"}', on_bucket, message["content"])
                            if new_message_content != message["content"]:
                                has_bucket = True
                                message["content"] = new_message_content
                    last_message = messages[-1]["content"]
                    if has_bucket and isinstance(last_message, str):
                        if "\nSource: " in last_message:
                            messages[-1]["content"] = last_message + BUCKET_INSTRUCTIONS
    
    # Process response chunks
    thinking_start_time = 0
    processor = ThinkingProcessor()
    model_info = model
    for chunk in provider.create_function(model=model, messages=messages, provider=provider, **kwargs):
        if isinstance(chunk, FinishReason):
            if sources is not None:
                yield sources
                sources = None
            yield chunk
            continue
        elif isinstance(chunk, Sources):
            sources = None
        elif isinstance(chunk, ProviderInfo):
            model_info = getattr(chunk, 'model', model_info)
        elif isinstance(chunk, Usage):
            usage = {"user": kwargs.get("user"), "model": model_info, "provider": provider.get_parent(), **chunk.get_dict()}
            usage_dir = Path(get_cookies_dir()) / ".usage"
            usage_file = usage_dir / f"{datetime.date.today()}.jsonl"
            usage_dir.mkdir(parents=True, exist_ok=True)
            with usage_file.open("a" if usage_file.exists() else "w") as f:
                f.write(f"{json.dumps(usage)}\n")

        if not isinstance(chunk, str):
            yield chunk
            continue
            
        thinking_start_time, results = processor.process_thinking_chunk(chunk, thinking_start_time)
        
        for result in results:
            yield result

    if sources is not None:
        yield sources
