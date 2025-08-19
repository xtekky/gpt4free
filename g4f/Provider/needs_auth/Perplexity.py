from __future__ import annotations

import json
import uuid
import random
import time
import re
from typing import Dict, List

from aiohttp import ClientSession, BaseConnector

from ...typing import AsyncResult, Messages, Cookies
from ...requests import raise_for_status, DEFAULT_HEADERS, StreamSession
from ...errors import ResponseError, MissingAuthError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_connector, format_cookies

class Perplexity(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Perplexity"
    url = "https://www.perplexity.ai"
    working = True
    needs_auth = True
    default_model = "claude-3.5-sonnet"
    
    # Models available with Pro subscription via web interface
    models = [
        "claude-3.5-sonnet",
        "claude-3.5-haiku", 
        "gpt-4o",
        "gpt-4o-mini",
        "gemini-2.0-flash-experimental",
        "o1-preview",
        "o1-mini",
        "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-large-128k-chat",
        "llama-3.1-sonar-huge-128k-online"
    ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: Cookies = None,
        **kwargs
    ) -> AsyncResult:
        if cookies is None:
            raise MissingAuthError("Cookies are required for Perplexity authentication. Please provide cookies from a logged-in Perplexity session.")
            
        headers = {
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': cls.url,
            'referer': f'{cls.url}/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        async with StreamSession(headers=headers, proxy=proxy, cookies=cookies, impersonate="chrome") as session:
            # First, get the main page to extract any necessary tokens or session info
            async with session.get(cls.url) as response:
                await raise_for_status(response, "Failed to load Perplexity homepage")
                page_content = await response.text()
            
            # Extract necessary tokens/session info from the page
            # This is a common pattern for web-based chat interfaces
            csrf_token = None
            session_id = None
            
            # Look for common token patterns in the page HTML
            csrf_match = re.search(r'"csrf[_-]?token["\']\s*:\s*["\']([^"\']+)', page_content, re.IGNORECASE)
            if csrf_match:
                csrf_token = csrf_match.group(1)
                
            session_match = re.search(r'"session[_-]?id["\']\s*:\s*["\']([^"\']+)', page_content, re.IGNORECASE)
            if session_match:
                session_id = session_match.group(1)

            # Try common chat API endpoints
            chat_endpoints = [
                f"{cls.url}/api/chat",
                f"{cls.url}/api/copilot/chat", 
                f"{cls.url}/api/ask",
                f"{cls.url}/api/query"
            ]
            
            # Format the message for the API
            formatted_messages = []
            for msg in messages:
                if isinstance(msg.get("content"), str):
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Prepare the request payload
            payload = {
                "messages": formatted_messages,
                "model": model,
                "stream": True
            }
            
            # Add tokens if found
            if csrf_token:
                headers['x-csrf-token'] = csrf_token
            if session_id:
                headers['x-session-id'] = session_id
                
            # Try each endpoint until we find one that works
            last_error = None
            
            for endpoint in chat_endpoints:
                try:
                    async with session.post(endpoint, json=payload, headers=headers) as response:
                        if response.status == 200:
                            # Handle streaming response
                            buffer = ""
                            async for line in response.content:
                                if not line:
                                    continue
                                    
                                line_text = line.decode('utf-8').strip()
                                if not line_text:
                                    continue
                                    
                                buffer += line_text
                                
                                # Handle different streaming formats
                                if line_text.startswith("data: "):
                                    data_str = line_text[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data = json.loads(data_str)
                                        if "choices" in data and data["choices"]:
                                            delta = data["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                yield delta["content"]
                                        elif "message" in data:
                                            yield data["message"]
                                        elif "text" in data:
                                            yield data["text"]
                                        elif isinstance(data, str):
                                            yield data
                                    except json.JSONDecodeError:
                                        # If it's not JSON, yield the raw text
                                        yield data_str
                                elif line_text.startswith("{"):
                                    # Handle JSON responses
                                    try:
                                        data = json.loads(line_text)
                                        if "choices" in data and data["choices"]:
                                            delta = data["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                yield delta["content"]
                                    except json.JSONDecodeError:
                                        pass
                                else:
                                    # Handle plain text responses
                                    yield line_text
                            return
                        else:
                            last_error = f"HTTP {response.status}: {await response.text()}"
                            continue
                            
                except Exception as e:
                    last_error = str(e)
                    continue
            
            # If we get here, none of the endpoints worked
            # Provide a helpful error message and basic functionality
            error_msg = f"Unable to connect to Perplexity chat API. Last error: {last_error}"
            yield f"Error: {error_msg}\n\n"
            yield "This provider requires cookies from a logged-in Perplexity.ai session. "
            yield "Please ensure you're using valid authentication cookies from your browser.\n\n"
            yield f"Your query was: {format_prompt(messages)}\n"
            yield f"Selected model: {model}\n"