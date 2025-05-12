from __future__ import annotations

import json
import base64
import time
import random
import asyncio
from datetime import datetime
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt, get_last_user_message
from ..providers.response import FinishReason, JsonConversation


class Conversation(JsonConversation):
    """Conversation class for DDG provider.
    
    Note: DDG doesn't actually support conversation history through its API,
    so we simulate it by including the history in the user message.
    """
    message_history: Messages = []
    
    def __init__(self, model: str):
        self.model = model
        self.message_history = []


class DDG(AsyncGeneratorProvider, ProviderModelMixin):
    label = "DuckDuckGo AI Chat"
    url = "https://duckduckgo.com"
    api_endpoint = "https://duckduckgo.com/duckchat/v1/chat"
    status_url = "https://duckduckgo.com/duckchat/v1/status"
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "gpt-4o-mini"
    model_aliases = {
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    }
    models = [default_model, "o3-mini"] + list(model_aliases.keys())

    @staticmethod
    def generate_fe_signals():
        """Generate a fake x-fe-signals header value"""
        current_time = int(time.time() * 1000)
        
        signals_data = {
            "start": current_time - 35000,
            "events": [
                {"name": "onboarding_impression_1", "delta": 383},
                {"name": "onboarding_impression_2", "delta": 6004},
                {"name": "onboarding_finish", "delta": 9690},
                {"name": "startNewChat", "delta": 10082},
                {"name": "initSwitchModel", "delta": 16586}
            ],
            "end": 35163
        }
        
        signals_json = json.dumps(signals_data)
        return base64.b64encode(signals_json.encode()).decode()

    @staticmethod
    def generate_fe_version():
        """Generate a fake x-fe-version header value"""
        return "serp_20250510_052906_ET-ed4f51dc2e106020bc4b"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        conversation: Conversation = None,
        return_conversation: bool = True,
        retry_count: int = 0,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        # Initialize conversation if not provided
        if conversation is None:
            conversation = Conversation(model)
            # Initialize message history from the provided messages
            conversation.message_history = messages.copy()
        else:
            # Update message history with the last user message
            last_message = None
            for msg in reversed(messages):
                if msg["role"] == "user":
                    last_message = msg
                    break
            
            if last_message and last_message not in conversation.message_history:
                conversation.message_history.append(last_message)
        
        # Base headers for all requests
        base_headers = {
            "accept-language": "en-US,en;q=0.9",
            "dnt": "1",
            "origin": "https://duckduckgo.com",
            "referer": "https://duckduckgo.com/",
            "sec-ch-ua": '"Chromium";v="135", "Not-A.Brand";v="8"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        }
        
        cookies = {'dcs': '1', 'dcm': '3'}
        
        # Format the conversation history as a single prompt using format_prompt
        if len(conversation.message_history) > 1:
            # If we have conversation history, format it as a single prompt
            formatted_prompt = format_prompt(conversation.message_history)
        else:
            # If we don't have conversation history, just use the last user message
            formatted_prompt = get_last_user_message(messages)
        
        # Prepare the request data
        data = {
            "model": model,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "canUseTools": False
        }
        
        # Create a new session for each request
        async with ClientSession(cookies=cookies) as session:
            # Step 1: Visit the main page to get initial cookies
            main_headers = base_headers.copy()
            main_headers.update({
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "priority": "u=0, i",
                "upgrade-insecure-requests": "1",
            })
            
            try:
                async with session.get(f"{cls.url}/?q=DuckDuckGo+AI+Chat&ia=chat&duckai=1", 
                                    headers=main_headers, 
                                    proxy=proxy) as main_response:
                    main_response.raise_for_status()
                    
                    # Extract fe_version from the page
                    page_content = await main_response.text()
                    fe_version = cls.generate_fe_version()
                    try:
                        xfe1 = page_content.split('__DDG_BE_VERSION__="', 1)[1].split('"', 1)[0]
                        xfe2 = page_content.split('__DDG_FE_CHAT_HASH__="', 1)[1].split('"', 1)[0]
                        fe_version = f"serp_20250510_052906_ET-{xfe2}"
                    except Exception:
                        pass
                
                # Step 2: Get the VQD token from the status endpoint
                status_headers = base_headers.copy()
                status_headers.update({
                    "accept": "*/*",
                    "cache-control": "no-store",
                    "priority": "u=1, i",
                    "x-vqd-accept": "1",
                })
                
                async with session.get(cls.status_url, 
                                    headers=status_headers, 
                                    proxy=proxy) as status_response:
                    status_response.raise_for_status()
                    
                    # Get VQD token from headers
                    vqd = status_response.headers.get("x-vqd-4", "")
                    
                    if not vqd:
                        # If we couldn't get a VQD token, try to generate one
                        vqd = f"4-{random.randint(10**29, 10**30 - 1)}"
                
                # Step 3: Send the chat request
                chat_headers = base_headers.copy()
                chat_headers.update({
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                    "priority": "u=1, i",
                    "x-fe-signals": cls.generate_fe_signals(),
                    "x-fe-version": fe_version,
                    "x-vqd-4": vqd,
                })
                
                async with session.post(cls.api_endpoint, 
                                       json=data, 
                                       headers=chat_headers, 
                                       proxy=proxy) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        
                        # If we get an ERR_INVALID_VQD error and haven't retried too many times, try again
                        if "ERR_INVALID_VQD" in error_text and retry_count < 3:
                            # Wait a bit before retrying
                            await asyncio.sleep(1)
                            # Try again with a new session
                            async for chunk in cls.create_async_generator(
                                model=model,
                                messages=messages,
                                proxy=proxy,
                                conversation=conversation,
                                return_conversation=return_conversation,
                                retry_count=retry_count + 1,
                                **kwargs
                            ):
                                yield chunk
                            return
                        
                        yield f"Error: HTTP {response.status} - {error_text}"
                        return
                    
                    full_message = ""
                    
                    async for line in response.content:
                        line_text = line.decode("utf-8").strip()
                        
                        if line_text.startswith("data:"):
                            data_content = line_text[5:].strip()
                            
                            # Handle [DONE] marker
                            if data_content == "[DONE]":
                                # Add the assistant's response to the conversation history
                                if full_message:
                                    conversation.message_history.append({
                                        "role": "assistant",
                                        "content": full_message
                                    })
                                
                                # Return the conversation if requested
                                if return_conversation:
                                    yield conversation
                                
                                yield FinishReason("stop")
                                break
                            
                            try:
                                message_data = json.loads(data_content)
                                
                                # Handle error responses
                                if message_data.get("action") == "error":
                                    error_type = message_data.get("type", "Unknown error")
                                    
                                    # If we get an ERR_INVALID_VQD error and haven't retried too many times, try again
                                    if error_type == "ERR_INVALID_VQD" and retry_count < 3:
                                        # Wait a bit before retrying
                                        await asyncio.sleep(1)
                                        # Try again with a new session
                                        async for chunk in cls.create_async_generator(
                                            model=model,
                                            messages=messages,
                                            proxy=proxy,
                                            conversation=conversation,
                                            return_conversation=return_conversation,
                                            retry_count=retry_count + 1,
                                            **kwargs
                                        ):
                                            yield chunk
                                        return
                                    
                                    yield f"Error: {error_type}"
                                    break
                                
                                # Extract message content
                                if "message" in message_data:
                                    message_content = message_data.get("message", "")
                                    if message_content:
                                        yield message_content
                                        full_message += message_content
                            
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                # If we get an exception and haven't retried too many times, try again
                if retry_count < 3:
                    # Wait a bit before retrying
                    await asyncio.sleep(1)
                    # Try again with a new session
                    async for chunk in cls.create_async_generator(
                        model=model,
                        messages=messages,
                        proxy=proxy,
                        conversation=conversation,
                        return_conversation=return_conversation,
                        retry_count=retry_count + 1,
                        **kwargs
                    ):
                        yield chunk
                else:
                    yield f"Error: {str(e)}"
