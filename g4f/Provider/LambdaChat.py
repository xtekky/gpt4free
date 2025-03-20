from __future__ import annotations

import json
import re
import uuid
from aiohttp import ClientSession, FormData

from ..typing import AsyncResult, Messages
from ..requests import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt, get_last_user_message
from ..providers.response import JsonConversation, TitleGeneration, Reasoning

class LambdaChat(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Lambda Chat"
    url = "https://lambda.chat"
    conversation_url = f"{url}/conversation"

    working = True

    default_model = "deepseek-llama3.3-70b"
    reasoning_model = "deepseek-r1"
    models = [
        default_model,
        reasoning_model,
        "hermes-3-llama-3.1-405b-fp8",
        "hermes3-405b-fp8-128k",
        "llama3.1-nemotron-70b-instruct",
        "lfm-40b",
        "llama3.3-70b-instruct-fp8"
    ]
    model_aliases = {
        "deepseek-v3": default_model,
        "hermes-3": "hermes-3-llama-3.1-405b-fp8",
        "hermes-3": "hermes3-405b-fp8-128k",
        "nemotron-70b": "llama3.1-nemotron-70b-instruct",
        "llama-3.3-70b": "llama3.3-70b-instruct-fp8"
    }

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages,
        api_key: str = None, 
        proxy: str = None,
        cookies: dict = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "Origin": cls.url,
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": cls.url,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Priority": "u=1, i",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        }
        
        # Initialize cookies if not provided
        if cookies is None:
            cookies = {
                "hf-chat": str(uuid.uuid4())  # Generate a session ID
            }
        
        async with ClientSession(headers=headers, cookies=cookies) as session:
            # Step 1: Create a new conversation
            data = {"model": model}
            async with session.post(cls.conversation_url, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                conversation_response = await response.json()
                conversation_id = conversation_response["conversationId"]
                
                # Update cookies with any new ones from the response
                for cookie_name, cookie in response.cookies.items():
                    cookies[cookie_name] = cookie.value
            
            # Step 2: Get data for this conversation to extract message ID
            async with session.get(
                f"{cls.conversation_url}/{conversation_id}/__data.json?x-sveltekit-invalidated=11", 
                proxy=proxy
            ) as response:
                await raise_for_status(response)
                response_text = await response.text()
                
                # Update cookies again
                for cookie_name, cookie in response.cookies.items():
                    cookies[cookie_name] = cookie.value
                
                try:
                    data_line = response_text.splitlines()[0]
                    data_json = json.loads(data_line)
                    
                    # Navigate to the data section containing message info
                    message_id = None
                    
                    # For debugging, print the JSON structure
                    if "nodes" in data_json and len(data_json["nodes"]) > 1:
                        node = data_json["nodes"][1]
                        if "data" in node:
                            data = node["data"]
                            
                            # Try to find the system message ID
                            if len(data) > 1 and isinstance(data[1], list) and len(data[1]) > 2:
                                for item in data[1]:
                                    if isinstance(item, dict) and "id" in item:
                                        # Found a potential message ID
                                        message_id = item["id"]
                                        break
                    
                    if not message_id:
                        # Fallback: Just search for any UUID in the response
                        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
                        uuids = re.findall(uuid_pattern, response_text)
                        if uuids:
                            message_id = uuids[0]
                    
                    if not message_id:
                        raise ValueError("Could not find message ID in response")
                        
                except (IndexError, KeyError, ValueError, json.JSONDecodeError) as e:
                    raise RuntimeError(f"Failed to parse conversation data: {str(e)}")
            
            # Step 3: Send the user message
            user_message = get_last_user_message(messages)
            
            # Prepare form data exactly
            form_data = FormData()
            form_data.add_field(
                "data",
                json.dumps({
                    "inputs": user_message, 
                    "id": message_id, 
                    "is_retry": False, 
                    "is_continue": False, 
                    "web_search": False, 
                    "tools": []
                }),
                content_type="application/json"
            )
            
            async with session.post(
                f"{cls.conversation_url}/{conversation_id}", 
                data=form_data,
                proxy=proxy
            ) as response:
                await raise_for_status(response)
                
                async for chunk in response.content:
                    if not chunk:
                        continue
                        
                    chunk_str = chunk.decode('utf-8', errors='ignore')
                    
                    try:
                        data = json.loads(chunk_str)
                    except json.JSONDecodeError:
                        continue
                        
                    # Handling different types of responses
                    if data.get("type") == "stream" and "token" in data:
                        token = data["token"].replace("\u0000", "")
                        if token:
                            yield token
                    elif data.get("type") == "title":
                        yield TitleGeneration(data.get("title", ""))
                    elif data.get("type") == "reasoning" and model == cls.reasoning_model:  # Only process reasoning for reasoning_model
                        subtype = data.get("subtype")
                        token = data.get("token", "").replace("\u0000", "")
                        status = data.get("status", "")
                        
                        if subtype == "stream" and token:
                            yield Reasoning(token=token)
                        elif subtype == "status" and status:
                            yield Reasoning(status=status)
                    elif data.get("type") == "finalAnswer":
                        break
