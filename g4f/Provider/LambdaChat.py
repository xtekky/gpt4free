from __future__ import annotations

import json
import re
import uuid
import random
from aiohttp import ClientSession, FormData

from ..typing import AsyncResult, Messages
from ..requests import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message
from ..providers.response import TitleGeneration, Reasoning, FinishReason
from ..errors import ModelNotFoundError
from .. import debug


class LambdaChat(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Lambda Chat"
    url = "https://lambda.chat"
    conversation_url = f"{url}/conversation"

    working = True
    active_by_default = True

    default_model = "deepseek-r1"
    models = [
        "deepseek-llama3.3-70b",
        "deepseek-r1",
        "deepseek-r1-0528",
        "apriel-5b-instruct",
        "hermes-3-llama-3.1-405b-fp8",
        "hermes3-405b-fp8-128k",
        "llama3.1-nemotron-70b-instruct",
        "lfm-40b",
        "llama3.3-70b-instruct-fp8",
        "qwen25-coder-32b-instruct",
        "deepseek-v3-0324",
        "llama-4-maverick-17b-128e-instruct-fp8",
        "llama-4-scout-17b-16e-instruct",
        "llama3.3-70b-instruct-fp8",
        "qwen3-32b-fp8",
    ]
    model_aliases = {
        "hermes-3": "hermes3-405b-fp8-128k",
        "hermes-3-405b": ["hermes3-405b-fp8-128k", "hermes-3-llama-3.1-405b-fp8"],
        "nemotron-70b": "llama3.1-nemotron-70b-instruct",
        "llama-3.3-70b": "llama3.3-70b-instruct-fp8",
        "qwen-2.5-coder-32b": "qwen25-coder-32b-instruct",
        "llama-4-maverick": "llama-4-maverick-17b-128e-instruct-fp8",
        "llama-4-scout": "llama-4-scout-17b-16e-instruct",
        "qwen-3-32b": "qwen3-32b-fp8"
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the internal model name from the user-provided model name."""
        
        if not model:
            return cls.default_model
        
        # Check if the model exists directly in our models list
        if model in cls.models:
            return model
        
        # Check if there's an alias for this model
        if model in cls.model_aliases:
            alias = cls.model_aliases[model]
            # If the alias is a list, randomly select one of the options
            if isinstance(alias, list):
                selected_model = random.choice(alias)
                debug.log(f"LambdaChat: Selected model '{selected_model}' from alias '{model}'")
                return selected_model
            debug.log(f"LambdaChat: Using model '{alias}' for alias '{model}'")
            return alias
        
        raise ModelNotFoundError(f"LambdaChat: Model {model} not found")

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
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
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
                
                # Parse the JSON response to find the message ID
                message_id = None
                try:
                    # Try to parse each line as JSON
                    for line in response_text.splitlines():
                        if not line.strip():
                            continue
                        
                        try:
                            data_json = json.loads(line)
                            if "type" in data_json and data_json["type"] == "data" and "nodes" in data_json:
                                for node in data_json["nodes"]:
                                    if "type" in node and node["type"] == "data" and "data" in node:
                                        # Look for system message ID
                                        for item in node["data"]:
                                            if isinstance(item, dict) and "id" in item and "from" in item and item.get("from") == "system":
                                                message_id = item["id"]
                                                break
                                        
                                        # If we found the ID, break out of the loop
                                        if message_id:
                                            break
                        except json.JSONDecodeError:
                            continue
                    
                    # If we still don't have a message ID, try to find any UUID in the response
                    if not message_id:
                        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
                        uuids = re.findall(uuid_pattern, response_text)
                        if uuids:
                            message_id = uuids[0]
                    
                    if not message_id:
                        raise ValueError("Could not find message ID in response")
                        
                except (IndexError, KeyError, ValueError) as e:
                    raise RuntimeError(f"Failed to parse conversation data: {str(e)}")
            
            # Step 3: Send the user message
            user_message = get_last_user_message(messages)
            
            # Prepare form data exactly as in the curl example
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
                if not response.ok:
                    debug.log(f"LambdaChat: Request Body: {form_data}")
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
                        # Remove null characters from the token
                        token = data["token"].replace("\u0000", "")
                        if token:
                            yield token
                    elif data.get("type") == "title":
                        yield TitleGeneration(data.get("title", ""))
                    elif data.get("type") == "reasoning":
                        subtype = data.get("subtype")
                        token = data.get("token", "").replace("\u0000", "")
                        status = data.get("status", "")
                        
                        if subtype == "stream" and token:
                            yield Reasoning(token=token)
                        elif subtype == "status" and status:
                            yield Reasoning(status=status)
                    elif data.get("type") == "finalAnswer":
                        yield FinishReason("stop")
                        break
                    elif data.get("type") == "status" and data.get("status") == "keepAlive":
                        # Just a keepalive, ignore
                        continue
