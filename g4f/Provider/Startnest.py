from __future__ import annotations

from aiohttp import ClientSession
import json
import time
import hashlib

from ..typing import AsyncResult, Messages, MediaListType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..tools.media import merge_media
from ..image import to_data_uri
from ..providers.response import FinishReason


class Startnest(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Startnest"
    url = "https://play.google.com/store/apps/details?id=starnest.aitype.aikeyboard.chatbot.chatgpt"
    api_endpoint = "https://api.startnest.uk/api/completions/stream"
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o-mini'
    default_vision_model = default_model
    vision_models = [default_model, "gpt-4o-mini"]
    models = vision_models

    @classmethod
    def generate_signature(cls, timestamp: int) -> str:
        """
        Generate signature for authorization header
        You may need to adjust this based on the actual signature algorithm
        """
        # This is a placeholder - the actual signature generation might involve:
        # - A secret key
        # - Specific string formatting
        # - Different hash input
        
        # Example implementation (adjust as needed):
        kid = "36ccfe00-78fc-4cab-9c5b-5460b0c78513"
        algorithm = "sha256"
        validity = 90
        user_id = ""
        
        # The actual signature generation logic needs to be determined
        # This is just a placeholder that creates a hash from timestamp
        signature_input = f"{kid}{timestamp}{validity}".encode()
        signature_value = hashlib.sha256(signature_input).hexdigest()
        
        return f"Signature kid={kid}&algorithm={algorithm}&timestamp={timestamp}&validity={validity}&userId={user_id}&value={signature_value}"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        media: MediaListType = None,
        stream: bool = True,
        max_tokens: int = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        # Generate current timestamp
        timestamp = int(time.time())
        
        headers = {
            "Accept-Encoding": "gzip",
            "app_name": "AIKEYBOARD",
            "Authorization": cls.generate_signature(timestamp),
            "Connection": "Keep-Alive",
            "Content-Type": "application/json; charset=UTF-8",
            "Host": "api.startnest.uk",
            "User-Agent": "okhttp/4.9.0",
        }
        
        async with ClientSession() as session:
            # Merge media with messages
            media = list(merge_media(media, messages))
            
            # Convert messages to the required format
            formatted_messages = []
            for i, msg in enumerate(messages):
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    # Create content array
                    content_array = []
                    
                    # Add images if this is the last user message and media exists
                    if media and role == "user" and i == len(messages) - 1:
                        for image, image_name in media:
                            image_data_uri = to_data_uri(image)
                            content_array.append({
                                "image_url": {
                                    "url": image_data_uri
                                },
                                "type": "image_url"
                            })
                    
                    # Add text content
                    if content:
                        content_array.append({
                            "text": content,
                            "type": "text"
                        })
                    
                    formatted_messages.append({
                        "role": role,
                        "content": content_array
                    })
            
            # If only one message and no media, use format_prompt as requested
            if len(messages) == 1 and not media:
                prompt_text = format_prompt(messages)
                formatted_messages = [{
                    "role": "user",
                    "content": [{"text": prompt_text, "type": "text"}]
                }]
            
            data = {
                "isVip": True,
                "max_tokens": max_tokens,
                "messages": formatted_messages,
                "stream": stream
            }
            
            # Add advanceToolType if media is present
            if media:
                data["advanceToolType"] = "upload_and_ask"
            
            async with session.post(cls.api_endpoint, json=data, headers=headers, proxy=proxy) as response:
                response.raise_for_status()
                
                if stream:
                    # Handle streaming response (SSE format)
                    async for line in response.content:
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    json_data = json.loads(data_str)
                                    if "choices" in json_data and len(json_data["choices"]) > 0:
                                        choice = json_data["choices"][0]
                                        
                                        # Handle content
                                        delta = choice.get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield content
                                        
                                        # Handle finish_reason
                                        if "finish_reason" in choice and choice["finish_reason"] is not None:
                                            yield FinishReason(choice["finish_reason"])
                                            break
                                            
                                except json.JSONDecodeError:
                                    continue
                else:
                    # Handle non-streaming response (regular JSON)
                    response_text = await response.text()
                    try:
                        json_data = json.loads(response_text)
                        if "choices" in json_data and len(json_data["choices"]) > 0:
                            choice = json_data["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                content = choice["message"]["content"]
                                if content:
                                    yield content.strip()
                            
                            # Handle finish_reason for non-streaming
                            if "finish_reason" in choice and choice["finish_reason"] is not None:
                                yield FinishReason(choice["finish_reason"])
                                return
                                
                    except json.JSONDecodeError:
                        # If it's still SSE format even when stream=False, handle it
                        lines = response_text.strip().split('\n')
                        full_content = []
                        finish_reason_value = None
                        
                        for line in lines:
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    json_data = json.loads(data_str)
                                    if "choices" in json_data and len(json_data["choices"]) > 0:
                                        choice = json_data["choices"][0]
                                        delta = choice.get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            full_content.append(content)
                                        
                                        # Store finish_reason
                                        if "finish_reason" in choice and choice["finish_reason"] is not None:
                                            finish_reason_value = choice["finish_reason"]
                                            
                                except json.JSONDecodeError:
                                    continue
                        
                        if full_content:
                            yield ''.join(full_content)
                        
                        if finish_reason_value:
                            yield FinishReason(finish_reason_value)
