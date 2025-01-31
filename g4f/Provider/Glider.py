from __future__ import annotations
import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests.raise_for_status import raise_for_status
from ..providers.response import FinishReason, Reasoning
from .helper import format_prompt

class Glider(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Glider"
    url = "https://glider.so"
    api_endpoint = "https://glider.so/api/chat"
    
    working = True
    needs_auth = False
    supports_stream = True 
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'chat-llama-3-1-70b'
    reasoning_models = ['deepseek-ai/DeepSeek-R1']
    models = [
        'chat-llama-3-1-70b',
        'chat-llama-3-1-8b',
        'chat-llama-3-2-3b',
    ] + reasoning_models
    
    model_aliases = {
        "llama-3.1-70b": "chat-llama-3-1-70b",
        "llama-3.1-8b": "chat-llama-3-1-8b",
        "llama-3.2-3b": "chat-llama-3-2-3b",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        }

        async with ClientSession(headers=headers) as session:
            data = {
                "messages": [{
                    "role": "user",
                    "content": format_prompt(messages),
                    "id": "",
                    "chatId": "",
                    "createdOn": "",
                    "model": None
                }],
                "model": model
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                
                is_reasoning = False
                current_reasoning = ""

                async for chunk in response.content:
                    if not chunk:
                        continue
                        
                    text = chunk.decode(errors="ignore")
                    
                    if not text.startswith("data: "):
                        continue
                        
                    if "[DONE]" in text:
                        if is_reasoning and current_reasoning:
                            yield Reasoning(status=current_reasoning.strip())
                        yield FinishReason("stop")
                        return
                        
                    try:
                        json_data = json.loads(text[6:])
                        content = json_data["choices"][0].get("delta", {}).get("content", "")
                        
                        if model in cls.reasoning_models:
                            if "<think>" in content:
                                content = content.replace("<think>", "")
                                is_reasoning = True
                                current_reasoning = content
                                continue
                                
                            if "</think>" in content:
                                content = content.replace("</think>", "")
                                is_reasoning = False
                                current_reasoning += content
                                yield Reasoning(status=current_reasoning.strip())
                                current_reasoning = ""
                                continue
                                
                            if is_reasoning:
                                current_reasoning += content
                                continue
                                
                        if content:
                            yield content
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        yield FinishReason("error")
                        return
