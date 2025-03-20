from __future__ import annotations
import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests.raise_for_status import raise_for_status
from ..providers.response import FinishReason
from .helper import format_prompt

class TypeGPT(AsyncGeneratorProvider, ProviderModelMixin):
    label = "TypeGpt"
    url = "https://chat.typegpt.net"
    api_endpoint = "https://chat.typegpt.net/api/openai/v1/chat/completions"
    
    working = True
    supports_stream = True 
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'o3-mini'
    models = [
        default_model,"gemini-1.5-flash","deepseek-r1","gemini-pro",
    ]
    
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
            "accept": "application/json, text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        }
        
        # Prepare the prompt from messages; adjust as needed.
        prompt = format_prompt(messages)
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "model": model,
            "temperature": 0.5,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "top_p": 1
        }
        
        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                
                async for chunk in response.content:
                    if not chunk:
                        continue
                        
                    text = chunk.decode(errors="ignore")
                    
                    # The stream returns lines prefixed with "data: "
                    if not text.startswith("data: "):
                        continue
                        
                    # Check for finish signal in the stream
                    if "[DONE]" in text:
                        yield FinishReason("stop")
                        return
                        
                    try:
                        json_data = json.loads(text[6:])
                        content = json_data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        yield FinishReason("error")
                        return
