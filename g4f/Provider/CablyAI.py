from __future__ import annotations

import json
from typing import AsyncGenerator
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests.raise_for_status import raise_for_status
from ..providers.response import FinishReason, Reasoning


class CablyAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "CablyAI"
    url = "https://cablyai.com"
    api_endpoint = "https://cablyai.com/v1/chat/completions"
    api_key = "sk-your-openai-api-key"
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o-mini'
    reasoning_models = ['deepseek-r1-uncensored']
    models = [
        default_model,
        'searchgpt',
        'llama-3.1-8b-instruct',
        'deepseek-v3',
        'tinyswallow1.5b',
        'andy-3.5',
        'o3-mini-low',
    ] + reasoning_models
    
    model_aliases = {
        "gpt-4o-mini": "searchgpt",
        "llama-3.1-8b": "llama-3.1-8b-instruct",
        "deepseek-r1": "deepseek-r1-uncensored",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        stream: bool = True,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        api_key = api_key or cls.api_key
        
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Referer": f"{cls.url}/chat",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }

        async with ClientSession(headers=headers) as session:
            data = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            
            async with session.post(
                cls.api_endpoint,
                json=data,
                proxy=proxy
            ) as response:
                await raise_for_status(response)
                
                if stream:
                    reasoning_buffer = []
                    in_reasoning = False

                    async for line in response.content:
                        if not line:
                            continue
                            
                        line = line.decode('utf-8').strip()
                        print(line)
                        
                        if not line.startswith("data: "):
                            continue
                            
                        if line == "data: [DONE]":
                            if in_reasoning and reasoning_buffer:
                                yield Reasoning(status="".join(reasoning_buffer).strip())
                            yield FinishReason("stop")
                            return
                            
                        try:
                            json_data = json.loads(line[6:])
                            delta = json_data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = json_data["choices"][0].get("finish_reason")
                            
                            if finish_reason:
                                if in_reasoning and reasoning_buffer:
                                    yield Reasoning(status="".join(reasoning_buffer).strip())
                                yield FinishReason(finish_reason)
                                return
                            
                            if model in cls.reasoning_models:
                                # Processing the beginning of a tag
                                if "<think>" in content:
                                    pre, _, post = content.partition("<think>")
                                    if pre:
                                        yield pre
                                    in_reasoning = True
                                    content = post
                                
                                # Tag end processing
                                if "</think>" in content:
                                    in_reasoning = False
                                    thought, _, post = content.partition("</think>")
                                    if thought:
                                        reasoning_buffer.append(thought)
                                    if reasoning_buffer:
                                        yield Reasoning(status="".join(reasoning_buffer).strip())
                                        reasoning_buffer.clear()
                                    if post:
                                        yield post
                                    continue
                                
                                # Buffering content inside tags
                                if in_reasoning:
                                    reasoning_buffer.append(content)
                                else:
                                    if content:
                                        yield content
                            else:
                                if content:
                                    yield content
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception:
                            yield FinishReason("error")
                            return
                else:
                    try:
                        response_data = await response.json()
                        message = response_data["choices"][0]["message"]
                        content = message["content"]
                        
                        if model in cls.reasoning_models and "<think>" in content:
                            think_start = content.find("<think>") + 7
                            think_end = content.find("</think>")
                            if think_start > 6 and think_end > 0:
                                reasoning = content[think_start:think_end].strip()
                                yield Reasoning(status=reasoning)
                                content = content[think_end + 8:].strip()
                        
                        yield content
                        yield FinishReason("stop")
                    except Exception:
                        yield FinishReason("error")
