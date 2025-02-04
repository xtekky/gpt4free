from __future__ import annotations
import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests.raise_for_status import raise_for_status
from ..providers.response import Reasoning
from .helper import format_prompt

class BlackboxAPI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Blackbox AI API"
    url = "https://api.blackbox.ai"
    api_endpoint = "https://api.blackbox.ai/api/chat"
    
    working = True
    needs_auth = False
    supports_stream = False
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'deepseek-ai/DeepSeek-V3'
    reasoning_models = ['deepseek-ai/DeepSeek-R1']
    models = [
        default_model,
        'mistralai/Mistral-Small-24B-Instruct-2501',
        'deepseek-ai/deepseek-llm-67b-chat',
        'databricks/dbrx-instruct',
        'Qwen/QwQ-32B-Preview',
        'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO'
    ] + reasoning_models
    
    model_aliases = {       
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-chat": "deepseek-ai/deepseek-llm-67b-chat",
        "mixtral-small-28b": "mistralai/Mistral-Small-24B-Instruct-2501",
        "dbrx-instruct": "databricks/dbrx-instruct",
        "qwq-32b": "Qwen/QwQ-32B-Preview",
        "hermes-2-dpo": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        max_tokens: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json",
        }
        
        async with ClientSession(headers=headers) as session:
            data = {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                
                is_reasoning = False
                current_reasoning = ""
                
                async for chunk in response.content:
                    if not chunk:
                        continue
                    
                    text = chunk.decode(errors='ignore')
                    
                    if model in cls.reasoning_models:
                        if "<think>" in text:
                            text = text.replace("<think>", "")
                            is_reasoning = True
                            current_reasoning = text
                            continue
                            
                        if "</think>" in text:
                            text = text.replace("</think>", "")
                            is_reasoning = False
                            current_reasoning += text
                            yield Reasoning(status=current_reasoning.strip())
                            current_reasoning = ""
                            continue
                            
                        if is_reasoning:
                            current_reasoning += text
                            continue
                    
                    try:
                        if text:
                            yield text
                    except Exception as e:
                        return

                if is_reasoning and current_reasoning:
                    yield Reasoning(status=current_reasoning.strip())
