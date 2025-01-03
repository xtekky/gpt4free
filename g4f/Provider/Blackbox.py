from __future__ import annotations

from aiohttp import ClientSession
import json
import uuid
import re
import aiohttp
from pathlib import Path
from functools import wraps
from typing import Optional, Callable, Any

from ..typing import AsyncResult, Messages, ImagesType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..image import ImageResponse, to_data_uri
from ..cookies import get_cookies_dir
from .. import debug


def cached_value(filename: str, cache_key: str = 'validated_value'):
    """Universal cache decorator for both memory and file caching"""
    def decorator(fetch_func: Callable) -> Callable:
        memory_cache: Optional[str] = None
        
        @wraps(fetch_func)
        async def wrapper(cls, *args, force_refresh=False, **kwargs) -> Any:
            nonlocal memory_cache
            
            # If force refresh, clear caches
            if force_refresh:
                memory_cache = None
                try:
                    cache_file = Path(get_cookies_dir()) / filename
                    if cache_file.exists():
                        cache_file.unlink()
                except Exception as e:
                    debug.log(f"Error clearing cache file: {e}")
            
            # Check memory cache first
            if memory_cache is not None:
                return memory_cache
            
            # Check file cache
            cache_file = Path(get_cookies_dir()) / filename
            try:
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        if data.get(cache_key):
                            memory_cache = data[cache_key]
                            return memory_cache
            except Exception as e:
                debug.log(f"Error reading cache file: {e}")
            
            # Fetch new value
            try:
                value = await fetch_func(cls, *args, **kwargs)
                memory_cache = value
                
                # Save to file
                cache_file.parent.mkdir(exist_ok=True)
                try:
                    with open(cache_file, 'w') as f:
                        json.dump({cache_key: value}, f)
                except Exception as e:
                    debug.log(f"Error writing to cache file: {e}")
                
                return value
            except Exception as e:
                debug.log(f"Error fetching value: {e}")
                raise
                
        return wrapper
    return decorator


class Blackbox(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Blackbox AI"
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/chat"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'blackboxai'
    default_vision_model = default_model
    default_image_model = 'ImageGeneration' 
    image_models = [default_image_model]
    vision_models = [default_vision_model, 'gpt-4o', 'gemini-pro', 'gemini-1.5-flash', 'llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-405b']
    
    userSelectedModel = ['gpt-4o', 'gemini-pro', 'claude-sonnet-3.5', 'blackboxai-pro']

    agentMode = {
        'ImageGeneration': {'mode': True, 'id': "ImageGenerationLV45LJp", 'name': "Image Generation"},
        #
        'meta-llama/Llama-3.3-70B-Instruct-Turbo': {'mode': True, 'id': "meta-llama/Llama-3.3-70B-Instruct-Turbo", 'name': "Meta-Llama-3.3-70B-Instruct-Turbo"},
        'mistralai/Mistral-7B-Instruct-v0.2': {'mode': True, 'id': "mistralai/Mistral-7B-Instruct-v0.2", 'name': "Mistral-(7B)-Instruct-v0.2"},
        'deepseek-ai/deepseek-llm-67b-chat': {'mode': True, 'id': "deepseek-ai/deepseek-llm-67b-chat", 'name': "DeepSeek-LLM-Chat-(67B)"},
        'databricks/dbrx-instruct': {'mode': True, 'id': "databricks/dbrx-instruct", 'name': "DBRX-Instruct"},
        'Qwen/QwQ-32B-Preview': {'mode': True, 'id': "Qwen/QwQ-32B-Preview", 'name': "Qwen-QwQ-32B-Preview"},
        'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO': {'mode': True, 'id': "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", 'name': "Nous-Hermes-2-Mixtral-8x7B-DPO"}
    }

    trendingAgentMode = {
        "gemini-1.5-flash": {'mode': True, 'id': 'Gemini'},
        "llama-3.1-8b": {'mode': True, 'id': "llama-3.1-8b"},
        'llama-3.1-70b': {'mode': True, 'id': "llama-3.1-70b"},
        'llama-3.1-405b': {'mode': True, 'id': "llama-3.1-405"},
        #
        'Python Agent': {'mode': True, 'id': "Python Agent"},
        'Java Agent': {'mode': True, 'id': "Java Agent"},
        'JavaScript Agent': {'mode': True, 'id': "JavaScript Agent"},
        'HTML Agent': {'mode': True, 'id': "HTML Agent"},
        'Google Cloud Agent': {'mode': True, 'id': "Google Cloud Agent"},
        'Android Developer': {'mode': True, 'id': "Android Developer"},
        'Swift Developer': {'mode': True, 'id': "Swift Developer"},
        'Next.js Agent': {'mode': True, 'id': "Next.js Agent"},
        'MongoDB Agent': {'mode': True, 'id': "MongoDB Agent"},
        'PyTorch Agent': {'mode': True, 'id': "PyTorch Agent"},
        'React Agent': {'mode': True, 'id': "React Agent"},
        'Xcode Agent': {'mode': True, 'id': "Xcode Agent"},
        'AngularJS Agent': {'mode': True, 'id': "AngularJS Agent"},
        #
        'blackboxai-pro': {'mode': True, 'id': "BLACKBOXAI-PRO"},
        #
        'repomap': {'mode': True, 'id': "repomap"},
        #
        'Heroku Agent': {'mode': True, 'id': "Heroku Agent"},
        'Godot Agent': {'mode': True, 'id': "Godot Agent"},
        'Go Agent': {'mode': True, 'id': "Go Agent"},
        'Gitlab Agent': {'mode': True, 'id': "Gitlab Agent"},
        'Git Agent': {'mode': True, 'id': "Git Agent"},
        'Flask Agent': {'mode': True, 'id': "Flask Agent"},
        'Firebase Agent': {'mode': True, 'id': "Firebase Agent"},
        'FastAPI Agent': {'mode': True, 'id': "FastAPI Agent"},
        'Erlang Agent': {'mode': True, 'id': "Erlang Agent"},
        'Electron Agent': {'mode': True, 'id': "Electron Agent"},
        'Docker Agent': {'mode': True, 'id': "Docker Agent"},
        'DigitalOcean Agent': {'mode': True, 'id': "DigitalOcean Agent"},
        'Bitbucket Agent': {'mode': True, 'id': "Bitbucket Agent"},
        'Azure Agent': {'mode': True, 'id': "Azure Agent"},
        'Flutter Agent': {'mode': True, 'id': "Flutter Agent"},
        'Youtube Agent': {'mode': True, 'id': "Youtube Agent"},
        'builder Agent': {'mode': True, 'id': "builder Agent"},
    }
    
    models = list(dict.fromkeys([default_model, *userSelectedModel, *list(agentMode.keys()), *list(trendingAgentMode.keys())]))

    model_aliases = {
        ### chat ###
        "gpt-4": "gpt-4o",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-pro": "gemini-pro",
        "claude-3.5-sonnet": "claude-sonnet-3.5",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "mixtral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "deepseek-chat": "deepseek-ai/deepseek-llm-67b-chat",
        "dbrx-instruct": "databricks/dbrx-instruct",
        "qwq-32b": "Qwen/QwQ-32B-Preview",
        "hermes-2-dpo": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        
        ### image ###
        "flux": "ImageGeneration",
    }

    @classmethod
    @cached_value(filename='blackbox.json')
    async def get_validated(cls) -> str:
        """Fetch validated value from website"""
        async with aiohttp.ClientSession() as session:
            async with session.get(cls.url) as response:
                if response.status != 200:
                    raise RuntimeError("Failed to get validated value")
                
                page_content = await response.text()
                js_files = re.findall(r'static/chunks/\d{4}-[a-fA-F0-9]+\.js', page_content)
                
                if not js_files:
                    js_files = re.findall(r'static/js/[a-zA-Z0-9-]+\.js', page_content)

                uuid_format = r'["\']([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})["\']'

                def is_valid_context(text_around):
                    return any(char + '=' in text_around for char in 'abcdefghijklmnopqrstuvwxyz')

                for js_file in js_files:
                    js_url = f"{cls.url}/_next/{js_file}"
                    try:
                        async with session.get(js_url) as js_response:
                            if js_response.status == 200:
                                js_content = await js_response.text()
                                for match in re.finditer(uuid_format, js_content):
                                    start = max(0, match.start() - 10)
                                    end = min(len(js_content), match.end() + 10)
                                    context = js_content[start:end]
                                    
                                    if is_valid_context(context):
                                        return match.group(1)
                    except Exception:
                        continue
                        
        raise RuntimeError("Failed to get validated value")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        web_search: bool = False,
        images: ImagesType = None,
        top_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> AsyncResult:      
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://www.blackbox.ai",
            "referer": "https://www.blackbox.ai/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        
        model = cls.get_model(model)
        
        conversation_id = str(uuid.uuid4())[:7]
        validated_value = await cls.get_validated()
        
        formatted_message = format_prompt(messages)
            
        first_message = next((msg for msg in messages if msg['role'] == 'user'), None)
        current_messages = [{"id": conversation_id, "content": formatted_message, "role": "user"}]

        if images is not None:
            current_messages[-1]['data'] = {
                "imagesData": [
                    {
                        "filePath": f"/{image_name}",
                        "contents": to_data_uri(image)
                    }
                    for image, image_name in images
                ],
                "fileText": "",
                "title": ""
            }
        
        while True:
            async with ClientSession(headers=headers) as session:
                data = {
                    "messages": current_messages,
                    "id": conversation_id,
                    "previewToken": None,
                    "userId": None,
                    "codeModelMode": True,
                    "agentMode": cls.agentMode.get(model, {}) if model in cls.agentMode else {},
                    "trendingAgentMode": cls.trendingAgentMode.get(model, {}) if model in cls.trendingAgentMode else {},
                    "isMicMode": False,
                    "userSystemPrompt": None,
                    "maxTokens": max_tokens,
                    "playgroundTopP": top_p,
                    "playgroundTemperature": temperature,
                    "isChromeExt": False,
                    "githubToken": "",
                    "clickedAnswer2": False,
                    "clickedAnswer3": False,
                    "clickedForceWebSearch": False,
                    "visitFromDelta": False,
                    "mobileClient": False,
                    "userSelectedModel": model if model in cls.userSelectedModel else None,
                    "validated": validated_value,
                    "imageGenerationMode": False,
                    "webSearchModePrompt": False,
                    "deepSearchMode": False,
                    "domains": None,
                    "webSearchMode": web_search
                }
                
                try:
                    async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                        response.raise_for_status()
                        first_chunk = True
                        content_received = False
                        async for chunk in response.content:
                            if chunk:
                                content_received = True
                                decoded = chunk.decode()
                                if first_chunk and "Generated by BLACKBOX.AI" in decoded:
                                    validated_value = await cls.get_validated(force_refresh=True)
                                    break
                                first_chunk = False
                                if model in cls.image_models and decoded.startswith("![]("):
                                    image_url = decoded.strip("![]()")
                                    prompt = messages[-1]["content"]
                                    yield ImageResponse(images=image_url, alt=prompt)
                                else:
                                    yield decoded
                        else:
                            if not content_received:
                                debug.log("Empty response received from Blackbox API, retrying...")
                                continue
                            return
                except Exception as e:
                    debug.log(f"Error in request: {e}")
                    raise
