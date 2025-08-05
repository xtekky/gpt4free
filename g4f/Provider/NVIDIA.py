from __future__ import annotations

import json
import os
import requests
import asyncio

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..providers.response import FinishReason, Sources


class NVIDIA(AsyncGeneratorProvider, ProviderModelMixin):
    label = "NVIDIA NIM API"
    url = "https://integrate.api.nvidia.com"
    api_endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
    models_endpoint = "https://integrate.api.nvidia.com/v1/models"
    working = True
    needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'meta/llama-3.1-8b-instruct'
    
    # Enhanced fallback models with detailed capability information
    _fallback_models = [
        {
            "id": "meta/llama-3.1-8b-instruct",
            "object": "model", 
            "created": 1640995200,
            "owned_by": "meta",
            "capabilities": {"text": True, "vision": False, "audio": False, "video": False, "image": False}
        },
        {
            "id": "meta/llama-3.1-70b-instruct",
            "object": "model",
            "created": 1640995200, 
            "owned_by": "meta",
            "capabilities": {"text": True, "vision": False, "audio": False, "video": False, "image": False}
        },
        {
            "id": "meta/llama-3.1-405b-instruct",
            "object": "model",
            "created": 1640995200,
            "owned_by": "meta", 
            "capabilities": {"text": True, "vision": False, "audio": False, "video": False, "image": False}
        },
        {
            "id": "mistralai/mixtral-8x7b-instruct-v0.1",
            "object": "model",
            "created": 1640995200,
            "owned_by": "mistralai",
            "capabilities": {"text": True, "vision": False, "audio": False, "video": False, "image": False}
        },
        {
            "id": "microsoft/phi-3-medium-4k-instruct", 
            "object": "model",
            "created": 1640995200,
            "owned_by": "microsoft",
            "capabilities": {"text": True, "vision": False, "audio": False, "video": False, "image": False}
        },
        {
            "id": "nvidia/nemotron-4-340b-instruct",
            "object": "model",
            "created": 1640995200,
            "owned_by": "nvidia",
            "capabilities": {"text": True, "vision": False, "audio": False, "video": False, "image": False}
        },
        {
            "id": "microsoft/phi-4-multimodal",
            "object": "model",
            "created": 1640995200,
            "owned_by": "microsoft",
            "capabilities": {"text": True, "vision": True, "audio": False, "video": False, "image": False}
        },
        {
            "id": "meta/llama-3.2-11b-vision-instruct",
            "object": "model",
            "created": 1640995200,
            "owned_by": "meta",
            "capabilities": {"text": True, "vision": True, "audio": False, "video": False, "image": False}
        },
        {
            "id": "meta/llama-3.2-90b-vision-instruct",
            "object": "model",
            "created": 1640995200,
            "owned_by": "meta",
            "capabilities": {"text": True, "vision": True, "audio": False, "video": False, "image": False}
        }
    ]
    
    model_aliases = {
        "llama": "meta/llama-3.1-8b-instruct",
        "llama-vision": "meta/llama-3.2-11b-vision-instruct",
        "mixtral": "mistralai/mixtral-8x7b-instruct-v0.1",
        "phi": "microsoft/phi-3-medium-4k-instruct",
        "phi-vision": "microsoft/phi-4-multimodal",
        "nemotron": "nvidia/nemotron-4-340b-instruct",
    }


    @classmethod
    async def fetch_models_with_capabilities(cls, api_key: str = None) -> list[dict]:
        """
        Fetch available models from NVIDIA API with full capability information
        """
        # Try to get API key from parameter first, then from environment
        api_key = api_key or os.getenv('NVIDIA_API_KEY')
        if not api_key:
            # Return fallback models if no API key provided
            return cls._fallback_models
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        
        try:
            async with ClientSession(headers=headers) as session:
                async with session.get(cls.models_endpoint) as response:
                    if response.status == 200:
                        data = await response.json()
                        models_data = []
                        
                        for model in data.get('data', []):
                            model_id = model.get('id', '')
                            model_name = model.get('name', model_id)
                            
                            # Determine capabilities based on model name/id patterns
                            capabilities = cls._determine_model_capabilities(model_id, model_name)
                            
                            model_info = {
                                "id": model_id,
                                "object": model.get('object', 'model'),
                                "created": model.get('created', 1640995200),
                                "owned_by": model.get('owned_by', model_id.split('/')[0] if '/' in model_id else 'nvidia'),
                                "capabilities": capabilities,
                                "name": model_name,
                                "description": model.get('description', ''),
                            }
                            models_data.append(model_info)
                        
                        # Cache the fetched models
                        cls._cached_models_data = models_data
                        cls._model_data = {model["id"]: model for model in models_data}
                        cls.models = [model["id"] for model in models_data]
                        
                        return models_data
                    else:
                        print(f"Error fetching NVIDIA models: HTTP {response.status}")
                        return cls._fallback_models
        except Exception as e:
            print(f"Error fetching NVIDIA models: {e}")
            return cls._fallback_models
    
    @classmethod
    def _determine_model_capabilities(cls, model_id: str, model_name: str = '') -> dict:
        """
        Determine model capabilities based on model ID and name patterns
        Enhanced with real NVIDIA API model analysis
        """
        model_lower = (model_id + ' ' + model_name).lower()
        
        capabilities = {
            "text": True,  # All models support text by default
            "vision": False,
            "audio": False,
            "video": False,
            "image": False,
            "embedding": False,
            "code": False,
            "reasoning": False,
            "safety": False
        }
        
        # Vision/Multimodal capabilities
        if any(keyword in model_lower for keyword in [
            'vision', 'multimodal', 'visual', 'vlm', 'fuyu', 'paligemma', 
            'kosmos', 'neva', 'vila', 'deplot'
        ]):
            capabilities["vision"] = True
        
        # Audio capabilities
        if any(keyword in model_lower for keyword in [
            'audio', 'speech', 'whisper', 'sound', 'tts', 'riva-translate'
        ]):
            capabilities["audio"] = True
        
        # Video capabilities  
        if any(keyword in model_lower for keyword in ['video', 'clip', 'nvclip']):
            capabilities["video"] = True
        
        # Image generation capabilities
        if any(keyword in model_lower for keyword in [
            'dalle', 'imagen', 'stable-diffusion', 'midjourney', 'flux', 'sdxl'
        ]):
            capabilities["image"] = True
        
        # Embedding models
        if any(keyword in model_lower for keyword in [
            'embed', 'embedding', 'retriever', 'bge-m3', 'arctic-embed'
        ]):
            capabilities["embedding"] = True
            capabilities["text"] = False  # Embedding models don't do text generation
        
        # Code-specific models
        if any(keyword in model_lower for keyword in [
            'code', 'starcoder', 'codestral', 'codegemma', 'granite-code', 
            'embedcode', 'usdcode'
        ]):
            capabilities["code"] = True
        
        # Reasoning models
        if any(keyword in model_lower for keyword in [
            'reasoning', 'deepseek-r1', 'qwq', 'mathstral', 'flash-reasoning'
        ]):
            capabilities["reasoning"] = True
            
        # Safety/Guard models
        if any(keyword in model_lower for keyword in [
            'guard', 'safety', 'nemoguard', 'shield', 'content-safety', 
            'topic-control', 'guardian'
        ]):
            capabilities["safety"] = True
        
        return capabilities

    @classmethod
    async def fetch_models(cls, api_key: str = None) -> list[str]:
        """
        Fetch available model IDs from NVIDIA API
        """
        models_data = await cls.fetch_models_with_capabilities(api_key)
        return [model['id'] for model in models_data]

    # Initialize with fallback models to ensure proper model resolution
    models = [model["id"] for model in _fallback_models]
    _model_data = {model["id"]: model for model in _fallback_models}
    
    @classmethod
    def get_model(cls, model: str) -> str:
        # First check if it's an exact match in our models list
        if model in cls.models:
            return model
        
        # Then check if it's an alias
        if model in cls.model_aliases:
            resolved_model = cls.model_aliases[model]
            print(f"NVIDIA: Resolved alias '{model}' to '{resolved_model}'")
            return resolved_model
        
        # Check if the model contains 'llama' and try to find the best match
        if 'llama' in model.lower():
            for available_model in cls.models:
                if 'llama' in available_model and 'meta' in available_model:
                    print(f"NVIDIA: Found Llama model match: '{available_model}' for requested '{model}'")
                    return available_model
        
        # Check if the model contains 'qwen' and try to find the best match
        if 'qwen' in model.lower():
            for available_model in cls.models:
                if 'qwen' in available_model:
                    print(f"NVIDIA: Found Qwen model match: '{available_model}' for requested '{model}'")
                    return available_model
        
        # Default fallback
        print(f"NVIDIA: No match found for '{model}', using default '{cls.default_model}'")
        return cls.default_model

    @classmethod
    def fetch_models_sync(cls, api_key: str = None) -> list[str]:
        """
        Fetch available models from NVIDIA API synchronously
        """
        # Only fetch models when explicitly requested, avoid frequent API calls
        if cls.models:
            return cls.models
        
        # Try to get API key from parameter first, then from environment
        api_key = api_key or os.getenv('NVIDIA_API_KEY')
        if not api_key:
            # Return default models if no API key provided
            return [cls.default_model]
        
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        
        # Use the models endpoint from NVIDIA API
        models_endpoint = cls.models_endpoint
        
        try:
            response = requests.get(models_endpoint, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['id'] for model in data.get('data', [])]
                cls.models = models
                return models
            else:
                # Return default models if API call fails
                return [cls.default_model]
        except Exception as e:
            # Return default models if there's an error
            print(f"Error fetching NVIDIA models: {e}")
            return [cls.default_model]
    
   

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        """
        Get available models, fetching them dynamically if needed
        """
        if not cls.models or len(cls.models) == 0:
            # Try to get API key from kwargs first, then from environment
            api_key = kwargs.get('api_key') or os.getenv('NVIDIA_API_KEY')
            if api_key:
                # Fetch models synchronously
                fetched = cls.fetch_models_sync(api_key=api_key)
                if fetched:
                    return fetched
            # Return fallback models if no API key provided or fetching failed
            return cls._fallback_models
        return cls.models
    
    @classmethod
    async def get_provider_models_async(cls, api_key: str = None) -> list[dict]:
        """
        Get models with capabilities for GUI dropdown selection (async version)
        Fetches fresh data from NVIDIA API every time to ensure latest models
        """
        # Always fetch fresh models with capabilities
        models_data = await cls.fetch_models_with_capabilities(api_key=api_key)
        
        # Convert to GUI format with capabilities
        gui_models = []
        for model_info in models_data:
            capabilities = model_info.get('capabilities', {})
            gui_model = {
                "model": model_info['id'],
                "label": model_info['id'].split("/")[-1] if "/" in model_info['id'] else model_info['id'],
                "default": model_info['id'] == cls.default_model,
                "vision": capabilities.get('vision', False),
                "audio": capabilities.get('audio', False),
                "video": capabilities.get('video', False),
                "image": capabilities.get('image', False),
                "count": None,
                "owned_by": model_info.get('owned_by', ''),
                "description": model_info.get('description', ''),
                "capabilities_summary": cls._format_capabilities_summary(capabilities)
            }
            gui_models.append(gui_model)
        
        # Sort models by capabilities for better organization
        return cls._categorize_models(gui_models)
    
    @classmethod
    def fetch_models_sync_with_capabilities(cls, api_key: str = None) -> list[dict]:
        """
        Fetch models with capabilities synchronously using requests
        """
        # Try to get API key from parameter first, then from environment
        api_key = api_key or os.getenv('NVIDIA_API_KEY')
        if not api_key:
            # Return fallback models if no API key provided
            return cls._fallback_models
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        
        try:
            response = requests.get(cls.models_endpoint, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models_data = []
                
                for model in data.get('data', []):
                    model_id = model.get('id', '')
                    model_name = model.get('name', model_id)
                    
                    # Determine capabilities based on model name/id patterns
                    capabilities = cls._determine_model_capabilities(model_id, model_name)
                    
                    model_info = {
                        "id": model_id,
                        "object": model.get('object', 'model'),
                        "created": model.get('created', 1640995200),
                        "owned_by": model.get('owned_by', model_id.split('/')[0] if '/' in model_id else 'nvidia'),
                        "capabilities": capabilities,
                        "name": model_name,
                        "description": model.get('description', ''),
                    }
                    models_data.append(model_info)
                
                # Cache the fetched models
                cls._cached_models_data = models_data
                cls._model_data = {model["id"]: model for model in models_data}
                cls.models = [model["id"] for model in models_data]
                
                return models_data
            else:
                print(f"Error fetching NVIDIA models: HTTP {response.status_code}")
                return cls._fallback_models
        except Exception as e:
            print(f"Error fetching NVIDIA models: {e}")
            return cls._fallback_models
    
    @classmethod
    def get_provider_models(cls, **kwargs) -> list[dict]:
        """
        Get models with capabilities for GUI dropdown selection (sync version)
        Uses synchronous requests to avoid event loop issues in WSGI threads
        """
        api_key = kwargs.get('api_key') or os.getenv('NVIDIA_API_KEY')
        
        try:
            # Use synchronous fetching to avoid event loop issues
            models_data = cls.fetch_models_sync_with_capabilities(api_key=api_key)
            
            # Convert to GUI format with capabilities
            gui_models = []
            for model_info in models_data:
                capabilities = model_info.get('capabilities', {})
                
                # Create a better label with company prefix if available
                model_id = model_info['id']
                if "/" in model_id:
                    # Keep the full company/model format for better clarity
                    label = model_id
                else:
                    label = model_id
                
                gui_model = {
                    "model": model_id,
                    "label": label,
                    "default": model_id == cls.default_model,
                    "vision": capabilities.get('vision', False),
                    "audio": capabilities.get('audio', False),
                    "video": capabilities.get('video', False),
                    "image": capabilities.get('image', False),
                    "count": None,
                    "owned_by": model_info.get('owned_by', ''),
                    "description": model_info.get('description', ''),
                    "capabilities_summary": cls._format_capabilities_summary(capabilities)
                }
                gui_models.append(gui_model)
            
            # Sort models by capabilities for better organization
            return cls._categorize_models(gui_models)
            
        except Exception as e:
            print(f"Error fetching models: {e}")
            # Fallback to static models with capabilities
            gui_models = []
            for model_info in cls._fallback_models:
                capabilities = model_info.get('capabilities', {})
                
                # Create a better label with company prefix if available
                model_id = model_info['id']
                if "/" in model_id:
                    # Keep the full company/model format for better clarity
                    label = model_id
                else:
                    label = model_id
                
                gui_model = {
                    "model": model_id,
                    "label": label,
                    "default": model_id == cls.default_model,
                    "vision": capabilities.get('vision', False),
                    "audio": capabilities.get('audio', False),
                    "video": capabilities.get('video', False),
                    "image": capabilities.get('image', False),
                    "count": None,
                    "owned_by": model_info.get('owned_by', ''),
                    "description": model_info.get('description', ''),
                    "capabilities_summary": cls._format_capabilities_summary(capabilities)
                }
                gui_models.append(gui_model)
            return cls._categorize_models(gui_models)
    
    @classmethod
    def _format_capabilities_summary(cls, capabilities: dict) -> str:
        """
        Format capabilities into a readable summary string with better categorization
        """
        caps = []
        
        # Primary capability first
        if capabilities.get('embedding', False):
            caps.append('Embedding')
        elif capabilities.get('safety', False):
            caps.append('Safety')
        elif capabilities.get('code', False):
            caps.append('Code')
        elif capabilities.get('reasoning', False):
            caps.append('Reasoning')
        elif capabilities.get('text', False):
            caps.append('Text')
        
        # Additional capabilities
        if capabilities.get('vision', False):
            caps.append('Vision')
        if capabilities.get('audio', False):
            caps.append('Audio')
        if capabilities.get('video', False):
            caps.append('Video')
        if capabilities.get('image', False):
            caps.append('Image Gen')
        
        return ' + '.join(caps) if caps else 'Text'
    
    @classmethod
    def _categorize_models(cls, models: list[dict]) -> list[dict]:
        """
        Categorize and sort models by capabilities based on real NVIDIA API data
        """
        # Define categories in order of preference, matching actual NVIDIA offerings
        categories = {
            'reasoning': [],     # Reasoning/Math models (DeepSeek-R1, QwQ, etc.)
            'vision': [],        # Vision/Multimodal models
            'code': [],          # Code-specific models
            'embedding': [],     # Embedding/Retrieval models
            'safety': [],        # Safety/Guard models
            'video': [],         # Video processing models
            'audio': [],         # Audio/Speech models
            'image': [],         # Image generation models
            'chat': []           # Regular chat/text models
        }
        
        for model in models:
            capabilities = model.get('capabilities_summary', 'Text').lower()
            model_name = model.get('label', '').lower()
            
            # Categorize based on capabilities and model names
            if 'reasoning' in capabilities or any(keyword in model_name for keyword in ['deepseek-r1', 'qwq', 'mathstral', 'flash-reasoning']):
                categories['reasoning'].append(model)
            elif 'vision' in capabilities or model.get('vision', False):
                categories['vision'].append(model)
            elif 'code' in capabilities or any(keyword in model_name for keyword in ['code', 'starcoder', 'codestral', 'codegemma', 'granite-code']):
                categories['code'].append(model)
            elif 'embedding' in capabilities or any(keyword in model_name for keyword in ['embed', 'retriever', 'bge-m3', 'arctic-embed']):
                categories['embedding'].append(model)
            elif 'safety' in capabilities or any(keyword in model_name for keyword in ['guard', 'shield', 'safety', 'guardian']):
                categories['safety'].append(model)
            elif 'video' in capabilities or model.get('video', False):
                categories['video'].append(model)
            elif 'audio' in capabilities or model.get('audio', False):
                categories['audio'].append(model)
            elif 'image gen' in capabilities or model.get('image', False):
                categories['image'].append(model)
            else:
                categories['chat'].append(model)
        
        # Combine categories in order, with separators for GUI
        result = []
        
        category_labels = {
            'reasoning': '🧠 Reasoning & Math',
            'vision': '👁️ Vision & Multimodal', 
            'code': '💻 Code & Programming',
            'embedding': '📊 Embedding & Retrieval',
            'safety': '🛡️ Safety & Moderation',
            'video': '🎬 Video Processing',
            'audio': '🎵 Audio & Speech', 
            'image': '🎨 Image Generation',
            'chat': '💬 Chat & Text'
        }
        
        for category_name, category_models in categories.items():
            if category_models:
                # Add category header (GUI can use this for grouping)
                # Always add separator for every category, including the first one
                result.append({
                    "model": f"__separator_{category_name}__",
                    "label": f"--- {category_labels.get(category_name, category_name.title())} ---",
                    "default": False,
                    "vision": False,
                    "audio": False,
                    "video": False,
                    "image": False,
                    "count": None,
                    "is_separator": True
                })
                
                # Sort models within category by name
                category_models.sort(key=lambda x: x['label'])
                result.extend(category_models)
        
        return result
    
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        # Get API key from kwargs first, then from environment variable
        api_key = kwargs.get('api_key') or os.getenv('NVIDIA_API_KEY')
        if not api_key:
            raise ValueError("API key is required for NVIDIA provider. Set NVIDIA_API_KEY environment variable or pass api_key parameter.")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        }
        
        # Convert messages to OpenAI format
        formatted_messages = []
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(message)
            else:
                formatted_messages.append({"role": "user", "content": str(message)})
                
        data = {
            "model": model,
            "messages": formatted_messages,
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 1024),
            "stream": kwargs.get('stream', False),
        }
        
        try:
            async with ClientSession(headers=headers) as session:
                async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                    if response.status == 404:
                        raise ValueError(f"NVIDIA API endpoint not found: {cls.api_endpoint}. The API might have changed or your model '{model}' might not be available.")
                    elif response.status == 401:
                        raise ValueError("Invalid NVIDIA API key. Please check your NVIDIA_API_KEY.")
                    elif response.status == 403:
                        raise ValueError("Access forbidden. Check your NVIDIA API key permissions.")
                    
                    response.raise_for_status()
                    response_text = await response.text()
                    
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON response from NVIDIA API: {response_text[:200]}...")
                    
                    # Handle error responses
                    if "error" in response_data:
                        error_msg = response_data["error"].get("message", "Unknown error")
                        raise ValueError(f"NVIDIA API error: {error_msg}")
                    
                    # Extract content from the response
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        choice = response_data["choices"][0]
                        message = choice.get("message", {})
                        content = message.get("content", "")
                        
                        if content:
                            yield content
                        
                        # Yield finish reason
                        finish_reason = choice.get("finish_reason", "stop")
                        yield FinishReason(finish_reason)
                    else:
                        yield "No response content received from NVIDIA API."
                        yield FinishReason("error")
                        
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                yield f"Error: NVIDIA API endpoint not available. The service might be temporarily down or the endpoint has changed.\n\nTried endpoint: {cls.api_endpoint}\nModel: {model}\n\nPlease check the NVIDIA API documentation for current endpoints."
            else:
                yield f"Error communicating with NVIDIA API: {str(e)}"
            yield FinishReason("error")

    # Make sure the function is properly closed
    # The function should end here with proper indentation