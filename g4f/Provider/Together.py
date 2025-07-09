from __future__ import annotations

import requests
import random
from typing import Union

from ..typing import AsyncResult, Messages, MediaListType
from .template import OpenaiTemplate
from ..requests import StreamSession, raise_for_status
from ..errors import ModelNotFoundError
from .. import debug

class Together(OpenaiTemplate):
    label = "Together"
    url = "https://together.xyz"
    login_url = "https://api.together.ai/"
    api_base = "https://api.together.xyz/v1"
    activation_endpoint = "https://www.codegeneration.ai/activate-v2"
    models_endpoint = "https://api.together.xyz/v1/models"

    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'
    default_vision_model = default_model
    default_image_model = 'black-forest-labs/FLUX.1.1-pro'
    vision_models = [
        'Qwen/Qwen2-VL-72B-Instruct',
        'Qwen/Qwen2.5-VL-72B-Instruct',
        'arcee-ai/virtuoso-medium-v2',
        'arcee_ai/arcee-spotlight',
        'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
        'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
        default_vision_model,
        'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'meta-llama/Llama-Vision-Free',
    ]
    image_models = []
    models = []
    model_configs = {}  # Store model configurations including stop tokens
    _models_cached = False
    _api_key_cache = None

    model_aliases = {
        ### Models Chat/Language ###
        # meta-llama
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "llama-2-70b": ["meta-llama/Llama-2-70b-hf", "meta-llama/Llama-2-70b-hf",],
        "llama-3-70b": ["meta-llama/Meta-Llama-3-70B-Instruct-Turbo", "meta-llama/Llama-3-70b-chat-hf", "Rrrr/meta-llama/Llama-3-70b-chat-hf-6f9ad551", "roberizk@gmail.com/meta-llama/Llama-3-70b-chat-hf-26ee936b", "roberizk@gmail.com/meta-llama/Meta-Llama-3-70B-Instruct-6feb41f7"],
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "llama-3.3-70b": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",],
        "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "llama-3.1-8b": ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "blackbox/meta-llama-3-1-8b"],
        "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        #"llama-vision": "meta-llama/Llama-Vision-Free",
        "llama-3-8b": ["meta-llama/Llama-3-8b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct-Lite", "roberizk@gmail.com/meta-llama/Meta-Llama-3-8B-Instruct-8ced8839",],
        "llama-3.1-70b": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "Rrrr/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-03dc18e1", "Rrrr/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-6c92f39d"],
        "llama-3.1-405b": ["meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "eddiehou/meta-llama/Llama-3.1-405B"],
        "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        
        # arcee-ai
        #"arcee-spotlight": "arcee_ai/arcee-spotlight",
        #"arcee-blitz": "arcee-ai/arcee-blitz",
        #"arcee-caller": "arcee-ai/caller",
        #"arcee-virtuoso-large": "arcee-ai/virtuoso-large",
        #"arcee-virtuoso-medium": "arcee-ai/virtuoso-medium-v2",
        #"arcee-coder-large": "arcee-ai/coder-large",
        #"arcee-maestro": "arcee-ai/maestro-reasoning",
        #"afm-4.5b": "arcee-ai/AFM-4.5B-Preview", 
        
        # deepseek-ai
        "deepseek-r1": ["deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-R1-0528-tput"],
        "deepseek-v3": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-p-dp"],
        "deepseek-r1-distill-llama-70b": ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"],
        "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        
        # Qwen
        "qwen-2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "qwen-2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
        "qwq-32b": "Qwen/QwQ-32B",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "qwen-3-235b": ["Qwen/Qwen3-235B-A22B-fp8", "Qwen/Qwen3-235B-A22B-fp8-tput"],
        "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
        "qwen-3-32b": "Qwen/Qwen3-32B-FP8",
        
        # mistralai
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistral-7b": ["mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3"],
        
        # google
        "gemma-2b": "google/gemma-2b-it",
        "gemma-2-27b": "google/gemma-2-27b-it",
        "gemma-3-27b": "google/gemma-3-27b-it",
        "gemma-3n-e4b": "google/gemma-3n-E4B-it",
        
        # nvidia
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        
        # togethercomputer
        #"refuel-v2": "togethercomputer/Refuel-Llm-V2",
        #"moa": "togethercomputer/MoA-1",
        #"refuel-v2-small": "togethercomputer/Refuel-Llm-V2-Small",
        #"moa-turbo": "togethercomputer/MoA-1-Turbo",
        
        # lgai
        #"exaone-3.5-32b": "lgai/exaone-3-5-32b-instruct",
        #"exaone-deep-32b": "lgai/exaone-deep-32b",
        
        # Gryphe
        #"mythomax-2-13b": "Gryphe/MythoMax-L2-13b",
        #"mythomax-2-13b-lite": "Gryphe/MythoMax-L2-13b-Lite",
        
        # NousResearch
        "hermes-2-dpo": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        
        # marin-community
        #"marin-8b": "marin-community/marin-8b-instruct",
        
        # scb10x
        #"typhoon-2-70b": "scb10x/scb10x-llama3-1-typhoon2-70b-instruct",
        #"typhoon-2.1": "scb10x/scb10x-typhoon-2-1-gemma3-12b",
        
        # perplexity-ai
        "r1-1776": "perplexity-ai/r1-1776",

        ### Models Image ###
        # black-forest-labs
        "flux": ["black-forest-labs/FLUX.1-schnell-Free", "black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1.1-pro", "black-forest-labs/FLUX.1-pro", "black-forest-labs/FLUX.1.1-pro", "black-forest-labs/FLUX.1-pro", "black-forest-labs/FLUX.1-redux", "black-forest-labs/FLUX.1-depth", "black-forest-labs/FLUX.1-canny", "black-forest-labs/FLUX.1-kontext-max", "black-forest-labs/FLUX.1-dev-lora", "black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-dev-lora", "black-forest-labs/FLUX.1-kontext-pro", "black-forest-labs/FLUX.1-kontext-dev"],
        
        "flux-schnell": ["black-forest-labs/FLUX.1-schnell-Free", "black-forest-labs/FLUX.1-schnell"],
        "flux-pro": ["black-forest-labs/FLUX.1.1-pro", "black-forest-labs/FLUX.1-pro"],
        "flux-redux": "black-forest-labs/FLUX.1-redux",
        "flux-depth": "black-forest-labs/FLUX.1-depth",
        "flux-canny": "black-forest-labs/FLUX.1-canny",
        "flux-kontext-max": "black-forest-labs/FLUX.1-kontext-max",
        "flux-dev-lora": "black-forest-labs/FLUX.1-dev-lora",
        "flux-dev": ["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-dev-lora"],
        "flux-kontext-pro": "black-forest-labs/FLUX.1-kontext-pro",
        "flux-kontext-dev": "black-forest-labs/FLUX.1-kontext-dev",
    }

    @classmethod
    async def get_activation_key(cls, proxy: str = None) -> str:
        """Get API key from activation endpoint"""
        if cls._api_key_cache:
            return cls._api_key_cache
            
        headers = {
            "Accept": "application/json",
        }
        
        async with StreamSession(proxy=proxy, headers=headers) as session:
            async with session.get(cls.activation_endpoint) as response:
                await raise_for_status(response)
                activation_data = await response.json()
                cls._api_key_cache = activation_data["openAIParams"]["apiKey"]
                return cls._api_key_cache

    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = None) -> list[str]:
        """Override to load models from Together API with proper categorization"""
        if cls._models_cached and cls.models:
            return cls.models
            
        try:
            # Get API key synchronously for model loading
            if api_key is None and cls._api_key_cache is None:
                # Make synchronous request to activation endpoint
                headers = {"Accept": "application/json"}
                response = requests.get(cls.activation_endpoint, headers=headers)
                raise_for_status(response)
                activation_data = response.json()
                cls._api_key_cache = activation_data["openAIParams"]["apiKey"]
                api_key = cls._api_key_cache
            elif api_key is None:
                api_key = cls._api_key_cache
            
            # Get models from Together API
            headers = {
                "Authorization": f"Bearer {api_key}",
            }
            
            response = requests.get(cls.models_endpoint, headers=headers)
            raise_for_status(response)
            models_data = response.json()
            
            # Clear existing model lists and configs
            cls.models = []
            cls.image_models = []
            cls.model_configs = {}
            
            # Categorize models by type
            for model in models_data:
                if isinstance(model, dict):
                    model_id = model.get("id")
                    model_type = model.get("type", "").lower()
                    
                    if not model_id:
                        continue
                    
                    # Extract model configuration
                    config = model.get("config", {})
                    if config:
                        cls.model_configs[model_id] = {
                            "stop": config.get("stop", []),
                            "chat_template": config.get("chat_template"),
                            "bos_token": config.get("bos_token"),
                            "eos_token": config.get("eos_token"),
                            "context_length": model.get("context_length")
                        }
                    
                    # Check if it's a vision model
                    if model_id in cls.vision_models:
                        cls.models.append(model_id)
                        continue
                    
                    # Categorize by type
                    if model_type == "chat":
                        cls.models.append(model_id)
                    elif model_type == "language":
                        # Filter language models - add to models if they support chat/completion
                        cls.models.append(model_id)
                    elif model_type == "image":
                        cls.image_models.append(model_id)
                        cls.models.append(model_id)  # Also add to main models list
                    # Skip embedding, moderation, and audio models
                    elif model_type in ["embedding", "moderation", "audio"]:
                        continue
                    # If no type specified, assume it's a chat model
                    elif model_type == "":
                        cls.models.append(model_id)
            
            # Ensure default model is in the list
            if cls.default_model not in cls.models:
                cls.models.insert(0, cls.default_model)
            
            # Ensure vision models are in the list
            for vision_model in cls.vision_models:
                if vision_model not in cls.models:
                    cls.models.append(vision_model)
            
            # Sort all model lists
            cls.models.sort()
            cls.image_models.sort()
            
            cls._models_cached = True
            return cls.models
            
        except Exception as e:
            debug.error(e)
            # Return default model on error
            cls.models = [cls.default_model]
            return cls.models

    @classmethod
    def get_model_config(cls, model: str) -> dict:
        """Get configuration for a specific model"""
        model = cls.get_model(model)
        return cls.model_configs.get(model, {})

    @classmethod
    def get_model(cls, model: str, api_key: str = None, api_base: str = None) -> str:
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
                import random  # Add this import at the top of the file
                selected_model = random.choice(alias)
                debug.log(f"Together: Selected model '{selected_model}' from alias '{model}'")
                return selected_model
            debug.log(f"Together: Using model '{alias}' for alias '{model}'")
            return alias
        
        raise ModelNotFoundError(f"Together: Model {model} not found")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        api_key: str = None,
        media: MediaListType = None,
        stop: Union[str, list[str]] = None,
        **kwargs
    ) -> AsyncResult:
        # Get API key from activation endpoint if not provided
        if api_key is None:
            api_key = await cls.get_activation_key(proxy)
        
        # Load models if not cached
        if not cls._models_cached:
            cls.get_models(api_key)
        
        # Get model configuration
        model_config = cls.get_model_config(model)
        
        # Use model's default stop tokens if not provided
        if stop is None and model_config.get("stop"):
            stop = model_config["stop"]
        
        # Use parent implementation with the obtained API key
        async for chunk in super().create_async_generator(
            model=model,
            messages=messages,
            proxy=proxy,
            api_key=api_key,
            media=media,
            stop=stop,
            **kwargs
        ):
            yield chunk
