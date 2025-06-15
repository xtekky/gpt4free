from __future__ import annotations

import random
import json
import uuid
import asyncio


from ..typing import AsyncResult, Messages, MediaListType
from ..requests import StreamSession, StreamResponse, FormData, raise_for_status
from ..providers.response import JsonConversation, FinishReason
from ..tools.media import merge_media
from ..image import to_bytes, is_accepted_format
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message
from ..errors import ModelNotFoundError, ResponseError
from .. import debug

class LegacyLMArena(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Legacy LM Arena"
    url = "https://legacy.lmarena.ai"
    api_endpoint = "/queue/join?"
    
    working = True
    
    default_model = "chatgpt-4o-latest-20250326"
    models = []
    
    # Models from HAR data (manually added)
    har_models = [
        "chatgpt-4o-latest-20250326", "gemini-2.5-pro-preview-05-06", "o3-2025-04-16", 
        "o4-mini-2025-04-16", "qwen3-235b-a22b", "mistral-medium-2505", 
        "gemini-2.5-flash-preview-04-17", "gpt-4.1-2025-04-14", 
        "llama-4-maverick-03-26-experimental", "grok-3-preview-02-24", 
        "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219-thinking-32k", 
        "deepseek-v3-0324", "llama-4-maverick-17b-128e-instruct", 
        "llama-4-scout-17b-16e-instruct", "gpt-4.1-mini-2025-04-14", 
        "gpt-4.1-nano-2025-04-14"
    ]

    # Models from JS data (manually added)
    js_models = [
        "gemini-2.0-flash-001", "gemini-2.0-flash-lite-preview-02-05", 
        "gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it", 
        "deepseek-r1", "claude-3-5-sonnet-20241022", "o3-mini"
    ]
    
    # Updated vision models list from JS data
    vision_models = [
        "gemini-2.5-pro-preview-05-06", "o3-2025-04-16", "o4-mini-2025-04-16", 
        "mistral-medium-2505", "gemini-2.5-flash-preview-04-17", "gpt-4.1-2025-04-14", 
        "claude-3-7-sonnet-20250219", "claude-3-7-sonnet-20250219-thinking-32k", 
        "llama-4-maverick-17b-128e-instruct", "llama-4-scout-17b-16e-instruct", 
        "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14", "gemini-2.0-flash-001", 
        "gemini-2.0-flash-lite-preview-02-05", "gemma-3-27b-it", "claude-3-5-sonnet-20241022", 
        "gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", 
        "gpt-4o-2024-05-13", "mistral-small-3.1-24b-instruct-2503", 
        "claude-3-5-sonnet-20240620", "amazon-nova-pro-v1.0", "amazon-nova-lite-v1.0", 
        "qwen2.5-vl-32b-instruct", "qwen2.5-vl-72b-instruct", "gemini-1.5-pro-002", 
        "gemini-1.5-flash-002", "gemini-1.5-flash-8b-001", "gemini-1.5-pro-001", 
        "gemini-1.5-flash-001", "pixtral-large-2411", "step-1o-vision-32k-highres", 
        "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229", 
        "qwen-vl-max-1119", "qwen-vl-max-0809", "reka-core-20240904", 
        "reka-flash-20240904", "c4ai-aya-vision-32b", "pixtral-12b-2409"
    ]
    
    model_aliases = {
        # Existing aliases
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-3.7-sonnet-thinking": "claude-3-7-sonnet-20250219-thinking-32k",
        "gpt-4o": ["chatgpt-4o-latest-20250326", "chatgpt-4o-latest-20250129", "chatgpt-4o-latest-20241120", "chatgpt-4o-latest-20240903", "chatgpt-4o-latest-20240808", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"],
        "grok-3": ["early-grok-3", "grok-3-preview-02-24",],
        "gemini-2.0-flash-thinking": ["gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking-exp-1219",],
        "gemini-2.0-pro": "gemini-2.0-pro-exp-02-05",
        #"gemini-exp": ["gemini-exp-1206", "gemini-exp-1121", "gemini-exp-1114",],
        #"deepseek-r1": "deepseek-r1",
        "gemini-2.0-flash": ["gemini-2.0-flash-001", "gemini-2.0-flash-exp", "gemini-2.0-flash-lite-preview-02-05",],
        "o1": ["o1-2024-12-17", "o1-preview",],
        "qwen-2.5-max": "qwen2.5-max",
        "o3-mini": ["o3-mini-high", "o3-mini",],
        "o3": "o3-2025-04-16",
        "o4-mini": "o4-mini-2025-04-16",
        "deepseek-v3": ["deepseek-v3", "deepseek-v3-0324"],
        #"deepseek-v3-0324": "deepseek-v3-0324",
        "qwen-plus": ["qwen-plus-0125", "qwen-plus-0828", "qwen-plus-0125-exp"],
        "glm-4-plus": ["glm-4-plus-0111", "glm-4-plus",],
        #"step-2": ["step-2-16k-exp-202412", "step-2-16k-202502"],
        #"step-1o": "step-1o-vision-32k-highres",
        #"o1-mini": "o1-mini",
        "gemini-1.5-pro": ["gemini-1.5-pro-002", "gemini-1.5-pro-exp-0827", "gemini-1.5-pro-exp-0801", "gemini-1.5-pro-001", "gemini-1.5-pro-api-0409-preview",],
        "grok-2": "grok-2-2024-08-13",
        #"yi": "yi-lightning",
        "claude-3.5-sonnet": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620"],
        "qwen-2.5-plus": "qwen2.5-plus-1127",
        "deepseek-v2.5": ["deepseek-v2.5-1210", "deepseek-v2.5",],
        #"athene-v2": "athene-v2-chat",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gemini-1.5-flash": ["gemini-1.5-flash-002", "gemini-1.5-flash-exp-0827", "gemini-1.5-flash-001", "gemini-1.5-flash-8b-001", "gemini-1.5-flash-8b-exp-0827"],
        "llama-3.1-405b": ["llama-3.1-405b-instruct-bf16", "llama-3.1-405b-instruct-fp8",],
        "nemotron-70b": "llama-3.1-nemotron-70b-instruct",
        #"gemini-advanced": "gemini-advanced-0514",
        "grok-2-mini": "grok-2-mini-2024-08-13",
        #"yi-lite": "yi-lightning-lite",
        "qwen-max": ["qwen-max-0919", "qwen-max-0428", "qwen-max-2025-01-25"],
        "qwen-2.5-72b": "qwen2.5-72b-instruct",
        "qwen-2.5-vl-32b": "qwen2.5-vl-32b-instruct",
        "qwen-2.5-vl-72b": "qwen2.5-vl-72b-instruct",
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "llama-3.3-70b": "llama-3.3-70b-instruct",
        "nemotron-49b": "llama-3.3-nemotron-49b-super-v1",
        "mistral-large": ["mistral-large-2407", "mistral-large-2411", "mistral-large-2402",],
        "mistral-medium": ["mistral-medium", "mistral-medium-2505",],
        "pixtral-large": "pixtral-large-2411",
        #"pixtral-12b": "pixtral-12b-2409",
        #"athene-70b": "athene-70b-0725",
        "gpt-4": ["gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-0314", "gpt-4-0613",],
        "gpt-4.1": "gpt-4.1-2025-04-14",
        "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
        "llama-3.1-70b": "llama-3.1-70b-instruct",
        "nemotron-253b": "llama-3.1-nemotron-ultra-253b-v1",
        "claude-3-opus": "claude-3-opus-20240229",
        #"amazon-nova-pro-v1.0": "amazon-nova-pro-v1.0",
        "tulu-3-70b": "llama-3.1-tulu-3-70b",
        #"yi-large-preview": "yi-large-preview",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "reka-core": ["reka-core-20240904", "reka-core-20240722", "reka-core-20240501",],
        "gemma-2-27b": "gemma-2-27b-it",
        "gemma-3-27b": "gemma-3-27b-it",
        "gemma-3-12b": "gemma-3-12b-it",
        "gemma-3-4b": "gemma-3-4b-it",
        "deepseek-v2": "deepseek-v2-api-0628",
        "qwen-2.5-coder-32b": "qwen2.5-coder-32b-instruct",
        #"amazon-nova-lite-v1.0": "amazon-nova-lite-v1.0",
        "gemma-2-9b": ["gemma-2-9b-it-simpo", "gemma-2-9b-it",],
        "command-r-plus": ["command-r-plus-08-2024",  "command-r-plus",],
        "command-a": "command-a-03-2025",
        "deepseek-coder-v2": ["deepseek-coder-v2-0724", "deepseek-coder-v2",],
        #"yi-large": "yi-large",
        "nemotron-51b": "llama-3.1-nemotron-51b-instruct",
        "mistral-small-24b": "mistral-small-24b-instruct-2501",
        "mistral-small-3.1-24b": "mistral-small-3.1-24b-instruct-2503",
        #"c4ai-aya-expanse-32b": "c4ai-aya-expanse-32b",
        #"c4ai-aya-vision-32b": "c4ai-aya-vision-32b",
        "nemotron-4-340b": ["nemotron-4-340b-instruct", "nemotron-4-340b"],
        #"bard-jan-24-gemini-pro": "bard-jan-24-gemini-pro",
        "glm-4": ["glm-4-0520", "glm-4-0116",],
        "llama-3-70b": "llama-3-70b-instruct",
        "llama-4-maverick": ["llama-4-maverick-03-26-experimental", "llama-4-maverick-17b-128e-instruct"],
        "llama-4-scout": "llama-4-scout-17b-16e-instruct",
        "reka-flash": ["reka-flash-20240904", "reka-flash-20240722", "reka-flash-preview-20240611", "reka-flash-21b-20240226-online", "reka-flash-21b-20240226",],
        "phi-4": "phi-4",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        #"amazon-nova-micro-v1.0": "amazon-nova-micro-v1.0",
        #"hunyuan-standard-256k": "hunyuan-standard-256k",
        #"hunyuan-turbos": ["hunyuan-turbos-20250416", "hunyuan-turbos-20250226"],
        "qwen-2-72b": "qwen2-72b-instruct",
        "qwen-3-235b": "qwen3-235b-a22b",
        "qwen-3-30b": "qwen3-30b-a3b",
        "qwen-3-32b": "qwen3-32b",
        "tulu-3-8b": "llama-3.1-tulu-3-8b",
        #"ministral-8b": "ministral-8b-2410",
        #"c4ai-aya-expanse-8b": "c4ai-aya-expanse-8b",
        "command-r": ["command-r-08-2024", "command-r",],
        "codestral": "codestral-2405",
        "claude-3-haiku": "claude-3-haiku-20240307",
        #"jamba-1.5-mini": "jamba-1.5-mini",
        "llama-3.1-8b": "llama-3.1-8b-instruct",
        "qwen-1.5-110b": "qwen1.5-110b-chat",
        #"yi-1.5-34b-chat": "yi-1.5-34b-chat",
        "qwq-32b": "qwq-32b-preview",
        #"p2l-router-7b": "p2l-router-7b",
        "llama-3-8b": "llama-3-8b-instruct",
        #"internlm2_5-20b-chat": "internlm2_5-20b-chat",
        #"olmo-2-32b": "olmo-2-0325-32b-instruct",
        #"claude-1": "claude-1",
        "qwen-1.5-72b": "qwen1.5-72b-chat",
        "gemma-2-2b": "gemma-2-2b-it",
        #"jamba-1.5-large": "jamba-1.5-large",
        #"eureka-chatbot": "eureka-chatbot",
        "qwen-vl-max": ["qwen-vl-max-1119", "qwen-vl-max-0809"],
        "gemini-2.5-pro": "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
        "mixtral-8x22b": "mixtral-8x22b-instruct-v0.1",
        #"claude-2": ["claude-2.0", "claude-2.1"],
        #"gemini-pro-dev": "gemini-pro-dev-api",
        #"zephyr-orpo-141b": "zephyr-orpo-141b-A35b-v0.1",
        "qwen-1.5-32b": "qwen1.5-32b-chat",
        "qwen-1.5-14b": "qwen1.5-14b-chat",
        "qwen-1.5-7b": "qwen1.5-7b-chat",
        "qwen-1.5-4b": "qwen1.5-4b-chat",
        "mistral-next": "mistral-next",
        "phi-3-medium": "phi-3-medium-4k-instruct",
        "phi-3-small": "phi-3-small-8k-instruct",
        "phi-3-mini": ["phi-3-mini-4k-instruct-june-2024", "phi-3-mini-4k-instruct", "phi-3-mini-128k-instruct"],
        "tulu-2-70b": "tulu-2-dpo-70b",
        "llama-2-70b": ["llama-2-70b-chat", "llama2-70b-steerlm-chat"],
        "llama-2-13b": "llama-2-13b-chat",
        "llama-2-7b": "llama-2-7b-chat",
        #"openchat-3.5": ["openchat-3.5-0106", "openchat-3.5"],
        #"vicuna-33b": "vicuna-33b",
        #"vicuna-13b": "vicuna-13b",
        #"vicuna-7b": "vicuna-7b",
        #"snowflake-arctic": "snowflake-arctic-instruct",
        #"starling-lm-7b": ["starling-lm-7b-beta", "starling-lm-7b-alpha"],
        "hermes-2-dpo": "nous-hermes-2-mixtral-8x7b-dpo",
        #"pplx-70b-online": "pplx-70b-online",
        "pplx-7b-online":"pplx-7b-online",
        "deepseek-67b": "deepseek-llm-67b-chat",
        "openhermes-2.5-7b": "openhermes-2.5-mistral-7b",
        "mistral-7b": ["mistral-7b-instruct-v0.2", "mistral-7b-instruct"],
        #"dolphin-2.2.1-7b": "dolphin-2.2.1-mistral-7b",
        #"solar-10.7b": "solar-10.7b-instruct-v1.0",
        #"wizardlm-70b": "wizardlm-70b",
        #"wizardlm-13b": "wizardlm-13b",
        "llama-3.2-3b": "llama-3.2-3b-instruct",
        "llama-3.2-1b": "llama-3.2-1b-instruct",
        #"zephyr-7b": ["zephyr-7b-beta", "zephyr-7b-alpha"],
        #"smollm-2-1.7b": "smollm2-1.7b-instruct",
        #"mpt-30b": "mpt-30b-chat",
        #"mpt-7b": "mpt-7b-chat",
        "codellama-34b": "codellama-34b-instruct",
        "codellama-70b": "codellama-70b-instruct",
        "qwen-14b": "qwen-14b-chat",
        #"falcon-180b": "falcon-180b-chat",
        #"guanaco-33b": "guanaco-33b",
        #"stripedhyena-nous-7b": "stripedhyena-nous-7b",
        #"olmo-7b": "olmo-7b-instruct",
        #"palm-2": "palm-2",
        "gpt-3.5-turbo": ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-0314", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"],
        #"glm-3-6b": "chatglm3-6b",
        #"glm-2-6b": "chatglm2-6b",
        #"glm-6b":  "chatglm-6b",
        #"gpt4all-13b": "gpt4all-13b-snoozy",
        "mixtral-8x7b": "mixtral-8x7b-instruct-v0.1",
        #"yi-34b": "yi-34b-chat",
        #"claude": "claude-instant-1",
        #"gemini-pro": "gemini-pro",
        "dbrx-instruct": "dbrx-instruct-preview",
        #"rwkv-4-14b": "RWKV-4-Raven-14B",
        #"alpaca-13b": "alpaca-13b",
        #"oasst-pythia-12b": "oasst-pythia-12b",
        #"fastchat-t5-3b": "fastchat-t5-3b",
        #"stablelm-7b": "stablelm-tuned-alpha-7b",
        #"dolly-v2-12b": "dolly-v2-12b",
        #"llama-13b": "llama-13b",
    }

    @classmethod
    def get_models(cls):
        """Get models with improved fallback sources"""
        if cls.models:  # Return cached models if already loaded
            return cls.models
            
        try:
            # Try to fetch models from Google Storage first
            url = "https://storage.googleapis.com/public-arena-no-cors/p2l-explorer/data/overall/arena.json"
            import requests
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            leaderboard_models = [model[0] for model in data.get("leaderboard", [])]
            
            # Combine models from all sources and remove duplicates
            all_models = list(set(leaderboard_models + cls.har_models + cls.js_models))
            
            if all_models:
                # Ensure default model is at index 0
                if cls.default_model in all_models:
                    all_models.remove(cls.default_model)
                all_models.insert(0, cls.default_model)
                cls.models = all_models
                return cls.models
        except Exception as e:
            # Log the error and fall back to alternative sources
            debug.log(f"Failed to fetch models from Google Storage: {str(e)}")
        
        # Fallback: Use combined har_models and js_models
        combined_models = list(set(cls.har_models + cls.js_models))
        if combined_models:
            if cls.default_model in combined_models:
                combined_models.remove(cls.default_model)
            combined_models.insert(0, cls.default_model)
            cls.models = combined_models
            return cls.models
        
        # Final fallback: Use vision_models
        models = cls.vision_models.copy()
        if cls.default_model not in models:
            models.insert(0, cls.default_model)
        cls.models = models
        
        return cls.models

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the internal model name from the user-provided model name."""
        if not model:
            return cls.default_model
        
        # Ensure models are loaded
        if not cls.models:
            cls.get_models()
        
        # Check if the model exists directly in our models list
        if model in cls.models:
            return model
        
        # Check if there's an alias for this model
        if model in cls.model_aliases:
            alias = cls.model_aliases[model]
            # If the alias is a list, randomly select one of the options
            if isinstance(alias, list):
                selected_model = random.choice(alias)
                debug.log(f"LegacyLMArena: Selected model '{selected_model}' from alias '{model}'")
                return selected_model
            debug.log(f"LegacyLMArena: Using model '{alias}' for alias '{model}'")
            return alias
        
        # If model still not found, check in all available model sources directly
        all_available_models = list(set(cls.har_models + cls.js_models + cls.vision_models))
        if model in all_available_models:
            return model
        
        raise ModelNotFoundError(f"LegacyLMArena: Model {model} not found")

    @classmethod
    def _build_payloads(cls, model_id: str, session_hash: str, text: str, files: list, max_tokens: int, temperature: float, top_p: float):
        """Build payloads for new conversations"""
        first_payload = {
            "data": [
                None,
                model_id,
                {"text": text, "files": files},
                {
                    "text_models": [model_id],
                    "all_text_models": [model_id],
                    "vision_models": [],
                    "all_vision_models": [],
                    "image_gen_models": [],
                    "all_image_gen_models": [],
                    "search_models": [],
                    "all_search_models": [],
                    "models": [model_id],
                    "all_models": [model_id],
                    "arena_type": "text-arena"
                }
            ],
            "event_data": None,
            "fn_index": 119,
            "trigger_id": 159,
            "session_hash": session_hash
        }

        second_payload = {
            "data": [],
            "event_data": None,
            "fn_index": 120,
            "trigger_id": 159,
            "session_hash": session_hash
        }

        third_payload = {
            "data": [None, temperature, top_p, max_tokens],
            "event_data": None,
            "fn_index": 121,
            "trigger_id": 159,
            "session_hash": session_hash
        }

        return first_payload, second_payload, third_payload

    @classmethod
    def _build_continuation_payloads(cls, model_id: str, session_hash: str, text: str, max_tokens: int, temperature: float, top_p: float):
        """Renamed from _build_second_payloads for clarity"""
        first_payload = {
            "data":[None,model_id,text,{
                "text_models":[model_id],
                "all_text_models":[model_id],
                "vision_models":[],
                "image_gen_models":[],
                "all_image_gen_models":[],
                "search_models":[],
                "all_search_models":[],
                "models":[model_id],
                "all_models":[model_id],
                "arena_type":"text-arena"}],
            "event_data": None,
            "fn_index": 122,
            "trigger_id": 157,
            "session_hash": session_hash
        }

        second_payload = {
            "data": [],
            "event_data": None,
            "fn_index": 123,
            "trigger_id": 157,
            "session_hash": session_hash
        }

        third_payload = {
            "data": [None, temperature, top_p, max_tokens],
            "event_data": None,
            "fn_index": 124,
            "trigger_id": 157,
            "session_hash": session_hash
        }

        return first_payload, second_payload, third_payload

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        media: MediaListType = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1,
        conversation: JsonConversation = None,
        return_conversation: bool = True,
        max_retries: int = 1,
        **kwargs
    ) -> AsyncResult:
        async def read_response(response: StreamResponse):
            returned_data = ""
            async for line in response.iter_lines():
                if not line:
                    continue
                    
                # Handle both "data: " prefix and raw JSON
                if line.startswith(b"data: "):
                    line = line[6:]
                
                # Skip empty lines or non-JSON data
                line = line.strip()
                if not line or line == b"[DONE]":
                    continue
                    
                try:
                    json_data = json.loads(line)
                    
                    # Process data based on message type
                    if json_data.get("msg") == "process_generating":
                        output_data = json_data.get("output", {}).get("data", [])
                        if len(output_data) > 1 and output_data[1]:
                            # Extract content from various response formats
                            data = output_data[1]
                            content = None
                            
                            if isinstance(data, list):
                                if data and data[0] == "replace" and len(data) > 2:
                                    content = data[2]
                                elif data and isinstance(data[0], list) and len(data[0]) > 2:
                                    content = data[0][2]
                            elif isinstance(data, str):
                                # Handle direct string responses
                                content = data
                            
                            if content:
                                # Clean up content
                                if isinstance(content, str):
                                    if content.endswith("▌"):
                                        content = content[:-1]
                                    if content in ['<span class="cursor"></span> ', 'update', '']:
                                        continue
                                    if content.startswith(returned_data):
                                        content = content[len(returned_data):]
                                    if content:
                                        returned_data += content
                                        yield content
                    
                    # Process completed messages
                    elif json_data.get("msg") == "process_completed":
                        output_data = json_data.get("output", {}).get("data", [])
                        if len(output_data) > 1:
                            # Handle both list and direct content
                            if isinstance(output_data[1], list):
                                for item in output_data[1]:
                                    if isinstance(item, list) and len(item) > 1:
                                        content = item[1]
                                    elif isinstance(item, str):
                                        content = item
                                    else:
                                        continue
                                        
                                    if content and content != returned_data and content != '<span class="cursor"></span> ':
                                        if "**NETWORK ERROR DUE TO HIGH TRAFFIC." in content:
                                            raise ResponseError(content)
                                        if content.endswith("▌"):
                                            content = content[:-1]
                                        new_content = content
                                        if content.startswith(returned_data):
                                            new_content = content[len(returned_data):]
                                        if new_content:
                                            returned_data = content
                                            yield new_content
                            elif isinstance(output_data[1], str) and output_data[1]:
                                # Direct string content
                                content = output_data[1]
                                if content != returned_data:
                                    if content.endswith("▌"):
                                        content = content[:-1]
                                    new_content = content
                                    if content.startswith(returned_data):
                                        new_content = content[len(returned_data):]
                                    if new_content:
                                        returned_data = content
                                        yield new_content
                                        
                    # Also check for other message types that might contain content
                    elif json_data.get("msg") in ["process_starts", "heartbeat"]:
                        # These are status messages, skip them but don't error
                        continue
                        
                except json.JSONDecodeError:
                    # Skip non-JSON lines
                    continue
                except Exception as e:
                    if max_retries == 1:
                        raise e
                    debug.log(f"Error parsing response: {str(e)}")
                    continue
        
        # Get the actual model name
        model = cls.get_model(model)
        prompt = get_last_user_message(messages)
        
        async with StreamSession(impersonate="chrome") as session:
            # Add retry logic for better reliability
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Handle new conversation
                    if conversation is None:
                        conversation = JsonConversation(session_hash=str(uuid.uuid4()).replace("-", ""))
                        media_objects = []
                        
                        # Process media if present
                        media = list(merge_media(media, messages))
                        if media:
                            data = FormData()
                            for i in range(len(media)):
                                media[i] = (to_bytes(media[i][0]), media[i][1])
                            for image, image_name in media:
                                data.add_field(f"files", image, filename=image_name)
                            
                            # Upload media files
                            async with session.post(f"{cls.url}/upload", params={"upload_id": conversation.session_hash}, data=data) as response:
                                await raise_for_status(response)
                                image_files = await response.json()
                            
                            # Format media objects for API request
                            media_objects = [{
                                "path": image_file,
                                "url": f"{cls.url}/file={image_file}",
                                "orig_name": media[i][1],
                                "size": len(media[i][0]),
                                "mime_type": is_accepted_format(media[i][0]),
                                "meta": {
                                    "_type": "gradio.FileData"
                                }
                            } for i, image_file in enumerate(image_files)]
                        
                        # Build payloads for new conversation
                        first_payload, second_payload, third_payload = cls._build_payloads(
                            model, conversation.session_hash, prompt, media_objects, 
                            max_tokens, temperature, top_p
                        )
                        
                        headers = {
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        }
                        
                        # Send the three required requests with small delays
                        async with session.post(f"{cls.url}{cls.api_endpoint}", json=first_payload, proxy=proxy, headers=headers) as response:
                            await raise_for_status(response)
                        
                        # Small delay between requests
                        await asyncio.sleep(0.1)
                        
                        async with session.post(f"{cls.url}{cls.api_endpoint}", json=second_payload, proxy=proxy, headers=headers) as response:
                            await raise_for_status(response)
                        
                        await asyncio.sleep(0.1)
                        
                        async with session.post(f"{cls.url}{cls.api_endpoint}", json=third_payload, proxy=proxy, headers=headers) as response:
                            await raise_for_status(response)
                        
                        # Small delay before streaming
                        await asyncio.sleep(0.2)
                        
                        # Stream the response
                        stream_url = f"{cls.url}/queue/data?session_hash={conversation.session_hash}"
                        async with session.get(stream_url, headers={"Accept": "text/event-stream"}, proxy=proxy) as response:
                            await raise_for_status(response)
                            count = 0
                            has_content = False
                            
                            # Add timeout for response
                            try:
                                async with asyncio.timeout(30):  # 30 second timeout
                                    async for chunk in read_response(response):
                                        count += 1
                                        has_content = True
                                        yield chunk
                            except asyncio.TimeoutError:
                                if not has_content:
                                    raise RuntimeError("Response timeout - no data received from server")
                        
                        # Only raise error if we truly got no content
                        if count == 0 and not has_content:
                            retry_count += 1
                            if retry_count < max_retries:
                                debug.log(f"No response received, retrying... (attempt {retry_count + 1}/{max_retries})")
                                await asyncio.sleep(1)  # Wait before retry
                                conversation = None  # Reset conversation for retry
                                continue
                            else:
                                raise RuntimeError("No response from server after multiple attempts")
                        
                        # Success - break retry loop
                        break
                        
                    # Handle continuation of existing conversation
                    else:
                        # Build payloads for conversation continuation
                        first_payload, second_payload, third_payload = cls._build_continuation_payloads(
                            model, conversation.session_hash, prompt, max_tokens, temperature, top_p
                        )
                        
                        headers = {
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        }
                        
                        # Send the three required requests with delays
                        async with session.post(f"{cls.url}{cls.api_endpoint}", json=first_payload, proxy=proxy, headers=headers) as response:
                            await raise_for_status(response)
                        
                        await asyncio.sleep(0.1)
                        
                        async with session.post(f"{cls.url}{cls.api_endpoint}", json=second_payload, proxy=proxy, headers=headers) as response:
                            await raise_for_status(response)
                        
                        await asyncio.sleep(0.1)
                        
                        async with session.post(f"{cls.url}{cls.api_endpoint}", json=third_payload, proxy=proxy, headers=headers) as response:
                            await raise_for_status(response)
                        
                        await asyncio.sleep(0.2)
                        
                        # Stream the response
                        stream_url = f"{cls.url}/queue/data?session_hash={conversation.session_hash}"
                        async with session.get(stream_url, headers={"Accept": "text/event-stream"}, proxy=proxy) as response:
                            await raise_for_status(response)
                            count = 0
                            has_content = False
                            
                            try:
                                async with asyncio.timeout(30):
                                    async for chunk in read_response(response):
                                        count += 1
                                        has_content = True
                                        yield chunk
                            except asyncio.TimeoutError:
                                if not has_content:
                                    raise RuntimeError("Response timeout - no data received from server")
                        
                        if count == 0 and not has_content:
                            raise RuntimeError("No response from server in conversation continuation")
                        
                        # Success - break retry loop
                        break
                        
                except Exception as e:
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        debug.log(f"Error occurred: {str(e)}, retrying... (attempt {retry_count + 1}/{max_retries})")
                        await asyncio.sleep(1)
                        conversation = None  # Reset for retry
                        continue
                    else:
                        raise
            
            # Return conversation object for future interactions
            if return_conversation and conversation:
                yield conversation
            
            # Yield finish reason if we hit token limit
            if count >= max_tokens:
                yield FinishReason("length")
