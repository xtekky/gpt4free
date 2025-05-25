from __future__ import annotations

from aiohttp import ClientSession
import os
import re
import json
import random
import string
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from ..typing import AsyncResult, Messages, MediaListType
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import to_data_uri
from .helper import render_messages
from ..providers.response import JsonConversation
from ..tools.media import merge_media
from ..errors import RateLimitError, ModelNotFoundError
from .. import debug

class Conversation(JsonConversation):
    validated_value: str = None
    chat_id: str = None
    message_history: Messages = []

    def __init__(self, model: str):
        self.model = model

class Blackbox(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Blackbox AI"
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/chat"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "blackboxai"
    default_vision_model = default_model

    # OpenRouter Free
    openrouter_models = [
        "Deepcoder 14B Preview",
        "DeepHermes 3 Llama 3 8B Preview",
        "DeepSeek R1 Zero",
        "Dolphin3.0 Mistral 24B",
        "Dolphin3.0 R1 Mistral 24B",
        "Flash 3", # FIX (<reasoning> ◁</reasoning>)
        "Gemini 2.0 Flash Experimental",
        "Gemma 2 9B",
        "Gemma 3 12B",
        "Gemma 3 1B",
        "Gemma 3 27B",
        "Gemma 3 4B",
        "Kimi VL A3B Thinking", # FIX (◁think▷ ◁/think▷)
        "Llama 3.1 8B Instruct",
        "Llama 3.1 Nemotron Ultra 253B v1",
        "Llama 3.2 11B Vision Instruct",
        "Llama 3.2 1B Instruct",
        "Llama 3.2 3B Instruct",
        "Llama 3.3 70B Instruct",
        "Llama 3.3 Nemotron Super 49B v1",
        "Llama 4 Maverick",
        "Llama 4 Scout",
        "Mistral 7B Instruct",
        "Mistral Nemo",
        "Mistral Small 3",
        "Mistral Small 3.1 24B",
        "Molmo 7B D",
        "Moonlight 16B A3B Instruct",
        "Qwen2.5 72B Instruct",
        "Qwen2.5 7B Instruct",
        "Qwen2.5 Coder 32B Instruct",
        "Qwen2.5 VL 32B Instruct",
        "Qwen2.5 VL 3B Instruct",
        "Qwen2.5 VL 72B Instruct",
        "Qwen2.5-VL 7B Instruct",
        "Qwerky 72B",
        "QwQ 32B",
        "QwQ 32B Preview",
        "QwQ 32B RpR v1",
        "R1",
        "R1 Distill Llama 70B",
        "R1 Distill Qwen 14B",
        "R1 Distill Qwen 32B",
    ]

    models = [
        default_model, 
        "o3-mini", 
        "gpt-4.1-mini", 
        "gpt-4.1-nano", 
        "Claude-sonnet-3.7", 
        "Claude-sonnet-3.5", 
        "DeepSeek-R1", 
        "Mistral-Small-24B-Instruct-2501",
        
        # OpenRouter Free
        *openrouter_models,
        
        # Trending agent modes
        'Python Agent',
        'HTML Agent',
        'Builder Agent',
        'Java Agent',
        'JavaScript Agent',
        'React Agent',
        'Android Agent',
        'Flutter Agent',
        'Next.js Agent',
        'AngularJS Agent',
        'Swift Agent',
        'MongoDB Agent',
        'PyTorch Agent',
        'Xcode Agent',
        'Azure Agent',
        'Bitbucket Agent',
        'DigitalOcean Agent',
        'Docker Agent',
        'Electron Agent',
        'Erlang Agent',
        'FastAPI Agent',
        'Firebase Agent',
        'Flask Agent',
        'Git Agent',
        'Gitlab Agent',
        'Go Agent',
        'Godot Agent',
        'Google Cloud Agent',
        'Heroku Agent'
    ]
    
    vision_models = [default_vision_model, 'o3-mini']

    userSelectedModel = ['o3-mini','Claude-sonnet-3.7', 'Claude-sonnet-3.5', 'DeepSeek-R1', 'Mistral-Small-24B-Instruct-2501'] + openrouter_models

    # Agent mode configurations
    agentMode = {
        # OpenRouter Free
        'Deepcoder 14B Preview': {'mode': True, 'id': "agentica-org/deepcoder-14b-preview:free", 'name': "Deepcoder 14B Preview"},
        'DeepHermes 3 Llama 3 8B Preview': {'mode': True, 'id': "nousresearch/deephermes-3-llama-3-8b-preview:free", 'name': "DeepHermes 3 Llama 3 8B Preview"},
        'DeepSeek R1 Zero': {'mode': True, 'id': "deepseek/deepseek-r1-zero:free", 'name': "DeepSeek R1 Zero"},
        'Dolphin3.0 Mistral 24B': {'mode': True, 'id': "cognitivecomputations/dolphin3.0-mistral-24b:free", 'name': "Dolphin3.0 Mistral 24B"},
        'Dolphin3.0 R1 Mistral 24B': {'mode': True, 'id': "cognitivecomputations/dolphin3.0-r1-mistral-24b:free", 'name': "Dolphin3.0 R1 Mistral 24B"},
        'Flash 3': {'mode': True, 'id': "rekaai/reka-flash-3:free", 'name': "Flash 3"},
        'Gemini 2.0 Flash Experimental': {'mode': True, 'id': "google/gemini-2.0-flash-exp:free", 'name': "Gemini 2.0 Flash Experimental"},
        'Gemma 2 9B': {'mode': True, 'id': "google/gemma-2-9b-it:free", 'name': "Gemma 2 9B"},
        'Gemma 3 12B': {'mode': True, 'id': "google/gemma-3-12b-it:free", 'name': "Gemma 3 12B"},
        'Gemma 3 1B': {'mode': True, 'id': "google/gemma-3-1b-it:free", 'name': "Gemma 3 1B"},
        'Gemma 3 27B': {'mode': True, 'id': "google/gemma-3-27b-it:free", 'name': "Gemma 3 27B"},
        'Gemma 3 4B': {'mode': True, 'id': "google/gemma-3-4b-it:free", 'name': "Gemma 3 4B"},
        'Kimi VL A3B Thinking': {'mode': True, 'id': "moonshotai/kimi-vl-a3b-thinking:free", 'name': "Kimi VL A3B Thinking"},
        'Llama 3.1 8B Instruct': {'mode': True, 'id': "meta-llama/llama-3.1-8b-instruct:free", 'name': "Llama 3.1 8B Instruct"},
        'Llama 3.1 Nemotron Ultra 253B v1': {'mode': True, 'id': "nvidia/llama-3.1-nemotron-ultra-253b-v1:free", 'name': "Llama 3.1 Nemotron Ultra 253B v1"},
        'Llama 3.2 11B Vision Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-11b-vision-instruct:free", 'name': "Llama 3.2 11B Vision Instruct"},
        'Llama 3.2 1B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-1b-instruct:free", 'name': "Llama 3.2 1B Instruct"},
        'Llama 3.2 3B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-3b-instruct:free", 'name': "Llama 3.2 3B Instruct"},
        'Llama 3.3 70B Instruct': {'mode': True, 'id': "meta-llama/llama-3.3-70b-instruct:free", 'name': "Llama 3.3 70B Instruct"},
        'Llama 3.3 Nemotron Super 49B v1': {'mode': True, 'id': "nvidia/llama-3.3-nemotron-super-49b-v1:free", 'name': "Llama 3.3 Nemotron Super 49B v1"},
        'Llama 4 Maverick': {'mode': True, 'id': "meta-llama/llama-4-maverick:free", 'name': "Llama 4 Maverick"},
        'Llama 4 Scout': {'mode': True, 'id': "meta-llama/llama-4-scout:free", 'name': "Llama 4 Scout"},
        'Mistral 7B Instruct': {'mode': True, 'id': "mistralai/mistral-7b-instruct:free", 'name': "Mistral 7B Instruct"},
        'Mistral Nemo': {'mode': True, 'id': "mistralai/mistral-nemo:free", 'name': "Mistral Nemo"},
        'Mistral Small 3': {'mode': True, 'id': "mistralai/mistral-small-24b-instruct-2501:free", 'name': "Mistral Small 3"},
        'Mistral Small 3.1 24B': {'mode': True, 'id': "mistralai/mistral-small-3.1-24b-instruct:free", 'name': "Mistral Small 3.1 24B"},
        'Molmo 7B D': {'mode': True, 'id': "allenai/molmo-7b-d:free", 'name': "Molmo 7B D"},
        'Moonlight 16B A3B Instruct': {'mode': True, 'id': "moonshotai/moonlight-16b-a3b-instruct:free", 'name': "Moonlight 16B A3B Instruct"},
        'Qwen2.5 72B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-72b-instruct:free", 'name': "Qwen2.5 72B Instruct"},
        'Qwen2.5 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-7b-instruct:free", 'name': "Qwen2.5 7B Instruct"},
        'Qwen2.5 Coder 32B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-coder-32b-instruct:free", 'name': "Qwen2.5 Coder 32B Instruct"},
        'Qwen2.5 VL 32B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-32b-instruct:free", 'name': "Qwen2.5 VL 32B Instruct"},
        'Qwen2.5 VL 3B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-3b-instruct:free", 'name': "Qwen2.5 VL 3B Instruct"},
        'Qwen2.5 VL 72B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-72b-instruct:free", 'name': "Qwen2.5 VL 72B Instruct"},
        'Qwen2.5-VL 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-vl-7b-instruct:free", 'name': "Qwen2.5-VL 7B Instruct"},
        'Qwerky 72B': {'mode': True, 'id': "featherless/qwerky-72b:free", 'name': "Qwerky 72B"},
        'QwQ 32B': {'mode': True, 'id': "qwen/qwq-32b:free", 'name': "QwQ 32B"},
        'QwQ 32B Preview': {'mode': True, 'id': "qwen/qwq-32b-preview:free", 'name': "QwQ 32B Preview"},
        'QwQ 32B RpR v1': {'mode': True, 'id': "arliai/qwq-32b-arliai-rpr-v1:free", 'name': "QwQ 32B RpR v1"},
        'R1': {'mode': True, 'id': "deepseek/deepseek-r1:free", 'name': "R1"},
        'R1 Distill Llama 70B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-llama-70b:free", 'name': "R1 Distill Llama 70B"},
        'R1 Distill Qwen 14B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-14b:free", 'name': "R1 Distill Qwen 14B"},
        'R1 Distill Qwen 32B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-32b:free", 'name': "R1 Distill Qwen 32B"},
        
        # Default
        'Claude-sonnet-3.7': {'mode': True, 'id': "Claude-sonnet-3.7", 'name': "Claude-sonnet-3.7"},
        'Claude-sonnet-3.5': {'mode': True, 'id': "Claude-sonnet-3.5", 'name': "Claude-sonnet-3.5"},
        'DeepSeek-R1': {'mode': True, 'id': "deepseek-reasoner", 'name': "DeepSeek-R1"},
        'Mistral-Small-24B-Instruct-2501': {'mode': True, 'id': "mistralai/Mistral-Small-24B-Instruct-2501", 'name': "Mistral-Small-24B-Instruct-2501"},
    }

    # Trending agent modes
    trendingAgentMode = {
        'Python Agent': {'mode': True, 'id': "python"},
        'HTML Agent': {'mode': True, 'id': "html"},
        'Builder Agent': {'mode': True, 'id': "builder"},
        'Java Agent': {'mode': True, 'id': "java"},
        'JavaScript Agent': {'mode': True, 'id': "javascript"},
        'React Agent': {'mode': True, 'id': "react"},
        'Android Agent': {'mode': True, 'id': "android"},
        'Flutter Agent': {'mode': True, 'id': "flutter"},
        'Next.js Agent': {'mode': True, 'id': "next.js"},
        'AngularJS Agent': {'mode': True, 'id': "angularjs"},
        'Swift Agent': {'mode': True, 'id': "swift"},
        'MongoDB Agent': {'mode': True, 'id': "mongodb"},
        'PyTorch Agent': {'mode': True, 'id': "pytorch"},
        'Xcode Agent': {'mode': True, 'id': "xcode"},
        'Azure Agent': {'mode': True, 'id': "azure"},
        'Bitbucket Agent': {'mode': True, 'id': "bitbucket"},
        'DigitalOcean Agent': {'mode': True, 'id': "digitalocean"},
        'Docker Agent': {'mode': True, 'id': "docker"},
        'Electron Agent': {'mode': True, 'id': "electron"},
        'Erlang Agent': {'mode': True, 'id': "erlang"},
        'FastAPI Agent': {'mode': True, 'id': "fastapi"},
        'Firebase Agent': {'mode': True, 'id': "firebase"},
        'Flask Agent': {'mode': True, 'id': "flask"},
        'Git Agent': {'mode': True, 'id': "git"},
        'Gitlab Agent': {'mode': True, 'id': "gitlab"},
        'Go Agent': {'mode': True, 'id': "go"},
        'Godot Agent': {'mode': True, 'id': "godot"},
        'Google Cloud Agent': {'mode': True, 'id': "googlecloud"},
        'Heroku Agent': {'mode': True, 'id': "heroku"},
    }
    
    # Complete list of all models (for authorized users)
    _all_models = list(dict.fromkeys([
        *models,  # Include all free models
        *list(agentMode.keys()),
        *list(trendingAgentMode.keys())
    ]))
    
    model_aliases = {
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "gpt-4o-mini": default_model,
        "claude-3.7-sonnet": "Claude-sonnet-3.7",
        "claude-3.5-sonnet": "Claude-sonnet-3.5",
        "deepseek-r1": "DeepSeek-R1",
        #
        "deepcoder-14b": "Deepcoder 14B Preview",
        "deephermes-3-8b": "DeepHermes 3 Llama 3 8B Preview",
        "deepseek-r1-zero": "DeepSeek R1 Zero",
        "deepseek-r1": "DeepSeek R1 Zero",
        "dolphin-3.0-24b": "Dolphin3.0 Mistral 24B",
        "dolphin-3.0-r1-24b": "Dolphin3.0 R1 Mistral 24B",
        "reka-flash":  "Flash 3",
        "gemini-2.0-flash":  "Gemini 2.0 Flash Experimental",
        "gemma-2-9b":  "Gemma 2 9B",
        "gemma-3-12b":  "Gemma 3 12B",
        "gemma-3-1b":  "Gemma 3 1B",
        "gemma-3-27b":  "Gemma 3 27B",
        "gemma-3-4b":  "Gemma 3 4B",
        "kimi-vl-thinking":  "Kimi VL A3B Thinking",
        "llama-3.1-8b":  "Llama 3.1 8B Instruct",
        "nemotron-253b":  "Llama 3.1 Nemotron Ultra 253B v1",
        "llama-3.2-11b":  "Llama 3.2 11B Vision Instruct",
        "llama-3.2-1b":  "Llama 3.2 1B Instruct",
        "llama-3.2-3b":  "Llama 3.2 3B Instruct",
        "llama-3.3-70b":  "Llama 3.3 70B Instruct",
        "nemotron-49b":  "Llama 3.3 Nemotron Super 49B v1",
        "llama-4-maverick":  "Llama 4 Maverick",
        "llama-4-scout":  "Llama 4 Scout",
        "mistral-7b":  "Mistral 7B Instruct",
        "mistral-nemo":  "Mistral Nemo",
        "mistral-small":  ["Mistral Small 3", "Mistral-Small-24B-Instruct-2501", "Mistral Small 3.1 24B"],
        "mistral-small-24b":  ["Mistral Small 3", "Mistral-Small-24B-Instruct-2501"],
        "mistral-small-3.1-24b":  "Mistral Small 3.1 24B",
        "molmo-7b":  "Molmo 7B D",
        "moonlight-16b":  "Moonlight 16B A3B Instruct",
        "qwen-2.5-72b":  "Qwen2.5 72B Instruct",
        "qwen-2.5-7b":  "Qwen2.5 7B Instruct",
        "qwen-2.5-coder-32b":  "Qwen2.5 Coder 32B Instruct",
        "qwen-2.5-vl-32b":  "Qwen2.5 VL 32B Instruct",
        "qwen-2.5-vl-3b":  "Qwen2.5 VL 3B Instruct",
        "qwen-2.5-vl-72b":  "Qwen2.5 VL 72B Instruct",
        "qwen-2.5-vl-7b":  "Qwen2.5-VL 7B Instruct",
        "qwerky-72b":  "Qwerky 72B",
        "qwq-32b":  "QwQ 32B",
        "qwq-32b-preview":  "QwQ 32B Preview",
        "qwq-32b-arliai":  "QwQ 32B RpR v1",
        "deepseek-r1":  "R1",
        "deepseek-r1-distill-llama-70b":  "R1 Distill Llama 70B",
        "deepseek-r1-distill-qwen-14b":  "R1 Distill Qwen 14B",
        "deepseek-r1-distill-qwen-32b":  "R1 Distill Qwen 32B",
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
                debug.log(f"Blackbox: Selected model '{selected_model}' from alias '{model}'")
                return selected_model
            debug.log(f"Blackbox: Using model '{alias}' for alias '{model}'")
            return alias
        
        raise ModelNotFoundError(f"Model {model} not found")

    @classmethod
    def generate_session(cls, id_length: int = 21, days_ahead: int = 365) -> dict:
        """
        Generate a dynamic session with proper ID and expiry format using a fixed email.
        
        Args:
            id_length: Length of the numeric ID (default: 21)
            days_ahead: Number of days ahead for expiry (default: 365)
        
        Returns:
            dict: A session dictionary with user information and expiry
        """
        
        # Generate numeric ID
        numeric_id = ''.join(random.choice('0123456789') for _ in range(id_length))
        
        # Generate future expiry date
        future_date = datetime.now() + timedelta(days=days_ahead)
        expiry = future_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
        # Generate random image ID for the new URL format
        chars = string.ascii_letters + string.digits + "-"
        random_img_id = ''.join(random.choice(chars) for _ in range(48))
        image_url = f"https://lh3.googleusercontent.com/a/ACg8oc{random_img_id}=s96-c"
        
        return {
            "user": {
                "name": "BLACKBOX AI", 
                "email": "blackboxapp@blackboxai.tech", 
                "image": image_url, 
                "id": numeric_id
            }, 
            "expires": expiry,
            "isNewUser": False
        }

    @classmethod
    async def fetch_validated(cls, url: str = "https://www.blackbox.ai", force_refresh: bool = False) -> Optional[str]:
        cache_path = Path(os.path.expanduser("~")) / ".g4f" / "cache"
        cache_file = cache_path / 'blackbox.json'
        
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if data.get('validated_value'):
                        return data['validated_value']
            except Exception as e:
                debug.log(f"Blackbox: Error reading cache: {e}")
        
        js_file_pattern = r'static/chunks/\d{4}-[a-fA-F0-9]+\.js'
        uuid_pattern = r'["\']([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})["\']'

        def is_valid_context(text: str) -> bool:
            return any(char + '=' in text for char in 'abcdefghijklmnopqrstuvwxyz')

        async with ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None

                    page_content = await response.text()
                    js_files = re.findall(js_file_pattern, page_content)

                for js_file in js_files:
                    js_url = f"{url}/_next/{js_file}"
                    async with session.get(js_url) as js_response:
                        if js_response.status == 200:
                            js_content = await js_response.text()
                            for match in re.finditer(uuid_pattern, js_content):
                                start = max(0, match.start() - 10)
                                end = min(len(js_content), match.end() + 10)
                                context = js_content[start:end]

                                if is_valid_context(context):
                                    validated_value = match.group(1)
                                    
                                    cache_file.parent.mkdir(exist_ok=True, parents=True)
                                    try:
                                        with open(cache_file, 'w') as f:
                                            json.dump({'validated_value': validated_value}, f)
                                    except Exception as e:
                                        debug.log(f"Blackbox: Error writing cache: {e}")
                                        
                                    return validated_value

            except Exception as e:
                debug.log(f"Blackbox: Error retrieving validated_value: {e}")

        return None

    @classmethod
    def generate_id(cls, length: int = 7) -> str:
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        media: MediaListType = None,
        top_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        conversation: Conversation = None,
        return_conversation: bool = True,
        **kwargs
    ) -> AsyncResult:      
        model = cls.get_model(model)
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': 'https://www.blackbox.ai',
            'referer': 'https://www.blackbox.ai/',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }
        
        async with ClientSession(headers=headers) as session:
            if conversation is None or not hasattr(conversation, "chat_id"):
                conversation = Conversation(model)
                conversation.validated_value = await cls.fetch_validated()
                conversation.chat_id = cls.generate_id()
                conversation.message_history = []

            current_messages = []
            for i, msg in enumerate(render_messages(messages)):
                msg_id = conversation.chat_id if i == 0 and msg["role"] == "user" else cls.generate_id()
                current_msg = {
                    "id": msg_id,
                    "content": msg["content"],
                    "role": msg["role"]
                }
                current_messages.append(current_msg)

            media = list(merge_media(media, messages))
            if media:
                current_messages[-1]['data'] = {
                    "imagesData": [
                        {
                            "filePath": f"/{image_name}",
                            "contents": to_data_uri(image)
                        }
                        for image, image_name in media
                    ],
                    "fileText": "",
                    "title": ""
                }
            
            data = {
                "messages": current_messages,
                "agentMode": cls.agentMode.get(model, {}) if model in cls.agentMode else {},
                "id": conversation.chat_id,
                "previewToken": None,
                "userId": None,
                "codeModelMode": True,
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
                "isMemoryEnabled": False,
                "mobileClient": False,
                "userSelectedModel": model if model in cls.userSelectedModel else None,
                "validated": conversation.validated_value,
                "imageGenerationMode": False,
                "webSearchModePrompt": False,
                "deepSearchMode": False,
                "designerMode": False,
                "domains": None,
                "vscodeClient": False,
                "codeInterpreterMode": False,
                "customProfile": {
                    "additionalInfo": "",
                    "enableNewChats": False,
                    "name": "",
                    "occupation": "",
                    "traits": []
                },
                "webSearchModeOption": {
                    "autoMode": False,
                    "webMode": False,
                    "offlineMode": False
                },
                "session": cls.generate_session(),
                "isPremium": True, 
                "subscriptionCache": {
                    "expiryTimestamp": None,
                    "isTrialSubscription": False,
                    "lastChecked": int(datetime.now().timestamp() * 1000),
                    "status": "FREE",
                    "customerId": None
                },
                "beastMode": False,
                "reasoningMode": False,
                "workspaceId": "",
                "asyncMode": False,
                "webSearchMode": False
            }

            # Continue with the API request and async generator behavior
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                
                # Collect the full response
                full_response = []
                async for chunk in response.content.iter_any():
                    if chunk:
                        chunk_text = chunk.decode()
                        if "You have reached your request limit for the hour" in chunk_text:
                            raise RateLimitError(chunk_text)
                        full_response.append(chunk_text)
                        yield chunk_text
                
                full_response_text = ''.join(full_response)
                
                # Handle conversation history
                if return_conversation:
                    conversation.message_history.append({"role": "assistant", "content": full_response_text})
                    yield conversation
