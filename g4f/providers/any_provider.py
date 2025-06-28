from __future__ import annotations

import os
import re
import json
from ..typing import AsyncResult, Messages, MediaListType, Union
from ..errors import ModelNotFoundError
from ..image import is_data_an_audio
from ..providers.retry_provider import IterListProvider
from ..Provider import ProviderUtils # Keep this for converting provider names to classes
from ..Provider.needs_auth import OpenaiChat, CopilotAccount
from ..Provider.hf_space import HuggingSpace
from ..Provider import Cloudflare, Gemini, GeminiPro, Grok, DeepSeekAPI, PerplexityLabs, LambdaChat, PollinationsAI, PuterJS
from ..Provider import Microsoft_Phi_4_Multimodal, DeepInfraChat, Blackbox, OIVSCodeSer2, OIVSCodeSer0501, TeachAnything
from ..Provider import Together, WeWordle, Yqcloud, Chatai, Free2GPT, ImageLabs, LegacyLMArena, LMArenaBeta
from ..Provider import EdgeTTS, gTTS, MarkItDown, OpenAIFM, Video
from ..Provider import HarProvider, HuggingFace, HuggingFaceMedia
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .. import Provider
from .. import models
from .. import debug
from .any_model_map import audio_models, image_models, vision_models, video_models, model_map, models_count, parents

# =================================================================================
# BEGIN: External Priority Routing from g4f_routing.json
# =================================================================================
_priority_routing_config = None

def load_priority_routing():
    """
    Loads the priority routing configuration from 'g4f_routing.json' located in the package root.
    Caches the configuration to avoid repeated disk reads.
    """
    global _priority_routing_config
    if _priority_routing_config is not None:
        return _priority_routing_config

    _priority_routing_config = {}  # Default to empty if file not found or invalid
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.join(current_dir, '..')
        config_path = os.path.join(package_root, 'g4f_routing.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                # Use the "routes" key from the JSON file
                routes = config_data.get("routes", {})
                # Convert provider names (strings) to actual Provider classes
                _priority_routing_config = {
                    model: ProviderUtils.convert.get(provider_name)
                    for model, provider_name in routes.items()
                    if ProviderUtils.convert.get(provider_name) is not None
                }
                debug.log(f"Priority routing config loaded successfully from: {config_path}")
        else:
            debug.log(f"Priority routing file not found at path: {config_path}. Skipping.")
            
    except (json.JSONDecodeError, IOError) as e:
        debug.error(f"Error loading or parsing g4f_routing.json: {e}")
    except Exception as e:
        debug.error(f"An unexpected error occurred in load_priority_routing: {e}")

    return _priority_routing_config
# =================================================================================
# END: External Priority Routing
# =================================================================================


PROVIERS_LIST_1 = [
    OpenaiChat, PollinationsAI, Cloudflare, PerplexityLabs, Gemini, Grok, DeepSeekAPI, Blackbox, OpenAIFM,
    OIVSCodeSer2, OIVSCodeSer0501, TeachAnything, Together, WeWordle, Yqcloud, Chatai, Free2GPT, ImageLabs,
    HarProvider, LegacyLMArena, LMArenaBeta, LambdaChat, CopilotAccount, DeepInfraChat,
    HuggingSpace, HuggingFace, HuggingFaceMedia, Together, GeminiPro
]

PROVIERS_LIST_2 = [
    OpenaiChat, CopilotAccount, PollinationsAI, PerplexityLabs, Gemini, Grok
]

PROVIERS_LIST_3 = [
    HarProvider, LambdaChat, DeepInfraChat, HuggingFace, HuggingFaceMedia, HarProvider, LegacyLMArena, LMArenaBeta,
    PuterJS, Together, Cloudflare, HuggingSpace
]


LABELS = {
    "default": "Default",
    "openai": "OpenAI: ChatGPT",
    "llama": "Meta: LLaMA",
    "deepseek": "DeepSeek",
    "qwen": "Alibaba: Qwen",
    "google": "Google: Gemini / Gemma",
    "grok": "xAI: Grok",
    "claude": "Anthropic: Claude",
    "command": "Cohere: Command",
    "phi": "Microsoft: Phi / WizardLM",
    "mistral": "Mistral",
    "PollinationsAI": "Pollinations AI",
    "voices": "Voices",
    "perplexity": "Perplexity Labs",
    "openrouter": "OpenRouter",
    "glm": "GLM",
    "tulu": "Tulu",
    "reka": "Reka",
    "hermes": "Hermes",
    "video": "Video Generation",
    "image": "Image Generation",
    "other": "Other Models",
}

class AnyModelProviderMixin(ProviderModelMixin):
    # ... (le reste de cette classe est inchangé)
    pass


class AnyProvider(AsyncGeneratorProvider, AnyModelProviderMixin):
    working = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        media: MediaListType = None,
        ignored: list[str] = [],
        api_key: Union[str, dict[str, str]] = None,
        **kwargs
    ) -> AsyncResult:
        # =================================================================================
        # BEGIN: Priority Routing Logic (Final Version)
        # =================================================================================
        routing_map = load_priority_routing()
        if model in routing_map:
            provider = routing_map[model]
            if provider and provider.working and provider.get_parent() not in ignored:
                debug.log(f"Priority routing: Using {provider.__name__} directly for model '{model}'")
                try:
                    # Get the actual model name for the provider, handling aliases
                    actual_model_name_for_provider = provider.get_model(model)
                    # Directly call the provider's generator
                    async for chunk in provider.create_async_generator(
                        model=actual_model_name_for_provider,
                        messages=messages,
                        stream=stream,
                        media=media,
                        api_key=api_key,
                        **kwargs
                    ):
                        yield chunk
                    return  # Stop execution here to bypass the default AnyProvider logic
                except Exception as e:
                    debug.error(f"Priority provider {provider.__name__} failed for '{model}': {e}")
                    # If the priority provider fails, we let the default logic take over.
            else:
                debug.log(f"Priority provider for model '{model}' is not working or is ignored. Falling back to default logic.")
        # =================================================================================
        # END: Priority Routing Logic
        # =================================================================================

        providers = []
        if not model or model == cls.default_model:
            model = ""
            # ... (le reste de la logique de AnyProvider reste inchangé)
            has_image = False
            has_audio = False
            if not has_audio and media is not None:
                for media_data, filename in media:
                    if is_data_an_audio(media_data, filename):
                        has_audio = True
                        break
                    has_image = True
            if "tools" in kwargs:
                providers = [PollinationsAI]
            elif "audio" in kwargs or "audio" in kwargs.get("modalities", []):
                if kwargs.get("audio", {}).get("language") is None:
                    providers = [PollinationsAI, OpenAIFM, Gemini]
                else:
                    providers = [PollinationsAI, OpenAIFM, EdgeTTS, gTTS]
            elif has_audio:
                providers = [PollinationsAI, Microsoft_Phi_4_Multimodal, MarkItDown]
            elif has_image:
                providers = models.default_vision.best_provider.providers
            else:
                providers = models.default.best_provider.providers
        elif model in Provider.__map__:
            provider = Provider.__map__[model]
            if provider.working and provider.get_parent() not in ignored:
                model = None
                providers.append(provider)
        elif model and ":" in model:
            provider, submodel = model.split(":", maxsplit=1)
            if hasattr(Provider, provider):
                provider = getattr(Provider, provider)
                if provider.working and provider.get_parent() not in ignored:
                    providers.append(provider)
                    model = submodel
        else:
            if model in cls.model_map:
                for provider, alias in cls.model_map[model].items():
                    provider = Provider.__map__[provider]
                    provider.model_aliases[model] = alias
                    providers.append(provider)
        providers = [provider for provider in providers if provider.working and provider.get_parent() not in ignored]
        providers = list({provider.__name__: provider for provider in providers}.values())

        if len(providers) == 0:
            raise ModelNotFoundError(f"AnyProvider: Model {model} not found in any provider.")

        debug.log(f"AnyProvider: Using providers: {[provider.__name__ for provider in providers]} for model '{model}'")

        async for chunk in IterListProvider(providers).create_async_generator(
            model,
            messages,
            stream=stream,
            media=media,
            api_key=api_key,
            **kwargs
        ):
            yield chunk

setattr(Provider, "AnyProvider", AnyProvider)
Provider.__map__["AnyProvider"] = AnyProvider
Provider.__providers__.append(AnyProvider)