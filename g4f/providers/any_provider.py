from __future__ import annotations

import re
from ..typing import AsyncResult, Messages, MediaListType, Union
from ..errors import ModelNotFoundError
from ..image import is_data_an_audio
from ..providers.retry_provider import IterListProvider
from ..providers.types import ProviderType
from ..Provider.needs_auth import OpenaiChat, CopilotAccount
from ..Provider.hf_space import HuggingSpace
from ..Provider import __map__
from ..Provider import Cloudflare, Gemini, Grok, DeepSeekAPI, PerplexityLabs, LambdaChat, PollinationsAI, PuterJS
from ..Provider import Microsoft_Phi_4_Multimodal, DeepInfraChat, Blackbox, OIVSCodeSer2, OIVSCodeSer0501, TeachAnything
from ..Provider import Together, WeWordle, Yqcloud, Chatai, Free2GPT, ImageLabs, LegacyLMArena, LMArenaBeta
from ..Provider import EdgeTTS, gTTS, MarkItDown, OpenAIFM, Video
from ..Provider import HarProvider, HuggingFace, HuggingFaceMedia
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .. import Provider
from .. import models
from .. import debug

PROVIERS_LIST_1 = [
    OpenaiChat, PollinationsAI, Cloudflare, PerplexityLabs, Gemini, Grok, DeepSeekAPI, Blackbox, OpenAIFM,
    OIVSCodeSer2, OIVSCodeSer0501, TeachAnything, Together, WeWordle, Yqcloud, Chatai, Free2GPT, ImageLabs,
    HarProvider, LegacyLMArena, LMArenaBeta, LambdaChat, CopilotAccount, DeepInfraChat,
    HuggingSpace, HuggingFace, HuggingFaceMedia, Together
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

class AnyProvider(AsyncGeneratorProvider, ProviderModelMixin):
    default_model = "default"
    working = True
    models_storage: dict[str, list[str]] = {}

    @classmethod
    def get_grouped_models(cls, ignored: list[str] = []) -> dict[str, list[str]]:
        unsorted_models = cls.get_models(ignored=ignored)
        groups = {key: [] for key in LABELS.keys()}

        # Always add default first
        groups["default"].append("default")

        for model in unsorted_models:
            if model == "default":
                continue  # Already added

            added = False
            # Check for models with prefix
            start = model.split(":")[0]
            if start in ("PollinationsAI", "openrouter"):
                submodel = model.split(":", maxsplit=1)[1]
                if submodel in OpenAIFM.voices or submodel in PollinationsAI.audio_models[PollinationsAI.default_audio_model]:
                    groups["voices"].append(submodel)
                else:
                    groups[start].append(model)
                added = True
            # Check for Mistral company models specifically
            elif model.startswith("mistral") and not any(x in model for x in ["dolphin", "nous", "openhermes"]):
                groups["mistral"].append(model)
                added = True
            elif model.startswith(("pixtral-", "ministral-", "codestral")) or "mistral" in model or "mixtral" in model:
                groups["mistral"].append(model)
                added = True
            # Check for Qwen models
            elif model.startswith(("qwen", "Qwen", "qwq", "qvq")):
                groups["qwen"].append(model)
                added = True
            # Check for Microsoft Phi models
            elif model.startswith(("phi-", "microsoft/")) or "wizardlm" in model.lower():
                groups["phi"].append(model)
                added = True
            # Check for Meta LLaMA models
            elif model.startswith(("llama-", "meta-llama/", "llama2-", "llama3")):
                groups["llama"].append(model)
                added = True
            elif model == "meta-ai" or model.startswith("codellama-"):
                groups["llama"].append(model)
                added = True
            # Check for Google models
            elif model.startswith(("gemini-", "gemma-", "google/", "bard-")):
                groups["google"].append(model)
                added = True
            # Check for Cohere Command models
            elif model.startswith(("command-", "CohereForAI/", "c4ai-command")):
                groups["command"].append(model)
                added = True
            # Check for DeepSeek models
            elif model.startswith(("deepseek-", "janus-")):
                groups["deepseek"].append(model)
                added = True
            # Check for Perplexity models
            elif model.startswith(("sonar", "sonar-", "pplx-")) or model == "r1-1776":
                groups["perplexity"].append(model)
                added = True
            # Check for image models - UPDATED to include flux check
            elif model in cls.image_models:
                groups["image"].append(model)
                added = True
            # Check for OpenAI models
            elif model.startswith(("gpt-", "chatgpt-", "o1", "o1-", "o3-", "o4-")) or model in ("auto", "searchgpt"):
                groups["openai"].append(model)
                added = True
            # Check for video models
            elif model in cls.video_models:
                groups["video"].append(model)
                added = True
            if not added:
                for group in LABELS.keys():
                    if model == group or group in model:
                        groups[group].append(model)
                        added = True
                        break
            # If not categorized, check for special cases then put in other
            if not added:
                groups["other"].append(model)
        return [
            {"group": LABELS[group], "models": names} for group, names in groups.items()
        ]

    @classmethod
    def get_models(cls, ignored: list[str] = []) -> list[str]:
        ignored_key = " ".join(ignored)
        if not cls.models_storage.get(ignored_key):
            cls.audio_models = {}
            cls.image_models = []
            cls.vision_models = []
            cls.video_models = []
            
            # Get models from the models registry
            model_with_providers = { 
                model: [
                    provider for provider in providers
                    if provider.working and provider.get_parent() not in ignored
                ] for model, (_, providers) in models.__models__.items()
            }
            model_with_providers = { 
                model: providers for model, providers in model_with_providers.items()
                if providers
            }
            cls.models_count = {
                model: len(providers) for model, providers in model_with_providers.items() if len(providers) > 1
            }
            all_models = [cls.default_model] + list(model_with_providers.keys())

            # Process special providers
            for provider in PROVIERS_LIST_2:
                provider: ProviderType = provider
                if not provider.working or provider.get_parent() in ignored:
                    continue
                try:
                    if provider == CopilotAccount:
                        all_models.extend(list(provider.model_aliases.keys()))
                    elif provider == PollinationsAI:
                        all_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model not in all_models])
                        cls.audio_models.update({f"{provider.__name__}:{model}": [] for model in provider.get_models() if model in provider.audio_models})
                        cls.image_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model in provider.image_models])
                        cls.vision_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model in provider.vision_models])
                        all_models.extend(list(provider.model_aliases.keys()))
                    else:
                        all_models.extend(provider.get_models())
                except Exception as e:
                    debug.error(f"Error getting models for provider {provider.__name__}:", e)
                    continue

                # Update special model lists
                if hasattr(provider, 'image_models'):
                    cls.image_models.extend(provider.image_models)
                if hasattr(provider, 'vision_models'):
                    cls.vision_models.extend(provider.vision_models)
                if hasattr(provider, 'video_models'):
                    cls.video_models.extend(provider.video_models)

            # Clean model names function
            def clean_name(name: str) -> str:
                name = name.split("/")[-1].split(":")[0].lower()
                # Date patterns
                name = re.sub(r'-\d{4}-\d{2}-\d{2}', '', name)
                name = re.sub(r'-\d{3,8}', '', name)
                name = re.sub(r'-\d{2}-\d{2}', '', name)
                name = re.sub(r'-[0-9a-f]{8}$', '', name)
                # Version patterns
                name = re.sub(r'-(instruct|chat|preview|experimental|v\d+|fp8|bf16|hf)$', '', name)
                # Other replacements
                name = name.replace("_", ".")
                name = name.replace("c4ai-", "")
                name = name.replace("meta-llama-", "llama-")
                name = name.replace("llama3", "llama-3")
                name = name.replace("flux.1-", "flux-")
                name = name.replace("-free", "")
                name = name.replace("qwen1-", "qwen-1")
                name = name.replace("qwen2-", "qwen-2")
                name = name.replace("qwen3-", "qwen-3")
                name = name.replace("stable-diffusion-3.5-large", "sd-3.5-large")
                return name

            # Process HAR providers
            for provider in PROVIERS_LIST_3:
                if not provider.working or provider.get_parent() in ignored:
                    continue
                try:
                    new_models = provider.get_models()
                except Exception as e:
                    debug.error(f"Error getting models for provider {provider.__name__}:", e)
                    continue
                if provider == HuggingFaceMedia:
                    new_models = provider.video_models
                model_map = {}
                for model in new_models:
                    clean_value = model if model.startswith("openrouter:") else clean_name(model)
                    if clean_value not in model_map:
                        model_map[clean_value] = model
                if provider.model_aliases:
                    model_map.update(provider.model_aliases)
                provider.model_aliases = model_map
                all_models.extend(list(model_map.keys()))

                # Update special model lists with both original and cleaned names
                if hasattr(provider, 'image_models'):
                    cls.image_models.extend(provider.image_models)
                    cls.image_models.extend([clean_name(model) for model in provider.image_models])
                if hasattr(provider, 'vision_models'):
                    cls.vision_models.extend(provider.vision_models)
                    cls.vision_models.extend([clean_name(model) for model in provider.vision_models])
                if hasattr(provider, 'video_models'):
                    cls.video_models.extend(provider.video_models)
                    cls.video_models.extend([clean_name(model) for model in provider.video_models])

            # Process audio providers
            for provider in [Microsoft_Phi_4_Multimodal, PollinationsAI]:
                if provider.working and provider.get_parent() not in ignored:
                    cls.audio_models.update(provider.audio_models)

            # Update model counts
            for model in all_models:
                count = all_models.count(model)
                if count > cls.models_count.get(model, 0):
                    cls.models_count.update({model: count})

            cls.video_models.append("video")
            all_models.append("video")

            # Deduplicate and store
            cls.models_storage[ignored_key] = list(dict.fromkeys([model if model else cls.default_model for model in all_models]))

        return cls.models_storage[ignored_key]

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
        cls.get_models(ignored=ignored)
        providers = []
        if model and ":" in model:
            provider, submodel = model.split(":", maxsplit=1)
            if hasattr(Provider, provider):
                provider = getattr(Provider, provider)
                if provider.working and provider.get_parent() not in ignored:
                    providers.append(provider)
                    model = submodel
        if not model or model == cls.default_model:
            model = ""
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
        else:
            extra_providers = []
            if isinstance(api_key, dict):
                for provider in api_key:
                    if api_key.get(provider):
                        if provider in __map__ and __map__[provider] not in PROVIERS_LIST_1:
                            extra_providers.append(__map__[provider])
            debug.log(f"Using extra providers: {[p.__name__ for p in extra_providers]}")
            for provider in PROVIERS_LIST_1 + extra_providers:
                if provider.working:
                    provider_api_key = api_key
                    if isinstance(api_key, dict):
                        provider_api_key = api_key.get(provider.get_parent())
                    try:
                        provider_models = provider.get_models(api_key=provider_api_key) if provider_api_key else provider.get_models()
                    except Exception as e:
                        debug.error(f"Error getting models for provider {provider.__name__}:", e)
                        continue
                    if model == "video":
                        providers.append(Video)
                    if model and provider == PuterJS:
                       providers.append(provider)
                    elif not model or model in provider_models or provider.model_aliases and model in provider.model_aliases or model in provider.model_aliases.values():
                       providers.append(provider)
                    elif provider.__name__ == "GeminiPro":
                        if model and "gemini" in model or "gemma" in model:
                            providers.append(provider)
            if model in models.__models__:
                for provider in models.__models__[model][1]:
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
