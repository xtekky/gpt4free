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
from ..Provider import Microsoft_Phi_4_Multimodal, DeepInfraChat, Blackbox, OIVSCodeSer2, OIVSCodeSer0501, TeachAnything, Together, WeWordle, Yqcloud, Chatai, Free2GPT, ARTA, ImageLabs, LegacyLMArena
from ..Provider import EdgeTTS, gTTS, MarkItDown, OpenAIFM
from ..Provider import HarProvider, HuggingFace, HuggingFaceMedia
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .. import Provider
from .. import models

MAIN_PROVIERS = [
    OpenaiChat, Cloudflare, HarProvider, PerplexityLabs, Gemini, Grok, DeepSeekAPI, Blackbox, 
    OIVSCodeSer2, OIVSCodeSer0501, TeachAnything, Together, WeWordle, Yqcloud, Chatai, Free2GPT, ARTA, ImageLabs, LegacyLMArena,
    HuggingSpace, LambdaChat, CopilotAccount, PollinationsAI, DeepInfraChat, HuggingFace, HuggingFaceMedia
]

SPECIAL_PROVIDERS = [OpenaiChat, CopilotAccount, PollinationsAI, HuggingSpace, Cloudflare, PerplexityLabs, Gemini, Grok, LegacyLMArena, ARTA]

SPECIAL_PROVIDERS2 = [HarProvider, LambdaChat, DeepInfraChat, HuggingFace, HuggingFaceMedia, PuterJS]

LABELS = {
    "default": "Default",
    "openai": "OpenAI: ChatGPT",
    "llama": "Meta: LLaMA",
    "deepseek": "DeepSeek",
    "qwen": "Alibaba: Qwen",
    "google": "Google: Gemini / Gemma / Bard",
    "grok": "xAI: Grok",
    "claude": "Anthropic: Claude",
    "command": "Cohere: Command",
    "phi": "Microsoft: Phi / WizardLM",
    "mistral": "Mistral",
    "PollinationsAI": "Pollinations AI",
    "perplexity": "Perplexity Labs",
    "openrouter": "OpenRouter",
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

            # Check for PollinationsAI models (with prefix)
            if model.startswith("PollinationsAI:"):
                groups["PollinationsAI"].append(model)
                added = True
            # Check for Mistral company models specifically
            elif model.startswith("mistral") and not any(x in model for x in ["dolphin", "nous", "openhermes"]):
                groups["mistral"].append(model)
                added = True
            elif model.startswith(("mistralai/", "mixtral-", "pixtral-", "ministral-", "codestral-")):
                groups["mistral"].append(model)
                added = True
            # Check for Qwen models
            elif model.startswith(("qwen", "Qwen/", "qwq", "qvq")):
                groups["qwen"].append(model)
                added = True
            # Check for Microsoft Phi models
            elif model.startswith(("phi-", "microsoft/")):
                groups["phi"].append(model)
                added = True
            # Check for Meta LLaMA models
            elif model.startswith(("llama-", "meta-llama/", "llama2-", "llama3")):
                groups["llama"].append(model)
                added = True
            elif model == "meta-ai":
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
            # Check for Claude models
            elif model.startswith("claude-"):
                groups["claude"].append(model)
                added = True
            # Check for Grok models
            elif model.startswith("grok-"):
                groups["grok"].append(model)
                added = True
            # Check for DeepSeek models
            elif model.startswith(("deepseek-", "janus-")):
                groups["deepseek"].append(model)
                added = True
            # Check for Perplexity models
            elif model.startswith(("sonar", "sonar-", "pplx-")) or model == "r1-1776":
                groups["perplexity"].append(model)
                added = True
            # Check for OpenAI models
            elif model.startswith(("gpt-", "chatgpt-", "o1", "o1-", "o3-", "o4-")) or model in ("auto", "dall-e-3", "searchgpt"):
                groups["openai"].append(model)
                added = True
            # Check for openrouter models
            elif model.startswith(("openrouter:")):
                groups["openrouter"].append(model)
                added = True
            # Check for video models
            elif model in cls.video_models:
                groups["video"].append(model)
                added = True
            # Check for image models - UPDATED to include flux check
            elif model in cls.image_models or "flux" in model.lower() or "stable-diffusion" in model.lower() or "sdxl" in model.lower() or "gpt-image" in model.lower():
                groups["image"].append(model)
                added = True

            # If not categorized, check for special cases then put in other
            if not added:
                # CodeLlama is Meta's model
                if model.startswith("codellama-"):
                    groups["llama"].append(model)
                # WizardLM is Microsoft's
                elif "wizardlm" in model.lower():
                    groups["phi"].append(model)
                else:
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
            for provider in SPECIAL_PROVIDERS:
                provider: ProviderType = provider
                if not provider.working or provider.get_parent() in ignored:
                    continue
                if provider == CopilotAccount:
                    all_models.extend(list(provider.model_aliases.keys()))
                elif provider == PollinationsAI:
                    all_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model not in all_models])
                    cls.audio_models.update({f"{provider.__name__}:{model}": [] for model in provider.get_models() if model in provider.audio_models})
                    cls.image_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model in provider.image_models])
                    cls.vision_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model in provider.vision_models])
                    all_models.extend(list(provider.model_aliases.keys()))
                elif provider == LegacyLMArena:
                    # Add models from LegacyLMArena
                    provider_models = provider.get_models()
                    all_models.extend(provider_models)
                    # Also add model aliases
                    all_models.extend(list(provider.model_aliases.keys()))
                    # Add vision models
                    cls.vision_models.extend(provider.vision_models)
                elif provider == ARTA:
                    # Add all ARTA models as image models
                    arta_models = provider.get_models()
                    all_models.extend(arta_models)
                    cls.image_models.extend(arta_models)
                else:
                    all_models.extend(provider.get_models())

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
                name = re.sub(r'-\d{8}', '', name)
                name = re.sub(r'-\d{4}', '', name)
                name = re.sub(r'-\d{2}-\d{2}', '', name)
                # Version patterns
                name = re.sub(r'-(instruct|chat|preview|experimental|v\d+|fp8|bf16|hf)$', '', name)
                # Other replacements
                name = name.replace("_", ".")
                name = name.replace("c4ai-", "")
                name = name.replace("meta-llama-", "llama-")
                name = name.replace("llama3", "llama-3")
                name = name.replace("flux.1-", "flux-")
                return name

            # Process HAR providers
            for provider in SPECIAL_PROVIDERS2:
                if not provider.working or provider.get_parent() in ignored:
                    continue
                new_models = provider.get_models()
                if provider == HuggingFaceMedia:
                    new_models = provider.video_models

                # Add original models too, not just cleaned names
                all_models.extend(new_models)

                model_map = {model if model.startswith("openrouter:") else clean_name(model): model for model in new_models}
                if not provider.model_aliases:
                    provider.model_aliases = {}
                provider.model_aliases.update(model_map)
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
            cls.models_count.update({model: all_models.count(model) for model in all_models if all_models.count(model) > cls.models_count.get(model, 0)})

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
            providers = model.split(":")
            model = providers.pop()
            providers = [getattr(Provider, provider) for provider in providers]
        elif not model or model == cls.default_model:
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
                        if provider in __map__ and __map__[provider] not in MAIN_PROVIERS:
                            extra_providers.append(__map__[provider])
            for provider in MAIN_PROVIERS + extra_providers:
                if provider.working:
                    if not model or model in provider.get_models() or model in provider.model_aliases:
                       providers.append(provider)
            if model in models.__models__:
                for provider in models.__models__[model][1]:
                    providers.append(provider)
        providers = [provider for provider in providers if provider.working and provider.get_parent() not in ignored]
        providers = list({provider.__name__: provider for provider in providers}.values())

        if len(providers) == 0:
            raise ModelNotFoundError(f"AnyProvider: Model {model} not found in any provider.")

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
