from __future__ import annotations

import os
import re
import json
from ..typing import AsyncResult, Messages, MediaListType, Union
from ..errors import ModelNotFoundError
from ..image import is_data_an_audio
from ..providers.retry_provider import RotatedProvider
from ..Provider.needs_auth import OpenaiChat, CopilotAccount
from ..Provider.hf_space import HuggingSpace
from ..Provider import Copilot, Cloudflare, Gemini, GeminiPro, Grok, DeepSeekAPI, PerplexityLabs, LambdaChat, PollinationsAI, PuterJS
from ..Provider import Microsoft_Phi_4_Multimodal, DeepInfraChat, Blackbox, OIVSCodeSer0501, OIVSCodeSer2, TeachAnything, OperaAria, Startnest
from ..Provider import WeWordle, Yqcloud, Chatai, ImageLabs, LegacyLMArena, LMArenaBeta, Free2GPT
from ..Provider import EdgeTTS, gTTS, MarkItDown, OpenAIFM
from ..Provider import HarProvider, HuggingFace, HuggingFaceMedia, Azure, Qwen, EasyChat, GLM
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .. import Provider
from .. import models
from .. import debug
from .any_model_map import audio_models, image_models, vision_models, video_models, model_map, models_count, parents, model_aliases

# Add all model aliases to the model map
PROVIERS_LIST_1 = [
    CopilotAccount, OpenaiChat, Cloudflare, PerplexityLabs, Gemini, Grok, DeepSeekAPI, Blackbox, OpenAIFM,
    OIVSCodeSer2, OIVSCodeSer0501, TeachAnything, WeWordle, Yqcloud, Chatai, Free2GPT, ImageLabs,
    # Has lazy loading model lists
    PollinationsAI, HarProvider, LegacyLMArena, LMArenaBeta, LambdaChat, DeepInfraChat,
    HuggingSpace, HuggingFace, HuggingFaceMedia, GeminiPro, PuterJS, OperaAria, Startnest
]

# Add all existing models to the model map
PROVIERS_LIST_2 = [
    OpenaiChat, Copilot, CopilotAccount, PollinationsAI, PerplexityLabs, Gemini, Grok, Azure, Qwen, EasyChat, GLM
]

# Add all models to the model map
PROVIERS_LIST_3 = [
    HarProvider, LambdaChat, DeepInfraChat, HuggingFace, HuggingFaceMedia, LegacyLMArena, LMArenaBeta,
    PuterJS, Cloudflare, HuggingSpace
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
    """Mixin to provide model-related methods for providers."""

    default_model = "default"
    audio_models = audio_models
    image_models = image_models
    vision_models = vision_models
    video_models = video_models
    models_count = models_count
    models = list(model_map.keys())
    model_map: dict[str, dict[str, str]] = model_map
    model_aliases: dict[str, str] = model_aliases

    @classmethod
    def extend_ignored(cls, ignored: list[str]) -> list[str]:
        """Extend the ignored list with parent providers."""
        for ignored_provider in ignored:
            if ignored_provider in parents and parents[ignored_provider] not in ignored:
                ignored.extend(parents[ignored_provider])
        return ignored

    @classmethod
    def get_models(cls, ignored: list[str] = []) -> list[str]:
        if not cls.models:
            cls.update_model_map()
        if not ignored:
            return cls.models
        ignored = cls.extend_ignored(ignored)
        filtered = []
        for model, providers in cls.model_map.items():
            for provider in providers.keys():
                if provider not in ignored:
                    filtered.append(model)
                    break
        return filtered

    @classmethod
    def update_model_map(cls):
        cls.create_model_map()
        file = os.path.join(os.path.dirname(__file__), "any_model_map.py")
        with open(file, "w", encoding="utf-8") as f:
            for key in ["audio_models", "image_models", "vision_models", "video_models", "model_map", "models_count", "parents", "model_aliases"]:
                value = getattr(cls, key)
                f.write(f"{key} = {json.dumps(value, indent=2) if isinstance(value, dict) else repr(value)}\n")

    @classmethod
    def create_model_map(cls):
        cls.audio_models = {}
        cls.image_models = []
        cls.vision_models = []
        cls.video_models = []

        # Get models from the models registry
        cls.model_map = {
            "default": {provider.__name__: "" for provider in models.default.best_provider.providers},
        }
        cls.model_map.update({ 
            name: {
                provider.__name__: model.get_long_name() for provider in providers
                if provider.working
            } for name, (model, providers) in models.__models__.items()
        })
        for name, (model, providers) in models.__models__.items():
            if isinstance(model, models.ImageModel):
                cls.image_models.append(name)

        # Process special providers
        for provider in PROVIERS_LIST_2:
            if not provider.working:
                continue
            try:
                if provider in [Copilot, CopilotAccount]:
                    for model in provider.model_aliases.keys():
                        if model not in cls.model_map:
                            cls.model_map[model] = {}
                        cls.model_map[model].update({provider.__name__: model})
                elif provider == PollinationsAI:
                    for model in provider.get_models():
                        pmodel = f"{provider.__name__}:{model}"
                        if pmodel not in cls.model_map:
                            cls.model_map[pmodel] = {}
                        cls.model_map[pmodel].update({provider.__name__: model})
                    cls.audio_models.update({f"{provider.__name__}:{model}": [] for model in provider.get_models() if model in provider.audio_models})
                    cls.image_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model in provider.image_models])
                    cls.vision_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model in provider.vision_models])
                    for model in provider.model_aliases.keys():
                        if model not in cls.model_map:
                            cls.model_map[model] = {}
                        cls.model_map[model].update({provider.__name__: model})
                else:
                    for model in provider.get_models():
                        cleaned = clean_name(model)
                        if cleaned not in cls.model_map:
                            cls.model_map[cleaned] = {}
                        cls.model_map[cleaned].update({provider.__name__: model})
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

        for provider in PROVIERS_LIST_3:
            if not provider.working:
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
            for alias, model in model_map.items():
                if alias not in cls.model_map:
                    cls.model_map[alias] = {}
                cls.model_map[alias].update({provider.__name__: model})

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

        for provider in PROVIERS_LIST_1:
            if provider.working:
                for model in provider.get_models():
                    if model in cls.model_map:
                        cls.model_map[model].update({provider.__name__: model})
                for alias, model in provider.model_aliases.items():
                    if alias in cls.model_map:
                        cls.model_map[alias].update({provider.__name__: model})
                if provider.__name__ == "GeminiPro":
                    for model in cls.model_map.keys():
                        if "gemini" in model or "gemma" in model:
                            cls.model_map[alias].update({provider.__name__: model})

        # Process audio providers
        for provider in [Microsoft_Phi_4_Multimodal, PollinationsAI]:
            if provider.working:
                cls.audio_models.update(provider.audio_models)

        # Update model counts
        for model, providers in cls.model_map.items():
            if len(providers) > 1:
                cls.models_count[model] = len(providers)

        cls.video_models.append("video")
        cls.model_map["video"] = {"Video": "video"}
        cls.audio_models = list(cls.audio_models.keys())

        # Create a mapping of parent providers to their children
        cls.parents = {}
        for provider in PROVIERS_LIST_1:
            if provider.working and provider.__name__ != provider.get_parent():
                if provider.get_parent() not in cls.parents:
                    cls.parents[provider.get_parent()] = [provider.__name__]
                elif provider.__name__ not in cls.parents[provider.get_parent()]:
                    cls.parents[provider.get_parent()].append(provider.__name__)

        for model, providers in cls.model_map.items():
            for provider, alias in providers.items():
                if alias != model and isinstance(alias, str) and alias not in cls.model_map:
                    cls.model_aliases[alias] = model

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
                if submodel in PollinationsAI.audio_models[PollinationsAI.default_audio_model]:
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
            elif model.startswith(("gpt-", "chatgpt-", "o1", "o1", "o3", "o4")) or model in ("auto", "searchgpt"):
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

class AnyProvider(AsyncGeneratorProvider, AnyModelProviderMixin):
    working = True
    active_by_default = True

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
        providers = []
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
        elif model and ":" in model:
            provider, submodel = model.split(":", maxsplit=1)
            if hasattr(Provider, provider):
                provider = getattr(Provider, provider)
                if provider.working and provider.get_parent() not in ignored:
                    providers.append(provider)
                    model = submodel
        else:
            if model not in cls.model_map:
                if model in cls.model_aliases:
                    model = cls.model_aliases[model]
            if model in cls.model_map:
                for provider, alias in cls.model_map[model].items():
                    provider = Provider.__map__[provider]
                    if model not in provider.model_aliases:
                        provider.model_aliases[model] = alias
                    providers.append(provider)
        if not providers:
            for provider in PROVIERS_LIST_1:
                if model in provider.get_models():
                    providers.append(provider)
                elif model in provider.model_aliases:
                    providers.append(provider)
        providers = [provider for provider in providers if provider.working and provider.get_parent() not in ignored]
        providers = list({provider.__name__: provider for provider in providers}.values())

        if len(providers) == 0:
            raise ModelNotFoundError(f"AnyProvider: Model {model} not found in any provider.")

        debug.log(f"AnyProvider: Using providers: {[provider.__name__ for provider in providers]} for model '{model}'")

        async for chunk in RotatedProvider(providers).create_async_generator(
            model,
            messages,
            stream=stream,
            media=media,
            api_key=api_key,
            **kwargs
        ):
            yield chunk

# Clean model names function
def clean_name(name: str) -> str:
    name = name.split("/")[-1].split(":")[0].lower()
    # Date patterns
    name = re.sub(r'-\d{4}-\d{2}-\d{2}', '', name)
    # name = re.sub(r'-\d{3,8}', '', name)
    name = re.sub(r'-\d{2}-\d{2}', '', name)
    name = re.sub(r'-[0-9a-f]{8}$', '', name)
    # Version patterns
    name = re.sub(r'-(instruct|chat|preview|experimental|v\d+|fp8|bf16|hf|free|tput)$', '', name)
    # Other replacements
    name = name.replace("_", ".")
    name = name.replace("c4ai-", "")
    name = name.replace("meta-llama-", "llama-")
    name = name.replace("llama3", "llama-3")
    name = name.replace("flux.1-", "flux-")
    name = name.replace("qwen1-", "qwen-1")
    name = name.replace("qwen2-", "qwen-2")
    name = name.replace("qwen3-", "qwen-3")
    name = name.replace("stable-diffusion-3.5-large", "sd-3.5-large")
    return name

setattr(Provider, "AnyProvider", AnyProvider)
Provider.__map__["AnyProvider"] = AnyProvider
Provider.__providers__.append(AnyProvider)
