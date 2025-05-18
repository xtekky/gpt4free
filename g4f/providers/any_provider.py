from __future__ import annotations

from ..typing import AsyncResult, Messages, MediaListType
from ..errors import ModelNotFoundError
from ..image import is_data_an_audio
from ..providers.retry_provider import IterListProvider
from ..providers.types import ProviderType
from ..Provider.needs_auth import OpenaiChat, CopilotAccount
from ..Provider.hf_space import HuggingSpace
from ..Provider import Cloudflare, Gemini, Grok, DeepSeekAPI, PerplexityLabs, LambdaChat, PollinationsAI, FreeRouter
from ..Provider import Microsoft_Phi_4_Multimodal, DeepInfraChat, Blackbox, EdgeTTS, gTTS, MarkItDown
from ..Provider import HarProvider, DDG, HuggingFace, HuggingFaceMedia
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .. import Provider
from .. import models

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
    "phi": "Microsoft: Phi",
    "mistral": "Mistral",
    "PollinationsAI": "Pollinations AI",
    "perplexity": "Perplexity Labs",
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
        for model in unsorted_models:
            added = False
            for group in groups:
                if group == "mistral":
                    if model.split("-")[0] in ("mistral", "mixtral", "mistralai", "pixtral", "ministral", "codestral"):
                        groups[group].append(model)
                        added = True
                        break
                elif group == "qwen":
                    if model.startswith("qwen") or model.startswith("qwq") or model.startswith("qvq"):
                        groups[group].append(model)
                        added = True
                        break
                elif group == "perplexity":
                    if model.startswith("sonar") or model == "r1-1776":
                        groups[group].append(model)
                        added = True
                        break
                elif group == "google":
                    if model.startswith("gemini-") or model.startswith("gemma-"):
                        groups[group].append(model)
                        added = True
                        break
                elif group == "openai":
                    if model.startswith(
                        "gpt-") or model.startswith(
                        "chatgpt-") or model.startswith(
                        "o1") or model.startswith(
                        "o3") or model.startswith(
                        "o4-") or model in ("auto", "dall-e-3", "searchgpt"):
                        groups[group].append(model)
                        added = True
                        break
                elif model.startswith(group):
                    groups[group].append(model)
                    added = True
                    break
                elif group == "video":
                    if model in cls.video_models:
                        groups[group].append(model)
                        added = True
                        break
                elif group == "image":
                    if model in cls.image_models:
                        groups[group].append(model)
                        added = True
                        break
            if not added:
                if model.startswith("janus"):
                    groups["deepseek"].append(model)
                elif model == "meta-ai":
                    groups["llama"].append(model)
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
            for provider in [OpenaiChat, CopilotAccount, PollinationsAI, HuggingSpace, Cloudflare, PerplexityLabs, Gemini, Grok, DDG]:
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
                else:
                    all_models.extend(provider.get_models())
                cls.image_models.extend(provider.image_models)
                cls.vision_models.extend(provider.vision_models)
                cls.video_models.extend(provider.video_models)
            def clean_name(name: str) -> str:
                return name.split("/")[-1].split(":")[0].lower(
                    ).replace("-instruct", ""
                    ).replace("-chat", ""
                    ).replace("-08-2024", ""
                    ).replace("-03-2025", ""
                    ).replace("-20241022", ""
                    ).replace("-20240904", ""
                    ).replace("-2025-04-16", ""
                    ).replace("-2025-04-14", ""
                    ).replace("-0125", ""
                    ).replace("-2407", ""
                    ).replace("-2501", ""
                    ).replace("-0324", ""
                    ).replace("-2409", ""
                    ).replace("-2410", ""
                    ).replace("-2411", ""
                    ).replace("-1119", ""
                    ).replace("-0919", ""
                    ).replace("-02-24", ""
                    ).replace("-03-25", ""
                    ).replace("-03-26", ""
                    ).replace("-01-21", ""
                    ).replace("-002", ""
                    ).replace("_", "."
                    ).replace("c4ai-", ""
                    ).replace("-preview", ""
                    ).replace("-experimental", ""
                    ).replace("-v1", ""
                    ).replace("-fp8", ""
                    ).replace("-bf16", ""
                    ).replace("-hf", ""
                    ).replace("flux.1-", "flux-"
                    ).replace("llama3", "llama-3"
                    ).replace("meta-llama-", "llama-")
            for provider in [HarProvider, LambdaChat, DeepInfraChat, HuggingFace, HuggingFaceMedia]:
                if not provider.working or provider.get_parent() in ignored:
                    continue
                new_models = provider.get_models()
                if provider == HuggingFaceMedia:
                    new_models = provider.video_models
                model_map = {clean_name(model): model for model in new_models}
                if not provider.model_aliases:
                    provider.model_aliases = {}
                provider.model_aliases.update(model_map)
                all_models.extend(list(model_map.keys()))
                cls.image_models.extend([clean_name(model) for model in provider.image_models])
                cls.vision_models.extend([clean_name(model) for model in provider.vision_models])
                cls.video_models.extend([clean_name(model) for model in provider.video_models])
            for provider in [Microsoft_Phi_4_Multimodal, PollinationsAI]:
                if provider.working and provider.get_parent() not in ignored:
                    cls.audio_models.update(provider.audio_models)
            cls.models_count.update({model: all_models.count(model) for model in all_models if all_models.count(model) > cls.models_count.get(model, 0)})
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
                providers = [PollinationsAI, EdgeTTS, gTTS]
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
            for provider in [
                OpenaiChat, Cloudflare, HarProvider, PerplexityLabs, Gemini, Grok, DeepSeekAPI, FreeRouter, Blackbox,
                HuggingSpace, LambdaChat, CopilotAccount, PollinationsAI, DeepInfraChat, DDG, HuggingFace, HuggingFaceMedia,
            ]:
                if provider.working:
                    if not model or model in provider.get_models() or model in provider.model_aliases:
                       providers.append(provider)
            if model in models.__models__:
                for provider in models.__models__[model][1]:
                    providers.append(provider)
        providers = [provider for provider in providers if provider.working and provider.get_parent() not in ignored]
        providers = list({provider.__name__: provider for provider in providers}.values())
        if len(providers) == 0:
            raise ModelNotFoundError(f"Model {model} not found in any provider.")
        async for chunk in IterListProvider(providers).create_async_generator(
            model,
            messages,
            stream=stream,
            media=media,
            **kwargs
        ):
            yield chunk

setattr(Provider, "AnyProvider", AnyProvider)
Provider.__map__["AnyProvider"] = AnyProvider
Provider.__providers__.append(AnyProvider)