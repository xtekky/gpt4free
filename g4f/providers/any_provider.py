from __future__ import annotations

from ..typing import AsyncResult, Messages, MediaListType
from ..errors import ModelNotFoundError
from ..providers.retry_provider import IterListProvider
from ..image import is_data_an_audio
from ..providers.response import JsonConversation, ProviderInfo
from ..Provider.needs_auth import OpenaiChat, CopilotAccount
from ..Provider.hf import HuggingFace, HuggingFaceMedia
from ..Provider.hf_space import HuggingSpace
from .. import Provider
from .. import models
from ..Provider import Cloudflare, LMArenaProvider, Gemini, Grok, DeepSeekAPI, PerplexityLabs, LambdaChat, PollinationsAI, FreeRouter
from ..Provider import Microsoft_Phi_4, DeepInfraChat, Blackbox
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class AnyProvider(AsyncGeneratorProvider, ProviderModelMixin):
    default_model = "default"
    working = True

    @classmethod
    def get_models(cls, ignored: list[str] = []) -> list[str]:
        cls.audio_models = {}
        cls.image_models = []
        cls.vision_models = []
        cls.video_models = []
        model_with_providers = { 
            model: [
                provider for provider in providers
                if provider.working and getattr(provider, "parent", provider.__name__) not in ignored
            ] for model, (_, providers) in models.__models__.items()
        }
        model_with_providers = { 
            model: providers for model, providers in model_with_providers.items()
            if providers
        }
        cls.models_count = {
            model: len(providers) for model, providers in model_with_providers.items() if len(providers) > 1
        }
        all_models = ["default"] + list(model_with_providers.keys())
        for provider in [OpenaiChat, PollinationsAI, HuggingSpace, Cloudflare, PerplexityLabs, Gemini, Grok]:
            if not provider.working or getattr(provider, "parent", provider.__name__) in ignored:
                continue
            if provider == PollinationsAI:
                all_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model not in all_models])
                cls.audio_models.update({f"{provider.__name__}:{model}": [] for model in provider.get_models() if model in provider.audio_models})
                cls.image_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model in provider.image_models])
                cls.vision_models.extend([f"{provider.__name__}:{model}" for model in provider.get_models() if model in provider.vision_models])
            else:
                all_models.extend(provider.get_models())
            cls.image_models.extend(provider.image_models)
            cls.vision_models.extend(provider.vision_models)
            cls.video_models.extend(provider.video_models)
        if CopilotAccount.working and CopilotAccount.parent not in ignored:
            all_models.extend(list(CopilotAccount.model_aliases.keys()))
        if PollinationsAI.working and PollinationsAI.__name__ not in ignored:
            all_models.extend(list(PollinationsAI.model_aliases.keys()))
        def clean_name(name: str) -> str:
            return name.split("/")[-1].split(":")[0].lower(
                ).replace("-instruct", ""
                ).replace("-chat", ""
                ).replace("-08-2024", ""
                ).replace("-03-2025", ""
                ).replace("-20250219", ""
                ).replace("-20241022", ""
                ).replace("-2025-04-16", ""
                ).replace("-2025-04-14", ""
                ).replace("-0125", ""
                ).replace("-2407", ""
                ).replace("-2501", ""
                ).replace("-0324", ""
                ).replace("-2409", ""
                ).replace("-2410", ""
                ).replace("-2411", ""
                ).replace("-02-24", ""
                ).replace("-03-25", ""
                ).replace("-03-26", ""
                ).replace("-01-21", ""
                ).replace(".1-", "-"
                ).replace("_", "."
                ).replace("c4ai-", ""
                ).replace("-preview", ""
                ).replace("-experimental", ""
                ).replace("-v1", ""
                ).replace("-fp8", ""
                ).replace("-bf16", ""
                ).replace("-hf", ""
                ).replace("llama3", "llama-3")
        for provider in [HuggingFace, HuggingFaceMedia, LMArenaProvider, LambdaChat, DeepInfraChat]:
            if not provider.working or getattr(provider, "parent", provider.__name__) in ignored:
                continue
            model_map = {clean_name(model): model for model in provider.get_models()}
            provider.model_aliases.update(model_map)
            all_models.extend(list(model_map.keys()))
            cls.image_models.extend([clean_name(model) for model in provider.image_models])
            cls.vision_models.extend([clean_name(model) for model in provider.vision_models])
            cls.video_models.extend([clean_name(model) for model in provider.video_models])
        for provider in [Microsoft_Phi_4, PollinationsAI]:
            if provider.working and getattr(provider, "parent", provider.__name__) not in ignored:
                cls.audio_models.update(provider.audio_models)
        cls.models_count.update({model: all_models.count(model) + cls.models_count.get(model, 0) for model in all_models})
        return list(dict.fromkeys([model if model else "default" for model in all_models]))

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        media: MediaListType = None,
        ignored: list[str] = [],
        conversation: JsonConversation = None,
        **kwargs
    ) -> AsyncResult:
        providers = []
        if ":" in model:
            providers = model.split(":")
            model = providers.pop()
            providers = [getattr(Provider, provider) for provider in providers]
        elif not model or model == "default":
            has_image = False
            has_audio = "audio" in kwargs
            if not has_audio and media is not None:
                for media_data, filename in media:
                    if is_data_an_audio(media_data, filename):
                        has_audio = True
                        break
                    has_image = True
            if has_audio:
                providers = [PollinationsAI, Microsoft_Phi_4]
            elif has_image:
                providers = models.default_vision.best_provider.providers
            else:
                providers = models.default.best_provider.providers
        else:
            for provider in [OpenaiChat, HuggingSpace, Cloudflare, LMArenaProvider, PerplexityLabs, Gemini, Grok, DeepSeekAPI, FreeRouter, Blackbox]:
                if provider.working and (model if model else "auto") in provider.get_models():
                    providers.append(provider)
            for provider in [HuggingFace, HuggingFaceMedia, LambdaChat, LMArenaProvider, CopilotAccount, PollinationsAI, DeepInfraChat]:
                if model in provider.model_aliases:
                    providers.append(provider)
            if model in models.__models__:
                for provider in models.__models__[model][1]:
                    providers.append(provider)
        providers = [provider for provider in providers if provider.working and getattr(provider, "parent", provider.__name__) not in ignored]
        if len(providers) == 0:
            raise ModelNotFoundError(f"Model {model} not found in any provider.")
        if len(providers) == 1:
            provider = providers[0]
            if conversation is not None:
                child_conversation = getattr(conversation, provider.__name__, None)
                if child_conversation is not None:
                    kwargs["conversation"] = JsonConversation(**child_conversation)
            yield ProviderInfo(**provider.get_dict(), model=model)
            async for chunk in provider.get_async_create_function()(
                model,
                messages,
                stream=stream,
                media=media,
                **kwargs
            ):
                if isinstance(chunk, JsonConversation):
                    if conversation is None:
                        conversation = JsonConversation()
                    setattr(conversation, provider.__name__, chunk.get_dict())
                    yield conversation
                else:
                    yield chunk
            return
        async for chunk in IterListProvider(providers).get_async_create_function()(
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