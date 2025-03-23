from __future__ import annotations

import requests

from ...providers.types import Messages
from ...typing import MediaListType
from ...requests import StreamSession, raise_for_status
from ...errors import ModelNotSupportedError
from ...providers.response import ProviderInfo
from ..template.OpenaiTemplate import OpenaiTemplate
from .models import model_aliases, vision_models, default_llama_model, default_vision_model, text_models
from ... import debug

class HuggingFaceAPI(OpenaiTemplate):
    label = "HuggingFace (Text Generation)"
    parent = "HuggingFace"
    url = "https://api-inference.huggingface.com"
    api_base = "https://api-inference.huggingface.co/v1"
    working = True
    needs_auth = True

    default_model = default_llama_model
    default_vision_model = default_vision_model
    vision_models = vision_models
    model_aliases = model_aliases
    fallback_models = text_models + vision_models

    provider_mapping: dict[str, dict] = {
        "google/gemma-3-27b-it": {
            "hf-inference/models/google/gemma-3-27b-it": {
                "task": "conversational",
                "providerId": "google/gemma-3-27b-it"}}}

    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        try:
            return super().get_model(model, **kwargs)
        except ModelNotSupportedError:
            return model

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls.models:
            url = "https://huggingface.co/api/models?inference=warm&&expand[]=inferenceProviderMapping"
            response = requests.get(url)
            if response.ok: 
                cls.models = [
                    model["id"]
                    for model in response.json()
                    if [
                        provider
                        for provider in model.get("inferenceProviderMapping")
                        if provider.get("status") == "live" and provider.get("task") == "conversational"
                    ]
                ] + list(cls.provider_mapping.keys())
            else:
                cls.models = cls.fallback_models
        return cls.models

    @classmethod
    async def get_mapping(cls, model: str, api_key: str = None):
        if model in cls.provider_mapping:
            return cls.provider_mapping[model]
        async with StreamSession(
            timeout=30,
            headers=cls.get_headers(False, api_key),
        ) as session:
            async with session.get(f"https://huggingface.co/api/models/{model}?expand[]=inferenceProviderMapping") as response:
                await raise_for_status(response)
                model_data = await response.json()
                cls.provider_mapping[model] = model_data.get("inferenceProviderMapping")
        return cls.provider_mapping[model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = None,
        api_key: str = None,
        max_tokens: int = 2048,
        max_inputs_lenght: int = 10000,
        media: MediaListType = None,
        **kwargs
    ):
        if not model and media is not None:
            model = cls.default_vision_model
        model = cls.get_model(model)
        provider_mapping = await cls.get_mapping(model, api_key)
        if not provider_mapping:
            raise ModelNotSupportedError(f"Model is not supported: {model} in: {cls.__name__}")
        for provider_key in provider_mapping:
            api_path = provider_key if provider_key == "novita" else f"{provider_key}/v1"
            api_base = f"https://router.huggingface.co/{api_path}"
            task = provider_mapping[provider_key]["task"]
            if task != "conversational":
                raise ModelNotSupportedError(f"Model is not supported: {model} in: {cls.__name__} task: {task}")
            model = provider_mapping[provider_key]["providerId"]
            yield ProviderInfo(**{**cls.get_dict(), "label": f"HuggingFace ({provider_key})"})
            break
        # start = calculate_lenght(messages)
        # if start > max_inputs_lenght:
        #     if len(messages) > 6:
        #         messages = messages[:3] + messages[-3:]
        #     if calculate_lenght(messages) > max_inputs_lenght:
        #         last_user_message = [{"role": "user", "content": get_last_user_message(messages)}]
        #         if len(messages) > 2:
        #             messages = [m for m in messages if m["role"] == "system"] + last_user_message
        #         if len(messages) > 1 and calculate_lenght(messages) > max_inputs_lenght:
        #             messages = last_user_message
        #     debug.log(f"Messages trimmed from: {start} to: {calculate_lenght(messages)}")
        async for chunk in super().create_async_generator(model, messages, api_base=api_base, api_key=api_key, max_tokens=max_tokens, media=media, **kwargs):
            yield chunk

def calculate_lenght(messages: Messages) -> int:
    return sum([len(message["content"]) + 16 for message in messages])