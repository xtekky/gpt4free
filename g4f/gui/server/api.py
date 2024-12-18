from __future__ import annotations

import logging
import os
import asyncio
from typing import Iterator
from flask import send_from_directory
from inspect import signature

from g4f import version, models
from g4f import ChatCompletion, get_model_and_provider
from g4f.errors import VersionNotFoundError
from g4f.image import ImagePreview, ImageResponse, copy_images, ensure_images_dir, images_dir
from g4f.Provider import ProviderType, __providers__, __map__
from g4f.providers.base_provider import ProviderModelMixin
from g4f.providers.retry_provider import IterListProvider
from g4f.providers.response import BaseConversation, FinishReason, SynthesizeData
from g4f.client.service import convert_to_provider
from g4f import debug

logger = logging.getLogger(__name__)
conversations: dict[dict[str, BaseConversation]] = {}

class Api:
    @staticmethod
    def get_models():
        return [{
            "name": model.name,
            "image": isinstance(model, models.ImageModel),
            "providers": [
                getattr(provider, "parent", provider.__name__)
                for provider in providers
                if provider.working
            ]
        }
        for model, providers in models.__models__.values()]

    @staticmethod
    def get_provider_models(provider: str, api_key: str = None):
        if provider in __map__:
            provider: ProviderType = __map__[provider]
            if issubclass(provider, ProviderModelMixin):
                if api_key is not None and "api_key" in signature(provider.get_models).parameters:
                    models = provider.get_models(api_key=api_key)
                else:
                    models = provider.get_models()
                return [
                    {
                        "model": model,
                        "default": model == provider.default_model,
                        "vision": getattr(provider, "default_vision_model", None) == model or model in getattr(provider, "vision_models", []),
                        "image": False if provider.image_models is None else model in provider.image_models,
                    }
                    for model in models
                ]
        return []

    @staticmethod
    def get_providers() -> dict[str, str]:
        return [{
            "name": provider.__name__,
            "label": provider.label if hasattr(provider, "label") else provider.__name__,
            "parent": getattr(provider, "parent", None),
            "image": getattr(provider, "image_models", None) is not None,
            "vision": getattr(provider, "default_vision_model", None) is not None,
            "webdriver": "webdriver" in provider.get_parameters(),
            "auth": provider.needs_auth,
        } for provider in __providers__ if provider.working]

    @staticmethod
    def get_version() -> dict:
        try:
            current_version = version.utils.current_version
        except VersionNotFoundError:
            current_version = None
        return {
            "version": current_version,
            "latest_version": version.utils.latest_version,
        }

    def serve_images(self, name):
        ensure_images_dir()
        return send_from_directory(os.path.abspath(images_dir), name)

    def _prepare_conversation_kwargs(self, json_data: dict, kwargs: dict):
        model = json_data.get('model') or models.default
        provider = json_data.get('provider')
        messages = json_data['messages']
        api_key = json_data.get("api_key")
        if api_key is not None:
            kwargs["api_key"] = api_key
        do_web_search = json_data.get('web_search')
        if do_web_search and provider:
            provider_handler = convert_to_provider(provider)
            if hasattr(provider_handler, "get_parameters"):
                if "web_search" in provider_handler.get_parameters():
                    kwargs['web_search'] = True
                    do_web_search = False
        if do_web_search:
            from .internet import get_search_message
            messages[-1]["content"] = get_search_message(messages[-1]["content"])
        if json_data.get("auto_continue"):
            kwargs['auto_continue'] = True

        conversation_id = json_data.get("conversation_id")
        if conversation_id and provider:
            if provider in conversations and conversation_id in conversations[provider]:
                kwargs["conversation"] = conversations[provider][conversation_id]

        if json_data.get("ignored"):
            kwargs["ignored"] = json_data["ignored"]

        return {
            "model": model,
            "provider": provider,
            "messages": messages,
            "stream": True,
            "ignore_stream": True,
            "return_conversation": True,
            **kwargs
        }

    def _create_response_stream(self, kwargs: dict, conversation_id: str, provider: str, download_images: bool = True) -> Iterator:
        def decorated_log(text: str):
            debug.logs.append(text)
            if debug.logging:
                debug.log_handler(text)
        debug.log = decorated_log
        proxy = os.environ.get("G4F_PROXY")
        provider = kwargs.get("provider")
        model, provider_handler = get_model_and_provider(
            kwargs.get("model"), provider,
            stream=True,
            ignore_stream=True,
            logging=False
        )
        first = True
        try:
            result = ChatCompletion.create(**{**kwargs, "model": model, "provider": provider_handler})
            for chunk in result:
                if first:
                    first = False
                    yield self.handle_provider(provider_handler, model)
                if isinstance(chunk, BaseConversation):
                    if provider is not None:
                        if provider not in conversations:
                            conversations[provider] = {}
                        conversations[provider][conversation_id] = chunk
                        yield self._format_json("conversation", conversation_id)
                elif isinstance(chunk, Exception):
                    logger.exception(chunk)
                    yield self._format_json("message", get_error_message(chunk))
                elif isinstance(chunk, ImagePreview):
                    yield self._format_json("preview", chunk.to_string())
                elif isinstance(chunk, ImageResponse):
                    images = chunk
                    if download_images:
                        images = asyncio.run(copy_images(chunk.get_list(), chunk.get("cookies"), proxy))
                        images = ImageResponse(images, chunk.alt)
                    yield self._format_json("content", str(images))
                elif isinstance(chunk, SynthesizeData):
                    yield self._format_json("synthesize", chunk.to_json())
                elif not isinstance(chunk, FinishReason):
                    yield self._format_json("content", str(chunk))
                if debug.logs:
                    for log in debug.logs:
                        yield self._format_json("log", str(log))
                    debug.logs = []
        except Exception as e:
            logger.exception(e)
            yield self._format_json('error', get_error_message(e))
        if first:
            yield self.handle_provider(provider_handler, model)

    def _format_json(self, response_type: str, content):
        return {
            'type': response_type,
            response_type: content
        }

    def handle_provider(self, provider_handler, model):
        if isinstance(provider_handler, IterListProvider) and provider_handler.last_provider is not None:
            provider_handler = provider_handler.last_provider
        if not model and hasattr(provider_handler, "last_model") and provider_handler.last_model is not None:
            model = provider_handler.last_model
        return self._format_json("provider", {**provider_handler.get_dict(), "model": model})

def get_error_message(exception: Exception) -> str:
    return f"{type(exception).__name__}: {exception}"