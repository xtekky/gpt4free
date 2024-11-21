from __future__ import annotations

import logging
import os
import asyncio
from typing import Iterator
from flask import send_from_directory
from inspect import signature

from g4f import version, models
from g4f import get_last_provider, ChatCompletion
from g4f.errors import VersionNotFoundError
from g4f.image import ImagePreview, ImageResponse, copy_images, ensure_images_dir, images_dir
from g4f.Provider import ProviderType, __providers__, __map__
from g4f.providers.base_provider import ProviderModelMixin
from g4f.providers.response import BaseConversation, FinishReason, SynthesizeData
from g4f.client.service import convert_to_provider
from g4f import debug

logger = logging.getLogger(__name__)
conversations: dict[dict[str, BaseConversation]] = {}

class Api:
    @staticmethod
    def get_models() -> list[str]:
        return models._all_models

    @staticmethod
    def get_provider_models(provider: str, api_key: str = None) -> list[dict]:
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
    def get_image_models() -> list[dict]:
        image_models = []
        index = []
        for provider in __providers__:
            if hasattr(provider, "image_models"):
                if hasattr(provider, "get_models"):
                    provider.get_models()
                parent = provider
                if hasattr(provider, "parent"):
                    parent = __map__[provider.parent]
                if parent.__name__ not in index:
                    for model in provider.image_models:
                        image_models.append({
                            "provider": parent.__name__,
                            "url": parent.url,
                            "label": parent.label if hasattr(parent, "label") else None,
                            "image_model": model,
                            "vision_model": getattr(parent, "default_vision_model", None)
                        })
                    index.append(parent.__name__)
            elif hasattr(provider, "default_vision_model") and provider.__name__ not in index:
                image_models.append({
                    "provider": provider.__name__,
                    "url": provider.url,
                    "label": provider.label if hasattr(provider, "label") else None,
                    "image_model": None,
                    "vision_model": provider.default_vision_model
                })
                index.append(provider.__name__)
        return image_models

    @staticmethod
    def get_providers() -> list[str]:
        return {
            provider.__name__: (provider.label if hasattr(provider, "label") else provider.__name__)
            + (" (Image Generation)" if getattr(provider, "image_models", None) else "")
            + (" (Image Upload)" if getattr(provider, "default_vision_model", None) else "")
            + (" (WebDriver)" if "webdriver" in provider.get_parameters() else "")
            + (" (Auth)" if provider.needs_auth else "")
            for provider in __providers__
            if provider.working
        }

    @staticmethod
    def get_version():
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
        if debug.logging:
            debug.logs = []
            print_callback = debug.log_handler
            def log_handler(text: str):
                debug.logs.append(text)
                print_callback(text)
            debug.log_handler = log_handler
        try:
            result = ChatCompletion.create(**kwargs)
            first = True
            if isinstance(result, ImageResponse):
                if first:
                    first = False
                    yield self._format_json("provider", get_last_provider(True))
                yield self._format_json("content", str(result))
            else:
                for chunk in result:
                    if first:
                        first = False
                        yield self._format_json("provider", get_last_provider(True))
                    if isinstance(chunk, BaseConversation):
                        if provider:
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
                            images = asyncio.run(copy_images(chunk.get_list(), chunk.options.get("cookies")))
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

    def _format_json(self, response_type: str, content):
        return {
            'type': response_type,
            response_type: content
        }

def get_error_message(exception: Exception) -> str:
    message = f"{type(exception).__name__}: {exception}"
    provider = get_last_provider()
    if provider is None:
        return message
    return f"{provider.__name__}: {message}"