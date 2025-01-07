from __future__ import annotations

import logging
import os
import asyncio
from typing import Iterator
from flask import send_from_directory
from inspect import signature

from ...errors import VersionNotFoundError
from ...image import ImagePreview, ImageResponse, copy_images, ensure_images_dir, images_dir
from ...tools.run_tools import iter_run_tools
from ...Provider import ProviderUtils, __providers__
from ...providers.base_provider import ProviderModelMixin
from ...providers.retry_provider import IterListProvider
from ...providers.response import BaseConversation, JsonConversation, FinishReason, Usage
from ...providers.response import SynthesizeData, TitleGeneration, RequestLogin, Parameters
from ... import version, models
from ... import ChatCompletion, get_model_and_provider
from ... import debug

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
        if provider in ProviderUtils.convert:
            provider = ProviderUtils.convert[provider]
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
            "auth": provider.needs_auth,
            "login_url": getattr(provider, "login_url", None),
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
        messages = json_data.get('messages')
        api_key = json_data.get("api_key")
        if api_key is not None:
            kwargs["api_key"] = api_key
        kwargs["tool_calls"] = [{
            "function": {
                "name": "bucket_tool"
            },
            "type": "function"
        }]
        do_web_search = json_data.get('web_search')
        if do_web_search and provider:
            kwargs["tool_calls"].append({
                "function": {
                    "name": "safe_search_tool"
                },
                "type": "function"
            })
        action = json_data.get('action')
        if action == "continue":
            kwargs["tool_calls"].append({
                "function": {
                    "name": "continue_tool"
                },
                "type": "function"
            })
        conversation = json_data.get("conversation")
        if conversation is not None:
            kwargs["conversation"] = JsonConversation(**conversation)
        else:
            conversation_id = json_data.get("conversation_id")
            if conversation_id and provider:
                if provider in conversations and conversation_id in conversations[provider]:
                    kwargs["conversation"] = conversations[provider][conversation_id]

        if json_data.get("ignored"):
            kwargs["ignored"] = json_data["ignored"]
        if json_data.get("action"):
            kwargs["action"] = json_data["action"]

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
        params = {
            **(provider_handler.get_parameters(as_json=True) if hasattr(provider_handler, "get_parameters") else {}),
            "model": model,
            "messages": kwargs.get("messages"),
            "web_search": kwargs.get("web_search")
        }
        if isinstance(kwargs.get("conversation"), JsonConversation):
            params["conversation"] = kwargs.get("conversation").get_dict()
        else:
            params["conversation_id"] = conversation_id
        if kwargs.get("api_key") is not None:
            params["api_key"] = kwargs["api_key"]
        yield self._format_json("parameters", params)
        first = True
        try:
            result = iter_run_tools(ChatCompletion.create, **{**kwargs, "model": model, "provider": provider_handler})
            for chunk in result:
                if first:
                    first = False
                    yield self.handle_provider(provider_handler, model)
                if isinstance(chunk, BaseConversation):
                    if provider is not None:
                        if provider not in conversations:
                            conversations[provider] = {}
                        conversations[provider][conversation_id] = chunk
                        if isinstance(chunk, JsonConversation):
                            yield self._format_json("conversation", {
                                provider: chunk.get_dict()
                            })
                        else:
                            yield self._format_json("conversation_id", conversation_id)
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
                    yield self._format_json("synthesize", chunk.get_dict())
                elif isinstance(chunk, TitleGeneration):
                    yield self._format_json("title", chunk.title)
                elif isinstance(chunk, RequestLogin):
                    yield self._format_json("login", str(chunk))
                elif isinstance(chunk, Parameters):
                    yield self._format_json("parameters", chunk.get_dict())
                elif isinstance(chunk, FinishReason):
                    yield self._format_json("finish", chunk.get_dict())
                elif isinstance(chunk, Usage):
                    yield self._format_json("usage", chunk.get_dict())
                else:
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