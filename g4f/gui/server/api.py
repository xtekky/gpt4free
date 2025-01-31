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
from ...providers.retry_provider import BaseRetryProvider
from ...providers.helper import format_image_prompt
from ...providers.response import *
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
            "vision": isinstance(model, models.VisionModel),
            "providers": [
                getattr(provider, "parent", provider.__name__)
                for provider in providers
                if provider.working
            ]
        }
        for model, providers in models.__models__.values()]

    @staticmethod
    def get_provider_models(provider: str, api_key: str = None, api_base: str = None):
        if provider in ProviderUtils.convert:
            provider = ProviderUtils.convert[provider]
            if issubclass(provider, ProviderModelMixin):
                if "api_key" in signature(provider.get_models).parameters:
                    models = provider.get_models(api_key=api_key, api_base=api_base)
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
            "image": bool(getattr(provider, "image_models", False)),
            "vision": getattr(provider, "default_vision_model", None) is not None,
            "nodriver": getattr(provider, "use_nodriver", False),
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
        model = json_data.get('model')
        provider = json_data.get('provider')
        messages = json_data.get('messages')
        api_key = json_data.get("api_key")
        if api_key:
            kwargs["api_key"] = api_key
        api_base = json_data.get("api_base")
        if api_base:
            kwargs["api_base"] = api_base
        kwargs["tool_calls"] = [{
            "function": {
                "name": "bucket_tool"
            },
            "type": "function"
        }]
        web_search = json_data.get('web_search')
        if web_search:
            kwargs["web_search"] = web_search
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
        try:
            model, provider_handler = get_model_and_provider(
                kwargs.get("model"), provider,
                stream=True,
                ignore_stream=True,
                logging=False,
                has_images="images" in kwargs,
            )
        except Exception as e:
            logger.exception(e)
            yield self._format_json('error', type(e).__name__, message=get_error_message(e))
            return
        if not isinstance(provider_handler, BaseRetryProvider):
            if not provider:
                provider = provider_handler.__name__
            yield self.handle_provider(provider_handler, model)
            if hasattr(provider_handler, "get_parameters"):
                yield self._format_json("parameters", provider_handler.get_parameters(as_json=True))
        try:
            result = iter_run_tools(ChatCompletion.create, **{**kwargs, "model": model, "provider": provider_handler})
            for chunk in result:
                if isinstance(chunk, ProviderInfo):
                    yield self.handle_provider(chunk, model)
                    provider = chunk.name
                elif isinstance(chunk, BaseConversation):
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
                    yield self._format_json('message', get_error_message(chunk), error=type(chunk).__name__)
                elif isinstance(chunk, (PreviewResponse, ImagePreview)):
                    yield self._format_json("preview", chunk.to_string(), images=chunk.images, alt=chunk.alt)
                elif isinstance(chunk, ImageResponse):
                    images = chunk
                    if download_images or chunk.get("cookies"):
                        alt = format_image_prompt(kwargs.get("messages"))
                        images = asyncio.run(copy_images(chunk.get_list(), chunk.get("cookies"), proxy, alt))
                        images = ImageResponse(images, chunk.alt)
                    yield self._format_json("content", str(images), images=chunk.get_list(), alt=chunk.alt)
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
                elif isinstance(chunk, Reasoning):
                    yield self._format_json("reasoning", token=chunk.token, status=chunk.status, is_thinking=chunk.is_thinking)
                elif isinstance(chunk, DebugResponse):
                    yield self._format_json("log", chunk.get_dict())
                elif isinstance(chunk, Notification):
                    yield self._format_json("notification", chunk.message)
                else:
                    yield self._format_json("content", str(chunk))
                if debug.logs:
                    for log in debug.logs:
                        yield self._format_json("log", str(log))
                    debug.logs = []
        except Exception as e:
            logger.exception(e)
            if debug.logs:
                for log in debug.logs:
                    yield self._format_json("log", str(log))
                debug.logs = []
            yield self._format_json('error', type(e).__name__, message=get_error_message(e))

    def _format_json(self, response_type: str, content = None, **kwargs):
        # Make sure it get be formated as JSON
        if content is not None and not isinstance(content, (str, dict)):
            content = str(content)
        kwargs = {
            key: value
            if value is isinstance(value, (str, dict))
            else str(value)
            for key, value in kwargs.items()
            if isinstance(key, str)}
        if content is not None:
            return {
                'type': response_type,
                response_type: content,
                **kwargs
            }
        return {
            'type': response_type,
            **kwargs
        }

    def handle_provider(self, provider_handler, model):
        if isinstance(provider_handler, BaseRetryProvider) and provider_handler.last_provider is not None:
            provider_handler = provider_handler.last_provider
        if model:
            return self._format_json("provider", {**provider_handler.get_dict(), "model": model})
        return self._format_json("provider", provider_handler.get_dict())

def get_error_message(exception: Exception) -> str:
    return f"{type(exception).__name__}: {exception}"