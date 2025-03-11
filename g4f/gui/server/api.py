from __future__ import annotations

import logging
import os
import asyncio
from typing import Iterator
from flask import send_from_directory
from inspect import signature

from ...errors import VersionNotFoundError
from ...image.copy_images import copy_images, ensure_images_dir, images_dir
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
        current_version = None
        latest_version = None
        try:
            current_version = version.utils.current_version
            latest_version = version.utils.latest_version
        except VersionNotFoundError:
            pass
        return {
            "version": current_version,
            "latest_version": latest_version,
        }

    def serve_images(self, name):
        ensure_images_dir()
        return send_from_directory(os.path.abspath(images_dir), name)

    def _prepare_conversation_kwargs(self, json_data: dict):
        kwargs = {**json_data}
        model = json_data.get('model')
        provider = json_data.get('provider')
        messages = json_data.get('messages')
        kwargs["tool_calls"] = [{
            "function": {
                "name": "bucket_tool"
            },
            "type": "function"
        }]
        action = json_data.get('action')
        if action == "continue":
            kwargs["tool_calls"].append({
                "function": {
                    "name": "continue_tool"
                },
                "type": "function"
            })
        conversation = json_data.get("conversation")
        if isinstance(conversation, dict):
            kwargs["conversation"] = JsonConversation(**conversation)
        else:
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
        def decorated_log(text: str, file = None):
            debug.logs.append(text)
            if debug.logging:
                debug.log_handler(text, file=file)
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
            debug.error(e)
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
                        if hasattr(provider, "__name__"):
                            provider = provider.__name__
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
                    debug.error(chunk)
                    yield self._format_json('message', get_error_message(chunk), error=type(chunk).__name__)
                elif isinstance(chunk, RequestLogin):
                    yield self._format_json("preview", chunk.to_string())
                elif isinstance(chunk, PreviewResponse):
                    yield self._format_json("preview", chunk.to_string())
                elif isinstance(chunk, ImagePreview):
                    yield self._format_json("preview", chunk.to_string(), images=chunk.images, alt=chunk.alt)
                elif isinstance(chunk, ImageResponse):
                    images = chunk
                    if download_images or chunk.get("cookies"):
                        chunk.alt = format_image_prompt(kwargs.get("messages"), chunk.alt)
                        images = asyncio.run(copy_images(chunk.get_list(), chunk.get("cookies"), chunk.get("headers"), proxy=proxy, alt=chunk.alt))
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
                    yield self._format_json("reasoning", **chunk.get_dict())
                elif isinstance(chunk, YouTube):
                    yield self._format_json("content", chunk.to_string())
                elif isinstance(chunk, Audio):
                    yield self._format_json("audio", str(chunk))
                elif isinstance(chunk, DebugResponse):
                    yield self._format_json("log", chunk.log)
                elif isinstance(chunk, RawResponse):
                    yield self._format_json(chunk.type, **chunk.get_dict())
                else:
                    yield self._format_json("content", str(chunk))
                yield from self._yield_logs()
        except Exception as e:
            logger.exception(e)
            debug.error(e)
            yield from self._yield_logs()
            yield self._format_json('error', type(e).__name__, message=get_error_message(e))

    def _yield_logs(self):
        if debug.logs:
            for log in debug.logs:
                yield self._format_json("log", log)
            debug.logs = []

    def _format_json(self, response_type: str, content = None, **kwargs):
        if content is not None and isinstance(response_type, str):
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