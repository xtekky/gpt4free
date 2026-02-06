from __future__ import annotations

import logging
import os
import asyncio
from typing import Iterator
from flask import send_from_directory, request
from inspect import signature

try:
    from PIL import Image 
    has_pillow = True
except ImportError:
    has_pillow = False

from ...errors import VersionNotFoundError, MissingAuthError
from ...image.copy_images import copy_media, ensure_media_dir, get_media_dir
from ...image import get_width_height
from ...tools.run_tools import iter_run_tools
from ...providers.base_provider import ProviderModelMixin
from ...providers.retry_provider import BaseRetryProvider
from ...providers.helper import format_media_prompt
from ...providers.response import *
from ...providers.any_model_map import model_map
from ...providers.any_provider import AnyProvider
from ...client.service import get_model_and_provider
from ... import Provider
from ... import version, models
from ... import debug

logger = logging.getLogger(__name__)

class Api:
    @staticmethod
    def get_models():
        return [{
            "name": model.name,
            "image": isinstance(model, models.ImageModel),
            "vision": isinstance(model, models.VisionModel),
            "audio": isinstance(model, models.AudioModel),
            "video": isinstance(model, models.VideoModel),
            "providers": [
                getattr(provider, "parent", provider.__name__)
                for provider in providers
                if provider.working
            ]
        }
        for model, providers in models.__models__.values()]

    @staticmethod
    def get_provider_models(provider: str, api_key: str = None, base_url: str = None, ignored: list = None):
        def get_model_data(provider: ProviderModelMixin, model: str, default: bool = False) -> dict:
            return {
                "model": model,
                "label": model.split(":")[-1] if provider.__name__ == "AnyProvider" and not model.startswith("openrouter:") else model,
                "default": default or model == provider.default_model,
                "vision": model in provider.vision_models,
                "audio": False if provider.audio_models is None else model in provider.audio_models,
                "video": model in provider.video_models,
                "image": model in provider.image_models,
                "count": False if provider.models_count is None else provider.models_count.get(model),
                "tags": [] if provider.models_tags is None else provider.models_tags.get(model, []),
            }
        if provider in Provider.__map__:
            provider = Provider.__map__[provider]
            if issubclass(provider, ProviderModelMixin):
                has_grouped_models = hasattr(provider, "get_grouped_models")
                method = provider.get_grouped_models if has_grouped_models else provider.get_models
                if "api_key" in signature(provider.get_models).parameters:
                    models = method(api_key=api_key, base_url=base_url)
                elif "ignored" in signature(provider.get_models).parameters:
                    models = method(ignored=ignored)
                else:
                    models = method()
                if has_grouped_models:
                    return [{
                        "group": model["group"],
                        "models": [get_model_data(provider, name) for name in model["models"]]
                    } for model in models]
                return [
                    get_model_data(provider, model)
                    for model in models
                ]
        elif provider in model_map:
            return [get_model_data(AnyProvider, provider, True)]

        return []

    @staticmethod
    def get_providers() -> dict[str, str]:
        def safe_get_models(provider: ProviderModelMixin):
            if not isinstance(provider, ProviderModelMixin):
                return True
            try:
                return provider.get_models()
            except Exception as e:
                logger.exception(e)
                return True
        return [{
            "name": provider.__name__,
            "label": getattr(provider, "label", provider.__name__),
            "parent": getattr(provider, "parent", None),
            "image": len(getattr(provider, "image_models", [])),
            "audio": len(getattr(provider, "audio_models", [])),
            "video": len(getattr(provider, "video_models", [])),
            "vision": getattr(provider, "default_vision_model", None) is not None,
            "nodriver": getattr(provider, "use_nodriver", False),
            "hf_space": getattr(provider, "hf_space", False),
            "active_by_default": False if provider.active_by_default is None else provider.active_by_default,
            "auth": provider.needs_auth,
            "login_url": getattr(provider, "login_url", None),
            "live": provider.live
        } for provider in Provider.__providers__ if provider.working and safe_get_models(provider)]

    def get_all_models(self) -> dict[str, list]:
        def safe_get_provider_models(provider: ProviderModelMixin) -> list[str]:
            try:
                return list(provider.get_models(timeout=10))
            except Exception as e:
                debug.error(f"{provider.__name__}: get_models error:", e)
                return []
        return {
            provider.__name__: safe_get_provider_models(provider)
            for provider in Provider.__providers__
            if provider.working and hasattr(provider, "get_models")
        }

    @staticmethod
    def get_version() -> dict:
        current_version = None
        latest_version = None
        try:
            current_version = version.utils.current_version
            try:
                if request.args.get("cache"):
                    latest_version = version.utils.latest_version_cached
            except RuntimeError:
                pass
            if latest_version is None:
                latest_version = version.utils.latest_version
        except VersionNotFoundError:
            pass
        return {
            "version": current_version,
            "latest_version": latest_version,
        }

    def serve_images(self, name):
        ensure_media_dir()
        return send_from_directory(os.path.abspath(get_media_dir()), name)

    def _prepare_conversation_kwargs(self, json_data: dict):
        kwargs = {**json_data}
        model = kwargs.pop('model', None)
        provider = kwargs.pop('provider', None)
        messages = kwargs.pop('messages', None)
        action = kwargs.get('action')
        if action == "continue":
            if "tool_calls" not in kwargs:
                kwargs["tool_calls"] = []
            kwargs["tool_calls"].append({
                "function": {
                    "name": "continue_tool"
                },
                "type": "function"
            })
        conversation = kwargs.pop("conversation", None)
        if isinstance(conversation, dict):
            kwargs["conversation"] = JsonConversation(**conversation)
        return {
            "model": model,
            "provider": provider,
            "messages": messages,
            "ignore_stream": True,
            **kwargs
        }

    def _create_response_stream(self, kwargs: dict, provider: str, download_media: bool = True, tempfiles: list[str] = []) -> Iterator:
        def decorated_log(*values: str, file = None):
            debug.logs.append(" ".join([str(value) for value in values]))
            if debug.logging:
                debug.log_handler(*values, file=file)
        if "user" not in kwargs:
            debug.log = decorated_log
        proxy = os.environ.get("G4F_PROXY")
        try:
            model, provider_handler = get_model_and_provider(
                kwargs.get("model"), provider or AnyProvider,
                has_images="media" in kwargs,
            )
            if "user" in kwargs:
                debug.error("User:", kwargs.get("user", "Unknown"))
                debug.error("Referrer:", kwargs.get("referer", ""))
                debug.error("User-Agent:", kwargs.get("user-agent", ""))
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
            result = iter_run_tools(provider_handler, **{**kwargs, "model": model, "download_media": download_media})
            for chunk in result:
                if isinstance(chunk, ProviderInfo):
                    model = getattr(chunk, "model", model)
                    provider = getattr(chunk, "provider", provider)
                    yield self.handle_provider(chunk, model)
                elif isinstance(chunk, JsonConversation):
                    if provider is not None:
                        yield self._format_json("conversation", chunk.get_dict() if provider == "AnyProvider" else {
                            provider: chunk.get_dict()
                        })
                elif isinstance(chunk, Exception):
                    logger.exception(chunk)
                    yield self._format_json('message', get_error_message(chunk), error=type(chunk).__name__)
                elif isinstance(chunk, RequestLogin):
                    yield self._format_json("preview", chunk.to_string())
                elif isinstance(chunk, PreviewResponse):
                    yield self._format_json("preview", chunk.to_string())
                elif isinstance(chunk, ImagePreview):
                    yield self._format_json("preview", chunk.to_string(), urls=chunk.urls, alt=chunk.alt)
                elif isinstance(chunk, MediaResponse):
                    media = chunk
                    if download_media or chunk.get("cookies") or chunk.get("headers"):
                        chunk.alt = format_media_prompt(kwargs.get("messages"), chunk.alt)
                        width, height = get_width_height(chunk.get("width"), chunk.get("height"))
                        tags = [model, kwargs.get("aspect_ratio"), kwargs.get("resolution")]
                        media = asyncio.run(copy_media(
                            chunk.get_list(),
                            chunk.get("cookies"),
                            chunk.get("headers"),
                            proxy=proxy,
                            alt=chunk.alt,
                            tags=tags,
                            add_url=True,
                            timeout=kwargs.get("timeout"),
                            return_target=True if isinstance(chunk, ImageResponse) else False,
                        ))
                        options = {}
                        target_paths, urls = get_target_paths_and_urls(media)
                        if target_paths:
                            if has_pillow:
                                try:
                                    with Image.open(target_paths[0]) as img:
                                        width, height = img.size
                                        options = {"width": width, "height": height}
                                except Exception as e:
                                    logger.exception(e)
                            options["target_paths"] = target_paths
                        media = ImageResponse(urls, chunk.alt, options) if isinstance(chunk, ImageResponse) else VideoResponse(media, chunk.alt)
                    yield self._format_json("content", str(media), urls=media.urls, alt=media.alt)
                elif isinstance(chunk, SynthesizeData):
                    yield self._format_json("synthesize", chunk.get_dict())
                elif isinstance(chunk, TitleGeneration):
                    yield self._format_json("title", chunk.title)
                elif isinstance(chunk, Parameters):
                    yield self._format_json("parameters", chunk.get_dict())
                elif isinstance(chunk, FinishReason):
                    yield self._format_json("finish", chunk.get_dict())
                elif isinstance(chunk, Usage):
                    yield self._format_json("usage", chunk.get_dict(), model=model, provider=provider)
                elif isinstance(chunk, Reasoning):
                    yield self._format_json("reasoning", **chunk.get_dict())
                elif isinstance(chunk, YouTubeResponse):
                    yield self._format_json("content", chunk.to_string())
                elif isinstance(chunk, AudioResponse):
                    yield self._format_json("content", str(chunk), data=chunk.data)
                elif isinstance(chunk, SuggestedFollowups):
                    yield self._format_json("suggestions", chunk.suggestions)
                elif isinstance(chunk, DebugResponse):
                    yield self._format_json("log", chunk.log)
                elif isinstance(chunk, ContinueResponse):
                    yield self._format_json("continue", chunk.text)
                elif isinstance(chunk, VariantResponse):
                    yield self._format_json("variant", chunk.text)
                elif isinstance(chunk, ToolCalls):
                    yield self._format_json("tool_calls", chunk.list)
                elif isinstance(chunk, RawResponse):
                    yield self._format_json(chunk.type, **chunk.get_dict())
                elif isinstance(chunk, JsonRequest):
                    yield self._format_json("request", chunk.get_dict())
                elif isinstance(chunk, JsonResponse):
                    yield self._format_json("response", chunk.get_dict())
                elif isinstance(chunk, PlainTextResponse):
                    yield self._format_json("response", chunk.text)
                elif isinstance(chunk, HeadersResponse):
                    yield self._format_json("headers", chunk.get_dict())
                else:
                    yield self._format_json("content", str(chunk))
        except MissingAuthError as e:
            yield self._format_json('auth', type(e).__name__, message=get_error_message(e))
        except (TimeoutError, asyncio.exceptions.CancelledError) as e:
            if "user" in kwargs:
                debug.error(e, "User:", kwargs.get("user", "Unknown"))
            yield self._format_json('error', type(e).__name__, message=get_error_message(e))
        except Exception as e:
            if "user" in kwargs:
                debug.error(e, "User:", kwargs.get("user", "Unknown"))
            logger.exception(e)
            yield self._format_json('error', type(e).__name__, message=get_error_message(e))
        finally:
            yield from self._yield_logs()
            for tempfile in tempfiles:
                try:
                    os.remove(tempfile)
                except Exception as e:
                    logger.exception(e)

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
        if not getattr(provider_handler, "model", False):
            return self._format_json("provider", {**provider_handler.get_dict(), "model": model})
        return self._format_json("provider", provider_handler.get_dict())

def get_error_message(exception: Exception) -> str:
    return f"{type(exception).__name__}: {exception}"

def get_target_paths_and_urls(media: list[Union[str, tuple[str, str]]]) -> tuple[list[str], list[str]]:
    target_paths = []
    urls = []
    for item in media:
        if isinstance(item, tuple):
            item, target_path = item
            target_paths.append(target_path)
        urls.append(item)
    return target_paths, urls
