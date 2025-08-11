from __future__ import annotations

import requests

from ..helper import filter_none, format_media_prompt
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin
from ...typing import Union, AsyncResult, Messages, MediaListType
from ...requests import StreamSession, StreamResponse, raise_for_status, sse_stream
from ...image import use_aspect_ratio
from ...image.copy_images import save_response_media
from ...providers.response import FinishReason, ToolCalls, Usage, ImageResponse, ProviderInfo, AudioResponse, Reasoning
from ...tools.media import render_messages
from ...tools.run_tools import AuthManager
from ...errors import MissingAuthError
from ... import debug

class OpenaiTemplate(AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin):
    api_base = ""
    api_key = None
    api_endpoint = None
    supports_message_history = True
    supports_system_message = True
    default_model = ""
    fallback_models = []
    sort_models = True
    models_needs_auth = False
    use_model_names = False
    ssl = None
    add_user = True
    use_image_size = False

    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = None) -> list[str]:
        if not cls.models:
            try:
                headers = {}
                if api_base is None:
                    api_base = cls.api_base
                if api_key is None and cls.api_key is not None:
                    api_key = cls.api_key
                if not api_key:
                    api_key = AuthManager.load_api_key(cls)
                if cls.models_needs_auth and not api_key:
                    raise MissingAuthError('Add a "api_key"')
                if api_key is not None:
                    headers["authorization"] = f"Bearer {api_key}"
                response = requests.get(f"{api_base}/models", headers=headers, verify=cls.ssl)
                raise_for_status(response)
                data = response.json()
                data = data.get("data") if isinstance(data, dict) else data
                cls.image_models = [model.get("name") if cls.use_model_names else model.get("id", model.get("name")) for model in data if model.get("image") or model.get("type") == "image" or model.get("supports_images")]
                cls.vision_models = cls.vision_models.copy()
                cls.vision_models += [model.get("name") if cls.use_model_names else model.get("id", model.get("name")) for model in data if model.get("vision")]
                cls.models = [model.get("name") if cls.use_model_names else model.get("id", model.get("name")) for model in data]
                cls.models_count = {model.get("name") if cls.use_model_names else model.get("id", model.get("name")): len(model.get("providers", [])) for model in data if len(model.get("providers", [])) > 1}
                if cls.sort_models:
                    cls.models.sort()
            except Exception as e:
                debug.error(e)
                return cls.fallback_models
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        media: MediaListType = None,
        api_key: str = None,
        api_endpoint: str = None,
        api_base: str = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        stop: Union[str, list[str]] = None,
        stream: bool = None,
        prompt: str = None,
        user: str = None,
        headers: dict = None,
        impersonate: str = None,
        download_media: bool = True,
        extra_parameters: list[str] = ["tools", "parallel_tool_calls", "tool_choice", "reasoning_effort", "logit_bias", "modalities", "audio", "stream_options"],
        extra_body: dict = None,
        **kwargs
    ) -> AsyncResult:
        if api_key is None and cls.api_key is not None:
            api_key = cls.api_key
        if cls.needs_auth and api_key is None:
            raise MissingAuthError('Add a "api_key"')
        async with StreamSession(
            proxy=proxy,
            headers=cls.get_headers(stream, api_key, headers),
            timeout=timeout,
            impersonate=impersonate,
        ) as session:
            model = cls.get_model(model, api_key=api_key, api_base=api_base)
            if api_base is None:
                api_base = cls.api_base

            # Proxy for image generation feature
            if model and model in cls.image_models:
                prompt = format_media_prompt(messages, prompt)
                size = use_aspect_ratio({"width": kwargs.get("width"), "height": kwargs.get("height")}, kwargs.get("aspect_ratio", None))
                size = {"size": f"{size['width']}x{size['height']}", **size} if cls.use_image_size and "width" in size and "height" in size else size
                data = {"prompt": prompt, "model": model, **size}

                # Handle media if provided
                if media is not None:
                    data["image_url"] = next(iter([data for data, _ in media if data and isinstance(data, str) and data.startswith("http://") or data.startswith("https://")]), None)
                async with session.post(f"{api_base.rstrip('/')}/images/generations", json=data, ssl=cls.ssl) as response:
                    data = await response.json()
                    cls.raise_error(data, response.status)
                    model = data.get("model")
                    if model:
                        yield ProviderInfo(**cls.get_dict(), model=model)
                    await raise_for_status(response)
                    yield ImageResponse([image["url"] for image in data["data"]], prompt)
                return

            extra_parameters = {key: kwargs[key] for key in extra_parameters if key in kwargs}
            if extra_body is None:
                extra_body = {}
            data = filter_none(
                messages=list(render_messages(messages, media)),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                stream="audio" not in extra_parameters if stream is None else stream,
                user=user if cls.add_user else None,
                **extra_parameters,
                **extra_body
            )
            if api_endpoint is None:
                if api_base:
                    api_endpoint = f"{api_base.rstrip('/')}/chat/completions"
                if api_endpoint is None:
                    api_endpoint = cls.api_endpoint
            async with session.post(api_endpoint, json=data, ssl=cls.ssl) as response:
                async for chunk in read_response(response, stream, prompt, cls.get_dict(), download_media):
                    yield chunk

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {api_key}"}
                if api_key else {}
            ),
            **({} if headers is None else headers)
        }
    
async def read_response(response: StreamResponse, stream: bool, prompt: str, provider_info: dict, download_media: bool):
    content_type = response.headers.get("content-type", "text/event-stream" if stream else "application/json")
    if content_type.startswith("text/plain"):
        yield await response.text()
    elif content_type.startswith("application/json"):
        data = await response.json()
        OpenaiTemplate.raise_error(data, response.status)
        await raise_for_status(response)
        model = data.get("model")
        if model:
            yield ProviderInfo(**provider_info, model=model)
        if "usage" in data:
            yield Usage(**data["usage"])
        if "choices" in data:
            choice = next(iter(data["choices"]), None)
            message = choice.get("message", {})
            if choice and "content" in message and message["content"]:
                yield message["content"].strip()
            if "tool_calls" in message:
                yield ToolCalls(message["tool_calls"])
            if choice:
                reasoning_content = choice.get("delta", {}).get("reasoning_content", choice.get("delta", {}).get("reasoning"))
                if reasoning_content:
                    yield Reasoning(reasoning_content, status="")
            audio = message.get("audio", {})
            if "data" in audio:
                if download_media:
                    async for chunk in save_response_media(audio, prompt, [model]):
                        yield chunk
                else:
                    yield AudioResponse(f"data:audio/mpeg;base64,{audio['data']}", transcript=audio.get("transcript"))
            if choice and "finish_reason" in choice and choice["finish_reason"] is not None:
                yield FinishReason(choice["finish_reason"])
                return
    elif content_type.startswith("text/event-stream"):
        await raise_for_status(response)
        reasoning = False
        first = True
        model_returned = False
        async for data in sse_stream(response):
            OpenaiTemplate.raise_error(data)
            model = data.get("model")
            if not model_returned and model:
                yield ProviderInfo(**provider_info, model=model)
                model_returned = True
            choice = next(iter(data["choices"]), None)
            if choice:
                content = choice.get("delta", {}).get("content")
                if content:
                    if first:
                        content = content.lstrip()
                    if content:
                        first = False
                        if reasoning:
                            yield Reasoning(status="")
                            reasoning = False
                        yield content
                tool_calls = choice.get("delta", {}).get("tool_calls")
                if tool_calls:
                    yield ToolCalls(tool_calls)
                reasoning_content = choice.get("delta", {}).get("reasoning_content", choice.get("delta", {}).get("reasoning"))
                if reasoning_content:
                    reasoning = True
                    yield Reasoning(reasoning_content)
            if "usage" in data and data["usage"]:
                yield Usage(**data["usage"])
            if choice and choice.get("finish_reason") is not None:
                yield FinishReason(choice["finish_reason"])
    else:
        await raise_for_status(response)
        async for chunk in save_response_media(response, prompt, [model]):
            yield chunk