from __future__ import annotations

import os
import json

from ...typing import Messages, AsyncResult, MediaListType
from ...errors import MissingAuthError, ModelNotFoundError
from ...requests import StreamSession, FormData, raise_for_status
from ...image import get_width_height, to_bytes
from ...image.copy_images import save_response_media
from ..template import OpenaiTemplate
from ..helper import format_media_prompt

class Azure(OpenaiTemplate):
    url = "https://ai.azure.com"
    api_base = "https://host.g4f.dev/api/Azure"
    working = True
    needs_auth = True
    models_needs_auth = True
    active_by_default = True
    login_url = "https://discord.gg/qXA4Wf4Fsm"
    routes: dict[str, str] = {}
    audio_models = ["gpt-4o-mini-audio-preview"]
    vision_models = ["gpt-4.1", "o4-mini", "model-router", "flux.1-kontext-pro"]
    image_models = ["flux-1.1-pro", "flux.1-kontext-pro"]
    model_aliases = {
        "flux-kontext": "flux.1-kontext-pro"
    }
    model_extra_body = {
        "gpt-4o-mini-audio-preview": {
            "audio": {
                "voice": "alloy",
                "format": "mp3"
            },
            "modalities": ["text", "audio"],
        }
    }
    api_keys: dict[str, str] = {}

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs) -> list[str]:
        api_keys = os.environ.get("AZURE_API_KEYS")
        if api_keys:
            try:
                cls.api_keys = json.loads(api_keys)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid AZURE_API_KEYS environment variable")
        routes = os.environ.get("AZURE_ROUTES")
        if routes:
            try:
                routes = json.loads(routes)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid AZURE_ROUTES environment variable format: {routes}")
            cls.routes = routes
        if cls.routes:
            return list(cls.routes.keys())
        return super().get_models(api_key=api_key, **kwargs)

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        media: MediaListType = None,
        extra_body: dict = None,
        api_key: str = None,
        api_endpoint: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = os.environ.get("AZURE_DEFAULT_MODEL", cls.default_model)
        if model in cls.model_aliases:
            model = cls.model_aliases[model]
        if not api_endpoint:
            if not cls.routes:
                cls.get_models()
            api_endpoint = cls.routes.get(model)
            if cls.routes and not api_endpoint:
                raise ModelNotFoundError(f"No API endpoint found for model: {model}")
        if not api_endpoint:
            api_endpoint = os.environ.get("AZURE_API_ENDPOINT")
        if cls.api_keys:
            api_key = cls.api_keys.get(model, cls.api_keys.get("default"))
            if not api_key:
                raise ValueError(f"API key is required for Azure provider. Ask for API key in the {cls.login_url} Discord server.")
        if api_endpoint and "/images/" in api_endpoint:
            prompt = format_media_prompt(messages, kwargs.get("prompt"))
            width, height = get_width_height(kwargs.get("aspect_ratio", "1:1"), kwargs.get("width"), kwargs.get("height"))
            output_format = kwargs.get("output_format", "webp")
            form = None
            data = None
            if media:
                form = FormData()
                form.add_field("prompt", prompt)
                form.add_field("width", str(width))
                form.add_field("height", str(height))
                output_format = "png"
                for i in range(len(media)):
                    if media[i][1] is None and isinstance(media[i][0], str):
                        media[i] = media[i][0], os.path.basename(media[i][0])
                    media[i] = (to_bytes(media[i][0]), media[i][1])
                for image, image_name in media:
                    form.add_field(f"image", image, filename=image_name)
            else:
                api_endpoint = api_endpoint.replace("/edits", "/generations")
                data = {
                    "prompt": prompt,
                    "n": 1,
                    "width": width,
                    "height": height,
                    "output_format": output_format,
                }
            async with StreamSession(proxy=kwargs.get("proxy"), headers={
                "Authorization": f"Bearer {api_key}",
                "x-ms-model-mesh-model-name": model,
            }) as session:
                async with session.post(api_endpoint, data=form, json=data) as response:
                    data = await response.json()
                    await raise_for_status(response, data)
                    async for chunk in save_response_media(data["data"][0]["b64_json"], prompt, content_type=f"image/{output_format}"):
                        yield chunk
            return
        if extra_body is None:
            if model in cls.model_extra_body:
                extra_body = cls.model_extra_body[model]
                stream = False
            else:
                extra_body = {}
        if stream:
            extra_body.setdefault("stream_options", {"include_usage": True})
        try:
            async for chunk in super().create_async_generator(
                model=model,
                messages=messages,
                stream=stream,
                media=media,
                api_key=api_key,
                api_endpoint=api_endpoint,
                extra_body=extra_body,
                **kwargs
            ):
                yield chunk
        except MissingAuthError as e:
            raise MissingAuthError(f"{e}. Ask for help in the {cls.login_url} Discord server.") from e