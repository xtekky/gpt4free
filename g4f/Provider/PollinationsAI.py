from __future__ import annotations

import time
import json
import random
import requests
import asyncio
from urllib.parse import quote_plus
from typing import Optional
from aiohttp import ClientSession

from .helper import filter_none, format_image_prompt
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages, MediaListType
from ..image import is_data_an_audio
from ..errors import ModelNotFoundError, ResponseError
from ..requests import see_stream
from ..requests.raise_for_status import raise_for_status
from ..requests.aiohttp import get_connector
from ..image.copy_images import save_response_media
from ..image import use_aspect_ratio
from ..providers.response import FinishReason, Usage, ToolCalls, ImageResponse, Reasoning, TitleGeneration, SuggestedFollowups
from ..tools.media import render_messages
from ..constants import STATIC_URL
from .. import debug

DEFAULT_HEADERS = {
    "accept": "*/*",
    'accept-language': 'en-US,en;q=0.9',
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "referer": "https://pollinations.ai/",
    "origin": "https://pollinations.ai",
}

FOLLOWUPS_TOOLS = [{
    "type": "function",
    "function": {
        "name": "options",
        "description": "Provides options for the conversation",
        "parameters": {
            "properties": {
                "title": {
                    "title": "Conversation Title",
                    "type": "string"
                },
                "followups": {
                    "items": {
                        "type": "string"
                    },
                    "title": "Suggested Followups",
                    "type": "array"
                }
            },
            "title": "Conversation",
            "type": "object"
        }
    }
}]

FOLLOWUPS_DEVELOPER_MESSAGE = [{
    "role": "developer",
    "content": "Prefix conversation title with one or more emojies. Suggested 4 Followups"
}]

class PollinationsAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Pollinations AI"
    url = "https://pollinations.ai"
    login_url = "https://auth.pollinations.ai"

    working = True
    supports_system_message = True
    supports_message_history = True

    # API endpoints
    text_api_endpoint = "https://text.pollinations.ai"
    openai_endpoint = "https://text.pollinations.ai/openai"
    image_api_endpoint = "https://image.pollinations.ai/"

    # Models configuration
    default_model = "openai"
    default_image_model = "flux"
    default_vision_model = default_model
    default_audio_model = "openai-audio"
    text_models = [default_model, "evil"]
    image_models = [default_image_model]
    audio_models = {default_audio_model: []}
    extra_image_models = ["flux-pro", "flux-dev", "flux-schnell", "midjourney", "dall-e-3", "turbo"]
    vision_models = [default_vision_model, "gpt-4o-mini", "openai", "openai-large", "openai-reasoning", "searchgpt"]
    _models_loaded = False
    # https://github.com/pollinations/pollinations/blob/master/text.pollinations.ai/generateTextPortkey.js#L15
    model_aliases = {
        ### Text Models ###
        "gpt-4o-mini": "openai",
        "gpt-4.1-nano": "openai-fast",
        "gpt-4": "openai-large",
        "gpt-4o": "openai-large",
        "gpt-4.1": "openai-large",
        "gpt-4o-audio": "openai-audio",
        "o4-mini": "openai-reasoning",
        "gpt-4.1-mini": "openai",
        "command-r-plus": "command-r",
        "gemini-2.5-flash": "gemini",
        "gemini-2.0-flash-thinking": "gemini-thinking",
        "qwen-2.5-coder-32b": "qwen-coder",
        "llama-3.3-70b": "llama",
        "llama-4-scout": "llamascout",
        "llama-4-scout-17b": "llamascout",
        "mistral-small-3.1-24b": "mistral",
        "deepseek-r1": "deepseek-reasoning-large",
        "deepseek-r1-distill-llama-70b": "deepseek-reasoning-large",
        #"deepseek-r1-distill-llama-70b": "deepseek-r1-llama",
        #"mistral-small-3.1-24b": "unity", # Personas
        #"mirexa": "mirexa", # Personas
        #"midijourney": "midijourney", # Personas
        #"rtist": "rtist", # Personas
        #"searchgpt": "searchgpt",
        #"evil": "evil", # Personas
        "deepseek-r1-distill-qwen-32b": "deepseek-reasoning",
        "phi-4": "phi",
        #"pixtral-12b": "pixtral",
        #"hormoz-8b": "hormoz",
        "qwq-32b": "qwen-qwq",
        #"hypnosis-tracy-7b": "hypnosis-tracy", # Personas
        #"mistral-?": "sur", # Personas
        "deepseek-v3": "deepseek",
        "deepseek-v3-0324": "deepseek",
        #"bidara": "bidara", # Personas
        "grok-3-mini": "grok",

        ### Audio Models ###
        "gpt-4o-mini-audio": "openai-audio",

        ### Image Models ###
        "sdxl-turbo": "turbo",
        "gpt-image": "gptimage",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the internal model name from the user-provided model name."""
        if not model:
            return cls.default_model
        
        # Check if the model exists directly in our model lists
        if model in cls.text_models or model in cls.image_models or model in cls.audio_models:
            return model
        
        # Check if there's an alias for this model
        if model in cls.model_aliases:
            return cls.model_aliases[model]
        
        # If no match is found, raise an error
        raise ModelNotFoundError(f"Model {model} not found")

    @classmethod
    def get_models(cls, **kwargs):
        if not cls._models_loaded:
            try:
                # Update of image models
                image_response = requests.get("https://image.pollinations.ai/models")
                if image_response.ok:
                    new_image_models = image_response.json()
                else:
                    new_image_models = []

                # Combine image models without duplicates
                all_image_models = [cls.default_image_model]  # Start with default model
                
                # Add extra image models if not already in the list
                for model in cls.extra_image_models + new_image_models:
                    if model not in all_image_models:
                        all_image_models.append(model)
                
                cls.image_models = all_image_models

                text_response = requests.get("https://text.pollinations.ai/models")
                text_response.raise_for_status()
                models = text_response.json()

                # Purpose of audio models
                cls.audio_models = {
                    model.get("name"): model.get("voices")
                    for model in models
                    if "output_modalities" in model and "audio" in model["output_modalities"] and model.get("name") != "gemini"
                }

                cls.vision_models.extend([
                    model.get("name")
                    for model in models
                    if model.get("vision") and model not in cls.vision_models
                ])
                for alias, model in cls.model_aliases.items():
                    if model in cls.vision_models and alias not in cls.vision_models:
                        cls.vision_models.append(alias)

                # Create a set of unique text models starting with default model
                unique_text_models = cls.text_models.copy()

                # Add models from vision_models
                unique_text_models.extend(cls.vision_models)

                # Add models from the API response
                for model in models:
                    model_name = model.get("name")
                    if model_name and "input_modalities" in model and "text" in model["input_modalities"]:
                        unique_text_models.append(model_name)

                # Convert to list and update text_models
                cls.text_models = list(dict.fromkeys(unique_text_models))

                cls._models_loaded = True

            except Exception as e:
                # Save default models in case of an error
                if not cls.text_models:
                    cls.text_models = [cls.default_model]
                if not cls.image_models:
                    cls.image_models = [cls.default_image_model]
                debug.error(f"Failed to fetch models: {e}")

        # Return unique models across all categories
        all_models = cls.text_models.copy()
        all_models.extend(cls.image_models)
        all_models.extend(cls.audio_models.keys())
        if cls.default_audio_model in cls.audio_models:
            all_models.extend(cls.audio_models[cls.default_audio_model])
        return list(dict.fromkeys(all_models))

    @classmethod
    def get_grouped_models(cls) -> dict[str, list[str]]:
        cls.get_models()
        return [
            {"group": "Text Generation", "models": cls.text_models},
            {"group": "Image Generation", "models": cls.image_models},
            {"group": "Audio Generation", "models": list(cls.audio_models.keys())},
            {"group": "Audio Voices", "models": cls.audio_models[cls.default_audio_model]}
        ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        cache: bool = False,
        referrer: str = STATIC_URL,
        api_key: str = None,
        extra_body: dict = {},
        # Image generation parameters
        prompt: str = None,
        aspect_ratio: str = "1:1",
        width: int = None,
        height: int = None,
        seed: Optional[int] = None,
        nologo: bool = True,
        private: bool = False,
        enhance: bool = False,
        safe: bool = False,
        n: int = 1,
        # Text generation parameters
        media: MediaListType = None,
        temperature: float = None,
        presence_penalty: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        response_format: Optional[dict] = None,
        extra_parameters: list[str] = ["tools", "parallel_tool_calls", "tool_choice", "reasoning_effort", "logit_bias", "voice", "modalities", "audio"],
        **kwargs
    ) -> AsyncResult:
        # Load model list
        cls.get_models()
        if not model:
            has_audio = "audio" in kwargs or "audio" in kwargs.get("modalities", [])
            if not has_audio and media is not None:
                for media_data, filename in media:
                    if is_data_an_audio(media_data, filename):
                        has_audio = True
                        break
            model = cls.default_audio_model if has_audio else model
        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            pass

        if model in cls.image_models:
            async for chunk in cls._generate_image(
                model=model,
                prompt=format_image_prompt(messages, prompt),
                proxy=proxy,
                aspect_ratio=aspect_ratio,
                width=width,
                height=height,
                seed=seed,
                cache=cache,
                nologo=nologo,
                private=private,
                enhance=enhance,
                safe=safe,
                n=n,
                referrer=referrer,
                api_key=api_key
            ):
                yield chunk
        else:
            if prompt is not None and len(messages) == 1:
                messages = [{
                    "role": "user",
                    "content": prompt
                }]
            if model and model in cls.audio_models[cls.default_audio_model]:
                kwargs["audio"] = {
                    "voice": model,
                }
                model = cls.default_audio_model
            async for result in cls._generate_text(
                model=model,
                messages=messages,
                media=media,
                proxy=proxy,
                temperature=temperature,
                presence_penalty=presence_penalty,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                response_format=response_format,
                seed=seed,
                cache=cache,
                stream=stream,
                extra_parameters=extra_parameters,
                referrer=referrer,
                api_key=api_key,
                extra_body=extra_body,
                **kwargs
            ):
                yield result

    @classmethod
    async def _generate_image(
        cls,
        model: str,
        prompt: str,
        proxy: str,
        aspect_ratio: str,
        width: int,
        height: int,
        seed: Optional[int],
        cache: bool,
        nologo: bool,
        private: bool,
        enhance: bool,
        safe: bool,
        n: int,
        referrer: str,
        api_key: str
    ) -> AsyncResult:
        params = use_aspect_ratio({
            "width": width,
            "height": height,
            "model": model,
            "nologo": str(nologo).lower(),
            "private": str(private).lower(),
            "enhance": str(enhance).lower(),
            "safe": str(safe).lower(),
        }, aspect_ratio)
        query = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items() if v is not None)
        prompt = quote_plus(prompt)[:2048-len(cls.image_api_endpoint)-len(query)-8]
        url = f"{cls.image_api_endpoint}prompt/{prompt}?{query}"
        def get_image_url(i: int, seed: Optional[int] = None):
            if i == 0:
                if not cache and seed is None:
                    seed = random.randint(0, 2**32)
            else:
                seed = random.randint(0, 2**32)
            return f"{url}&seed={seed}" if seed else url
        headers = {"referer": referrer}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            responses = set()
            responses.add(Reasoning(status=f"Generating {n} {'image' if n == 1 else 'images'}..."))
            finished = 0
            start = time.time()
            async def get_image(responses: set, i: int, seed: Optional[int] = None):
                nonlocal finished
                async with session.get(get_image_url(i, seed), allow_redirects=False, headers=headers) as response:
                    try:
                        await raise_for_status(response)
                    except Exception as e:
                        debug.error(f"Error fetching image: {e}")
                    responses.add(ImageResponse(str(response.url), prompt))
                    finished += 1
                    responses.add(Reasoning(status=f"Image {finished}/{n} generated in {time.time() - start:.2f}s"))
            tasks = []
            for i in range(int(n)):
                tasks.append(asyncio.create_task(get_image(responses, i, seed)))
            while finished < n or len(responses) > 0:
                while len(responses) > 0:
                    yield responses.pop()
                await asyncio.sleep(0.1)
            await asyncio.gather(*tasks)

    @classmethod
    async def _generate_text(
        cls,
        model: str,
        messages: Messages,
        media: MediaListType,
        proxy: str,
        temperature: float,
        presence_penalty: float,
        top_p: float,
        frequency_penalty: float,
        response_format: Optional[dict],
        seed: Optional[int],
        cache: bool,
        stream: bool,
        extra_parameters: list[str],
        referrer: str,
        api_key: str,
        extra_body: dict,
        **kwargs
    ) -> AsyncResult:
        if not cache and seed is None:
            seed = random.randint(0, 2**32)

        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            if model in cls.audio_models:
                if "audio" in kwargs and kwargs.get("audio", {}).get("voice") is None:
                    kwargs["audio"]["voice"] = cls.audio_models[model][0]
                url = cls.text_api_endpoint
                stream = False
            else:
                url = cls.openai_endpoint
            extra_body.update({param: kwargs[param] for param in extra_parameters if param in kwargs})
            data = filter_none(
                messages=list(render_messages(messages, media)),
                model=model,
                temperature=temperature,
                presence_penalty=presence_penalty,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                response_format=response_format,
                stream=stream,
                seed=seed,
                cache=cache,
                **extra_body
            )
            headers = {"referer": referrer}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            async with session.post(url, json=data, headers=headers) as response:
                if response.status == 400:
                    debug.error(f"Error: 400 - Bad Request: {data}")
                await raise_for_status(response)
                if response.headers["content-type"].startswith("text/plain"):
                    yield await response.text()
                    return
                elif response.headers["content-type"].startswith("text/event-stream"):
                    reasoning = False
                    async for result in see_stream(response.content):
                        if "error" in result:
                            raise ResponseError(result["error"].get("message", result["error"]))
                        if result.get("usage") is not None:
                            yield Usage(**result["usage"])
                        choices = result.get("choices", [{}])
                        choice = choices.pop() if choices else {}
                        content = choice.get("delta", {}).get("content")
                        if content:
                            yield content
                        tool_calls = choice.get("delta", {}).get("tool_calls")
                        if tool_calls:
                            yield ToolCalls(choice["delta"]["tool_calls"])
                        reasoning_content = choice.get("delta", {}).get("reasoning_content")
                        if reasoning_content:
                            reasoning = True
                            yield Reasoning(reasoning_content)
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            yield FinishReason(finish_reason)
                    if reasoning:
                        yield Reasoning(status="Done")
                    if kwargs.get("action") == "next":
                        data = {
                            "model": "openai",
                            "messages": messages + FOLLOWUPS_DEVELOPER_MESSAGE,
                            "tool_choice": "required",
                            "tools": FOLLOWUPS_TOOLS
                        }
                        async with session.post(url, json=data, headers=headers) as response:
                            try:
                                await raise_for_status(response)
                                tool_calls = (await response.json()).get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                                if tool_calls:
                                    arguments = json.loads(tool_calls.pop().get("function", {}).get("arguments"))
                                    if arguments.get("title"):
                                        yield TitleGeneration(arguments.get("title"))
                                    if arguments.get("followups"):
                                        yield SuggestedFollowups(arguments.get("followups"))
                            except Exception as e:
                                debug.error("Error generating title and followups")
                                debug.error(e)
                elif response.headers["content-type"].startswith("application/json"):
                    result = await response.json()
                    if "choices" in result:
                        choice = result["choices"][0]
                        message = choice.get("message", {})
                        content = message.get("content", "")
                        if content:
                            yield content
                        if "tool_calls" in message:
                            yield ToolCalls(message["tool_calls"])
                    else:
                        raise ResponseError(result)
                    if result.get("usage") is not None:
                        yield Usage(**result["usage"])
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        yield FinishReason(finish_reason)
                else:
                    async for chunk in save_response_media(response, format_image_prompt(messages), [model, extra_body.get("audio", {}).get("voice")]):
                        yield chunk
                        return
