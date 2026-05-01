from __future__ import annotations

from aiohttp import ClientSession, FormData
from urllib.parse import quote

from curl_cffi import requests

from ...typing import AsyncResult, Messages, MediaListType
from ...image import is_data_an_audio
from ...image.copy_images import save_response_media
from ...providers.response import AudioResponse
from ...providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...requests.aiohttp import get_connector
from ...requests.defaults import DEFAULT_HEADERS
from ...requests.raise_for_status import raise_for_status
from ...tools.run_tools import AuthManager
from ...image import to_bytes
from ...tools.media import merge_media
from ..helper import filter_none, format_media_prompt
from ..PollinationsAI import PollinationsAI
from ... import debug

class PollinationsAudio(AsyncGeneratorProvider, ProviderModelMixin):
    label = "PollinationsAudio"
    parent = PollinationsAI.__name__
    active_by_default = False
    working = True
    supports_stream = False
    default_model = "elevenlabs"
    default_transcription_model = "openai-audio"
    models_endpoint = "https://gen.pollinations.ai/audio/models"
    speech_api_endpoint = "https://gen.pollinations.ai/v1/audio/speech"
    transcription_api_endpoint = "https://gen.pollinations.ai/v1/audio/transcriptions"
    simple_audio_endpoint = "https://gen.pollinations.ai/audio/{}"
    public_key = "".join(["pk", "_", "pqjxQN7C", "oSJUShHF"])
    available_voices = [
        "alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "ballad", "coral", "sage", "verse",
        "rachel", "domi", "bella", "elli", "charlotte", "dorothy", "sarah", "emily", "lily", "matilda",
        "adam", "antoni", "arnold", "josh", "sam", "daniel", "charlie", "james", "fin", "callum", "liam",
        "george", "brian", "bill"
    ]
    documented_audio_models = [
        "openai-audio", "elevenlabs", "elevenmusic", "whisper", "whisper-large-v3", "whisper-1",
        "scribe", "acestep", "qwen-tts", "qwen-tts-instruct"
    ]

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls.models:
            try:
                response = requests.get(cls.models_endpoint, timeout=kwargs.get("timeout", 15))
                response.raise_for_status()
                models = response.json()
                cls.models = {data.get("name"): {"id": data.get("name"), **data} for data in models}
            except Exception as e:
                debug.error(e)
                cls.models = {model: {"id": model} for model in cls.documented_audio_models}
        return cls.models
    
    @classmethod
    def _get_audio_voices(cls) -> list[str]:
        for model in cls.get_models().values():
            if "voices" in model:
                return model["voices"]
        return cls.available_voices

    @classmethod
    def get_grouped_models(cls) -> list[dict[str, list[str]]]:
        return [
            {"group": model.get("id"), "models": model.get("voices")} if model.get("voices") else model
            for model in
            cls.get_models().values()
        ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        media: MediaListType = None,
        proxy: str = None,
        api_key: str = None,
        prompt: str = None,
        audio: dict = None,
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:
        api_key = api_key or kwargs.get("api_key") or AuthManager.load_api_key(cls) or cls.public_key
        audio = {} if audio is None else dict(audio)
        if model in cls._get_audio_voices() and audio.get("voice") is None:
            audio["voice"] = model
            model = cls.default_model

        # Any audio media input is treated as a transcription request.
        media = list(merge_media(media, messages)) if model != cls.default_model else media
        if media and any(is_data_an_audio(media_data, filename) for media_data, filename in media):
            async for chunk in cls._create_transcription(
                media=media,
                api_key=api_key,
                proxy=proxy,
                model=model,
                **kwargs,
            ):
                yield chunk
            return
        if model == cls.default_transcription_model:
            model = cls.default_model
        async for chunk in cls._create_speech(
            model=model if model else cls.default_model,
            messages=messages,
            prompt=prompt,
            api_key=api_key,
            proxy=proxy,
            audio=audio,
            **kwargs,
        ):
            yield chunk

    @classmethod
    async def _create_speech(
        cls,
        model: str,
        messages: Messages,
        prompt: str,
        api_key: str,
        proxy: str,
        audio: dict,
        **kwargs,
    ) -> AsyncResult:
        text = format_media_prompt(messages, prompt)
        voice = audio.get("voice") or kwargs.get("voice")
        response_format = audio.get("format") or kwargs.get("response_format")
        payload = filter_none(
            model=model,
            input=text[:4096],
            voice=voice,
            response_format=response_format,
            speed=audio.get("speed") if "speed" in audio else kwargs.get("speed"),
            duration=audio.get("duration") if "duration" in audio else kwargs.get("duration"),
            instrumental=audio.get("instrumental") if "instrumental" in audio else kwargs.get("instrumental"),
            seed=audio.get("seed") if "seed" in audio else kwargs.get("seed"),
            style=audio.get("style") if "style" in audio else kwargs.get("style"),
            instruct=audio.get("instruct") if "instruct" in audio else kwargs.get("instruct"),
        )

        if not kwargs.get("download_media", True) and api_key.startswith("pk_"):
            encoded_text = quote(text[:4096])
            query = "&".join(f"{key}={quote(str(value))}" for key, value in payload.items() if key not in {"input"} and value is not None)
            if query:
                query = f"{query}&key={quote(api_key)}"
            else:
                query = f"key={quote(api_key)}"
            yield AudioResponse(f"{cls.simple_audio_endpoint.format(encoded_text)}?{query}", text, headers=headers)
            return

        headers = {
            **DEFAULT_HEADERS,
            "authorization": f"Bearer {api_key}",
        }
        async with ClientSession(headers=headers, connector=get_connector(proxy=proxy)) as session:
            async with session.post(cls.speech_api_endpoint, json=payload) as response:
                await raise_for_status(response)
                async for chunk in save_response_media(response, text, [model, voice]):
                    yield chunk

    @classmethod
    async def _create_transcription(
        cls,
        media: MediaListType,
        api_key: str,
        proxy: str,
        model: str = None,
        **kwargs,
    ) -> AsyncResult:
        media_data, filename = media[0]
        file_bytes = to_bytes(media_data)
        if not file_bytes:
            raise ValueError("No valid audio data found for transcription")

        form = FormData()
        form.add_field("file", file_bytes, filename=filename or "audio.wav", content_type="application/octet-stream")

        transcription_model = model
        if transcription_model in (None, "openai-audio"):
            transcription_model = cls.default_transcription_model
        form_fields = filter_none(
            model=transcription_model,
            language=kwargs.get("language"),
            prompt=kwargs.get("prompt"),
            response_format=kwargs.get("response_format"),
            temperature=kwargs.get("temperature"),
        )
        for key, value in form_fields.items():
            form.add_field(key, str(value))

        headers = {"authorization": f"Bearer {api_key}"}
        async with ClientSession(headers=headers, connector=get_connector(proxy=proxy)) as session:
            async with session.post(cls.transcription_api_endpoint, data=form) as response:
                await raise_for_status(response)
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    data = await response.json()
                    yield data.get("text", "")
                else:
                    yield await response.text()
