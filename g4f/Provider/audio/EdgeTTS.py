from __future__ import annotations

import os
import random
import asyncio

try:
    import edge_tts
    from edge_tts import VoicesManager
    has_edge_tts = True
except ImportError:
    has_edge_tts = False

from ...typing import AsyncResult, Messages
from ...providers.response import AudioResponse
from ...image.copy_images import get_filename, get_media_dir, ensure_media_dir
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_last_message

class EdgeTTS(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Edge TTS"
    working = has_edge_tts
    model_id = "edge-tts"
    default_language = "en"
    default_locale = "en-US"
    default_format = "mp3"

    @classmethod
    def get_models(cls) -> list[str]:
        if not cls.models:
            voices = asyncio.run(VoicesManager.create())
            cls.default_model = voices.find(Locale=cls.default_locale)[0]["Name"]
            cls.models = [voice["Name"] for voice in voices.voices]
            cls.audio_models = cls.models
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        audio: dict = {},
        **kwargs
    ) -> AsyncResult:
        prompt = get_last_message(messages, prompt)
        if not prompt:
            raise ValueError("Prompt is empty.")
        voice = audio.get("voice", model if model and model != cls.model_id else None)
        if not voice:
            voices = await VoicesManager.create()
            if "locale" in audio:
                voices = voices.find(Locale=audio["locale"])
            elif audio.get("language", cls.default_language) != cls.default_language:
                if "-" in audio.get("language"):
                    voices = voices.find(Locale=audio.get("language"))
                else:
                    voices = voices.find(Language=audio.get("language"))
            else:
                voices = voices.find(Locale=cls.default_locale)
            if not voices:
                raise ValueError(f"No voices found for language '{audio.get('language')}' and locale '{audio.get('locale')}'.")
            voice = random.choice(voices)["Name"]

        format = audio.get("format", cls.default_format)
        filename = get_filename([cls.model_id], prompt, f".{format}", prompt)
        target_path = os.path.join(get_media_dir(), filename)
        ensure_media_dir()

        extra_parameters = {param: audio[param] for param in ["rate", "volume", "pitch"] if param in audio}
        communicate = edge_tts.Communicate(prompt, voice=voice, proxy=proxy, **extra_parameters)

        await communicate.save(target_path)
        yield AudioResponse(f"/media/{filename}", voice=voice, text=prompt)