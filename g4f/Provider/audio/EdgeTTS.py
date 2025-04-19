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
from ..helper import format_image_prompt

class EdgeTTS(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Edge TTS"
    working = has_edge_tts
    default_model = "edge-tts"
    default_locale = "en-US"

    @classmethod
    def get_models(cls) -> list[str]:
        if not cls.models:
            voices = asyncio.run(VoicesManager.create())
            cls.default_model = voices.find(Locale=cls.default_locale)[0]["Name"]
            cls.models = [voice["Name"] for voice in voices.voices]
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        audio: dict = {"voice": None, "format": "mp3"},
        **kwargs
    ) -> AsyncResult:
        prompt = format_image_prompt(messages, prompt)
        if not prompt:
            raise ValueError("Prompt is empty.")
        voice = audio.get("voice", model)
        if not voice:
            voices = await VoicesManager.create()
            if "locale" in audio:
                voices = voices.find(Locale=audio["locale"])
            elif "language" in audio:
                if "-" in audio["language"]:
                    voices = voices.find(Locale=audio["language"])
                else:
                    voices = voices.find(Language=audio["language"])
            else:
                voices = voices.find(Locale=cls.default_locale)
            if not voices:
                raise ValueError(f"No voices found for language '{audio.get('language')}' and locale '{audio.get('locale')}'.")
            voice = random.choice(voices)["Name"]

        format = audio.get("format", "mp3")
        filename = get_filename([cls.default_model], prompt, f".{format}", prompt)
        target_path = os.path.join(get_media_dir(), filename)
        ensure_media_dir()

        extra_parameters = {param: audio[param] for param in ["rate", "volume", "pitch"] if param in audio}
        communicate = edge_tts.Communicate(prompt, voice=voice, proxy=proxy, **extra_parameters)

        await communicate.save(target_path)
        yield AudioResponse(f"/media/{filename}", voice=voice, prompt=prompt)