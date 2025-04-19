from __future__ import annotations

import os
import random
import asyncio

try:
    from gtts import gTTS as gTTS_Service
    has_gtts = True
except ImportError:
    has_gtts = False

from ...typing import AsyncResult, Messages
from ...providers.response import AudioResponse
from ...image.copy_images import get_filename, get_media_dir, ensure_media_dir
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_image_prompt

locals = {
    "en-AU": ["English (Australia)", "en", "com.au"],
    "en-GB": ["English (United Kingdom)", "en", "co.uk"],
    "en-US": ["English (United States)", "en", "us"],
    "en-CA": ["English (Canada)", "en", "ca"],
    "en-IN": ["English (India)", "en", "co.in"],
    "en-IE": ["English (Ireland)", "en", "ie"],
    "en-ZA": ["English (South Africa)", "en", "co.za"],
    "en-NG": ["English (Nigeria)", "en", "com.ng"],
    "fr-CA": ["French (Canada)", "fr", "ca"],
    "fr-FR": ["French (France)", "fr", "fr"],
    "de-DE": ["German (Germany)", "de", "de"],
    "zh-CN": ["Mandarin (China Mainland)", "zh-CN", "com"],
    "zh-TW": ["Mandarin (Taiwan)", "zh-TW", "com"],
    "pt-BR": ["Portuguese (Brazil)", "pt", "com.br"],
    "pt-PT": ["Portuguese (Portugal)", "pt", "pt"],
    "es-MX": ["Spanish (Mexico)", "es", "com.mx"],
    "es-ES": ["Spanish (Spain)", "es", "es"],
    "es-US": ["Spanish (United States)", "es", "us"],
}
models = {locale[0]: {"lang": locale[1], "tld": locale[2]} for locale in locals.values()}

class gTTS(AsyncGeneratorProvider, ProviderModelMixin):
    label = "gTTS (Google Text-to-Speech)"
    working = has_gtts
    model_id = "google-tts"
    default_language = "en"
    default_tld = "com"
    default_format = "mp3"
    models = list(models.keys())

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        audio: dict = {},
        **kwargs
    ) -> AsyncResult:
        prompt = format_image_prompt(messages, prompt)
        if not prompt:
            raise ValueError("Prompt is empty.")
        format = audio.get("format", cls.default_format)
        filename = get_filename([cls.model_id], prompt, f".{format}", prompt)
        target_path = os.path.join(get_media_dir(), filename)
        ensure_media_dir()

        gTTS_Service(
            prompt,
            **{
                "lang": audio.get("language", cls.default_language),
                "tld": audio.get("tld", cls.default_tld),
                "slow": audio.get("slow", False),
                **models.get(model, {})
            }
        ).save(target_path)

        yield AudioResponse(f"/media/{filename}", audio=audio, text=prompt)