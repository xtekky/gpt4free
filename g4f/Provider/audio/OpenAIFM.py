from __future__ import annotations

from urllib.parse import quote
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_media_prompt, get_system_prompt
from ...image.copy_images import save_response_media
from ...providers.response import AudioResponse
from ...requests.raise_for_status import raise_for_status
from ...requests.aiohttp import get_connector
from ...requests import DEFAULT_HEADERS

class OpenAIFM(AsyncGeneratorProvider, ProviderModelMixin):
    label = "OpenAI.fm"
    url = "https://www.openai.fm"
    api_endpoint = "https://www.openai.fm/api/generate"
    working = True

    default_model = 'coral'
    voices = ['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer', 'verse']
    styles = ['friendly', 'patient_teacher', 'noir_detective', 'cowboy', 'calm', 'scientific_style']
    audio_models = {"gpt-4o-mini-tts": voices}
    model_aliases = {"gpt-4o-mini-tts": default_model}
    models = styles + voices

    @classmethod
    def get_grouped_models(cls):
        return [
            {"group":"Styles", "models": cls.styles},
            {"group":"Voices", "models": cls.voices},
        ]

    friendly = """Affect/personality: A cheerful guide 

Tone: Friendly, clear, and reassuring, creating a calm atmosphere and making the listener feel confident and comfortable.

Pronunciation: Clear, articulate, and steady, ensuring each instruction is easily understood while maintaining a natural, conversational flow.

Pause: Brief, purposeful pauses after key instructions (e.g., "cross the street" and "turn right") to allow time for the listener to process the information and follow along.

Emotion: Warm and supportive, conveying empathy and care, ensuring the listener feels guided and safe throughout the journey."""

    patient_teacher = """Accent/Affect: Warm, refined, and gently instructive, reminiscent of a friendly art instructor.

Tone: Calm, encouraging, and articulate, clearly describing each step with patience.

Pacing: Slow and deliberate, pausing often to allow the listener to follow instructions comfortably.

Emotion: Cheerful, supportive, and pleasantly enthusiastic; convey genuine enjoyment and appreciation of art.

Pronunciation: Clearly articulate artistic terminology (e.g., "brushstrokes," "landscape," "palette") with gentle emphasis.

Personality Affect: Friendly and approachable with a hint of sophistication; speak confidently and reassuringly, guiding users through each painting step patiently and warmly."""

    noir_detective = """Affect: a mysterious noir detective

Tone: Cool, detached, but subtly reassuring—like they've seen it all and know how to handle a missing package like it's just another case.

Delivery: Slow and deliberate, with dramatic pauses to build suspense, as if every detail matters in this investigation.

Emotion: A mix of world-weariness and quiet determination, with just a hint of dry humor to keep things from getting too grim.

Punctuation: Short, punchy sentences with ellipses and dashes to create rhythm and tension, mimicking the inner monologue of a detective piecing together clues."""

    cowboy = """Voice: Warm, relaxed, and friendly, with a steady cowboy drawl that feels approachable.

Punctuation: Light and natural, with gentle pauses that create a conversational rhythm without feeling rushed.

Delivery: Smooth and easygoing, with a laid-back pace that reassures the listener while keeping things clear.

Phrasing: Simple, direct, and folksy, using casual, familiar language to make technical support feel more personable.

Tone: Lighthearted and welcoming, with a calm confidence that puts the caller at ease."""

    calm = """Voice Affect: Calm, composed, and reassuring; project quiet authority and confidence.

Tone: Sincere, empathetic, and gently authoritative—express genuine apology while conveying competence.

Pacing: Steady and moderate; unhurried enough to communicate care, yet efficient enough to demonstrate professionalism.

Emotion: Genuine empathy and understanding; speak with warmth, especially during apologies ("I'm very sorry for any disruption...").

Pronunciation: Clear and precise, emphasizing key reassurances ("smoothly," "quickly," "promptly") to reinforce confidence.

Pauses: Brief pauses after offering assistance or requesting details, highlighting willingness to listen and support."""

    scientific_style = """Voice: Authoritative and precise, with a measured, academic tone.

Tone: Formal and analytical, maintaining objectivity while conveying complex information.

Pacing: Moderate and deliberate, allowing time for complex concepts to be processed.

Pronunciation: Precise articulation of technical terms and scientific vocabulary.

Pauses: Strategic pauses after introducing new concepts to allow for comprehension.

Emotion: Restrained enthusiasm for discoveries and findings, conveying intellectual curiosity."""

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        audio: dict = {},
        download_media: bool = True,
        **kwargs
    ) -> AsyncResult:
        default_instructions = get_system_prompt(messages)
        if model and hasattr(cls, model):
            default_instructions = getattr(cls, model)
            model = ""
        model = cls.get_model(model)
        voice = audio.get("voice", kwargs.get("voice", model))
        instructions = audio.get("instructions", kwargs.get("instructions", default_instructions))
        headers = {
            **DEFAULT_HEADERS,
            "referer": f"{cls.url}/"
        }
        prompt = format_media_prompt(messages, prompt)
        params = {
            "input": prompt,
            "prompt": instructions,
            "voice": voice
        }
        if not download_media:
            query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items() if v is not None)
            yield AudioResponse(f"{cls.api_endpoint}?{query}")
            return
        async with ClientSession(headers=headers, connector=get_connector(proxy=proxy)) as session:            
            async with session.get(
                cls.api_endpoint,
                params=params
            ) as response:
                await raise_for_status(response)                
                async for chunk in save_response_media(response, prompt, [model, voice]):
                    yield chunk
