from __future__ import annotations

try:
    has_openaifm = True
except ImportError:
    has_openaifm = False

from aiohttp import ClientSession
from urllib.parse import urlencode
import json

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_last_message
from ...image.copy_images import save_response_media


class OpenAIFM(AsyncGeneratorProvider, ProviderModelMixin):
    label = "OpenAI.fm"
    url = "https://www.openai.fm"
    api_endpoint = "https://www.openai.fm/api/generate"
    
    working = has_openaifm
    
    default_model = 'gpt-4o-mini-tts'
    default_audio_model = default_model
    default_voice = 'coral'
    voices = ['alloy', 'ash', 'ballad', default_voice, 'echo', 'fable', 'onyx', 'nova', 'sage', 'shimmer', 'verse']
    audio_models = {default_audio_model: voices}
    models = [default_audio_model]

    
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
        **kwargs
    ) -> AsyncResult:
        
        # Retrieve parameters from the audio dictionary
        voice = audio.get("voice", kwargs.get("voice", cls.default_voice))
        instructions = audio.get("instructions", kwargs.get("instructions", cls.friendly))
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "sec-fetch-dest": "audio",
            "sec-fetch-mode": "no-cors", 
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
            "referer": cls.url
        }

        # Using prompts or formatting messages
        text = get_last_message(messages, prompt)
        
        params = {
            "input": text,
            "prompt": instructions,
            "voice": voice
        }
        
        async with ClientSession(headers=headers) as session:
            
            # Print the full URL with parameters
            full_url = f"{cls.api_endpoint}?{urlencode(params)}"
            
            async with session.get(
                cls.api_endpoint,
                params=params,
                proxy=proxy
            ) as response:
                
                response.raise_for_status()
                
                async for chunk in save_response_media(response, text, [model, voice]):
                    yield chunk
