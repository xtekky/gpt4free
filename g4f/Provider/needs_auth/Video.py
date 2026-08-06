from __future__ import annotations

import json
from typing import Optional
from aiohttp import ClientSession, ClientTimeout
from urllib.parse import quote
from urllib.parse import quote

from ...typing import Messages, AsyncResult
from ...providers.response import VideoResponse, Reasoning, ProviderInfo
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_media_prompt
from ... import debug

FLIM_SEARCH_URL = "https://api.flim.ai/2.0.0/search"
TRANSLATE_URL = "https://g4f.space/ai/auto/Translate%20the%20given%20prompt%20into%20English%20and%20reply%20with%20only%20the%20translated%20text%E2%80%94no%20explanations,%20introductions,%20or%20additional%20content:%20{prompt}"
FLIM_HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "origin": "https://app.flim.ai",
    "referer": "https://app.flim.ai/",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
}


class RequestConfig:
    urls: dict[str, list[str]] = {}

    @classmethod
    async def get_response(
        cls, prompt: str, search: bool = False
    ) -> Optional[VideoResponse]:
        if prompt in cls.urls and cls.urls[prompt]:
            unique_list = list(set(cls.urls[prompt]))[:10]
            return VideoResponse(unique_list, prompt)
        if search:
            found_urls = await cls.search_flim(prompt)
            if found_urls:
                return VideoResponse(found_urls, prompt)

    @classmethod
    async def translate_prompt(cls, prompt: str) -> str:
        """Translate the prompt into English via the g4f.space auto endpoint."""
        try:
            url = TRANSLATE_URL.format(prompt=quote(prompt, safe=""))
            async with ClientSession() as session:
                async with session.get(
                    url, timeout=ClientTimeout(total=15)
                ) as response:
                    if not response.ok:
                        debug.error(f"Translate prompt failed: {response.status}")
                        return prompt
                    text = (await response.text()).strip()
            return text[:100] if text else prompt
        except Exception as e:
            debug.error(f"Error translating prompt:", e)
            return prompt

    @classmethod
    async def search_flim(cls, prompt: str) -> list[str]:
        payload = {
            "search": {
                "saved_images": False,
                "full_text": prompt,
                "similar_picture_id": "",
                "movie_id": "",
                "dop": "",
                "director": "",
                "brand": "",
                "agency": "",
                "production_company": "",
                "actor": "",
                "creator": "",
                "artist": "",
                "collection_id": "",
                "board_id": "",
                "filters": {
                    "genres": [],
                    "colors": [],
                    "number_of_persons": [],
                    "years": [],
                    "shot_types": [],
                    "movie_types": [],
                    "aspect_ratio": [],
                    "safety_content": [],
                    "has_video_cuts": True,
                    "camera_motions": [],
                },
                "negative_filters": {
                    "aspect_ratio": [],
                    "genres": ["ANIMATION"],
                    "movie_types": [],
                    "colors": [],
                    "shot_types": [],
                    "number_of_persons": [],
                    "years": [],
                    "safety_content": ["nudity", "violence"],
                },
            },
            "page": 0,
            "sort_by": "",
            "order_by": "",
            "number_per_pages": 100,
        }
        urls = []
        try:
            async with ClientSession(headers=FLIM_HEADERS) as session:
                async with session.post(
                    FLIM_SEARCH_URL, json=payload, timeout=ClientTimeout(total=15)
                ) as response:
                    if not response.ok:
                        debug.error(f"Flim search failed: {response.status}")
                        return urls
                    data = await response.json()
            for img in data.get("query_response", {}).get("images", []):
                if img.get("has_video_urls"):
                    video_urls = img.get("video_urls", {})
                    url = video_urls.get("url_thumbnail") or video_urls.get("url")
                    if url and url not in urls:
                        urls.append(url)
        except Exception as e:
            debug.error(f"Error fetching flim video URLs:", e)
        return urls[:10]


class Video(AsyncGeneratorProvider, ProviderModelMixin):
    urls = {
        "search": "https://app.flim.ai/?ft={0}",
    }

    active_by_default = True
    default_model = "search"
    models = list(urls.keys())
    video_models = models

    needs_auth = False
    working = True

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages, prompt: str = None, **kwargs
    ) -> AsyncResult:
        if not model:
            model = cls.default_model
        if model not in cls.video_models:
            raise ValueError(
                f"Model '{model}' is not supported by {cls.__name__}. Supported models: {cls.models}"
            )
        yield ProviderInfo(**cls.get_dict(), model=model)
        prompt = (
            format_media_prompt(messages, prompt)
            .encode()[:100]
            .decode("utf-8", "ignore")
            .strip()
        )
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        prompt = await RequestConfig.translate_prompt(prompt)
        response = await RequestConfig.get_response(prompt, model == "search")
        if response:
            yield Reasoning(label=f"Found {len(response.urls)} Video(s)", status="")
            yield response
            return
        raise RuntimeError("Failed to find any videos for the prompt.")
