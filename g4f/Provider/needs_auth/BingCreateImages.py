from __future__ import annotations

from ...cookies import get_cookies
from ...image import ImageResponse
from ...errors import MissingAuthError
from ...typing import AsyncResult, Messages, Cookies
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..bing.create_images import create_images, create_session

class BingCreateImages(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Microsoft Designer in Bing"
    parent = "Bing"
    url = "https://www.bing.com/images/create"
    working = True
    needs_auth = True
    image_models = ["dall-e"]

    def __init__(self, cookies: Cookies = None, proxy: str = None, api_key: str = None) -> None:
        if api_key is not None:
            if cookies is None:
                cookies = {}
            cookies["_U"] = api_key
        self.cookies = cookies
        self.proxy = proxy

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        cookies: Cookies = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        session = BingCreateImages(cookies, proxy, api_key)
        yield await session.generate(messages[-1]["content"])

    async def generate(self, prompt: str) -> ImageResponse:
        """
        Asynchronously creates a markdown formatted string with images based on the prompt.

        Args:
            prompt (str): Prompt to generate images.

        Returns:
            str: Markdown formatted string with images.
        """
        cookies = self.cookies or get_cookies(".bing.com", False)
        if cookies is None or "_U" not in cookies:
            raise MissingAuthError('Missing "_U" cookie')
        async with create_session(cookies, self.proxy) as session:
            images = await create_images(session, prompt)
            return ImageResponse(images, prompt, {"preview": "{image}?w=200&h=200"} if len(images) > 1 else {})