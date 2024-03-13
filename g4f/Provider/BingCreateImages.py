from __future__ import annotations

import asyncio
import os
from typing import Iterator, Union

from ..cookies import get_cookies
from ..image import ImageResponse
from ..errors import MissingRequirementsError, MissingAuthError
from ..typing import Cookies
from .bing.create_images import create_images, create_session, get_cookies_from_browser

class BingCreateImages:
    """A class for creating images using Bing."""

    def __init__(self, cookies: Cookies = None, proxy: str = None) -> None:
        self.cookies: Cookies = cookies
        self.proxy: str = proxy

    def create(self, prompt: str) -> Iterator[Union[ImageResponse, str]]:
        """
        Generator for creating imagecompletion based on a prompt.

        Args:
            prompt (str): Prompt to generate images.

        Yields:
            Generator[str, None, None]: The final output as markdown formatted string with images.
        """
        cookies = self.cookies or get_cookies(".bing.com", False)
        if "_U" not in cookies:
            login_url = os.environ.get("G4F_LOGIN_URL")
            if login_url:
                yield f"Please login: [Bing]({login_url})\n\n"
            try:
                self.cookies = get_cookies_from_browser(self.proxy)
            except MissingRequirementsError as e:
                raise MissingAuthError(f'Missing "_U" cookie. {e}')
        yield asyncio.run(self.create_async(prompt))

    async def create_async(self, prompt: str) -> ImageResponse:
        """
        Asynchronously creates a markdown formatted string with images based on the prompt.

        Args:
            prompt (str): Prompt to generate images.

        Returns:
            str: Markdown formatted string with images.
        """
        cookies = self.cookies or get_cookies(".bing.com", False)
        if "_U" not in cookies:
            raise MissingAuthError('Missing "_U" cookie')
        proxy = self.proxy or os.environ.get("G4F_PROXY")
        async with create_session(cookies, proxy) as session:
            images = await create_images(session, prompt, proxy)
            return ImageResponse(images, prompt, {"preview": "{image}?w=200&h=200"} if len(images) > 1 else {})