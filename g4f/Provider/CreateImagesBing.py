from __future__ import annotations

import asyncio
import time
import os
from typing import Generator

from ..cookies import get_cookies
from ..webdriver import WebDriver, get_driver_cookies, get_browser
from ..image import ImageResponse
from ..errors import MissingRequirementsError, MissingAuthError
from .bing.create_images import BING_URL, create_images, create_session

BING_URL = "https://www.bing.com"
TIMEOUT_LOGIN = 1200

def wait_for_login(driver: WebDriver, timeout: int = TIMEOUT_LOGIN) -> None:
    """
    Waits for the user to log in within a given timeout period.

    Args:
        driver (WebDriver): Webdriver for browser automation.
        timeout (int): Maximum waiting time in seconds.

    Raises:
        RuntimeError: If the login process exceeds the timeout.
    """
    driver.get(f"{BING_URL}/")
    start_time = time.time()
    while not driver.get_cookie("_U"):
        if time.time() - start_time > timeout:
            raise RuntimeError("Timeout error")
        time.sleep(0.5)

def get_cookies_from_browser(proxy: str = None) -> dict[str, str]:
    """
    Retrieves cookies from the browser using webdriver.

    Args:
        proxy (str, optional): Proxy configuration.

    Returns:
        dict[str, str]: Retrieved cookies.
    """
    with get_browser(proxy=proxy) as driver:
        wait_for_login(driver)
        time.sleep(1)
        return get_driver_cookies(driver)

class CreateImagesBing:
    """A class for creating images using Bing."""

    def __init__(self, cookies: dict[str, str] = {}, proxy: str = None) -> None:
        self.cookies = cookies
        self.proxy = proxy

    def create_completion(self, prompt: str) -> Generator[ImageResponse, None, None]:
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
            return ImageResponse(images, prompt, {"preview": "{image}?w=200&h=200"})