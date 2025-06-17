from __future__ import annotations

import asyncio
import random
from aiohttp import ClientSession, ClientTimeout

from urllib.parse import quote, quote_plus
from aiohttp import ClientSession

try:
    import nodriver
except:
    pass

from ...typing import Messages, AsyncResult
from ...providers.response import VideoResponse
from ...requests import get_nodriver
from ...errors import MissingRequirementsError
from ..base_provider import AsyncGeneratorProvider
from ..helper import format_media_prompt
from ... import debug

class RequestConfig:
    urls: list[str] = []
    headers: dict = {}

class Video(AsyncGeneratorProvider):
    urls = [
        "https://sora.chatgpt.com/explore",
        #"https://aistudio.google.com/generate-video"
    ]
    pub_url = "https://home.g4f.dev"
    api_url = f"{pub_url}/backend-api/v2/create?provider=Video&cache=true&prompt="
    search_url = f"{pub_url}/search/video+"
    drive_url = "https://www.googleapis.com/drive/v3/"

    needs_auth = True
    working = True

    browser = None
    stop_browser = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        **kwargs
    ) -> AsyncResult:
        prompt = format_media_prompt(messages, prompt)
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        try:
            browser, stop_browser = await get_nodriver(proxy=proxy, user_data_dir="gemini")
        except Exception as e:
            debug.error(f"Error getting nodriver:", e)
            async with ClientSession() as session:
                async with session.get(cls.search_url + quote_plus(prompt) + f"&min={prompt.count(' ') + 1}", timeout=ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        yield VideoResponse(str(response.url), prompt)
                        return
                async with session.post(cls.api_url + quote(prompt)) as response:
                    if not response.ok:
                        debug.error(f"Failed to connect to Video API: {response.status}")
                    else:
                        if response.headers.get("content-type", "text/plain").startswith("text/plain"):
                            data = (await response.text()).split("\n")
                            yield VideoResponse([f"{cls.pub_url}{url}" if url.startswith("/") else url for url in data], prompt)
                            return
                        yield VideoResponse(str(response.url), prompt)
                        return
            raise MissingRequirementsError("Video provider requires a browser to be installed.")
        RequestConfig.urls = []
        try:
            cls.page = await browser.get(random.choice(cls.urls))
        except Exception as e:
            debug.error(f"Error opening page:", e)
        try:
            page = cls.page
            await asyncio.sleep(3)
            await page.select("textarea", 240)
            def on_request(event: nodriver.cdp.network.RequestWillBeSent, page=None):
                if "mp4" in event.request.url:
                    RequestConfig.headers = {}
                    for key, value in event.request.headers.items():
                        RequestConfig.headers[key.lower()] = value
                    RequestConfig.urls.append(event.request.url)
                elif event.request.url.startswith(cls.drive_url):
                    RequestConfig.headers = {}
                    for key, value in event.request.headers.items():
                        RequestConfig.headers[key.lower()] = value
                    RequestConfig.urls.append(event.request.url)
            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)

            try:
                button = await page.find("Image")
                if button:
                    await button.click()
                else:
                    debug.error("No 'Image' button found.")
                    button = await page.find("Video")
                    if button:
                        await button.click()
                    else:
                        debug.error("No 'Video' button found.")
            except Exception as e:
                debug.error(f"Error clicking button:", e)
            debug.log(f"Using prompt: {prompt}")
            textarea = await page.select("textarea", 180)
            await textarea.send_keys(prompt)
            # try:
            #     button = await page.select('button[type="submit"]', 5)
            #     if button:
            #         await button.click()
            # finally:
            try:
                button = await page.find("Create")
                if button:
                    await button.click()
            except Exception as e:
                debug.error(f"Error clicking 'Create' button:", e)
            try:
                button = await page.find("Activity")
                if button:
                    await button.click()
            except Exception as e:
                debug.error(f"Error clicking 'Activity' button:", e)
            try:
                await asyncio.sleep(15)
                button = await page.find("Queued", timeout=30)
                if button:
                    await button.click()
            except Exception as e:
                debug.error(f"Error clicking 'Queued' button:", e)
            debug.log(f"Waiting for Video URL...")
            for idx in range(600):
                await asyncio.sleep(1)
                if RequestConfig.urls:
                    await asyncio.sleep(2)
                    RequestConfig.urls = list(set(RequestConfig.urls))
                    debug.log(f"Video URL: {len(RequestConfig.urls)}")
                    yield VideoResponse(RequestConfig.urls, prompt, {
                        "headers": {"authorization": RequestConfig.headers.get("authorization")} if RequestConfig.headers.get("authorization") else {}
                    })
                    RequestConfig.urls = []
                    break
                if idx == 599:
                    raise RuntimeError("Failed to get Video URL")
        finally:
            stop_browser()