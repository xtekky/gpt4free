from __future__ import annotations

import time
import asyncio
import random
from aiohttp import ClientSession, ClientTimeout

from urllib.parse import quote, quote_plus
from aiohttp import ClientSession

try:
    import nodriver
    from nodriver.core.connection import ProtocolException
except:
    pass

from ...typing import Messages, AsyncResult
from ...providers.response import VideoResponse, Reasoning, ContinueResponse, ProviderInfo
from ...requests import get_nodriver
from ...errors import MissingRequirementsError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_media_prompt
from ... import debug

class RequestConfig:
    urls: dict[str, list[str]] = {}
    headers: dict = {}

    @classmethod
    def get_response(cls, prompt: str) -> VideoResponse | None:
        if prompt in cls.urls and cls.urls[prompt]:
            cls.urls[prompt] = list(set(cls.urls[prompt]))
            debug.log(f"Video URL: {len(cls.urls[prompt])}")
            return VideoResponse(cls.urls[prompt], prompt, {
                "headers": {"authorization": cls.headers.get("authorization")} if cls.headers.get("authorization") else {},
                "preview": [url.replace("md.mp4", "thumb.webp") for url in cls.urls[prompt]]
            })

class Video(AsyncGeneratorProvider, ProviderModelMixin):
    urls = [
        "https://sora.chatgpt.com/explore",
        #"https://aistudio.google.com/generate-video"
    ]
    pub_url = "https://home.g4f.dev"
    api_url = f"{pub_url}/backend-api/v2/create?provider=Video&cache=true&prompt="
    search_url = f"{pub_url}/search/video+"
    drive_url = "https://www.googleapis.com/drive/v3/"

    active_by_default = True
    default_model = "sora"
    video_models = [default_model]

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
        aspect_ratio: str = None,
        **kwargs
    ) -> AsyncResult:
        yield ProviderInfo(**cls.get_dict(), model="sora")
        started = time.time()
        prompt = format_media_prompt(messages, prompt)
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        async with ClientSession() as session:
            yield Reasoning(label="Lookup")
            found_urls = []
            for skip in range(0, 9):
                async with session.get(cls.search_url + quote_plus(prompt) + f"?skip={skip}", timeout=ClientTimeout(total=10)) as response:
                    if response.ok:
                        yield Reasoning(label=f"Found {skip+1}", status="")
                        found_urls.append(str(response.url))
                    else:
                        break
            if found_urls:
                yield Reasoning(label=f"Finished", status="")
                yield VideoResponse(found_urls, prompt)
                return
        response = RequestConfig.get_response(prompt)
        if response:
            yield Reasoning(label="Found cached Video", status="")
            yield response
            return
        try:
            yield Reasoning(label="Open browser")
            browser, stop_browser = await get_nodriver(proxy=proxy, user_data_dir="gemini")
        except Exception as e:
            debug.error(f"Error getting nodriver:", e)
            async with ClientSession() as session:
                yield Reasoning(label="Generating")
                async with session.post(cls.api_url + quote(prompt)) as response:
                    if not response.ok:
                        debug.error(f"Failed to generate Video: {response.status}")
                    else:
                        yield Reasoning(label="Finished", status="")
                        if response.headers.get("content-type", "text/plain").startswith("text/plain"):
                            data = (await response.text()).split("\n")
                            yield VideoResponse([f"{cls.pub_url}{url}" if url.startswith("/") else url for url in data], prompt)
                            return
                        yield VideoResponse(str(response.url), prompt)
                        return
            raise MissingRequirementsError("Video provider requires a browser to be installed.")
        try:
            cls.page = await browser.get(random.choice(cls.urls))
        except Exception as e:
            debug.error(f"Error opening page:", e)
        response = RequestConfig.get_response(prompt)
        if response:
            yield Reasoning(label="Found", status="")
            yield response
            return
        try:
            page = cls.page
            await asyncio.sleep(3)
            await page.select("textarea", 240)
            try:
                button = await page.find("Image")
                if button:
                    await button.click()
                else:
                    debug.error("No 'Image' button found.")
                    button = await page.find("Video")
                    if button:
                        await button.click()
                        yield Reasoning(label=f"Clicked 'Video' button")
                    else:
                        debug.error("No 'Video' button found.")
            except Exception as e:
                debug.error(f"Error clicking button:", e)
            try:
                if aspect_ratio:
                    button = await page.find("2:3")
                    if button:
                        await button.click()
                    else:
                        debug.error("No '2:3' button found.")
                        button = await page.find(aspect_ratio)
                        if button:
                            await button.click()
                            yield Reasoning(label=f"Clicked '{aspect_ratio}' button")
                        else:
                            debug.error(f"No '{aspect_ratio}' button found.")
            except Exception as e:
                debug.error(f"Error clicking button:", e)
            debug.log(f"Using prompt: {prompt}")
            textarea = await page.select("textarea", 180)
            await textarea.send_keys(prompt)
            yield Reasoning(label=f"Sending prompt", token=prompt)
            # try:
            #     button = await page.select('button[type="submit"]', 5)
            #     if button:
            #         await button.click()
            # finally:
            try:
                button = await page.find("Create")
                if button:
                    await button.click()
                    yield Reasoning(label=f"Clicked 'Create' button")
            except Exception as e:
                debug.error(f"Error clicking 'Create' button:", e)
            try:
                button = await page.find("Activity")
                if button:
                    await button.click()
                    yield Reasoning(label=f"Clicked 'Activity' button")
            except Exception as e:
                debug.error(f"Error clicking 'Activity' button:", e)
            for idx in range(60):
                await asyncio.sleep(1)
                try:
                    button = await page.find("Queued")
                    if button:
                        await button.click()
                        yield Reasoning(label=f"Clicked 'Queued' button")
                        break
                except ProtocolException as e:
                    pass
            if prompt not in RequestConfig.urls:
                RequestConfig.urls[prompt] = []
            def on_request(event: nodriver.cdp.network.RequestWillBeSent, page=None):
                if ".mp4" in event.request.url:
                    RequestConfig.headers = {}
                    for key, value in event.request.headers.items():
                        RequestConfig.headers[key.lower()] = value
                    RequestConfig.urls[prompt].append(event.request.url)
                elif event.request.url.startswith(cls.drive_url):
                    RequestConfig.headers = {}
                    for key, value in event.request.headers.items():
                        RequestConfig.headers[key.lower()] = value
                    RequestConfig.urls[prompt].append(event.request.url)
            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            for idx in range(600):
                yield Reasoning(label=f"Waiting for Video... {idx+1}/600")
                if time.time() - started > 30:
                    yield ContinueResponse("Timeout waiting for Video URL")
                await asyncio.sleep(1)
                if RequestConfig.urls[prompt]:
                    await asyncio.sleep(2)
                    response = RequestConfig.get_response(prompt)
                    if response:
                        yield Reasoning(label="Finished", status="")
                        yield response
                        return
                if idx == 599:
                    raise RuntimeError("Failed to get Video URL")
        finally:
            stop_browser()