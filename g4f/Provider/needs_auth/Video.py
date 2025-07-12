from __future__ import annotations

import asyncio
from typing import Optional
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

PUBLIC_URL = "https://home.g4f.dev"
SEARCH_URL = f"{PUBLIC_URL}/search/video+"

class RequestConfig:
    urls: dict[str, list[str]] = {}
    headers: dict = {}

    @classmethod
    async def get_response(cls, prompt: str, search: bool = False) -> Optional[VideoResponse]:
        if prompt in cls.urls and cls.urls[prompt]:
            unique_list = list(set(cls.urls[prompt]))[:10]
            return VideoResponse(unique_list, prompt, {
                "headers": {"authorization": cls.headers.get("authorization")} if cls.headers.get("authorization") else {},
            })
        if search:
            async with ClientSession() as session:
                found_urls = []
                for skip in range(0, 9):
                    async with session.get(SEARCH_URL + quote_plus(prompt) + f"?skip={skip}", timeout=ClientTimeout(total=10)) as response:
                        if response.ok:
                            found_urls.append(str(response.url))
                        else:
                            break
                if found_urls:
                    return VideoResponse(found_urls, prompt)

class Video(AsyncGeneratorProvider, ProviderModelMixin):
    urls = {
        "search": "https://sora.chatgpt.com/explore?query={0}",
        "sora": "https://sora.chatgpt.com/explore",
        #"veo": "https://aistudio.google.com/generate-video"
    }
    api_url = f"{PUBLIC_URL}/backend-api/v2/create?provider=Video&cache=true&prompt="
    drive_url = "https://www.googleapis.com/drive/v3/"

    active_by_default = True
    default_model = "search"
    models = list(urls.keys())
    video_models = models

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
        if not model:
            model = cls.default_model
        if model not in cls.video_models:
            raise ValueError(f"Model '{model}' is not supported by {cls.__name__}. Supported models: {cls.models}")
        yield ProviderInfo(**cls.get_dict(), model="sora")
        prompt = format_media_prompt(messages, prompt).encode()[:100].decode("utf-8", "ignore").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        response = await RequestConfig.get_response(prompt, model=="search")
        if response:
            yield Reasoning(label=f"Found {len(response.urls)} Video(s)", status="")
            yield response
            return
        try:
            yield Reasoning(label="Open browser")
            browser, stop_browser = await get_nodriver(proxy=proxy)
        except Exception as e:
            debug.error(f"Error getting nodriver:", e)
            async with ClientSession() as session:
                yield Reasoning(label="Generating")
                async with session.get(cls.api_url + quote(prompt)) as response:
                    if not response.ok:
                        debug.error(f"Failed to generate Video: {response.status}")
                    else:
                        yield Reasoning(label="Finished", status="")
                        if response.headers.get("content-type", "text/plain").startswith("text/plain"):
                            data = (await response.text()).split("\n")
                            yield VideoResponse([f"{PUBLIC_URL}{url}" if url.startswith("/") else url for url in data], prompt)
                            return
                        yield VideoResponse(str(response.url), prompt)
                        return
            raise MissingRequirementsError("Video provider requires a browser to be installed.")
        page = None
        try:
            yield ContinueResponse("Timeout waiting for Video URL")
            page = await browser.get(cls.urls[model].format(quote(prompt)))
        except Exception as e:
            stop_browser()
            debug.error(f"Error opening page:", e)
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
                for _, urls in RequestConfig.urls.items():
                    if event.request.url in urls:
                        return
                RequestConfig.urls[prompt].append(event.request.url)
        if model == "search" and page is not None:
            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            for _ in range(5):
                await page.scroll_down(5)
                await asyncio.sleep(1)
        response = await RequestConfig.get_response(prompt, True)
        if response:
            stop_browser()
            yield Reasoning(label="Found", status="")
            yield response
            return
        if page is None:
            raise RuntimeError("Failed to open page or get response.")
        try:
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
                    button = await page.find(":")
                    if button:
                        await button.click()
                    else:
                        debug.error("No 'x:x' button found.")
                    await asyncio.sleep(1)
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
            await textarea.click()
            await textarea.clear_input()
            await textarea.send_keys(prompt)
            yield Reasoning(label=f"Sending prompt", token=prompt)
            try:
                button = await page.select('button[type="submit"]', 5)
                if button:
                    await button.click()
            except Exception as e:
                debug.error(f"Error clicking submit button:", e)
            try:
                button = await page.find("Create video")
                if button:
                    await button.click()
                    yield Reasoning(label=f"Clicked 'Create video' button")
            except Exception as e:
                debug.error(f"Error clicking 'Create video' button:", e)
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
                    if idx == 59:
                        debug.error(e)
                        raise RuntimeError("Failed to click 'Queued' button")
            await asyncio.sleep(3)
            if model != "search" and page is not None:
                await page.send(nodriver.cdp.network.enable())
                page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            for idx in range(300):
                yield Reasoning(label="Waiting for Video...", status=f"{idx+1}/300")
                await asyncio.sleep(1)
                if RequestConfig.urls[prompt]:
                    await asyncio.sleep(2)
                    response = await RequestConfig.get_response(prompt, model=="search")
                    if response:
                        stop_browser()
                        yield Reasoning(label="Finished", status="")
                        yield response
                        return
                if idx == 299:
                    stop_browser()
                    raise RuntimeError("Failed to get Video URL")
        finally:
            stop_browser()