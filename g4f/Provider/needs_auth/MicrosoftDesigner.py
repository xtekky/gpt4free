from __future__ import annotations

import uuid
import aiohttp
import random
import asyncio
import json

from ...image import ImageResponse
from ...errors import MissingRequirementsError, NoValidHarFileError
from ...typing import AsyncResult, Messages
from ...requests.raise_for_status import raise_for_status
from ...requests.aiohttp import get_connector
from ...requests import get_nodriver
from ..Copilot import get_headers, get_har_files
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_random_hex, format_image_prompt
from ... import debug

class MicrosoftDesigner(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Microsoft Designer"
    url = "https://designer.microsoft.com"
    working = True
    use_nodriver = True
    needs_auth = True
    default_image_model = "dall-e-3"
    image_models = [default_image_model, "1024x1024", "1024x1792", "1792x1024"]
    models = image_models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        image_size = "1024x1024"
        if model != cls.default_image_model and model in cls.image_models:
            image_size = model
        yield await cls.generate(format_image_prompt(messages, prompt), image_size, proxy)

    @classmethod
    async def generate(cls, prompt: str, image_size: str, proxy: str = None) -> ImageResponse:
        try:
            access_token, user_agent = readHAR("https://designerapp.officeapps.live.com")
        except NoValidHarFileError as h:
            debug.log(f"{cls.__name__}: {h}")
            try:
                access_token, user_agent = await get_access_token_and_user_agent(cls.url, proxy)
            except MissingRequirementsError:
                raise h
        images = await create_images(prompt, access_token, user_agent, image_size, proxy)
        return ImageResponse(images, prompt)

async def create_images(prompt: str, access_token: str, user_agent: str, image_size: str, proxy: str = None, seed: int = None):
    url = 'https://designerapp.officeapps.live.com/designerapp/DallE.ashx?action=GetDallEImagesCogSci'
    if seed is None:
        seed = random.randint(0, 10000)

    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US",
        'Authorization': f'Bearer {access_token}',
        "AudienceGroup": "Production",
        "Caller": "DesignerApp",
        "ClientId": "b5c2664a-7e9b-4a7a-8c9a-cd2c52dcf621",
        "SessionId": str(uuid.uuid4()),
        "UserId": get_random_hex(16),
        "ContainerId": "1e2843a7-2a98-4a6c-93f2-42002de5c478",
        "FileToken": "9f1a4cb7-37e7-4c90-b44d-cb61cfda4bb8",
        "x-upload-to-storage-das": "1",
        "traceparent": "",
        "X-DC-Hint": "FranceCentral",
        "Platform": "Web",
        "HostApp": "DesignerApp",
        "ReleaseChannel": "",
        "IsSignedInUser": "true",
        "Locale": "de-DE",
        "UserType": "MSA",
        "x-req-start": "2615401",
        "ClientBuild": "1.0.20241120.9",
        "ClientName": "DesignerApp",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "Referer": "https://designer.microsoft.com/"
    }

    form_data = aiohttp.FormData()
    form_data.add_field('dalle-caption', prompt)
    form_data.add_field('dalle-scenario-name', 'TextToImage')
    form_data.add_field('dalle-batch-size', '4')
    form_data.add_field('dalle-image-response-format', 'UrlWithBase64Thumbnail')
    form_data.add_field('dalle-seed', seed)
    form_data.add_field('ClientFlights', 'EnableBICForDALLEFlight')
    form_data.add_field('dalle-hear-back-in-ms', 1000)
    form_data.add_field('dalle-include-b64-thumbnails', 'true')
    form_data.add_field('dalle-aspect-ratio-scaling-factor-b64-thumbnails', 0.3)
    form_data.add_field('dalle-image-size', image_size)

    async with aiohttp.ClientSession(connector=get_connector(proxy=proxy)) as session:
        async with session.post(url, headers=headers, data=form_data) as response:
            await raise_for_status(response)
            response_data = await response.json()
        form_data.add_field('dalle-boost-count', response_data.get('dalle-boost-count', 0))
        polling_meta_data = response_data.get('polling_response', {}).get('polling_meta_data', {})
        form_data.add_field('dalle-poll-url', polling_meta_data.get('poll_url', ''))

        while True:
            await asyncio.sleep(polling_meta_data.get('poll_interval', 1000) / 1000)
            async with session.post(url, headers=headers, data=form_data) as response:
                await raise_for_status(response)
                response_data = await response.json()
            images = [image["ImageUrl"] for image in response_data.get('image_urls_thumbnail', [])]
            if images:
                return images

def readHAR(url: str) -> tuple[str, str]:
    api_key = None
    user_agent = None
    for path in get_har_files():
        with open(path, 'rb') as file:
            try:
                harFile = json.loads(file.read())
            except json.JSONDecodeError:
                # Error: not a HAR file!
                continue
            for v in harFile['log']['entries']:
                if v['request']['url'].startswith(url):
                    v_headers = get_headers(v)
                    if "authorization" in v_headers:
                        api_key = v_headers["authorization"].split(maxsplit=1).pop()
                    if "user-agent" in v_headers:
                        user_agent = v_headers["user-agent"]
    if api_key is None:
        raise NoValidHarFileError("No access token found in .har files")

    return api_key, user_agent

async def get_access_token_and_user_agent(url: str, proxy: str = None):
    browser, stop_browser = await get_nodriver(proxy=proxy, user_data_dir="designer")
    try:
        page = await browser.get(url)
        user_agent = await page.evaluate("navigator.userAgent")
        access_token = None
        while access_token is None:
            access_token = await page.evaluate("""
                (() => {
                    for (var i = 0; i < localStorage.length; i++) {
                        try {
                            item = JSON.parse(localStorage.getItem(localStorage.key(i)));
                            if (item.credentialType == "AccessToken" 
                                && item.expiresOn > Math.floor(Date.now() / 1000)
                                && item.target.includes("designerappservice")) {
                                return item.secret;
                            }
                        } catch(e) {}
                    }
                })()
            """)
            if access_token is None:
                await asyncio.sleep(1)
        await page.close()
        return access_token, user_agent
    finally:
        stop_browser()