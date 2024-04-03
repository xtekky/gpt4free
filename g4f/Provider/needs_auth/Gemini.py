from __future__ import annotations

import os
import json
import random
import re

from aiohttp import ClientSession, BaseConnector

from ..helper import get_connector

try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    pass

from ...typing import Messages, Cookies, ImageType, AsyncResult
from ..base_provider import AsyncGeneratorProvider
from ..helper import format_prompt, get_cookies
from ...errors import MissingAuthError, MissingRequirementsError
from ...image import to_bytes, ImageResponse
from ...webdriver import get_browser, get_driver_cookies

REQUEST_HEADERS = {
    "authority": "gemini.google.com",
    "origin": "https://gemini.google.com",
    "referer": "https://gemini.google.com/",
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'x-same-domain': '1',
}
REQUEST_BL_PARAM = "boq_assistant-bard-web-server_20240201.08_p8"
REQUEST_URL = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
UPLOAD_IMAGE_URL = "https://content-push.googleapis.com/upload/"
UPLOAD_IMAGE_HEADERS = {
    "authority": "content-push.googleapis.com",
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.7",
    "authorization": "Basic c2F2ZXM6cyNMdGhlNmxzd2F2b0RsN3J1d1U=",
    "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
    "origin": "https://gemini.google.com",
    "push-id": "feeds/mcudyrk2a4khkz",
    "referer": "https://gemini.google.com/",
    "x-goog-upload-command": "start",
    "x-goog-upload-header-content-length": "",
    "x-goog-upload-protocol": "resumable",
    "x-tenant-id": "bard-storage",
}

class Gemini(AsyncGeneratorProvider):
    url = "https://gemini.google.com"
    needs_auth = True
    working = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: Cookies = None,
        connector: BaseConnector = None,
        image: ImageType = None,
        image_name: str = None,
        **kwargs
    ) -> AsyncResult:
        prompt = format_prompt(messages)
        cookies = cookies if cookies else get_cookies(".google.com", False, True)
        base_connector = get_connector(connector, proxy)
        async with ClientSession(
            headers=REQUEST_HEADERS,
            connector=base_connector
        ) as session:
            snlm0e  = await cls.fetch_snlm0e(session, cookies) if cookies else None
            if not snlm0e:
                driver = None
                try:
                    driver = get_browser(proxy=proxy)
                    try:
                        driver.get(f"{cls.url}/app")
                        WebDriverWait(driver, 5).until(
                            EC.visibility_of_element_located((By.CSS_SELECTOR, "div.ql-editor.textarea"))
                        )
                    except:
                        login_url = os.environ.get("G4F_LOGIN_URL")
                        if login_url:
                            yield f"Please login: [Google Gemini]({login_url})\n\n"
                        WebDriverWait(driver, 240).until(
                            EC.visibility_of_element_located((By.CSS_SELECTOR, "div.ql-editor.textarea"))
                        )
                    cookies = get_driver_cookies(driver)
                except MissingRequirementsError:
                    pass
                finally:
                    if driver:
                        driver.close()

            if not snlm0e:
                if "__Secure-1PSID" not in cookies:
                    raise MissingAuthError('Missing "__Secure-1PSID" cookie')
                snlm0e = await cls.fetch_snlm0e(session, cookies)
            if not snlm0e:
                raise RuntimeError("Invalid auth. SNlM0e not found")

            image_url = await cls.upload_image(base_connector, to_bytes(image), image_name) if image else None
        
            async with ClientSession(
                cookies=cookies,
                headers=REQUEST_HEADERS,
                connector=base_connector,
            ) as client:
                params = {
                    'bl': REQUEST_BL_PARAM,
                    '_reqid': random.randint(1111, 9999),
                    'rt': 'c'
                }
                data = {
                    'at': snlm0e,
                    'f.req': json.dumps([None, json.dumps(cls.build_request(
                        prompt,
                        image_url=image_url,
                        image_name=image_name
                    ))])
                }
                async with client.post(
                    REQUEST_URL,
                    data=data,
                    params=params,
                ) as response:
                    response = await response.text()
                    response_part = json.loads(json.loads(response.splitlines()[-5])[0][2])
                    if response_part[4] is None:
                        response_part = json.loads(json.loads(response.splitlines()[-7])[0][2])

                    content = response_part[4][0][1][0]
                    image_prompt = None
                    match = re.search(r'\[Imagen of (.*?)\]', content)
                    if match:
                        image_prompt = match.group(1)
                        content = content.replace(match.group(0), '')

                    yield content
                    if image_prompt:
                        images = [image[0][3][3] for image in response_part[4][0][12][7][0]]
                        resolved_images = []
                        preview = []
                        for image in images:
                            async with client.get(image, allow_redirects=False) as fetch:
                                image = fetch.headers["location"]
                            async with client.get(image, allow_redirects=False) as fetch:
                                image = fetch.headers["location"]
                            resolved_images.append(image)
                            preview.append(image.replace('=s512', '=s200'))
                        yield ImageResponse(resolved_images, image_prompt, {"orginal_links": images, "preview": preview})

    def build_request(
        prompt: str,
        conversation_id: str = "",
        response_id: str = "",
        choice_id: str = "",
        image_url: str = None,
        image_name: str = None,
        tools: list[list[str]] = []
    ) -> list:
        image_list = [[[image_url, 1], image_name]] if image_url else []
        return [
            [prompt, 0, None, image_list, None, None, 0],
            ["en"],
            [conversation_id, response_id, choice_id, None, None, []],
            None,
            None,
            None,
            [1],
            0,
            [],
            tools,
            1,
            0,
        ]

    async def upload_image(connector: BaseConnector, image: bytes, image_name: str = None):
        async with ClientSession(
            headers=UPLOAD_IMAGE_HEADERS,
            connector=connector
        ) as session:
            async with session.options(UPLOAD_IMAGE_URL) as reponse:
                reponse.raise_for_status()

            headers = {
                "size": str(len(image)),
                "x-goog-upload-command": "start"
            }
            data = f"File name: {image_name}" if image_name else None
            async with session.post(
                UPLOAD_IMAGE_URL, headers=headers, data=data
            ) as response:
                response.raise_for_status()
                upload_url = response.headers["X-Goog-Upload-Url"]

            async with session.options(upload_url, headers=headers) as response:
                response.raise_for_status()

            headers["x-goog-upload-command"] = "upload, finalize"
            headers["X-Goog-Upload-Offset"] = "0"
            async with session.post(
                upload_url, headers=headers, data=image
            ) as response:
                response.raise_for_status()
                return await response.text()

    @classmethod
    async def fetch_snlm0e(cls, session: ClientSession, cookies: Cookies):
        async with session.get(cls.url, cookies=cookies) as response:
            text = await response.text()
        match = re.search(r'SNlM0e\":\"(.*?)\"', text)
        if match:
            return match.group(1)