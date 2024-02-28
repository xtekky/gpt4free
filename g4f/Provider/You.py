from __future__ import annotations

import json
import base64
import uuid
from aiohttp import ClientSession, FormData, BaseConnector

from ..typing import AsyncResult, Messages, ImageType, Cookies
from .base_provider import AsyncGeneratorProvider
from ..providers.helper import get_connector, format_prompt
from ..image import to_bytes
from ..requests.defaults import DEFAULT_HEADERS

class You(AsyncGeneratorProvider):
    url = "https://you.com"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    _cookies = None
    _cookies_used = 0

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        image: ImageType = None,
        image_name: str = None,
        connector: BaseConnector = None,
        proxy: str = None,
        chat_mode: str = "default",
        **kwargs,
    ) -> AsyncResult:
        async with ClientSession(
            connector=get_connector(connector, proxy),
            headers=DEFAULT_HEADERS
        ) as client:
            if image:
                chat_mode = "agent"
            elif model == "gpt-4":
                chat_mode = model
            cookies = await cls.get_cookies(client) if chat_mode != "default" else None
            upload = json.dumps([await cls.upload_file(client, cookies, to_bytes(image), image_name)]) if image else ""
            #questions = [message["content"] for message in messages if message["role"] == "user"]
            # chat = [
            #     {"question": questions[idx-1], "answer": message["content"]}
            #     for idx, message in enumerate(messages)
            #     if message["role"] == "assistant"
            #     and idx < len(questions)
            # ]
            headers = {
                "Accept": "text/event-stream",
                "Referer": f"{cls.url}/search?fromSearchBar=true&tbm=youchat",
            }
            data = {
                "userFiles": upload,
                "q": format_prompt(messages),
                "domain": "youchat",
                "selectedChatMode": chat_mode,
                #"chat": json.dumps(chat),
            }
            params = {
                "userFiles": upload,
                "selectedChatMode": chat_mode,
            }
            async with (client.post if chat_mode == "default" else client.get)(
                f"{cls.url}/api/streamingSearch",
                data=data,
                params=params,
                headers=headers,
                cookies=cookies
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b'event: '):
                        event = line[7:-1].decode()
                    elif line.startswith(b'data: '):
                        if event in ["youChatUpdate", "youChatToken"]:
                            data = json.loads(line[6:-1])
                        if event == "youChatToken" and event in data:
                            yield data[event]
                        elif event == "youChatUpdate" and "t" in data:
                            yield data["t"]                         

    @classmethod
    async def upload_file(cls, client: ClientSession, cookies: Cookies, file: bytes, filename: str = None) -> dict:
        async with client.get(
            f"{cls.url}/api/get_nonce",
            cookies=cookies,
        ) as response:
            response.raise_for_status()
            upload_nonce = await response.text()
        data = FormData()
        data.add_field('file', file, filename=filename)
        async with client.post(
            f"{cls.url}/api/upload",
            data=data,
            headers={
                "X-Upload-Nonce": upload_nonce,
            },
            cookies=cookies
        ) as response:
            if not response.ok:
                raise RuntimeError(f"Response: {await response.text()}")
            result = await response.json()
        result["user_filename"] = filename
        result["size"] = len(file)
        return result

    @classmethod
    async def get_cookies(cls, client: ClientSession) -> Cookies:
        if not cls._cookies or cls._cookies_used >= 5:
            cls._cookies = await cls.create_cookies(client)
            cls._cookies_used = 0
        cls._cookies_used += 1
        return cls._cookies        

    @classmethod
    def get_sdk(cls) -> str:
        return base64.standard_b64encode(json.dumps({
            "event_id":f"event-id-{str(uuid.uuid4())}",
            "app_session_id":f"app-session-id-{str(uuid.uuid4())}",
            "persistent_id":f"persistent-id-{uuid.uuid4()}",
            "client_sent_at":"","timezone":"",
            "stytch_user_id":f"user-live-{uuid.uuid4()}",
            "stytch_session_id":f"session-live-{uuid.uuid4()}",
            "app":{"identifier":"you.com"},
            "sdk":{"identifier":"Stytch.js Javascript SDK","version":"3.3.0"
        }}).encode()).decode()

    def get_auth() -> str:
        auth_uuid = "507a52ad-7e69-496b-aee0-1c9863c7c8"
        auth_token = f"public-token-live-{auth_uuid}bb:public-token-live-{auth_uuid}19"
        auth = base64.standard_b64encode(auth_token.encode()).decode()
        return f"Basic {auth}"

    @classmethod
    async def create_cookies(cls, client: ClientSession) -> Cookies:
        user_uuid = str(uuid.uuid4())
        async with client.post(
            "https://web.stytch.com/sdk/v1/passwords",
            headers={
                "Authorization": cls.get_auth(),
                "X-SDK-Client": cls.get_sdk(),
                "X-SDK-Parent-Host": cls.url
            },
            json={
                "email": f"{user_uuid}@gmail.com",
                "password": f"{user_uuid}#{user_uuid}",
                "session_duration_minutes": 129600
            }
        ) as response:
            if not response.ok:
                raise RuntimeError(f"Response: {await response.text()}")
            session = (await response.json())["data"]
        return {
            "stytch_session": session["session_token"],
            'stytch_session_jwt': session["session_jwt"],
            'ydc_stytch_session': session["session_token"],
            'ydc_stytch_session_jwt': session["session_jwt"],
        }