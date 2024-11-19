from __future__ import annotations

import json
import asyncio
from http.cookiejar import CookieJar
from urllib.parse import quote

try:
    from curl_cffi.requests import Session, CurlWsFlag
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False
try:
    import nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False
try:
    from platformdirs import user_config_dir
    has_platformdirs = True
except ImportError:
    has_platformdirs = False

from .base_provider import AbstractProvider, BaseConversation
from .helper import format_prompt
from ..typing import CreateResult, Messages, ImageType
from ..errors import MissingRequirementsError
from ..requests.raise_for_status import raise_for_status
from ..image import to_bytes, is_accepted_format
from .. import debug

class Conversation(BaseConversation):
    conversation_id: str
    cookie_jar: CookieJar
    access_token: str

    def __init__(self, conversation_id: str, cookie_jar: CookieJar, access_token: str = None):
        self.conversation_id = conversation_id
        self.cookie_jar = cookie_jar
        self.access_token = access_token

class Copilot(AbstractProvider):
    label = "Microsoft Copilot"
    url = "https://copilot.microsoft.com"
    working = True
    supports_stream = True
    default_model = "Copilot"

    websocket_url = "wss://copilot.microsoft.com/c/api/chat?api-version=2"
    conversation_url = f"{url}/c/api/conversations"

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        timeout: int = 900,
        image: ImageType = None,
        conversation: Conversation = None,
        return_conversation: bool = False,
        **kwargs
    ) -> CreateResult:
        if not has_curl_cffi:
            raise MissingRequirementsError('Install or update "curl_cffi" package | pip install -U curl_cffi')

        websocket_url = cls.websocket_url
        access_token = None
        headers = None
        cookies = conversation.cookie_jar if conversation is not None else None
        if cls.needs_auth or image is not None:
            if conversation is None or conversation.access_token is None:
                access_token, cookies = asyncio.run(cls.get_access_token_and_cookies(proxy))
            else:
                access_token = conversation.access_token
            websocket_url = f"{websocket_url}&acessToken={quote(access_token)}"
            headers = {"Authorization": f"Bearer {access_token}"}
    
        with Session(
            timeout=timeout,
            proxy=proxy,
            impersonate="chrome",
            headers=headers,
            cookies=cookies
        ) as session:
            response = session.get(f"{cls.url}/")
            raise_for_status(response)
            if conversation is None:
                response = session.post(cls.conversation_url)
                raise_for_status(response)
                conversation_id = response.json().get("id")
                if return_conversation:
                    yield Conversation(conversation_id, session.cookies.jar, access_token)
                prompt = format_prompt(messages)
                debug.log(f"Copilot: Created conversation: {conversation_id}")
            else:
                conversation_id = conversation.conversation_id
                prompt = messages[-1]["content"]
                debug.log(f"Copilot: Use conversation: {conversation_id}")

            images = []
            if image is not None:
                data = to_bytes(image)
                response = session.post(
                    "https://copilot.microsoft.com/c/api/attachments",
                    headers={"content-type": is_accepted_format(data)},
                    data=data
                )
                raise_for_status(response)
                images.append({"type":"image", "url": response.json().get("url")})

            wss = session.ws_connect(cls.websocket_url)
            wss.send(json.dumps({
                "event": "send",
                "conversationId": conversation_id,
                "content": [*images, {
                    "type": "text",
                    "text": prompt,
                }],
                "mode": "chat"
            }).encode(), CurlWsFlag.TEXT)

            is_started = False
            msg = None
            while True:
                try:
                    msg = wss.recv()[0]
                    msg = json.loads(msg)
                except:
                    break
                if msg.get("event") == "appendText":
                    yield msg.get("text")
                elif msg.get("event") in ["done", "partCompleted"]:
                    break
            if not is_started:
                raise RuntimeError(f"Last message: {msg}")

    @classmethod
    async def get_access_token_and_cookies(cls, proxy: str = None):
        if not has_nodriver:
            raise MissingRequirementsError('Install "nodriver" package | pip install -U nodriver')
        user_data_dir = user_config_dir("g4f-nodriver") if has_platformdirs else None
        debug.log(f"Copilot: Open nodriver with user_dir: {user_data_dir}")
        browser = await nodriver.start(
            user_data_dir=user_data_dir,
            browser_args=None if proxy is None else [f"--proxy-server={proxy}"],
        )
        page = await browser.get(cls.url)
        access_token = None
        while access_token is None:
            access_token = await page.evaluate("""
                (() => {
                    for (var i = 0; i < localStorage.length; i++) {
                        try {
                            item = JSON.parse(localStorage.getItem(localStorage.key(i)));
                            if (item.credentialType == "AccessToken") {
                                return item.secret;
                            }
                        } catch(e) {}
                    }
                })()
            """)
            if access_token is None:
                asyncio.sleep(1)
        cookies = {}
        for c in await page.send(nodriver.cdp.network.get_cookies([cls.url])):
            cookies[c.name] = c.value
        await page.close()
        return access_token, cookies