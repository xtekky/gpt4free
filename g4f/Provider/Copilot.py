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

from .base_provider import AbstractProvider, BaseConversation
from .helper import format_prompt
from ..typing import CreateResult, Messages, ImageType
from ..errors import MissingRequirementsError
from ..requests.raise_for_status import raise_for_status
from ..providers.helper import format_cookies
from ..requests import get_nodriver
from ..image import ImageResponse, to_bytes, is_accepted_format
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
        web_search: bool = True,
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
            debug.log(f"Copilot: Access token: {access_token[:7]}...{access_token[-5:]}")
            debug.log(f"Copilot: Cookies: {';'.join([*cookies])}")
            websocket_url = f"{websocket_url}&accessToken={quote(access_token)}"
            headers = {"authorization": f"Bearer {access_token}", "cookie": format_cookies(cookies)}
    
        with Session(
            timeout=timeout,
            proxy=proxy,
            impersonate="chrome",
            headers=headers,
            cookies=cookies,
        ) as session:
            response = session.get("https://copilot.microsoft.com/c/api/user")
            raise_for_status(response)
            debug.log(f"Copilot: User: {response.json().get('firstName', 'null')}")
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
            image_prompt: str = None
            last_msg = None
            while True:
                try:
                    msg = wss.recv()[0]
                    msg = json.loads(msg)
                except:
                    break
                last_msg = msg
                if msg.get("event") == "appendText":
                    is_started = True
                    yield msg.get("text")
                elif msg.get("event") == "generatingImage":
                    image_prompt = msg.get("prompt")
                elif msg.get("event") == "imageGenerated":
                    yield ImageResponse(msg.get("url"), image_prompt, {"preview": msg.get("thumbnailUrl")})
                elif msg.get("event") == "done":
                    break
                elif msg.get("event") == "error":
                    raise RuntimeError(f"Error: {msg}")
                elif msg.get("event") not in ["received", "startMessage", "citation", "partCompleted"]:
                    debug.log(f"Copilot Message: {msg}")
            if not is_started:
                raise RuntimeError(f"Invalid response: {last_msg}")

    @classmethod
    async def get_access_token_and_cookies(cls, proxy: str = None):
        browser = await get_nodriver(proxy=proxy)
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
                await asyncio.sleep(1)
        cookies = {}
        for c in await page.send(nodriver.cdp.network.get_cookies([cls.url])):
            cookies[c.name] = c.value
        await page.close()
        return access_token, cookies