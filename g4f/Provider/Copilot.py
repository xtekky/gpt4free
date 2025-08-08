from __future__ import annotations

import os
import json
import asyncio
import base64
from typing import AsyncIterator
from urllib.parse import quote

try:
    from curl_cffi.requests import AsyncSession
    from curl_cffi import CurlWsFlag
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False
try:
    import nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False

from .base_provider import AsyncAuthedProvider, ProviderModelMixin
from .helper import format_prompt_max_length
from .openai.har_file import get_headers, get_har_files
from ..typing import AsyncResult, Messages, MediaListType
from ..errors import MissingRequirementsError, NoValidHarFileError, MissingAuthError
from ..providers.response import *
from ..tools.media import merge_media
from ..requests import get_nodriver
from ..image import to_bytes, is_accepted_format
from .helper import get_last_user_message
from .. import debug

class Conversation(JsonConversation):
    conversation_id: str

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id

class Copilot(AsyncAuthedProvider, ProviderModelMixin):
    label = "Microsoft Copilot"
    url = "https://copilot.microsoft.com"
    
    working = True
    supports_stream = True
    active_by_default = True
    
    default_model = "Copilot"
    models = [default_model, "Think Deeper", "Smart (GPT-5)"]
    model_aliases = {
        "o1": "Think Deeper",
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "gpt-5": "GPT-5",
    }

    websocket_url = "wss://copilot.microsoft.com/c/api/chat?api-version=2"
    conversation_url = f"{url}/c/api/conversations"

    _access_token: str = None
    _cookies: dict = {}

    @classmethod
    async def on_auth_async(cls, **kwargs) -> AsyncIterator:
        yield AuthResult(
            api_key=cls._access_token,
            cookies=cls.cookies_to_dict()
        )

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        **kwargs
    ) -> AsyncResult:
        cls._access_token = getattr(auth_result, "api_key")
        cls._cookies = getattr(auth_result, "cookies")
        async for chunk in cls.create(model, messages, **kwargs):
            yield chunk
        auth_result.cookies = cls.cookies_to_dict()

    @classmethod
    def cookies_to_dict(cls):
        return cls._cookies if isinstance(cls._cookies, dict) else {c.name: c.value for c in cls._cookies}

    @classmethod
    async def create(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 30,
        prompt: str = None,
        media: MediaListType = None,
        conversation: BaseConversation = None,
        return_conversation: bool = True,
        api_key: str = None,
        **kwargs
    ) -> AsyncResult:
        if not has_curl_cffi:
            raise MissingRequirementsError('Install or update "curl_cffi" package | pip install -U curl_cffi')
        model = cls.get_model(model)
        websocket_url = cls.websocket_url
        headers = None
        if cls._access_token:
            if api_key is not None:
                cls._access_token = api_key
            if cls._access_token is None:
                try:
                    cls._access_token, cls._cookies = readHAR(cls.url)
                except NoValidHarFileError as h:
                    debug.log(f"Copilot: {h}")
                    if has_nodriver:
                        yield RequestLogin(cls.label, os.environ.get("G4F_LOGIN_URL", ""))
                        cls._access_token, cls._cookies = await get_access_token_and_cookies(cls.url, proxy)
                    else:
                        raise h
            websocket_url = f"{websocket_url}&accessToken={quote(cls._access_token)}"
            headers = {"authorization": f"Bearer {cls._access_token}"}

        async with AsyncSession(
            timeout=timeout,
            proxy=proxy,
            impersonate="chrome",
            headers=headers,
            cookies=cls._cookies,
        ) as session:
            if cls._access_token is not None:
                cls._cookies = session.cookies.jar if hasattr(session.cookies, "jar") else session.cookies
            response = await session.get("https://copilot.microsoft.com/c/api/user")
            if response.status_code == 401:
                raise MissingAuthError("Status 401: Invalid access token")
            response.raise_for_status()
            user = response.json().get('firstName')
            if user is None:
                if cls.needs_auth:
                    raise MissingAuthError("No user found, please login first")
                cls._access_token = None
            else:
                debug.log(f"Copilot: User: {user}")
            if conversation is None:
                response = await session.post(cls.conversation_url)
                response.raise_for_status()
                conversation_id = response.json().get("id")
                conversation = Conversation(conversation_id)
                if prompt is None:
                    prompt = format_prompt_max_length(messages, 10000)
                debug.log(f"Copilot: Created conversation: {conversation_id}")
            else:
                conversation_id = conversation.conversation_id
                if prompt is None:
                    prompt = get_last_user_message(messages)
                debug.log(f"Copilot: Use conversation: {conversation_id}")
            if return_conversation:
                yield conversation

            uploaded_images = []
            for media, _ in merge_media(media, messages):
                if not isinstance(media, str):
                    data = to_bytes(media)
                    response = await session.post(
                        "https://copilot.microsoft.com/c/api/attachments",
                        headers={
                            "content-type": is_accepted_format(data),
                            "content-length": str(len(data)),
                        },
                        data=data
                    )
                    response.raise_for_status()
                    media = response.json().get("url")
                uploaded_images.append({"type":"image", "url": media})

            wss = await session.ws_connect(cls.websocket_url, timeout=3)
            if "Think" in model:
                mode = "reasoning"
            elif model.startswith("gpt-5") or "GPT-5" in model:
                mode = "smart"
            else:
                mode = "chat"
            await wss.send(json.dumps({
                "event": "send",
                "conversationId": conversation_id,
                "content": [*uploaded_images, {
                    "type": "text",
                    "text": prompt,
                }],
                "mode": mode,
            }).encode(), CurlWsFlag.TEXT)

            done = False
            msg = None
            image_prompt: str = None
            last_msg = None
            sources = {}
            while not wss.closed:
                try:
                    msg_txt, _ = await asyncio.wait_for(wss.recv(), 3 if done else timeout)
                    msg = json.loads(msg_txt)
                except:
                    break
                last_msg = msg
                if msg.get("event") == "appendText":
                    yield msg.get("text")
                elif msg.get("event") == "generatingImage":
                    image_prompt = msg.get("prompt")
                elif msg.get("event") == "imageGenerated":
                    yield ImageResponse(msg.get("url"), image_prompt, {"preview": msg.get("thumbnailUrl")})
                elif msg.get("event") == "done":
                    yield FinishReason("stop")
                    done = True
                elif msg.get("event") == "suggestedFollowups":
                    yield SuggestedFollowups(msg.get("suggestions"))
                    break
                elif msg.get("event") == "replaceText":
                    yield msg.get("text")
                elif msg.get("event") == "titleUpdate":
                    yield TitleGeneration(msg.get("title"))
                elif msg.get("event") == "citation":
                    sources[msg.get("url")] = msg
                    yield SourceLink(list(sources.keys()).index(msg.get("url")), msg.get("url"))
                elif msg.get("event") == "partialImageGenerated":
                    mime_type = is_accepted_format(base64.b64decode(msg.get("content")[:12]))
                    yield ImagePreview(f"data:{mime_type};base64,{msg.get('content')}", image_prompt)
                elif msg.get("event") == "error":
                    raise RuntimeError(f"Error: {msg}")
                elif msg.get("event") not in ["received", "startMessage", "partCompleted", "connected"]:
                    debug.log(f"Copilot Message: {msg_txt[:100]}...")
            if not done:
                raise RuntimeError(f"Invalid response: {last_msg}")
            if sources:
                yield Sources(sources.values())
            if not wss.closed:
                await wss.close()

async def get_access_token_and_cookies(url: str, proxy: str = None, target: str = "ChatAI",):
    browser, stop_browser = await get_nodriver(proxy=proxy, user_data_dir="copilot")
    try:
        page = await browser.get(url)
        access_token = None
        while access_token is None:
            access_token = await page.evaluate("""
                (() => {
                    for (var i = 0; i < localStorage.length; i++) {
                        try {
                            item = JSON.parse(localStorage.getItem(localStorage.key(i)));
                            if (item.credentialType == "AccessToken" 
                                && item.expiresOn > Math.floor(Date.now() / 1000)
                                && item.target.includes("target")) {
                                return item.secret;
                            }
                        } catch(e) {}
                    }
                })()
            """.replace('"target"', json.dumps(target)))
            if access_token is None:
                await asyncio.sleep(1)
        cookies = {}
        for c in await page.send(nodriver.cdp.network.get_cookies([url])):
            cookies[c.name] = c.value
        await page.close()
        return access_token, cookies
    finally:
        stop_browser()

def readHAR(url: str):
    api_key = None
    cookies = None
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
                    if v['request']['cookies']:
                        cookies = {c['name']: c['value'] for c in v['request']['cookies']}
    if api_key is None:
        raise NoValidHarFileError("No access token found in .har files")

    return api_key, cookies