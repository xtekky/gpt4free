from __future__ import annotations

import os
import json
import asyncio
import base64
import random
import string
import urllib.parse
from typing import AsyncIterator
from urllib.parse import quote

try:
    from curl_cffi.requests import AsyncSession
    from curl_cffi import CurlWsFlag, CurlMime
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False
try:
    import nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False

from .base_provider import AsyncAuthedProvider, ProviderModelMixin
from .openai.har_file import get_headers, get_har_files
from ..typing import AsyncResult, Messages, MediaListType
from ..errors import MissingRequirementsError, NoValidHarFileError, MissingAuthError
from ..providers.response import *
from ..tools.media import merge_media
from ..requests import get_nodriver, DEFAULT_HEADERS
from ..image import to_bytes, is_accepted_format
from .helper import get_last_user_message
from ..files import get_bucket_dir
from ..tools.files import read_bucket
from ..cookies import get_cookies
from pathlib import Path
from .. import debug

class Conversation(JsonConversation):
    conversation_id: str

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id

def extract_bucket_items(messages: Messages) -> list[dict]:
    """Extract bucket items from messages content."""
    bucket_items = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            for content_item in message["content"]:
                if isinstance(content_item, dict) and "bucket_id" in content_item and "name" not in content_item:
                    bucket_items.append(content_item)
        if message.get("role") == "assistant":
            bucket_items = []
    return bucket_items

def random_hex(length):
    return ''.join(random.choices('0123456789ABCDEF', k=length))

def random_base64(length):
    chars = string.ascii_letters + string.digits + '+/='
    return ''.join(random.choices(chars, k=length))

def get_fake_cookie():
    return {
        "_C_ETH": "1",
        "_C_Auth": "",
        "MUID": random_hex(32),
        "MUIDB": random_hex(32),
        "_EDGE_S": f"F=1&SID={random_hex(32)}",
        "_EDGE_V": "1",
        "ak_bmsc": f"{random_hex(32)}~{'0'*48}~{urllib.parse.quote(random_base64(300))}"
    }

class Copilot(AsyncAuthedProvider, ProviderModelMixin):
    label = "Microsoft Copilot"
    url = "https://copilot.microsoft.com"
    cookie_domain = ".microsoft.com"
    anon_cookie_name = "__Host-copilot-anon"
    
    working = True
    use_nodriver = has_nodriver
    active_by_default = True
    
    default_model = "Copilot"
    models = [default_model, "Think Deeper", "Smart (GPT-5)", "Study"]
    model_aliases = {
        "o1": "Think Deeper",
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "gpt-5": "GPT-5",
        "study": "Study",
    }

    websocket_url = "wss://copilot.microsoft.com/c/api/chat?api-version=2"
    conversation_url = f"{url}/c/api/conversations"

    @classmethod
    async def on_auth_async(cls, cookies: dict = None, proxy: str = None, **kwargs) -> AsyncIterator:
        if cookies is None:
            cookies = get_fake_cookie() or get_cookies(cls.cookie_domain, False, cache_result=False)
        access_token = None
        useridentitytype = None
        if cls.needs_auth or cls.anon_cookie_name not in cookies:
            try:
                access_token, useridentitytype, cookies = readHAR(cls.url)
            except NoValidHarFileError as h:
                debug.log(f"Copilot: {h}")
                if has_nodriver:
                    yield RequestLogin(cls.label, os.environ.get("G4F_LOGIN_URL", ""))
                    access_token, useridentitytype, cookies = await get_access_token_and_cookies(cls.url, proxy, cls.needs_auth)
                else:
                    raise h
        yield AuthResult(
            access_token=access_token,
            useridentitytype=useridentitytype,
            cookies=cookies
        )

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        proxy: str = None,
        timeout: int = 30,
        prompt: str = None,
        media: MediaListType = None,
        conversation: BaseConversation = None,
        return_conversation: bool = True,
        **kwargs
    ) -> AsyncResult:
        if not has_curl_cffi:
            raise MissingRequirementsError('Install or update "curl_cffi" package | pip install -U curl_cffi')
        model = cls.get_model(model)
        websocket_url = cls.websocket_url
        headers = DEFAULT_HEADERS.copy()
        headers["origin"] = cls.url
        headers["referer"] = cls.url + "/"
        if getattr(auth_result, "access_token", None):
            websocket_url = f"{websocket_url}&accessToken={quote(auth_result.access_token)}" + (f"&X-UserIdentityType={quote(auth_result.useridentitytype)}" if getattr(auth_result, "useridentitytype", None) else "")
            headers["authorization"] = f"Bearer {auth_result.access_token}"

        async with AsyncSession(
            timeout=timeout,
            proxy=proxy,
            impersonate="chrome",
            headers=headers,
            cookies=auth_result.cookies
        ) as session:
            if conversation is None:
                # har_file = os.path.join(os.path.dirname(__file__), "copilot", "copilot.microsoft.com.har")
                # with open(har_file, "r") as f:
                #     har_entries = json.load(f).get("log", {}).get("entries", [])
                # conversationId = ""
                # for har_entry in har_entries:
                #     if har_entry.get("request"):
                #         if "/c/api/" in har_entry.get("request").get("url", ""):
                #             try:
                #                 response = await getattr(session, har_entry.get("request").get("method").lower())(
                #                     har_entry.get("request").get("url", "").replace("cvqBJw7kyPAp1RoMTmzC6", conversationId),
                #                     data=har_entry.get("request").get("postData", {}).get("text"),
                #                     headers={header["name"]: header["value"] for header in har_entry.get("request").get("headers")}
                #                 )
                #                 response.raise_for_status()
                #                 if response.headers.get("content-type", "").startswith("application/json"):
                #                     conversationId = response.json().get("currentConversationId", conversationId)
                #             except Exception as e:
                #                 debug.log(f"Copilot: Failed request to {har_entry.get('request').get('url', '')}: {e}")
                data = {
                    "timeZone": "America/Los_Angeles",
                    "startNewConversation": True,
                    "teenSupportEnabled": True,
                    "correctPersonalizationSetting": True,
                    "performUserMerge": True,
                    "deferredDataUseCapable": True
                }
                response = await session.post(
                    "https://copilot.microsoft.com/c/api/start",
                    headers={
                        "content-type": "application/json",
                        **({"x-useridentitytype": auth_result.useridentitytype} if getattr(auth_result, "useridentitytype", None) else {}),
                        **(headers or {})
                    },
                    json=data
                )
                if response.status_code == 401:
                    raise MissingAuthError("Status 401: Invalid session")
                response.raise_for_status()
                debug.log(f"Copilot: Update cookies: [{', '.join(key for key in response.cookies)}]")
                auth_result.cookies.update({key: value for key, value in response.cookies.items()})
                if not getattr(auth_result, "access_token", None) and not cls.needs_auth and cls.anon_cookie_name not in auth_result.cookies:
                    raise MissingAuthError(f"Missing cookie: {cls.anon_cookie_name}")
                conversation = Conversation(response.json().get("currentConversationId"))
                debug.log(f"Copilot: Created conversation: {conversation.conversation_id}")
            else:
                debug.log(f"Copilot: Use conversation: {conversation.conversation_id}")

            # response = await session.get("https://copilot.microsoft.com/c/api/user?api-version=4", headers={"x-useridentitytype": useridentitytype} if cls._access_token else {})
            # if response.status_code == 401:
            #     raise MissingAuthError("Status 401: Invalid session")
            # response.raise_for_status()
            # print(response.json())
            # user = response.json().get('firstName')
            # if user is None:
            #     if cls.needs_auth:
            #         raise MissingAuthError("No user found, please login first")
            #     cls._access_token = None
            # else:
            #     debug.log(f"Copilot: User: {user}")

            uploaded_attachments = []
            if auth_result.access_token:
                # Upload regular media (images)
                for media, _ in merge_media(media, messages):
                    if not isinstance(media, str):
                        data = to_bytes(media)
                        response = await session.post(
                            "https://copilot.microsoft.com/c/api/attachments",
                            headers={
                                "content-type": is_accepted_format(data),
                                "content-length": str(len(data)),
                                **({"x-useridentitytype": auth_result.useridentitytype} if getattr(auth_result, "useridentitytype", None) else {})
                            },
                            data=data
                        )
                        response.raise_for_status()
                        media = response.json().get("url")
                    uploaded_attachments.append({"type":"image", "url": media})

                # Upload bucket files
                bucket_items = extract_bucket_items(messages)
                for item in bucket_items:
                    try:
                        # Handle plain text content from bucket
                        bucket_path = Path(get_bucket_dir(item["bucket_id"]))
                        for text_chunk in read_bucket(bucket_path):
                            if text_chunk.strip():
                                # Upload plain text as a text file
                                text_data = text_chunk.encode('utf-8')
                                data = CurlMime()
                                data.addpart("file", filename=f"bucket_{item['bucket_id']}.txt", content_type="text/plain", data=text_data)
                                response = await session.post(
                                    "https://copilot.microsoft.com/c/api/attachments",
                                    multipart=data,
                                    headers={"x-useridentitytype": auth_result.useridentitytype} if getattr(auth_result, "useridentitytype", None) else {}
                                )
                                response.raise_for_status()
                                data = response.json()
                                uploaded_attachments.append({"type": "document", "attachmentId": data.get("id")})
                                debug.log(f"Copilot: Uploaded bucket text content: {item['bucket_id']}")
                            else:
                                debug.log(f"Copilot: No text content found in bucket: {item['bucket_id']}")
                    except Exception as e:
                        debug.log(f"Copilot: Failed to upload bucket item: {item}")
                        debug.error(e)

            if prompt is None:
                prompt = get_last_user_message(messages, False)

            wss = await session.ws_connect(websocket_url, timeout=3)
            if "Think" in model:
                mode = "reasoning"
            elif model.startswith("gpt-5") or "GPT-5" in model:
                mode = "smart"
            else:
                mode = "chat"
            await wss.send(json.dumps({
                "event": "send",
                "conversationId": conversation.conversation_id,
                "content": [*uploaded_attachments, {
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
                    msg_txt, _ = await asyncio.wait_for(wss.recv(), 1 if done else timeout)
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
                elif msg.get("event") == "chainOfThought":
                    yield Reasoning(msg.get("text"))
                elif msg.get("event") == "error":
                    raise RuntimeError(f"Error: {msg}")
                elif msg.get("event") not in ["received", "startMessage", "partCompleted", "connected"]:
                    debug.log(f"Copilot Message: {msg_txt[:100]}...")
            if not done:
                raise MissingAuthError(f"Invalid response: {last_msg}")
            if return_conversation:
                yield conversation
            if sources:
                yield Sources(sources.values())
            if not wss.closed:
                await wss.close()

async def get_access_token_and_cookies(url: str, proxy: str = None, needs_auth: bool = False):
    browser, stop_browser = await get_nodriver(proxy=proxy)
    try:
        page = await browser.get(url)
        access_token = None
        useridentitytype = None
        while access_token is None:
            for _ in range(2):
                await asyncio.sleep(3)
                access_token = await page.evaluate("""
                    (() => {
                        for (var i = 0; i < localStorage.length; i++) {
                            try {
                                const key = localStorage.key(i);
                                const item = JSON.parse(localStorage.getItem(key));
                                if (item?.body?.access_token) {
                                    return ["" + item?.body?.access_token, "google"];
                                } else if (key.includes("chatai")) {
                                    return "" + item.secret;
                                }
                            } catch(e) {}
                        }
                    })()
                """)
                if access_token is None:
                    await asyncio.sleep(1)
                    continue
                if isinstance(access_token, list):
                    access_token, useridentitytype = access_token
                access_token = access_token.get("value") if isinstance(access_token, dict) else access_token
                useridentitytype = useridentitytype.get("value") if isinstance(useridentitytype, dict) else None
                debug.log(f"Got access token: {access_token[:10]}..., useridentitytype: {useridentitytype}")
                break
            if not needs_auth:
                debug.log("No access token found, but authentication not required.")
                break
        if not needs_auth:
            textarea = await page.select("textarea")
            if textarea is not None:
                debug.log("Filling textarea to generate anon cookie.")
                await textarea.send_keys("Hello")
                await asyncio.sleep(1)
                button = await page.select("[data-testid=\"submit-button\"]")
                if button:
                    debug.log("Clicking submit button to generate anon cookie.")
                    await button.click()
                    turnstile = await page.select('#cf-turnstile')
                    if turnstile:
                        debug.log("Found Element: 'cf-turnstile'")
                        await asyncio.sleep(3)
                        await click_trunstile(page)
        cookies = {}
        while not access_token and Copilot.anon_cookie_name not in cookies:
            await asyncio.sleep(2)
            cookies = {c.name: c.value for c in await page.send(nodriver.cdp.network.get_cookies([url]))}
            if not needs_auth and Copilot.anon_cookie_name in cookies:
                break
            elif needs_auth and next(filter(lambda x: "auth0" in x, cookies), None):
                break
        stop_browser()
        return access_token, useridentitytype, cookies
    finally:
        stop_browser()

def readHAR(url: str):
    api_key = None
    useridentitytype = None
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
                    if "x-useridentitytype" in v_headers:
                        useridentitytype = v_headers["x-useridentitytype"]
                    if v['request']['cookies']:
                        cookies = {c['name']: c['value'] for c in v['request']['cookies']}
    if not cookies:
        raise NoValidHarFileError("No session found in .har files")

    return api_key, useridentitytype, cookies

if has_nodriver:
    async def click_trunstile(page: nodriver.Tab, element='document.getElementById("cf-turnstile")'):
        for _ in range(3):
            size = None
            for idx in range(15):
                size = await page.js_dumps(f'{element}?.getBoundingClientRect()||{{}}')
                debug.log(f"Found size: {size.get('x'), size.get('y')}")
                if "x" not in size:
                    break
                await page.flash_point(size.get("x") + idx * 3, size.get("y") + idx * 3)
                await page.mouse_click(size.get("x") + idx * 3, size.get("y") + idx * 3)
                await asyncio.sleep(2)
            if "x" not in size:
                break
        debug.log("Finished clicking trunstile.")