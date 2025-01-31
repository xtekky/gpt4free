from __future__ import annotations

import os
import json
import asyncio
import base64
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

from .base_provider import AbstractProvider, ProviderModelMixin
from .helper import format_prompt_max_length
from .openai.har_file import get_headers, get_har_files
from ..typing import CreateResult, Messages, ImagesType
from ..errors import MissingRequirementsError, NoValidHarFileError, MissingAuthError
from ..requests.raise_for_status import raise_for_status
from ..providers.response import BaseConversation, JsonConversation, RequestLogin, Parameters
from ..providers.asyncio import get_running_loop
from ..requests import get_nodriver
from ..image import ImageResponse, to_bytes, is_accepted_format
from .helper import get_last_user_message
from .. import debug

class Conversation(JsonConversation):
    conversation_id: str

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id

class Copilot(AbstractProvider, ProviderModelMixin):
    label = "Microsoft Copilot"
    url = "https://copilot.microsoft.com"
    
    working = True
    supports_stream = True
    
    default_model = "Copilot"
    models = [default_model]
    model_aliases = {
        "gpt-4": default_model,
        "gpt-4o": default_model,
    }

    websocket_url = "wss://copilot.microsoft.com/c/api/chat?api-version=2"
    conversation_url = f"{url}/c/api/conversations"

    _access_token: str = None
    _cookies: CookieJar = None

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        timeout: int = 900,
        prompt: str = None,
        images: ImagesType = None,
        conversation: BaseConversation = None,
        return_conversation: bool = False,
        api_key: str = None,
        **kwargs
    ) -> CreateResult:
        if not has_curl_cffi:
            raise MissingRequirementsError('Install or update "curl_cffi" package | pip install -U curl_cffi')

        websocket_url = cls.websocket_url
        headers = None
        if cls.needs_auth or images is not None:
            if api_key is not None:
                cls._access_token = api_key
            if cls._access_token is None:
                try:
                    cls._access_token, cls._cookies = readHAR(cls.url)
                except NoValidHarFileError as h:
                    debug.log(f"Copilot: {h}")
                    if has_nodriver:
                        login_url = os.environ.get("G4F_LOGIN_URL")
                        if login_url:
                            yield RequestLogin(cls.label, login_url)
                        get_running_loop(check_nested=True)
                        cls._access_token, cls._cookies = asyncio.run(get_access_token_and_cookies(cls.url, proxy))
                    else:
                        raise h
            websocket_url = f"{websocket_url}&accessToken={quote(cls._access_token)}"
            headers = {"authorization": f"Bearer {cls._access_token}"}

        with Session(
            timeout=timeout,
            proxy=proxy,
            impersonate="chrome",
            headers=headers,
            cookies=cls._cookies,
        ) as session:
            if cls._access_token is not None:
                cls._cookies = session.cookies.jar
            # if cls._access_token is None:
            #     try:
            #         url = "https://copilot.microsoft.com/cl/eus-sc/collect"
            #         headers = {
            #             "Accept": "application/x-clarity-gzip",
            #             "referrer": "https://copilot.microsoft.com/onboarding"
            #         }
            #         response = session.post(url, headers=headers, data=get_clarity())
            #         clarity_token = json.loads(response.text.split(" ", maxsplit=1)[-1])[0]["value"]
            #         debug.log(f"Copilot: Clarity Token: ...{clarity_token[-12:]}")
            #     except Exception as e:
            #         debug.log(f"Copilot: {e}")
            # else:
            #     clarity_token = None
            response = session.get("https://copilot.microsoft.com/c/api/user")
            if response.status_code == 401:
                raise MissingAuthError("Status 401: Invalid access token")
            raise_for_status(response)
            user = response.json().get('firstName')
            if user is None:
                cls._access_token = None
            debug.log(f"Copilot: User: {user or 'null'}")
            if conversation is None:
                response = session.post(cls.conversation_url)
                raise_for_status(response)
                conversation_id = response.json().get("id")
                conversation = Conversation(conversation_id)
                if return_conversation:
                    yield conversation
                if prompt is None:
                    prompt = format_prompt_max_length(messages, 10000)
                debug.log(f"Copilot: Created conversation: {conversation_id}")
            else:
                conversation_id = conversation.conversation_id
                if prompt is None:
                    prompt = get_last_user_message(messages)
                debug.log(f"Copilot: Use conversation: {conversation_id}")

            uploaded_images = []
            if images is not None:
                for image, _ in images:
                    data = to_bytes(image)
                    response = session.post(
                        "https://copilot.microsoft.com/c/api/attachments",
                        headers={"content-type": is_accepted_format(data)},
                        data=data
                    )
                    raise_for_status(response)
                    uploaded_images.append({"type":"image", "url": response.json().get("url")})
                    break

            wss = session.ws_connect(cls.websocket_url)
            # if clarity_token is not None:
            #     wss.send(json.dumps({
            #         "event": "challengeResponse",
            #         "token": clarity_token,
            #         "method":"clarity"
            #     }).encode(), CurlWsFlag.TEXT)
            wss.send(json.dumps({
                "event": "send",
                "conversationId": conversation_id,
                "content": [*uploaded_images, {
                    "type": "text",
                    "text": prompt,
                }],
                "mode": "chat"
            }).encode(), CurlWsFlag.TEXT)

            is_started = False
            msg = None
            image_prompt: str = None
            last_msg = None
            try:
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
                    elif msg.get("event") == "replaceText":
                        yield msg.get("text")
                    elif msg.get("event") == "error":
                        raise RuntimeError(f"Error: {msg}")
                    elif msg.get("event") not in ["received", "startMessage", "citation", "partCompleted"]:
                        debug.log(f"Copilot Message: {msg}")
                if not is_started:
                    raise RuntimeError(f"Invalid response: {last_msg}")
            finally:
                yield Parameters(**{"conversation": conversation.get_dict(), "user": user, "prompt": prompt})
                yield Parameters(**{"cookies": {c.name: c.value for c in session.cookies.jar}})

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

def get_clarity() -> bytes:
     #{"e":["0.7.58",5,7284,4779,"n59ae4ieqq","aln5en","1upufhz",1,0,0],"a":[[7323,12,65,217,324],[7344,12,65,214,329],[7385,12,65,211,334],[7407,12,65,210,337],[7428,12,65,209,338],[7461,12,65,209,339],[7497,12,65,209,339],[7531,12,65,208,340],[7545,12,65,208,342],[11654,13,65,208,342],[11728,14,65,208,342],[11728,9,65,208,342,17535,19455,0,0,0,"Annehmen",null,"52w7wqv1r.8ovjfyrpu",1],[7284,4,1,393,968,393,968,0,0,231,310,939,0],[12063,0,2,147,3,4,4,18,5,1,10,79,25,15],[12063,36,6,[11938,0]]]}
    body = base64.b64decode("H4sIAAAAAAAAA23RwU7DMAwG4HfJ2aqS2E5ibjxH1cMOnQYqYZvUTQPx7vyJRGGAemj01XWcP+9udg+j80MetDhSyrEISc5GrqrtZnmaTydHbrdUnSsWYT2u+8Obo0Ce/IQvaDBmjkwhUlKKIRNHmQgosqEArWPRDQMx90rxeUMPzB1j+UJvwNIxhTvsPcXyX1T+rizE4juK3mEEhpAUg/JvzW1/+U/tB1LATmhqotoiweMea50PLy2vui4LOY3XfD1dwnkor5fn/e18XBFgm6fHjSzZmCyV7d3aRByAEYextaTHEH3i5pgKGVP/s+DScE5PuLKIpW6FnCi1gY3Rbpqmj0/DI/+L7QEAAA==")
    return body
