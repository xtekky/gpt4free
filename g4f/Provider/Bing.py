from __future__ import annotations

import random
import json
import uuid
import time
import asyncio
from urllib import parse
from datetime import datetime, date

from ..typing import AsyncResult, Messages, ImageType, Cookies
from ..image import ImageRequest
from ..errors import ResponseError, ResponseStatusError, RateLimitError
from ..requests import DEFAULT_HEADERS
from ..requests.aiohttp import StreamSession
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_random_hex
from .bing.upload_image import upload_image
from .bing.conversation import Conversation, create_conversation, delete_conversation
from .BingCreateImages import BingCreateImages
from .. import debug

class Tones:
    """
    Defines the different tone options for the Bing provider.
    """
    creative = "Creative"
    balanced = "Balanced"
    precise = "Precise"
    copilot = "Copilot"

class Bing(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Bing provider for generating responses using the Bing API.
    """
    label = "Microsoft Copilot in Bing"
    url = "https://bing.com/chat"
    working = True
    supports_message_history = True
    supports_gpt_4 = True
    default_model = "Balanced"
    default_vision_model = "gpt-4-vision"
    models = [getattr(Tones, key) for key in Tones.__dict__ if not key.startswith("__")]

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 900,
        api_key: str = None,
        cookies: Cookies = None,
        tone: str = None,
        image: ImageType = None,
        web_search: bool = False,
        context: str = None,
        **kwargs
    ) -> AsyncResult:
        """
        Creates an asynchronous generator for producing responses from Bing.

        :param model: The model to use.
        :param messages: Messages to process.
        :param proxy: Proxy to use for requests.
        :param timeout: Timeout for requests.
        :param cookies: Cookies for the session.
        :param tone: The tone of the response.
        :param image: The image type to be used.
        :param web_search: Flag to enable or disable web search.
        :return: An asynchronous result object.
        """
        prompt = messages[-1]["content"]
        if context is None:
            context = create_context(messages[:-1]) if len(messages) > 1 else None
        if tone is None:
            tone = tone if model.startswith("gpt-4") else model
        tone = cls.get_model("" if tone is None else tone)
        gpt4_turbo = True if model.startswith("gpt-4-turbo") else False

        return stream_generate(
            prompt, tone, image, context, cookies, api_key,
            proxy, web_search, gpt4_turbo, timeout,
            **kwargs
        )

def create_context(messages: Messages) -> str:
    """
    Creates a context string from a list of messages.

    :param messages: A list of message dictionaries.
    :return: A string representing the context created from the messages.
    """
    return "".join(
        f"[{message['role']}]" + ("(#message)"
        if message['role'] != "system"
        else "(#additional_instructions)") + f"\n{message['content']}"
        for message in messages
    ) + "\n\n"

def get_ip_address() -> str:
    return f"13.{random.randint(104, 107)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def get_default_cookies():
    #muid = get_random_hex().upper()
    sid = get_random_hex().upper()
    guid = get_random_hex().upper()
    isodate = date.today().isoformat()
    timestamp = int(time.time())
    zdate = "0001-01-01T00:00:00.0000000"
    return {
        "_C_Auth": "",
        #"MUID": muid,
        #"MUIDB":  muid,
        "_EDGE_S": f"F=1&SID={sid}",
        "_EDGE_V": "1",
        "SRCHD": "AF=hpcodx",
        "SRCHUID": f"V=2&GUID={guid}&dmnchg=1",
        "_RwBf": (
            f"r=0&ilt=1&ihpd=0&ispd=0&rc=3&rb=0&gb=0&rg=200&pc=0&mtu=0&rbb=0&g=0&cid="
            f"&clo=0&v=1&l={isodate}&lft={zdate}&aof=0&ard={zdate}"
            f"&rwdbt={zdate}&rwflt={zdate}&o=2&p=&c=&t=0&s={zdate}"
            f"&ts={isodate}&rwred=0&wls=&wlb="
            "&wle=&ccp=&cpt=&lka=0&lkt=0&aad=0&TH="
        ),
        '_Rwho': f'u=d&ts={isodate}',
        "_SS": f"SID={sid}&R=3&RB=0&GB=0&RG=200&RP=0",
        "SRCHUSR": f"DOB={date.today().strftime('%Y%m%d')}&T={timestamp}",
        "SRCHHPGUSR": f"HV={int(time.time())}",
        "BCP": "AD=1&AL=1&SM=1",
        "ipv6": f"hit={timestamp}",
        '_C_ETH' : '1',
    }

async def create_headers(cookies: Cookies = None, api_key: str = None) -> dict:
    if cookies is None:
        # import nodriver as uc
        # browser = await uc.start(headless=False)
        # page = await browser.get(Defaults.home)
        # await asyncio.sleep(10)
        # cookies = {}
        # for c in await page.browser.cookies.get_all():
        #     if c.domain.endswith(".bing.com"):
        #         cookies[c.name] = c.value
        # user_agent = await page.evaluate("window.navigator.userAgent")
        # await page.close()
        cookies = get_default_cookies()
    if api_key is not None:
        cookies["_U"] = api_key
    headers = Defaults.headers.copy()
    headers["cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
    return headers

class Defaults:
    """
    Default settings and configurations for the Bing provider.
    """
    delimiter = "\x1e"

    # List of allowed message types for Bing responses
    allowedMessageTypes = [
        "ActionRequest","Chat",
        "ConfirmationCard", "Context",
        "InternalSearchQuery", #"InternalSearchResult",
        #"Disengaged", "InternalLoaderMessage",
        "Progress", "RenderCardRequest",
        "RenderContentRequest", "AdsQuery",
        "SemanticSerp", "GenerateContentQuery",
        "SearchQuery", "GeneratedCode",
        "InternalTasksMessage"
    ]

    sliceIds = {
        "balanced": [
            "supllmnfe","archnewtf",
            "stpstream", "stpsig", "vnextvoicecf", "scmcbase", "cmcpupsalltf", "sydtransctrl",
            "thdnsrch", "220dcl1s0", "0215wcrwips0", "0305hrthrots0", "0130gpt4t",
            "bingfc", "0225unsticky1", "0228scss0",
            "defquerycf", "defcontrol", "3022tphpv"
        ],
        "creative": [
            "bgstream", "fltltst2c",
            "stpstream", "stpsig", "vnextvoicecf", "cmcpupsalltf", "sydtransctrl",
            "0301techgnd", "220dcl1bt15", "0215wcrwip", "0305hrthrot", "0130gpt4t",
            "bingfccf", "0225unsticky1", "0228scss0",
            "3022tpvs0"
        ],
        "precise": [
            "bgstream", "fltltst2c",
            "stpstream", "stpsig", "vnextvoicecf", "cmcpupsalltf", "sydtransctrl",
            "0301techgnd", "220dcl1bt15", "0215wcrwip", "0305hrthrot", "0130gpt4t",
            "bingfccf", "0225unsticky1", "0228scss0",
            "defquerycf", "3022tpvs0"
        ],
        "copilot": []
    }

    optionsSets = {
        "balanced": {
            "default": [
                "nlu_direct_response_filter", "deepleo",
                "disable_emoji_spoken_text", "responsible_ai_policy_235",
                "enablemm", "dv3sugg", "autosave",
                "iyxapbing", "iycapbing",
                "galileo", "saharagenconv5", "gldcl1p",
                "gpt4tmncnp"
            ],
            "nosearch": [
                "nlu_direct_response_filter", "deepleo",
                "disable_emoji_spoken_text", "responsible_ai_policy_235",
                "enablemm", "dv3sugg", "autosave",
                "iyxapbing", "iycapbing",
                "galileo", "sunoupsell", "base64filter", "uprv4p1upd",
                "hourthrot", "noctprf", "gndlogcf", "nosearchall"
            ]
        },
        "creative": {
            "default": [
                "nlu_direct_response_filter", "deepleo",
                "disable_emoji_spoken_text", "responsible_ai_policy_235",
                "enablemm", "dv3sugg",
                "iyxapbing", "iycapbing",
                "h3imaginative", "techinstgnd", "hourthrot", "clgalileo", "gencontentv3",
                "gpt4tmncnp"
            ],
            "nosearch": [
                "nlu_direct_response_filter", "deepleo",
                "disable_emoji_spoken_text", "responsible_ai_policy_235",
                "enablemm", "dv3sugg", "autosave",
                "iyxapbing", "iycapbing",
                "h3imaginative", "sunoupsell", "base64filter", "uprv4p1upd",
                "hourthrot", "noctprf", "gndlogcf", "nosearchall",
                "clgalileo", "nocache", "up4rp14bstcst"
            ]
        },
        "precise": {
            "default": [
                "nlu_direct_response_filter", "deepleo",
                "disable_emoji_spoken_text", "responsible_ai_policy_235",
                "enablemm", "dv3sugg",
                "iyxapbing", "iycapbing",
                "h3precise", "techinstgnd", "hourthrot", "techinstgnd", "hourthrot",
                "clgalileo", "gencontentv3"
            ],
            "nosearch": [
                "nlu_direct_response_filter", "deepleo",
                "disable_emoji_spoken_text", "responsible_ai_policy_235",
                "enablemm", "dv3sugg", "autosave",
                "iyxapbing", "iycapbing",
                "h3precise", "sunoupsell", "base64filter", "uprv4p1upd",
                "hourthrot", "noctprf", "gndlogcf", "nosearchall",
                "clgalileo", "nocache", "up4rp14bstcst"
            ]
        },
        "copilot": [
            "nlu_direct_response_filter", "deepleo",
            "disable_emoji_spoken_text", "responsible_ai_policy_235",
            "enablemm", "dv3sugg",
            "iyxapbing", "iycapbing",
            "h3precise", "clgalileo", "gencontentv3", "prjupy"
        ],
    }

    # Default location settings
    location = {
        "locale": "en-US", "market": "en-US", "region": "US",
        "location":"lat:34.0536909;long:-118.242766;re=1000m;",
        "locationHints": [{
            "country": "United States", "state": "California", "city": "Los Angeles",
            "timezoneoffset": 8, "countryConfidence": 8,
            "Center": {"Latitude": 34.0536909, "Longitude": -118.242766},
            "RegionType": 2, "SourceType": 1
        }],
    }

    # Default headers for requests
    home = "https://www.bing.com/chat?q=Microsoft+Copilot&FORM=hpcodx"
    headers = {
        **DEFAULT_HEADERS,
        "accept": "application/json",
        "referer": home,
        "x-ms-client-request-id": str(uuid.uuid4()),
        "x-ms-useragent": "azsdk-js-api-client-factory/1.0.0-beta.1 core-rest-pipeline/1.15.1 OS/Windows",
    }

def format_message(msg: dict) -> str:
    """
    Formats a message dictionary into a JSON string with a delimiter.

    :param msg: The message dictionary to format.
    :return: A formatted string representation of the message.
    """
    return json.dumps(msg, ensure_ascii=False) + Defaults.delimiter

def create_message(
    conversation: Conversation,
    prompt: str,
    tone: str,
    context: str = None,
    image_request: ImageRequest = None,
    web_search: bool = False,
    gpt4_turbo: bool = False,
    new_conversation: bool = True
) -> str:
    """
    Creates a message for the Bing API with specified parameters.

    :param conversation: The current conversation object.
    :param prompt: The user's input prompt.
    :param tone: The desired tone for the response.
    :param context: Additional context for the prompt.
    :param image_request: The image request with the url.
    :param web_search: Flag to enable web search.
    :param gpt4_turbo: Flag to enable GPT-4 Turbo.
    :return: A formatted string message for the Bing API.
    """

    options_sets = Defaults.optionsSets[tone.lower()]
    if not web_search and "nosearch" in options_sets:
        options_sets = options_sets["nosearch"]
    elif "default" in options_sets:
        options_sets = options_sets["default"]
    options_sets = options_sets.copy()
    if gpt4_turbo:
        options_sets.append("dlgpt4t")

    request_id = str(uuid.uuid4())
    struct = {
        "arguments":[{
            "source": "cib",
            "optionsSets": options_sets,
            "allowedMessageTypes": Defaults.allowedMessageTypes,
            "sliceIds": Defaults.sliceIds[tone.lower()],
            "verbosity": "verbose",
            "scenario": "CopilotMicrosoftCom" if tone == Tones.copilot else "SERP",
            "plugins": [{"id": "c310c353-b9f0-4d76-ab0d-1dd5e979cf68", "category": 1}] if web_search else [],
            "traceId": get_random_hex(40),
            "conversationHistoryOptionsSets": ["autosave","savemem","uprofupd","uprofgen"],
            "gptId": "copilot",
            "isStartOfSession": new_conversation,
            "requestId": request_id,
            "message":{
                **Defaults.location,
                "userIpAddress": get_ip_address(),
                "timestamp": datetime.now().isoformat(),
                "author": "user",
                "inputMethod": "Keyboard",
                "text": prompt,
                "messageType": "Chat",
                "requestId": request_id,
                "messageId": request_id
            },
            "tone": "Balanced" if tone == Tones.copilot else tone,
            "spokenTextMode": "None",
            "conversationId": conversation.conversationId,
            "participant": {"id": conversation.clientId}
        }],
        "invocationId": "0",
        "target": "chat",
        "type": 4
    }

    if image_request and image_request.get('imageUrl') and image_request.get('originalImageUrl'):
        struct['arguments'][0]['message']['originalImageUrl'] = image_request.get('originalImageUrl')
        struct['arguments'][0]['message']['imageUrl'] = image_request.get('imageUrl')
        struct['arguments'][0]['experienceType'] = None
        struct['arguments'][0]['attachedFileInfo'] = {"fileName": None, "fileType": None}

    if context:
        struct['arguments'][0]['previousMessages'] = [{
            "author": "user",
            "description": context,
            "contextType": "ClientApp",
            "messageType": "Context",
            "messageId": "discover-web--page-ping-mriduna-----"
        }]

    return format_message(struct)

async def stream_generate(
    prompt: str,
    tone: str,
    image: ImageType = None,
    context: str = None,
    cookies: dict = None,
    api_key: str = None,
    proxy: str = None,
    web_search: bool = False,
    gpt4_turbo: bool = False,
    timeout: int = 900,
    conversation: Conversation = None,
    return_conversation: bool = False,
    raise_apology: bool = False,
    max_retries: int = None,
    sleep_retry: int = 15,
    **kwargs
):
    """
    Asynchronously streams generated responses from the Bing API.

    :param prompt: The user's input prompt.
    :param tone: The desired tone for the response.
    :param image: The image type involved in the response.
    :param context: Additional context for the prompt.
    :param cookies: Cookies for the session.
    :param web_search: Flag to enable web search.
    :param gpt4_turbo: Flag to enable GPT-4 Turbo.
    :param timeout: Timeout for the request.
    :return: An asynchronous generator yielding responses.
    """
    headers = await create_headers(cookies, api_key)
    new_conversation = conversation is None
    max_retries = (5 if new_conversation else 0) if max_retries is None else max_retries
    first = True
    while first or conversation is None:
        async with StreamSession(timeout=timeout, proxy=proxy) as session:
            first = False
            do_read = True
            try:
                if conversation is None:
                    conversation = await create_conversation(session, headers, tone)
                if return_conversation:
                    yield conversation
            except (ResponseStatusError, RateLimitError) as e:
                max_retries -= 1
                if max_retries < 1:
                    raise e
                if debug.logging:
                    print(f"Bing: Retry: {e}")
                headers = await create_headers()
                await asyncio.sleep(sleep_retry)
                continue

            image_request = await upload_image(
                session,
                image,
                "Balanced" if tone == Tones.copilot else tone,
                headers
            ) if image else None
            async with session.ws_connect(
                'wss://s.copilot.microsoft.com/sydney/ChatHub'
                if tone == "Copilot" else
                'wss://sydney.bing.com/sydney/ChatHub',
                autoping=False,
                params={'sec_access_token': conversation.conversationSignature},
                headers=headers
            ) as wss:
                await wss.send_str(format_message({'protocol': 'json', 'version': 1}))
                await wss.send_str(format_message({"type": 6}))
                await wss.receive_str()
                await wss.send_str(create_message(
                    conversation, prompt, tone,
                    context if new_conversation else None,
                    image_request, web_search, gpt4_turbo,
                    new_conversation
                ))
                response_txt = ''
                returned_text = ''
                message_id = None
                while do_read:
                    try:
                        msg = await wss.receive_str()
                    except TypeError:
                        continue
                    objects = msg.split(Defaults.delimiter)
                    for obj in objects:
                        if not obj:
                            continue
                        try:
                            response = json.loads(obj)
                        except ValueError:
                            continue
                        if response and response.get('type') == 1 and response['arguments'][0].get('messages'):
                            message = response['arguments'][0]['messages'][0]
                            if message_id is not None and message_id != message["messageId"]:
                                returned_text = ''
                            message_id = message["messageId"]
                            image_response = None
                            if (raise_apology and message['contentOrigin'] == 'Apology'):
                                raise ResponseError("Apology Response Error")
                            if 'adaptiveCards' in message:
                                card = message['adaptiveCards'][0]['body'][0]
                                if "text" in card:
                                    response_txt = card.get('text')
                                if message.get('messageType') and "inlines" in card:
                                    inline_txt = card['inlines'][0].get('text')
                                    response_txt += f"{inline_txt}\n"
                            elif message.get('contentType') == "IMAGE":
                                prompt = message.get('text')
                                try:
                                    image_client = BingCreateImages(cookies, proxy, api_key)
                                    image_response = await image_client.create_async(prompt)
                                except Exception as e:
                                    if debug.logging:
                                        print(f"Bing: Failed to create images: {e}")
                                    image_response = f"\nhttps://www.bing.com/images/create?q={parse.quote(prompt)}"
                            if response_txt.startswith(returned_text):
                                new = response_txt[len(returned_text):]
                                if new not in ("", "\n"):
                                    yield new
                                    returned_text = response_txt
                            if image_response is not None:
                                yield image_response
                        elif response.get('type') == 2:
                            result = response['item']['result']
                            do_read = False
                            if result.get('error'):
                                max_retries -= 1
                                if max_retries < 1:
                                    if result["value"] == "CaptchaChallenge":
                                        raise RateLimitError(f"{result['value']}: Use other cookies or/and ip address")
                                    else:
                                        raise RuntimeError(f"{result['value']}: {result['message']}")
                                if debug.logging:
                                    print(f"Bing: Retry: {result['value']}: {result['message']}")
                                headers = await create_headers()
                                conversation = None
                                await asyncio.sleep(sleep_retry)
                            break
                        elif response.get('type') == 3:
                            do_read = False
                            break
            if conversation is not None:
                await delete_conversation(session, conversation, headers)
