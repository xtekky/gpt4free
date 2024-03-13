from __future__ import annotations

import random
import json
import uuid
import time
import asyncio
from urllib import parse
from datetime import datetime
from aiohttp import ClientSession, ClientTimeout, BaseConnector, WSMsgType

from ..typing import AsyncResult, Messages, ImageType, Cookies
from ..image import ImageRequest
from ..errors import ResponseStatusError
from .base_provider import AsyncGeneratorProvider
from .helper import get_connector, get_random_hex
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

class Bing(AsyncGeneratorProvider):
    """
    Bing provider for generating responses using the Bing API.
    """
    url = "https://bing.com/chat"
    working = True
    supports_message_history = True
    supports_gpt_4 = True
        
    @staticmethod
    def create_async_generator(
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 900,
        cookies: Cookies = None,
        connector: BaseConnector = None,
        tone: str = Tones.balanced,
        image: ImageType = None,
        web_search: bool = False,
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
        if len(messages) < 2:
            prompt = messages[0]["content"]
            context = None
        else:
            prompt = messages[-1]["content"]
            context = create_context(messages[:-1])

        gpt4_turbo = True if model.startswith("gpt-4-turbo") else False

        return stream_generate(
            prompt, tone, image, context, cookies,
            get_connector(connector, proxy, True),
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
        f"[{message['role']}]" + ("(#message)" if message['role'] != "system" else "(#additional_instructions)") + f"\n{message['content']}"
        for message in messages
    ) + "\n\n"

def get_ip_address() -> str:
    return f"13.{random.randint(104, 107)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def get_default_cookies():
    return {
        'SRCHD'         : 'AF=NOFORM',
        'PPLState'      : '1',
        'KievRPSSecAuth': '',
        'SUID'          : '',
        'SRCHUSR'       : '',
        'SRCHHPGUSR'    : f'HV={int(time.time())}',
    }

def create_headers(cookies: Cookies = None) -> dict:
    if cookies is None:
        cookies = get_default_cookies()
    headers = Defaults.headers.copy()
    headers["cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
    headers["x-forwarded-for"] = get_ip_address()
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
        "Disengaged", #"InternalLoaderMessage",
        "Progress", "RenderCardRequest",
        "RenderContentRequest", "AdsQuery",
        "SemanticSerp", "GenerateContentQuery",
        "SearchQuery", "GeneratedCode",
        "InternalTasksMessage"
    ]

    sliceIds = {
        "Balanced": [
            "supllmnfe","archnewtf",
            "stpstream", "stpsig", "vnextvoicecf", "scmcbase", "cmcpupsalltf", "sydtransctrl",
            "thdnsrch", "220dcl1s0", "0215wcrwips0", "0305hrthrots0", "0130gpt4t",
            "bingfc", "0225unsticky1", "0228scss0",
            "defquerycf", "defcontrol", "3022tphpv"
        ],
        "Creative": [
            "bgstream", "fltltst2c",
            "stpstream", "stpsig", "vnextvoicecf", "cmcpupsalltf", "sydtransctrl",
            "0301techgnd", "220dcl1bt15", "0215wcrwip", "0305hrthrot", "0130gpt4t",
            "bingfccf", "0225unsticky1", "0228scss0",
            "3022tpvs0"
        ],
        "Precise": [
            "bgstream", "fltltst2c",
            "stpstream", "stpsig", "vnextvoicecf", "cmcpupsalltf", "sydtransctrl",
            "0301techgnd", "220dcl1bt15", "0215wcrwip", "0305hrthrot", "0130gpt4t",
            "bingfccf", "0225unsticky1", "0228scss0",
            "defquerycf", "3022tpvs0"
        ],
    }

    optionsSets = {
        "Balanced": [
             "nlu_direct_response_filter", "deepleo",
            "disable_emoji_spoken_text", "responsible_ai_policy_235",
            "enablemm", "dv3sugg", "autosave",
            "iyxapbing", "iycapbing",
            "galileo", "saharagenconv5", "gldcl1p",
            "gpt4tmncnp"
        ],
        "Creative": [
            "nlu_direct_response_filter", "deepleo",
            "disable_emoji_spoken_text", "responsible_ai_policy_235",
            "enablemm", "dv3sugg",
            "iyxapbing", "iycapbing",
            "h3imaginative", "techinstgnd", "hourthrot", "clgalileo", "gencontentv3",
            "gpt4tmncnp"
        ],
        "Precise": [
            "nlu_direct_response_filter", "deepleo",
            "disable_emoji_spoken_text", "responsible_ai_policy_235",
            "enablemm", "dv3sugg",
            "iyxapbing", "iycapbing",
            "h3precise", "techinstgnd", "hourthrot", "techinstgnd", "hourthrot",
            "clgalileo", "gencontentv3"
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
    home = 'https://www.bing.com/chat?q=Bing+AI&FORM=hpcodx'
    headers = {
        'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        'sec-ch-ua-mobile': '?0',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'sec-ch-ua-arch': '"x86"',
        'sec-ch-ua-full-version': '"122.0.6261.69"',
        'accept': 'application/json',
        'sec-ch-ua-platform-version': '"15.0.0"',
        "x-ms-client-request-id": str(uuid.uuid4()),
        'sec-ch-ua-full-version-list': '"Chromium";v="122.0.6261.69", "Not(A:Brand";v="24.0.0.0", "Google Chrome";v="122.0.6261.69"',
        'x-ms-useragent': 'azsdk-js-api-client-factory/1.0.0-beta.1 core-rest-pipeline/1.12.3 OS/Windows',
        'sec-ch-ua-model': '""',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': home,
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
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
    gpt4_turbo: bool = False
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

    options_sets = []
    if gpt4_turbo:
        options_sets.append("dlgpt4t")

    request_id = str(uuid.uuid4())
    struct = {
        "arguments":[{
            "source": "cib",
            "optionsSets": [*Defaults.optionsSets[tone], *options_sets],
            "allowedMessageTypes": Defaults.allowedMessageTypes,
            "sliceIds": Defaults.sliceIds[tone],
            "verbosity": "verbose",
            "scenario": "SERP",
            "plugins": [{"id": "c310c353-b9f0-4d76-ab0d-1dd5e979cf68", "category": 1}] if web_search else [],
            "traceId": get_random_hex(40),
            "conversationHistoryOptionsSets": ["autosave","savemem","uprofupd","uprofgen"],
            "gptId": "copilot",
            "isStartOfSession": True,
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
            "tone": tone,
            "extraExtensionParameters": {"gpt-creator-persona": {"personaId": "copilot"}},
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
            "contextType": "WebPage",
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
    connector: BaseConnector = None,
    proxy: str = None,
    web_search: bool = False,
    gpt4_turbo: bool = False,
    timeout: int = 900,
    conversation: Conversation = None,
    raise_apology: bool = False,
    max_retries: int = 5,
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
    headers = create_headers(cookies)
    async with ClientSession(
        timeout=ClientTimeout(total=timeout), connector=connector
    ) as session:
        while conversation is None:
            do_read = True
            try:
                conversation = await create_conversation(session, headers)
            except ResponseStatusError as e:
                max_retries -= 1
                if max_retries < 1:
                    raise e
                if debug.logging:
                    print(f"Bing: Retry: {e}")
                headers = create_headers()
                await asyncio.sleep(sleep_retry)
                continue

            image_request = await upload_image(session, image, tone, headers) if image else None
            async with session.ws_connect(
                'wss://sydney.bing.com/sydney/ChatHub',
                autoping=False,
                params={'sec_access_token': conversation.conversationSignature},
                headers=headers
            ) as wss:
                await wss.send_str(format_message({'protocol': 'json', 'version': 1}))
                await wss.send_str(format_message({"type": 6}))
                await wss.receive(timeout=timeout)
                await wss.send_str(create_message(conversation, prompt, tone, context, image_request, web_search, gpt4_turbo))
                response_txt = ''
                returned_text = ''
                message_id = None
                while do_read:
                    msg = await wss.receive(timeout=timeout)
                    if msg.type == WSMsgType.CLOSED:
                        break
                    if msg.type != WSMsgType.TEXT or not msg.data:
                        continue
                    objects = msg.data.split(Defaults.delimiter)
                    for obj in objects:
                        if obj is None or not obj:
                            continue
                        response = json.loads(obj)
                        if response and response.get('type') == 1 and response['arguments'][0].get('messages'):
                            message = response['arguments'][0]['messages'][0]
                            if message_id is not None and message_id != message["messageId"]:
                                returned_text = ''
                            message_id = message["messageId"]
                            image_response = None
                            if (raise_apology and message['contentOrigin'] == 'Apology'):
                                raise RuntimeError("Apology Response Error")
                            if 'adaptiveCards' in message:
                                card = message['adaptiveCards'][0]['body'][0]
                                if "text" in card:
                                    response_txt = card.get('text')
                                if message.get('messageType') and "inlines" in card:
                                    inline_txt = card['inlines'][0].get('text')
                                    response_txt += inline_txt + '\n'
                            elif message.get('contentType') == "IMAGE":
                                prompt = message.get('text')
                                try:
                                    image_client = BingCreateImages(cookies, proxy)
                                    image_response = await image_client.create_async(prompt)
                                except Exception as e:
                                    response_txt += f"\nhttps://www.bing.com/images/create?q={parse.quote(prompt)}"
                                do_read = False
                            if response_txt.startswith(returned_text):
                                new = response_txt[len(returned_text):]
                                if new != "\n":
                                    yield new
                                    returned_text = response_txt
                            if image_response:
                                yield image_response
                        elif response.get('type') == 2:
                            result = response['item']['result']
                            if result.get('error'):
                                max_retries -= 1
                                if max_retries < 1:
                                    if result["value"] == "CaptchaChallenge":
                                        raise RuntimeError(f"{result['value']}: Use other cookies or/and ip address")
                                    else:
                                        raise RuntimeError(f"{result['value']}: {result['message']}")
                                if debug.logging:
                                    print(f"Bing: Retry: {result['value']}: {result['message']}")
                                headers = create_headers()
                                do_read = False
                                conversation = None
                                await asyncio.sleep(sleep_retry)
                                break
                            return
        await delete_conversation(session, conversation, headers)
