from __future__ import annotations

import random
import json
import os
import uuid
import time
from urllib import parse
from aiohttp import ClientSession, ClientTimeout, BaseConnector

from ..typing import AsyncResult, Messages, ImageType
from ..image import ImageResponse, ImageRequest
from .base_provider import AsyncGeneratorProvider
from .helper import get_connector
from .bing.upload_image import upload_image
from .bing.create_images import create_images
from .bing.conversation import Conversation, create_conversation, delete_conversation

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
        cookies: dict = None,
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
        
        cookies = {**Defaults.cookies, **cookies} if cookies else Defaults.cookies

        gpt4_turbo = True if model.startswith("gpt-4-turbo") else False

        return stream_generate(prompt, tone, image, context, cookies, get_connector(connector, proxy), web_search, gpt4_turbo, timeout)

def create_context(messages: Messages) -> str:
    """
    Creates a context string from a list of messages.

    :param messages: A list of message dictionaries.
    :return: A string representing the context created from the messages.
    """
    return "".join(
        f"[{message['role']}]" + ("(#message)" if message['role'] != "system" else "(#additional_instructions)") + f"\n{message['content']}\n\n"
        for message in messages
    )

class Defaults:
    """
    Default settings and configurations for the Bing provider.
    """
    delimiter = "\x1e"
    ip_address = f"13.{random.randint(104, 107)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

    # List of allowed message types for Bing responses
    allowedMessageTypes = [
        "ActionRequest", "Chat", "Context", "Progress", "SemanticSerp",
        "GenerateContentQuery", "SearchQuery", "RenderCardRequest"
    ]

    sliceIds = [
        'abv2', 'srdicton', 'convcssclick', 'stylewv2', 'contctxp2tf',
        '802fluxv1pc_a', '806log2sphs0', '727savemem', '277teditgnds0', '207hlthgrds0'
    ]

    # Default location settings
    location = {
        "locale": "en-US", "market": "en-US", "region": "US",
        "locationHints": [{
            "country": "United States", "state": "California", "city": "Los Angeles",
            "timezoneoffset": 8, "countryConfidence": 8,
            "Center": {"Latitude": 34.0536909, "Longitude": -118.242766},
            "RegionType": 2, "SourceType": 1
        }],
    }

    # Default headers for requests
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'cache-control': 'max-age=0',
        'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"',
        'sec-ch-ua-arch': '"x86"',
        'sec-ch-ua-bitness': '"64"',
        'sec-ch-ua-full-version': '"110.0.1587.69"',
        'sec-ch-ua-full-version-list': '"Chromium";v="110.0.5481.192", "Not A(Brand";v="24.0.0.0", "Microsoft Edge";v="110.0.1587.69"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-model': '""',
        'sec-ch-ua-platform': '"Windows"',
        'sec-ch-ua-platform-version': '"15.0.0"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.69',
        'x-edge-shopping-flag': '1',
        'x-forwarded-for': ip_address,
    }

    optionsSets = [
        'nlu_direct_response_filter', 'deepleo', 'disable_emoji_spoken_text',
        'responsible_ai_policy_235', 'enablemm', 'iyxapbing', 'iycapbing',
        'gencontentv3', 'fluxsrtrunc', 'fluxtrunc', 'fluxv1', 'rai278',
        'replaceurl', 'eredirecturl', 'nojbfedge'
    ]
    
    # Default cookies
    cookies = {
        'SRCHD'         : 'AF=NOFORM',
        'PPLState'      : '1',
        'KievRPSSecAuth': '',
        'SUID'          : '',
        'SRCHUSR'       : '',
        'SRCHHPGUSR'    : f'HV={int(time.time())}',
    }

class ConversationStyleOptionSets():
    CREATIVE = ["h3imaginative", "clgalileo", "gencontentv3"]
    BALANCED = ["galileo"]
    PRECISE = ["h3precise", "clgalileo"]

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
    options_sets = Defaults.optionsSets.copy()
    # Append tone-specific options
    if tone == Tones.creative:
        options_sets.extend(ConversationStyleOptionSets.CREATIVE)
    elif tone == Tones.precise:
        options_sets.extend(ConversationStyleOptionSets.PRECISE)
    elif tone == Tones.balanced:
        options_sets.extend(ConversationStyleOptionSets.BALANCED)
    else:
        options_sets.append("harmonyv3")

    # Additional configurations based on parameters
    if not web_search:
        options_sets.append("nosearchall")
    if gpt4_turbo:
        options_sets.append("dlgpt4t")

    request_id = str(uuid.uuid4())
    struct = {
        'arguments': [{
            'source': 'cib',
            'optionsSets': options_sets,
            'allowedMessageTypes': Defaults.allowedMessageTypes,
            'sliceIds': Defaults.sliceIds,
            'traceId': os.urandom(16).hex(),
            'isStartOfSession': True,
            'requestId': request_id,
            'message': {
                **Defaults.location,
                'author': 'user',
                'inputMethod': 'Keyboard',
                'text': prompt,
                'messageType': 'Chat',
                'requestId': request_id,
                'messageId': request_id
            },
            "verbosity": "verbose",
            "scenario": "SERP",
            "plugins": [{"id": "c310c353-b9f0-4d76-ab0d-1dd5e979cf68", "category": 1}] if web_search else [],
            'tone': tone,
            'spokenTextMode': 'None',
            'conversationId': conversation.conversationId,
            'participant': {'id': conversation.clientId},
        }],
        'invocationId': '1',
        'target': 'chat',
        'type': 4
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
    web_search: bool = False,
    gpt4_turbo: bool = False,
    timeout: int = 900
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
    headers = Defaults.headers
    if cookies:
        headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())

    async with ClientSession(
        timeout=ClientTimeout(total=timeout), headers=headers, connector=connector
    ) as session:
        conversation = await create_conversation(session)
        image_request = await upload_image(session, image, tone) if image else None

        try:
            async with session.ws_connect(
                'wss://sydney.bing.com/sydney/ChatHub',
                autoping=False,
                params={'sec_access_token': conversation.conversationSignature}
            ) as wss:
                await wss.send_str(format_message({'protocol': 'json', 'version': 1}))
                await wss.receive(timeout=timeout)
                await wss.send_str(create_message(conversation, prompt, tone, context, image_request, web_search, gpt4_turbo))

                response_txt = ''
                returned_text = ''
                final = False
                while not final:
                    msg = await wss.receive(timeout=timeout)
                    if not msg.data:
                        continue
                    objects = msg.data.split(Defaults.delimiter)
                    for obj in objects:
                        if obj is None or not obj:
                            continue
                        response = json.loads(obj)
                        if response and response.get('type') == 1 and response['arguments'][0].get('messages'):
                            message = response['arguments'][0]['messages'][0]
                            image_response = None
                            if (message['contentOrigin'] != 'Apology'):
                                if 'adaptiveCards' in message:
                                    card = message['adaptiveCards'][0]['body'][0]
                                    if "text" in card:
                                        response_txt = card.get('text')
                                    if message.get('messageType'):
                                        inline_txt = card['inlines'][0].get('text')
                                        response_txt += inline_txt + '\n'
                                elif message.get('contentType') == "IMAGE":
                                    prompt = message.get('text')
                                    try:
                                        image_response = ImageResponse(await create_images(session, prompt), prompt, {"preview": "{image}?w=200&h=200"})
                                    except:
                                        response_txt += f"\nhttps://www.bing.com/images/create?q={parse.quote(prompt)}"
                                    final = True
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
                                if result["value"] == "CaptchaChallenge":
                                    raise Exception(f"{result['value']}: Use other cookies or/and ip address")
                                else:
                                    raise Exception(f"{result['value']}: {result['message']}")
                            return
        finally:
            await delete_conversation(session, conversation)
