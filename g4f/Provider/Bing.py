from __future__ import annotations

import random
import json
import os
import uuid
import time
from urllib import parse
from aiohttp import ClientSession, ClientTimeout

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from ..webdriver import get_browser, get_driver_cookies
from .bing.upload_image import upload_image
from .bing.create_images import create_images, format_images_markdown, wait_for_login
from .bing.conversation import Conversation, create_conversation, delete_conversation

class Tones():
    creative = "Creative"
    balanced = "Balanced"
    precise = "Precise"

class Bing(AsyncGeneratorProvider):
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
        tone: str = Tones.creative,
        image: str = None,
        web_search: bool = False,
        **kwargs
    ) -> AsyncResult:
        if len(messages) < 2:
            prompt = messages[0]["content"]
            context = None
        else:
            prompt = messages[-1]["content"]
            context = create_context(messages[:-1])
        
        if not cookies:
            cookies = Defaults.cookies
        else:
            for key, value in Defaults.cookies.items():
                if key not in cookies:
                    cookies[key] = value

        gpt4_turbo = True if model.startswith("gpt-4-turbo") else False

        return stream_generate(prompt, tone, image, context, proxy, cookies, web_search, gpt4_turbo, timeout)

def create_context(messages: Messages):
    return "".join(
        f"[{message['role']}]" + ("(#message)" if message['role']!="system" else "(#additional_instructions)") + f"\n{message['content']}\n\n"
        for message in messages
    )

class Defaults:
    delimiter = "\x1e"
    ip_address = f"13.{random.randint(104, 107)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

    allowedMessageTypes = [
        "ActionRequest",
        "Chat",
        "Context",
        # "Disengaged", unwanted
        "Progress",
        # "AdsQuery", unwanted
        "SemanticSerp",
        "GenerateContentQuery",
        "SearchQuery",
        # The following message types should not be added so that it does not flood with
        # useless messages (such as "Analyzing images" or "Searching the web") while it's retrieving the AI response
        # "InternalSearchQuery",
        # "InternalSearchResult",
        "RenderCardRequest",
        # "RenderContentRequest"
    ]

    sliceIds = [
        'abv2',
        'srdicton',
        'convcssclick',
        'stylewv2',
        'contctxp2tf',
        '802fluxv1pc_a',
        '806log2sphs0',
        '727savemem',
        '277teditgnds0',
        '207hlthgrds0',
    ]

    location = {
        "locale": "en-US",
        "market": "en-US",
        "region": "US",
        "locationHints": [
            {
                "country": "United States",
                "state": "California",
                "city": "Los Angeles",
                "timezoneoffset": 8,
                "countryConfidence": 8,
                "Center": {"Latitude": 34.0536909, "Longitude": -118.242766},
                "RegionType": 2,
                "SourceType": 1,
            }
        ],
    }

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
        'nlu_direct_response_filter',
        'deepleo',
        'disable_emoji_spoken_text',
        'responsible_ai_policy_235',
        'enablemm',
        'iyxapbing',
        'iycapbing',
        'gencontentv3',
        'fluxsrtrunc',
        'fluxtrunc',
        'fluxv1',
        'rai278',
        'replaceurl',
        'eredirecturl',
        'nojbfedge'
    ]
    
    cookies = {
        'SRCHD'         : 'AF=NOFORM',
        'PPLState'      : '1',
        'KievRPSSecAuth': '',
        'SUID'          : '',
        'SRCHUSR'       : '',
        'SRCHHPGUSR'    : f'HV={int(time.time())}',
    }

def format_message(msg: dict) -> str:
    return json.dumps(msg, ensure_ascii=False) + Defaults.delimiter

def create_message(
    conversation: Conversation,
    prompt: str,
    tone: str,
    context: str = None,
    image_info: dict = None,
    web_search: bool = False,
    gpt4_turbo: bool = False
) -> str:
    options_sets = Defaults.optionsSets
    if tone == Tones.creative:
        options_sets.append("h3imaginative")
    elif tone == Tones.precise:
        options_sets.append("h3precise")
    elif tone == Tones.balanced:
        options_sets.append("galileo")
    else:
        options_sets.append("harmonyv3")
        
    if not web_search:
        options_sets.append("nosearchall")

    if gpt4_turbo:
        options_sets.append("dlgpt4t")
    
    request_id = str(uuid.uuid4())
    struct = {
        'arguments': [
            {
                'source': 'cib',
                'optionsSets': options_sets,
                'allowedMessageTypes': Defaults.allowedMessageTypes,
                'sliceIds': Defaults.sliceIds,
                'traceId': os.urandom(16).hex(),
                'isStartOfSession': True,
                'requestId': request_id,
                'message': {**Defaults.location, **{
                    'author': 'user',
                    'inputMethod': 'Keyboard',
                    'text': prompt,
                    'messageType': 'Chat',
                    'requestId': request_id,
                    'messageId': request_id,
                }},
                "scenario": "SERP",
                'tone': tone,
                'spokenTextMode': 'None',
                'conversationId': conversation.conversationId,
                'participant': {
                    'id': conversation.clientId
                },
            }
        ],
        'invocationId': '1',
        'target': 'chat',
        'type': 4
    }
    if image_info and "imageUrl" in image_info and "originalImageUrl" in image_info:
        struct['arguments'][0]['message']['originalImageUrl'] = image_info['originalImageUrl']
        struct['arguments'][0]['message']['imageUrl'] = image_info['imageUrl']
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
        image: str = None,
        context: str = None,
        proxy: str = None,
        cookies: dict = None,
        web_search: bool = False,
        gpt4_turbo: bool = False,
        timeout = int = 900
    ):
    headers = Defaults.headers
    if cookies:
        headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
    async with ClientSession(
        timeout=ClientTimeout(total=timeout),
        headers=headers
    ) as session:
        conversation = await create_conversation(session, proxy)
        image_info = None
        if image:
            image_info = await upload_image(session, image, tone, proxy)
        try:
            async with session.ws_connect(
                'wss://sydney.bing.com/sydney/ChatHub',
                autoping=False,
                params={'sec_access_token': conversation.conversationSignature},
                proxy=proxy
            ) as wss:
                await wss.send_str(format_message({'protocol': 'json', 'version': 1}))
                await wss.receive(timeout=timeout)
                await wss.send_str(create_message(conversation, prompt, tone, context, image_info, web_search, gpt4_turbo))

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
                        if response.get('type') == 1 and response['arguments'][0].get('messages'):
                            message = response['arguments'][0]['messages'][0]
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
                                        response_txt += format_images_markdown(await create_images(session, prompt, proxy), prompt)
                                    except:
                                        response_txt += f"\nhttps://www.bing.com/images/create?q={parse.quote(prompt)}"
                                    final = True
                            if response_txt.startswith(returned_text):
                                new = response_txt[len(returned_text):]
                                if new != "\n":
                                    yield new
                                    returned_text = response_txt
                        elif response.get('type') == 2:
                            result = response['item']['result']
                            if result.get('error'):
                                if result["value"] == "CaptchaChallenge":
                                    driver = get_browser(proxy=proxy)
                                    try:
                                        wait_for_login(driver)
                                        cookies = get_driver_cookies(driver)
                                    finally:
                                        driver.quit()
                                    async for chunk in stream_generate(prompt, tone, image, context, proxy, cookies, web_search, gpt4_turbo, timeout):
                                        yield chunk
                                else:
                                    raise Exception(f"{result['value']}: {result['message']}")
                            return
        finally:
            await delete_conversation(session, conversation, proxy)
