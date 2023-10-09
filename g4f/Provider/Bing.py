from __future__ import annotations

import random
import uuid
import json
import os
import uuid
import urllib.parse
from aiohttp        import ClientSession, ClientTimeout
from ..typing       import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

class Tones():
    creative = "Creative"
    balanced = "Balanced"
    precise = "Precise"

default_cookies = {
    'SRCHD'         : 'AF=NOFORM',
    'PPLState'      : '1',
    'KievRPSSecAuth': '',
    'SUID'          : '',
    'SRCHUSR'       : '',
    'SRCHHPGUSR'    : '',
}

class Bing(AsyncGeneratorProvider):
    url             = "https://bing.com/chat"
    working         = True
    supports_gpt_4  = True
        
    @staticmethod
    def create_async_generator(
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: dict = None,
        tone: str = Tones.creative,
        **kwargs
    ) -> AsyncResult:
        if len(messages) < 2:
            prompt = messages[0]["content"]
            context = None
        else:
            prompt = messages[-1]["content"]
            context = create_context(messages[:-1])
        
        if not cookies or "SRCHD" not in cookies:
            cookies = default_cookies
        return stream_generate(prompt, tone, context, proxy, cookies)

def create_context(messages: Messages):
    context = "".join(f"[{message['role']}](#message)\n{message['content']}\n\n" for message in messages)

    return context

class Conversation():
    def __init__(self, conversationId: str, clientId: str, conversationSignature: str) -> None:
        self.conversationId = conversationId
        self.clientId = clientId
        self.conversationSignature = conversationSignature

async def create_conversation(session: ClientSession, proxy: str = None) -> Conversation:
    url = 'https://www.bing.com/turing/conversation/create?bundleVersion=1.1150.3'
    
    async with await session.get(url, proxy=proxy) as response:
        data = await response.json()
        
        conversationId = data.get('conversationId')
        clientId = data.get('clientId')
        conversationSignature = response.headers.get('X-Sydney-Encryptedconversationsignature')

        if not conversationId or not clientId or not conversationSignature:
            raise Exception('Failed to create conversation.')
        
        return Conversation(conversationId, clientId, conversationSignature)

async def list_conversations(session: ClientSession) -> list:
    url = "https://www.bing.com/turing/conversation/chats"
    async with session.get(url) as response:
        response = await response.json()
        return response["chats"]
        
async def delete_conversation(session: ClientSession, conversation: Conversation, proxy: str = None) -> list:
    url = "https://sydney.bing.com/sydney/DeleteSingleConversation"
    json = {
        "conversationId": conversation.conversationId,
        "conversationSignature": conversation.conversationSignature,
        "participant": {"id": conversation.clientId},
        "source": "cib",
        "optionsSets": ["autosave"]
    }
    async with session.post(url, json=json, proxy=proxy) as response:
        response = await response.json()
        return response["result"]["value"] == "Success"

class Defaults:
    delimiter = "\x1e"
    ip_address = f"13.{random.randint(104, 107)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

    allowedMessageTypes = [
        "Chat",
        "Disengaged",
        "AdsQuery",
        "SemanticSerp",
        "GenerateContentQuery",
        "SearchQuery",
        "ActionRequest",
        "Context",
        "Progress",
        "AdsQuery",
        "SemanticSerp",
    ]

    sliceIds = [
        "winmuid3tf",
        "osbsdusgreccf",
        "ttstmout",
        "crchatrev",
        "winlongmsgtf",
        "ctrlworkpay",
        "norespwtf",
        "tempcacheread",
        "temptacache",
        "505scss0",
        "508jbcars0",
        "515enbotdets0",
        "5082tsports",
        "515vaoprvs",
        "424dagslnv1s0",
        "kcimgattcf",
        "427startpms0",
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
        'saharasugg',
        'enablenewsfc',
        'clgalileo',
        'gencontentv3',
        "nlu_direct_response_filter",
        "deepleo",
        "disable_emoji_spoken_text",
        "responsible_ai_policy_235",
        "enablemm",
        "h3precise"
        "dtappid",
        "cricinfo",
        "cricinfov2",
        "dv3sugg",
        "nojbfedge"
    ]

def format_message(msg: dict) -> str:
    return json.dumps(msg, ensure_ascii=False) + Defaults.delimiter

def create_message(conversation: Conversation, prompt: str, tone: str, context: str=None) -> str:
    request_id = str(uuid.uuid4())
    struct = {
        'arguments': [
            {
                'source': 'cib',
                'optionsSets': Defaults.optionsSets,
                'allowedMessageTypes': Defaults.allowedMessageTypes,
                'sliceIds': Defaults.sliceIds,
                'traceId': os.urandom(16).hex(),
                'isStartOfSession': True,
                'requestId': request_id,
                'message': Defaults.location | {
                    'author': 'user',
                    'inputMethod': 'Keyboard',
                    'text': prompt,
                    'messageType': 'Chat',
                    'requestId': request_id,
                    'messageId': request_id,
                },
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
        context: str = None,
        proxy: str = None,
        cookies: dict = None
    ):
    async with ClientSession(
        timeout=ClientTimeout(total=900),
        cookies=cookies,
        headers=Defaults.headers,
    ) as session:
        conversation = await create_conversation(session, proxy)
        try:
            async with session.ws_connect(
                f'wss://sydney.bing.com/sydney/ChatHub',
                autoping=False,
                params={'sec_access_token': conversation.conversationSignature},
                proxy=proxy
            ) as wss:
                
                await wss.send_str(format_message({'protocol': 'json', 'version': 1}))
                await wss.receive(timeout=900)
                await wss.send_str(create_message(conversation, prompt, tone, context))

                response_txt = ''
                returned_text = ''
                final = False

                while not final:
                    msg = await wss.receive(timeout=900)
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
                                    query = urllib.parse.quote(message.get('text'))
                                    url = f"\nhttps://www.bing.com/images/create?q={query}"
                                    response_txt += url
                                    final = True
                            if response_txt.startswith(returned_text):
                                new = response_txt[len(returned_text):]
                                if new != "\n":
                                    yield new
                                    returned_text = response_txt
                        elif response.get('type') == 2:
                            result = response['item']['result']
                            if result.get('error'):
                                raise Exception(f"{result['value']}: {result['message']}")
                            return
        finally:
            await delete_conversation(session, conversation, proxy)