from __future__ import annotations

import string
import random
import json
import os
import re
import io
import base64
import numpy as np
import uuid
import urllib.parse
import time
from PIL import Image
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
    'SRCHHPGUSR'    : f'HV={int(time.time())}',
}

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
            cookies = default_cookies
        else:
            for key, value in default_cookies.items():
                if key not in cookies:
                    cookies[key] = value

        gpt4_turbo = True if model.startswith("gpt-4-turbo") else False

        return stream_generate(prompt, tone, image, context, proxy, cookies, web_search, gpt4_turbo)

def create_context(messages: Messages):
    return "".join(
        f"[{message['role']}]" + ("(#message)" if message['role']!="system" else "(#additional_instructions)") + f"\n{message['content']}\n\n"
        for message in messages
    )

class Conversation():
    def __init__(self, conversationId: str, clientId: str, conversationSignature: str, imageInfo: dict=None) -> None:
        self.conversationId = conversationId
        self.clientId = clientId
        self.conversationSignature = conversationSignature
        self.imageInfo = imageInfo

async def create_conversation(session: ClientSession, tone: str, image: str = None, proxy: str = None) -> Conversation:
    url = 'https://www.bing.com/turing/conversation/create?bundleVersion=1.1199.4'
    async with session.get(url, proxy=proxy) as response:
        data = await response.json()

        conversationId = data.get('conversationId')
        clientId = data.get('clientId')
        conversationSignature = response.headers.get('X-Sydney-Encryptedconversationsignature')

        if not conversationId or not clientId or not conversationSignature:
            raise Exception('Failed to create conversation.')
        conversation = Conversation(conversationId, clientId, conversationSignature, None)
        if isinstance(image,str):
            try:
                config = {
                    "visualSearch": {
                        "maxImagePixels": 360000,
                        "imageCompressionRate": 0.7,
                        "enableFaceBlurDebug": 0,
                    }
                }
                is_data_uri_an_image(image)
                img_binary_data = extract_data_uri(image)
                is_accepted_format(img_binary_data)
                img = Image.open(io.BytesIO(img_binary_data))
                width, height = img.size
                max_image_pixels = config['visualSearch']['maxImagePixels']
                compression_rate = config['visualSearch']['imageCompressionRate']

                if max_image_pixels / (width * height) < 1:
                    new_width = int(width * np.sqrt(max_image_pixels / (width * height)))
                    new_height = int(height * np.sqrt(max_image_pixels / (width * height)))
                else:
                    new_width = width
                    new_height = height
                try:
                    orientation = get_orientation(img)
                except Exception:
                    orientation = None
                new_img = process_image(orientation, img, new_width, new_height)
                new_img_binary_data = compress_image_to_base64(new_img, compression_rate)
                data, boundary = build_image_upload_api_payload(new_img_binary_data, conversation, tone)
                headers = session.headers.copy()
                headers["content-type"] = f'multipart/form-data; boundary={boundary}'
                headers["referer"] = 'https://www.bing.com/search?q=Bing+AI&showconv=1&FORM=hpcodx'
                headers["origin"] = 'https://www.bing.com'
                async with session.post("https://www.bing.com/images/kblob", data=data, headers=headers, proxy=proxy) as image_upload_response:
                    if image_upload_response.status != 200:
                        raise Exception("Failed to upload image.")

                    image_info = await image_upload_response.json()
                    if not image_info.get('blobId'):
                        raise Exception("Failed to parse image info.")
                    result = {'bcid': image_info.get('blobId', "")}
                    result['blurredBcid'] = image_info.get('processedBlobId', "")
                    if result['blurredBcid'] != "":
                        result["imageUrl"] = "https://www.bing.com/images/blob?bcid=" + result['blurredBcid']
                    elif result['bcid'] != "":
                        result["imageUrl"] = "https://www.bing.com/images/blob?bcid=" + result['bcid']
                    result['originalImageUrl'] = (
                        "https://www.bing.com/images/blob?bcid="
                        + result['blurredBcid']
                        if config['visualSearch']["enableFaceBlurDebug"]
                        else "https://www.bing.com/images/blob?bcid="
                        + result['bcid']
                    )
                    conversation.imageInfo = result
            except Exception as e:
                print(f"An error happened while trying to send image: {str(e)}")
        return conversation

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
        try:
            response = await response.json()
            return response["result"]["value"] == "Success"
        except:
            return False

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

def format_message(msg: dict) -> str:
    return json.dumps(msg, ensure_ascii=False) + Defaults.delimiter

def build_image_upload_api_payload(image_bin: str, conversation: Conversation, tone: str):
    payload = {
        'invokedSkills': ["ImageById"],
        'subscriptionId': "Bing.Chat.Multimodal",
        'invokedSkillsRequestData': {
            'enableFaceBlur': True
        },
        'convoData': {
            'convoid': "",
            'convotone': tone
        }
    }
    knowledge_request = {
        'imageInfo': {},
        'knowledgeRequest': payload
    }
    boundary="----WebKitFormBoundary" + ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    data = (
        f'--{boundary}'
        + '\r\nContent-Disposition: form-data; name="knowledgeRequest"\r\n\r\n'
        + json.dumps(knowledge_request, ensure_ascii=False)
        + "\r\n--"
        + boundary
        + '\r\nContent-Disposition: form-data; name="imageBase64"\r\n\r\n'
        + image_bin
        + "\r\n--"
        + boundary
        + "--\r\n"
    )
    return data, boundary

def is_data_uri_an_image(data_uri: str):
    try:
        # Check if the data URI starts with 'data:image' and contains an image format (e.g., jpeg, png, gif)
        if not re.match(r'data:image/(\w+);base64,', data_uri):
            raise ValueError("Invalid data URI image.")
            # Extract the image format from the data URI
        image_format = re.match(r'data:image/(\w+);base64,', data_uri).group(1)
        # Check if the image format is one of the allowed formats (jpg, jpeg, png, gif)
        if image_format.lower() not in ['jpeg', 'jpg', 'png', 'gif']:
            raise ValueError("Invalid image format (from mime file type).")
    except Exception as e:
        raise e

def is_accepted_format(binary_data: bytes) -> bool:
        try:
            check = False
            if binary_data.startswith(b'\xFF\xD8\xFF'):
                check = True  # It's a JPEG image
            elif binary_data.startswith(b'\x89PNG\r\n\x1a\n'):
                check = True  # It's a PNG image
            elif binary_data.startswith(b'GIF87a') or binary_data.startswith(b'GIF89a'):
                check = True  # It's a GIF image
            elif binary_data.startswith(b'\x89JFIF') or binary_data.startswith(b'JFIF\x00'):
                check = True  # It's a JPEG image
            elif binary_data.startswith(b'\xFF\xD8'):
                check = True  # It's a JPEG image
            elif binary_data.startswith(b'RIFF') and binary_data[8:12] == b'WEBP':
                check = True  # It's a WebP image
            # else we raise ValueError
            if not check:
                raise ValueError("Invalid image format (from magic code).")
        except Exception as e:
            raise e
    
def extract_data_uri(data_uri: str) -> bytes:
    try:
        data = data_uri.split(",")[1]
        data = base64.b64decode(data)
        return data
    except Exception as e:
        raise e

def get_orientation(data: bytes) -> int:
    try:
        if data[:2] != b'\xFF\xD8':
            raise Exception('NotJpeg')
        with Image.open(data) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                orientation = exif_data.get(274)  # 274 corresponds to the orientation tag in EXIF
                if orientation is not None:
                    return orientation
    except Exception:
        pass

def process_image(orientation: int, img: Image.Image, new_width: int, new_height: int) -> Image.Image:
    try:
        # Initialize the canvas
        new_img = Image.new("RGB", (new_width, new_height), color="#FFFFFF")
        if orientation:
            if orientation > 4:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if orientation in [3, 4]:
                img = img.transpose(Image.ROTATE_180)
            if orientation in [5, 6]:
                img = img.transpose(Image.ROTATE_270)
            if orientation in [7, 8]:
                img = img.transpose(Image.ROTATE_90)
        new_img.paste(img, (0, 0))
        return new_img
    except Exception as e:
        raise e
    
def compress_image_to_base64(img, compression_rate) -> str:
    try:
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=int(compression_rate * 100))
        return base64.b64encode(output_buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise e

def create_message(conversation: Conversation, prompt: str, tone: str, context: str = None, web_search: bool = False, gpt4_turbo: bool = False) -> str:
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
    if conversation.imageInfo != None and "imageUrl" in conversation.imageInfo and "originalImageUrl" in conversation.imageInfo:
        struct['arguments'][0]['message']['originalImageUrl'] = conversation.imageInfo['originalImageUrl']
        struct['arguments'][0]['message']['imageUrl'] = conversation.imageInfo['imageUrl']
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
        gpt4_turbo: bool = False
    ):
    async with ClientSession(
            timeout=ClientTimeout(total=900),
            headers=Defaults.headers if not cookies else {**Defaults.headers, "Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items())},
        ) as session:
        conversation = await create_conversation(session, tone, image, proxy)
        try:
            async with session.ws_connect('wss://sydney.bing.com/sydney/ChatHub', autoping=False, params={'sec_access_token': conversation.conversationSignature}, proxy=proxy) as wss:

                await wss.send_str(format_message({'protocol': 'json', 'version': 1}))
                await wss.receive(timeout=900)
                await wss.send_str(create_message(conversation, prompt, tone, context, web_search, gpt4_turbo))

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
