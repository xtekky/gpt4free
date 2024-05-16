from __future__     import annotations

import os, requests, time, json
from ..typing       import CreateResult, Messages, ImageType
from .base_provider import AbstractProvider
from ..cookies      import get_cookies
from ..image        import to_bytes

class Reka(AbstractProvider):
    url             = "https://chat.reka.ai/"
    working         = True
    needs_auth      = True
    supports_stream = True
    default_vision_model = "reka"
    cookies         = {}

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        api_key: str = None,
        image: ImageType = None,
        **kwargs
    ) -> CreateResult:
        cls.proxy = proxy

        if not api_key:
            cls.cookies = get_cookies("chat.reka.ai")
            if not cls.cookies:
                raise ValueError("No cookies found for chat.reka.ai")
            elif "appSession" not in cls.cookies:
                raise ValueError("No appSession found in cookies for chat.reka.ai, log in or provide bearer_auth")
            api_key = cls.get_access_token(cls)

        conversation = []
        for message in messages:
            conversation.append({
                "type": "human",
                "text": message["content"],
            })

        if image:
            image_url = cls.upload_image(cls, api_key, image)
            conversation[-1]["image_url"] = image_url
            conversation[-1]["media_type"] = "image"

        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': f'Bearer {api_key}',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'origin': 'https://chat.reka.ai',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'sec-ch-ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        }

        json_data = {
            'conversation_history': conversation,
            'stream': True,
            'use_search_engine': False,
            'use_code_interpreter': False,
            'model_name': 'reka-core',
            'random_seed': int(time.time() * 1000),
        }

        tokens = ''

        response = requests.post('https://chat.reka.ai/api/chat', 
                                cookies=cls.cookies, headers=headers, json=json_data, proxies=cls.proxy, stream=True)

        for completion in response.iter_lines():
            if b'data' in completion:
                token_data = json.loads(completion.decode('utf-8')[5:])['text']

                yield (token_data.replace(tokens, ''))

                tokens = token_data

    def upload_image(cls, access_token, image: ImageType) -> str:
        boundary_token = os.urandom(8).hex()

        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control': 'no-cache',
            'authorization': f'Bearer {access_token}',
            'content-type': f'multipart/form-data; boundary=----WebKitFormBoundary{boundary_token}',
            'origin': 'https://chat.reka.ai',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://chat.reka.ai/chat/hPReZExtDOPvUfF8vCPC',
            'sec-ch-ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        }

        image_data = to_bytes(image)

        boundary = f'----WebKitFormBoundary{boundary_token}'
        data = f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="image.png"\r\nContent-Type: image/png\r\n\r\n'
        data += image_data.decode('latin-1')
        data += f'\r\n--{boundary}--\r\n'

        response = requests.post('https://chat.reka.ai/api/upload-image', 
                                    cookies=cls.cookies, headers=headers, proxies=cls.proxy, data=data.encode('latin-1'))

        return response.json()['media_url']

    def get_access_token(cls):
        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control': 'no-cache',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://chat.reka.ai/chat',
            'sec-ch-ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        }

        try:
            response = requests.get('https://chat.reka.ai/bff/auth/access_token', 
                                    cookies=cls.cookies, headers=headers, proxies=cls.proxy)

            return response.json()['accessToken']

        except Exception as e:
            raise ValueError(f"Failed to get access token: {e}, refresh your cookies / log in into chat.reka.ai")