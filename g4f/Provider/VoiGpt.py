import json
from .base_provider import BaseProvider
from .typing import Messages, CreateResult
from .helper import get_cookies
import requests



class VoiGpt(BaseProvider):
    url = "https://voigpt.com"
    working = True
    supports_gpt_35_turbo = True
    supports_message_history = True
    supports_stream = False

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        **kwargs
    ) -> CreateResult:
        
        if not model:
            model = "gpt-3.5-turbo"

        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7,hi;q=0.6',
            'Content-Type': 'application/json',
            'Host': 'voigpt.com',
            'Origin': 'https://voigpt.com',
            'Referer': 'https://voigpt.com/',
            'Sec-Ch-Ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        }
        
        cookies = get_cookies(cls.url)

        payload = {
            'messages': messages,
        }

        request_url = cls.url + "/generate_response/"
        req_response = requests.post(request_url, headers=headers, json=payload, cookies=cookies)

        response = json.loads(req_response.text)
        return response["response"]
    
