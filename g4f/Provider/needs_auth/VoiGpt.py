from __future__ import annotations

import json
import requests
from ..base_provider import BaseProvider
from ...typing import Messages, CreateResult
from ..helper import get_cookies



class VoiGpt(BaseProvider):
    """
    VoiGpt - A provider for VoiGpt.com

    **Note** : to use this provider you have to get your csrf token/cookie from the voigpt.com website

    Args:
        model: The model to use
        messages: The messages to send
        stream: Whether to stream the response
        proxy: The proxy to use
        access_token: The access token to use
        **kwargs: Additional keyword arguments

    Returns:
        A CreateResult object
    """
    url = "https://voigpt.com"
    working = True
    supports_gpt_35_turbo = True
    supports_message_history = True
    supports_stream = False
    needs_auth = True
    _access_token: str = None

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        access_token: str = None,
        **kwargs
    ) -> CreateResult:
        
        if not model:
            model = "gpt-3.5-turbo"
        if not access_token:
            access_token = cls._access_token

        headers = {
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7,hi;q=0.6",
            "Content-Type": "application/json",
            "Cookie": f"csrftoken={access_token};",
            "Host": "voigpt.com",
            "Origin": "https://voigpt.com",
            "Referer": "https://voigpt.com/",
            "Sec-Ch-Ua": "'Google Chrome';v='119', 'Chromium';v='119', 'Not?A_Brand';v='24'",
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "'Windows'",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "X-Csrftoken": f"{access_token}",
        }

        payload = {
            "messages": messages,
        }

        request_url = cls.url + "/generate_response/"
        req_response = requests.post(request_url, headers=headers, json=payload)

        response = json.loads(req_response.text)
        return response["response"]
    
