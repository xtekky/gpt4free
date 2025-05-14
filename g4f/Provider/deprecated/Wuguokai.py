from __future__ import annotations

import random

import requests

from ...typing import Any, CreateResult
from ..base_provider import AbstractProvider, format_prompt


class Wuguokai(AbstractProvider):
    url = 'https://chat.wuguokai.xyz'
    supports_gpt_35_turbo = True
    working = False

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        headers = {
            'authority': 'ai-api.wuguokai.xyz',
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
            'content-type': 'application/json',
            'origin': 'https://chat.wuguokai.xyz',
            'referer': 'https://chat.wuguokai.xyz/',
            'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        data ={
            "prompt": format_prompt(messages),
            "options": {},
            "userId": f"#/chat/{random.randint(1,99999999)}",
            "usingContext": True
        }
        response = requests.post(
            "https://ai-api20.wuguokai.xyz/api/chat-process",
            headers=headers,
            timeout=3,
            json=data,
            proxies=kwargs.get('proxy', {}),
        )
        _split = response.text.split("> 若回答失败请重试或多刷新几次界面后重试")
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.reason}")
        if len(_split) > 1:
            yield _split[1].strip()
        else:
            yield _split[0].strip()