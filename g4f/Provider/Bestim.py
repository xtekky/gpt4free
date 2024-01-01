from __future__         import annotations

from ..typing           import Messages, List, Dict
from .base_provider     import BaseProvider, CreateResult
from uuid               import uuid4
import requests

def format_prompt(messages) -> List[Dict[str, str]]:

    return [{"id": str(uuid4()), "content": '\n'.join(f'{m["role"]}: {m["content"]}' for m in messages), "from": "you"}]

class Bestim(BaseProvider):
    url = "https://chatgpt.bestim.org"
    supports_gpt_35_turbo = True
    supports_message_history = True
    working = True
    supports_stream = True

    @staticmethod
    def create_completion(
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None, 
        **kwargs
    ) -> CreateResult:
        
        headers = {
            'POST': '/chat/send2/ HTTP/3',
            'Host': 'chatgpt.bestim.org',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Accept': 'application/json, text/event-stream',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://chatgpt.bestim.org/chat/',
            'Content-Type': 'application/json',
            'Content-Length': '109',
            'Origin': 'https://chatgpt.bestim.org',
            'Cookie': 'NpZAER=qKkRHguMIOraVbJAWpoyzGLFjZwYlm; qKkRHguMIOraVbJAWpoyzGLFjZwYlm=8ebb5ae1561bde05354de5979b52c6e1-1704058188-1704058188; NpZAER_hits=2; _csrf-front=fcf20965823c0a152ae8f9cdf15b23022bb26cdc6bf32a9d4c8bfe78dcc6b807a%3A2%3A%7Bi%3A0%3Bs%3A11%3A%22_csrf-front%22%3Bi%3A1%3Bs%3A32%3A%22a5wP6azsc7dxV8rmwAXaNsl8XS1yvW5V%22%3B%7D',
            'Alt-Used': 'chatgpt.bestim.org',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'TE': 'trailers'
        }

        data = {

            "messagesHistory": format_prompt(messages),
            "type": "chat",
        }
            
        response = requests.post(
            url="https://chatgpt.bestim.org/chat/send2/",
            headers=headers,
            json=data,
            proxies={"https": proxy}
        )

        response.raise_for_status()

        for chunk in response.iter_lines():

            if b"event: trylimit" not in chunk:

                yield chunk.decode().removeprefix("data: ")






            




