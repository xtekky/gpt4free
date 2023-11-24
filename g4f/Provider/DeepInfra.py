from __future__ import annotations

import requests, json
from ..typing           import CreateResult, Messages
from .base_provider     import BaseProvider

class DeepInfra(BaseProvider):
    url: str = "https://deepinfra.com"
    working: bool = True
    supports_stream: bool = True
    supports_message_history: bool = True

    @staticmethod
    def create_completion(model: str,
                          messages: Messages,
                          stream: bool,
                          **kwargs) -> CreateResult:
        
        headers = {
            'Accept-Language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Pragma': 'no-cache',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-embed',
            'accept': 'text/event-stream',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        }

        json_data = json.dumps({
            'model'   : 'meta-llama/Llama-2-70b-chat-hf',
            'messages': messages,
            'stream'  : True}, separators=(',', ':'))

        response = requests.post('https://api.deepinfra.com/v1/openai/chat/completions', 
                                headers=headers, data=json_data, stream=True)

        response.raise_for_status()
        first = True

        for line in response.iter_content(chunk_size=1024):
            if line.startswith(b"data: [DONE]"):
                break
            
            elif line.startswith(b"data: "):
                chunk = json.loads(line[6:])["choices"][0]["delta"].get("content")
                
                if chunk:
                    if first:
                        chunk = chunk.lstrip()
                        if chunk:
                            first = False
                    
                    yield (chunk) 