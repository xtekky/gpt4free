from __future__ import annotations

import re
import json
from urllib import parse
from datetime import datetime

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ...requests import StreamSession

class Phind(AsyncGeneratorProvider):
    url = "https://www.phind.com"
    working = False
    lockdown = True
    supports_stream = True
    supports_message_history = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        creative_mode: bool = False,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Accept": "*/*",
            "Origin": cls.url,
            "Referer": f"{cls.url}/search",
            "Sec-Fetch-Dest": "empty", 
            "Sec-Fetch-Mode": "cors", 
            "Sec-Fetch-Site": "same-origin",
        }
        async with StreamSession(
            headers=headers,
            impersonate="chrome",
            proxies={"https": proxy},
            timeout=timeout
        ) as session:
            url = "https://www.phind.com/search?home=true"
            async with session.get(url) as response:
                text = await response.text()
                match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(?P<json>[\S\s]+?)</script>', text)
                data = json.loads(match.group("json"))
                challenge_seeds = data["props"]["pageProps"]["challengeSeeds"]
                
            prompt = messages[-1]["content"]
            data = {
                "question": prompt,
                "question_history": [
                    message["content"] for message in messages[:-1] if message["role"] == "user"
                ],
                "answer_history": [
                    message["content"] for message in messages if message["role"] == "assistant"
                ],
                "webResults": [],
                "options": {
                    "date": datetime.now().strftime("%d.%m.%Y"),
                    "language": "en-US",
                    "detailed": True,
                    "anonUserId": "",
                    "answerModel": "GPT-4" if model.startswith("gpt-4") else "Phind-34B",
                    "creativeMode": creative_mode,
                    "customLinks": []
                },
                "context": "\n".join([message["content"] for message in messages if message["role"] == "system"]),
            }
            data["challenge"] = generate_challenge(data, **challenge_seeds)
            async with session.post(f"https://https.api.phind.com/infer/", headers=headers, json=data) as response:
                new_line = False
                async for line in response.iter_lines():
                    if line.startswith(b"data: "):
                        chunk = line[6:]
                        if chunk.startswith(b'<PHIND_DONE/>'):
                            break
                        if chunk.startswith(b'<PHIND_BACKEND_ERROR>'):
                            raise RuntimeError(f"Response: {chunk.decode()}")
                        if chunk.startswith(b'<PHIND_WEBRESULTS>') or chunk.startswith(b'<PHIND_FOLLOWUP>'):
                            pass
                        elif chunk.startswith(b"<PHIND_METADATA>") or chunk.startswith(b"<PHIND_INDICATOR>"):
                            pass
                        elif chunk.startswith(b"<PHIND_SPAN_BEGIN>") or chunk.startswith(b"<PHIND_SPAN_END>"):
                            pass
                        elif chunk:
                            yield chunk.decode()
                        elif new_line:
                            yield "\n"
                            new_line = False
                        else:
                            new_line = True

def deterministic_stringify(obj):
    def handle_value(value):
        if isinstance(value, (dict, list)):
            if isinstance(value, list):
                return '[' + ','.join(sorted(map(handle_value, value))) + ']'
            else:  # It's a dict
                return '{' + deterministic_stringify(value) + '}'
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            return format(value, '.8f').rstrip('0').rstrip('.')
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return 'null'

    items = sorted(obj.items(), key=lambda x: x[0])
    return ','.join([f'{k}:{handle_value(v)}' for k, v in items if handle_value(v) is not None])

def prng_general(seed, multiplier, addend, modulus):
    a = seed * multiplier + addend
    if a < 0:
        return ((a%modulus)-modulus)/modulus
    else:
        return a%modulus/modulus

def generate_challenge_seed(l):
    I = deterministic_stringify(l)
    d = parse.quote(I, safe='')
    return simple_hash(d)

def simple_hash(s):
    d = 0
    for char in s:
        if len(char) > 1 or ord(char) >= 256:
            continue
        d = ((d << 5) - d + ord(char[0])) & 0xFFFFFFFF
        if d > 0x7FFFFFFF: # 2147483647
            d -= 0x100000000 # Subtract 2**32
    return d

def generate_challenge(obj, **kwargs):
    return prng_general(
        seed=generate_challenge_seed(obj),
        **kwargs
    )