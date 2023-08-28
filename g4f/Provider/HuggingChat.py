import json
from aiohttp import ClientSession

from ..typing import AsyncGenerator
from .base_provider import AsyncGeneratorProvider, get_cookies, format_prompt


class HuggingChat(AsyncGeneratorProvider):
    url = "https://huggingface.co/chat/"
    needs_auth = True
    working = True
    model = "OpenAssistant/oasst-sft-6-llama-30b-xor"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = True,
        proxy: str = None,
        cookies: dict = None,
        **kwargs
    ) -> AsyncGenerator:
        if not cookies:
            cookies = get_cookies(".huggingface.co")
        model = model if model else cls.model
        if proxy and "://" not in proxy:
            proxy = f"http://{proxy}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        }
        async with ClientSession(
            cookies=cookies,
            headers=headers
        ) as session:
            async with session.post("https://huggingface.co/chat/conversation", proxy=proxy, json={"model": model}) as response:
                conversation_id = (await response.json())["conversationId"]

            send = {
                "inputs": format_prompt(messages),
                "parameters": {
                    "temperature": 0.2,
                    "truncate": 1000,
                    "max_new_tokens": 1024,
                    "stop": ["</s>"],
                    "top_p": 0.95,
                    "repetition_penalty": 1.2,
                    "top_k": 50,
                    "return_full_text": False,
                    **kwargs
                },
                "stream": stream,
                "options": {
                    "id": "9e9b8bc4-6604-40c6-994e-8eb78fa32e37",
                    "response_id": "04ce2602-3bea-45e8-8efc-cef00680376a",
                    "is_retry": False,
                    "use_cache": False,
                    "web_search_id": ""
                }
            }
            start = "data:"
            first = True
            async with session.post(f"https://huggingface.co/chat/conversation/{conversation_id}", proxy=proxy, json=send) as response:
                async for line in response.content:
                    line = line.decode("utf-8")
                    if not line:
                        continue
                    if not stream:
                        try:
                            data = json.loads(line)
                        except json.decoder.JSONDecodeError:
                            raise RuntimeError(f"No json: {line}")
                        if "error" in data:
                            raise RuntimeError(data["error"])
                        elif isinstance(data, list):
                            yield data[0]["generated_text"]
                        else:
                            raise RuntimeError(f"Response: {line}")
                    elif line.startswith(start):
                        line = json.loads(line[len(start):-1])
                        if not line:
                            continue
                        if "token" not in line:
                            raise RuntimeError(f"Response: {line}")
                        if not line["token"]["special"]:
                            if first:
                                yield line["token"]["text"].lstrip()
                                first = False
                            else:
                                yield line["token"]["text"]
                
            async with session.delete(f"https://huggingface.co/chat/conversation/{conversation_id}", proxy=proxy) as response:
                response.raise_for_status()


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
