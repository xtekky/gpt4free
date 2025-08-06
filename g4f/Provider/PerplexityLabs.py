from __future__ import annotations

import random
import json

from ..typing import AsyncResult, Messages
from ..requests import StreamSession, raise_for_status
from ..errors import ResponseError
from ..providers.response import FinishReason, Sources
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

API_URL = "https://www.perplexity.ai/socket.io/"
WS_URL = "wss://www.perplexity.ai/socket.io/"

class PerplexityLabs(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Perplexity Labs"
    url = "https://labs.perplexity.ai"
    working = True
    active_by_default = True

    default_model = "r1-1776"
    models = [
        default_model,
        "sonar-pro",
        "sonar",
        "sonar-reasoning",
        "sonar-reasoning-pro",
    ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
        }
        async with StreamSession(headers=headers, proxy=proxy, impersonate="chrome") as session:
            t = format(random.getrandbits(32), "08x")
            async with session.get(
                f"{API_URL}?EIO=4&transport=polling&t={t}"
            ) as response:
                await raise_for_status(response)
                text = await response.text()
            assert text.startswith("0")
            sid = json.loads(text[1:])["sid"]
            post_data = '40{"jwt":"anonymous-ask-user"}'
            async with session.post(
                f"{API_URL}?EIO=4&transport=polling&t={t}&sid={sid}",
                data=post_data
            ) as response:
                await raise_for_status(response)
                assert await response.text() == "OK"
            async with session.get(
                f"{API_URL}?EIO=4&transport=polling&t={t}&sid={sid}",
                data=post_data
            ) as response:
                await raise_for_status(response)
                assert (await response.text()).startswith("40")
            async with session.ws_connect(f"{WS_URL}?EIO=4&transport=websocket&sid={sid}", autoping=False) as ws:
                await ws.send_str("2probe")
                assert(await ws.receive_str() == "3probe")
                await ws.send_str("5")
                assert(await ws.receive_str() == "6")
                format_messages = []
                last_is_assistant = False
                for message in messages:
                    if message["role"] == "assistant":
                        if last_is_assistant:
                            continue
                        last_is_assistant = True
                    else:
                        last_is_assistant = False
                    if isinstance(message["content"], str):
                        format_messages.append({
                            "role": message["role"],
                            "content": message["content"]
                        })
                message_data = {
                    "version": "2.18",
                    "source": "default",
                    "model": model,
                    "messages": format_messages
                }
                await ws.send_str("42" + json.dumps(["perplexity_labs", message_data]))
                last_message = 0
                while True:
                    message = await ws.receive_str()
                    if message == "2":
                        if last_message == 0:
                            raise RuntimeError("Unknown error")
                        await ws.send_str("3")
                        continue
                    try:
                        if not message.startswith("42"):
                            continue
                            
                        parsed_data = json.loads(message[2:])
                        message_type = parsed_data[0]
                        data = parsed_data[1]
                        
                        # Handle error responses
                        if message_type.endswith("_query_progress") and data.get("status") == "failed":
                            error_message = data.get("text", "Unknown API error")
                            raise ResponseError(f"API Error: {error_message}\n")
                        
                        # Handle normal responses
                        if "output" in data:
                            if last_message == 0 and model == cls.default_model:
                                yield "<think>"
                            yield data["output"][last_message:]
                            last_message = len(data["output"])
                            if data["final"]:
                                if data["citations"]:
                                    yield Sources(data["citations"])
                                yield FinishReason("stop")
                                break
                    except ResponseError as e:
                        # Re-raise ResponseError directly
                        raise e
                    except Exception as e:
                        raise ResponseError(f"Error processing message: {message}") from e
