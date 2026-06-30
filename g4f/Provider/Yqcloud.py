from __future__ import annotations
import time
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..providers.response import FinishReason, JsonConversation

class Conversation(JsonConversation):
    userId: str = None
    message_history: Messages = []

    def __init__(self, model: str):
        self.model = model
        self.userId = f"#/chat/{int(time.time() * 1000)}"

class Yqcloud(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chat9.yqcloud.top"
    api_endpoint = "https://api.binjie.fun/api/generateStream"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "gpt-4"
    models = [default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        conversation: Conversation = None,
        return_conversation: bool = True,
        **kwargs
    ) -> AsyncResult:      
        model = cls.get_model(model)
        headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": f"{cls.url}",
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        
        if conversation is None:
            conversation = Conversation(model)
            conversation.message_history = messages
        else:
            conversation.message_history.append(messages[-1])

        # Extract system message if present
        system_message = ""
        current_messages = conversation.message_history
        if current_messages and current_messages[0]["role"] == "system":
            system_message = current_messages[0]["content"]
            current_messages = current_messages[1:]
        
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(current_messages)
            data = {
                "prompt": prompt,
                "userId": conversation.userId,
                "network": True,
                "system": system_message,
                "withoutContext": False,
                "stream": stream
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                full_message = ""
                async for chunk in response.content:
                    if chunk:
                        message = chunk.decode()
                        yield message
                        full_message += message

                if return_conversation:
                    conversation.message_history.append({"role": "assistant", "content": full_message})
                    yield conversation
                
                yield FinishReason("stop")
