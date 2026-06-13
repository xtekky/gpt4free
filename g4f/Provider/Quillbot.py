import json
import uuid
from aiohttp import ClientSession
from typing import AsyncGenerator

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..providers.helper import format_prompt

class Quillbot(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://quillbot.com/ai-chat"
    api_endpoint = "https://quillbot.com/api/ai-chat/chat/conversation/{}"
    working = True
    supports_message_history = True
    supports_system_message = True

    default_model = "quillbot"
    models = ["quillbot", "quillbot-search"]
    
    model_aliases = {
        "quillbot": "quillbot",
        "quillbot-search": "quillbot-search",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator:
        conversation_id = str(uuid.uuid4())
        api_url = cls.api_endpoint.format(conversation_id)
        
        headers = {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://quillbot.com",
            "platform-type": "webapp",
            "priority": "u=1, i",
            "qb-product": "AI-CHAT",
            "referer": f"https://quillbot.com/ai-chat/c/{conversation_id}",
            "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
            "useridtoken": "empty-token",
            "webapp-version": "42.61.1"
        }

        # Format messages into a single prompt string since this is what the API expects for a single conversation request
        prompt = format_prompt(messages) + "\n\n"

        payload = {
            "message": {"content": prompt},
            "context": {
                "editorContext": "",
                "selectionContext": "",
                "userDialect": "en-us",
                "apiVersion": 2
            },
            "origin": {"name": "ai-chat.chat", "url": "https://quillbot.com"}
        }

        # Add tools if web search model is selected or explicitly requested via kwargs
        if model == "quillbot-search" or kwargs.get("web_search"):
            payload["tools"] = {"web_search_builtin": {}}

        async with ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload, proxy=proxy) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        if data.get("type") == "content" and "content" in data:
                            yield data["content"]
                    except json.JSONDecodeError:
                        pass
