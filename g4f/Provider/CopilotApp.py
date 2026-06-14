from __future__ import annotations

import uuid
import json
import aiohttp

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages
from .helper import format_prompt
from ..providers.response import SourceLink, Sources

class CopilotApp(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Copilot App"
    url = "https://play.google.com/store/apps/details?id=com.microsoft.copilot"
    working = True
    supports_stream = True

    default_model = "smart"
    models = ["smart", "reasoning", "chat", "study", "search", "gpt-4", "gpt-4o"]
    model_aliases = {
        "Copilot": default_model,
        "gpt-4": "chat", 
        "gpt-4o": "chat",
        "o1": "reasoning",
        "o3-mini": "reasoning",
        "gpt-5": "smart",
        "think-deeper": "reasoning"
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        prompt = format_prompt(messages)
        
        headers = {
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Host": "copilot.microsoft.com",
            "User-Agent": "CopilotNative/30.0.440505001-prod (Android 14; Google; Pixel 8 Pro)",
            "X-Search-UILang": "en-US"
        }

        start_payload = {
            "timeZone": "Europe/Kiev",
            "startNewConversation": True,
            "teenSupportEnabled": True,
            "correctPersonalizationSetting": True,
            "deferredDataUseCapable": True
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://copilot.microsoft.com/c/api/start", 
                headers=headers, 
                json=start_payload,
                proxy=proxy
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to start conversation: {await resp.text()}")
                
                start_data = await resp.json()
                conversation_id = start_data.get("currentConversationId")

            client_session_id = str(uuid.uuid4())
            ws_url = f"wss://copilot.microsoft.com/c/api/chat?api-version=2&clientSessionId={client_session_id}"
            
            async with session.ws_connect(ws_url, headers=headers, proxy=proxy) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("event") == "connected":
                            break
                
                model_lower = model.lower()
                if "reasoning" in model_lower or "think" in model_lower or "o1" in model_lower or "o3" in model_lower:
                    mode = "reasoning"
                elif "smart" in model_lower or "gpt-5" in model_lower:
                    mode = "smart"
                elif "study" in model_lower:
                    mode = "study"
                elif "search" in model_lower:
                    mode = "search"
                else:
                    mode = "smart"
                
                send_payload = {
                    "event": "send",
                    "content": [{"type": "text", "text": prompt}],
                    "conversationId": conversation_id,
                    "mode": mode
                }
                await ws.send_json(send_payload)
                
                sources = {}
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        event = data.get("event")
                        if event == "appendText":
                            yield data.get("text", "")
                        elif event == "citation":
                            sources[data.get("url")] = data
                            yield SourceLink(list(sources.keys()).index(data.get("url")), data.get("url"))
                        elif event == "done":
                            if sources:
                                yield Sources(sources.values())
                            break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(f"WebSocket Error: {ws.exception()}")
