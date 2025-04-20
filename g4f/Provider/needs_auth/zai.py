from __future__ import annotations
from aiohttp import ClientSession
from g4f.typing import AsyncResult, Messages
from g4f.requests.raise_for_status import raise_for_status
from g4f.Provider.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from g4f.Provider.helper import format_prompt
from g4f.providers.response import FinishReason, JsonConversation

class Conversation(JsonConversation):
    pass

class ZAI(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chat.z.ai"
    api_endpoint = "https://chat.z.ai/api/chat/completions"
    working = True
    needs_auth = True
    supports_stream = True
    supports_message_history = True
    supports_system_message = True

    # Add all available models
    default_model = "main_chat"
    models = ["main_chat", "zero", "deep-research"]
    model_display_names = {
        "main_chat": "GLM-4-32B",
        "zero": "Z1-32B",
        "deep-research": "Z1-Rumination"
    }
    model_descriptions = {
        "main_chat": "Great for everyday tasks",
        "zero": "Proficient in reasoning",
        "deep-research": "Deep Research, expert in synthesizing insights from the web"
    }
    model_aliases = {
        "GLM-4-32B": "main_chat",
        "Z1-32B": "zero",
        "Z1-Rumination": "deep-research"
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        conversation: Conversation = None,
        return_conversation: bool = False,
        token: str = None,
        user_id: str = "7080a6c5-5fcc-4ea4-a85f-3b3fac905cf2",
        chat_id: str = "local",
        request_id: str = "633dcec1-adb4-4a2e-bfc4-c4599cc564be",
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        display_name = cls.model_display_names.get(model, model)
        description = cls.model_descriptions.get(model, "")
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,mr;q=0.8",
            **({"authorization": f"Bearer {token}"} if token else {}),
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://chat.z.ai",
            "referer": "https://chat.z.ai/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        }
        cookies = kwargs.get("cookies") or {
            "token": token
        }
        payload = {
            "stream": True,
            "model": model,
            "messages": [{"role": "user", "content": messages[-1]["content"]}],
            "params": {},
            "tool_servers": [],
            "features": {
                "image_generation": False,
                "code_interpreter": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": False
            },
            "variables": {
                "{{USER_NAME}}": "Guest-1745003417617",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": "2025-04-21 00:04:12",
                "{{CURRENT_DATE}}": "2025-04-21",
                "{{CURRENT_TIME}}": "00:04:12",
                "{{CURRENT_WEEKDAY}}": "Monday",
                "{{CURRENT_TIMEZONE}}": "Asia/Calcutta",
                "{{USER_LANGUAGE}}": "en-US"
            },
            "model_item": {
                "id": model,
                "name": display_name,
                "owned_by": "openai",
                "openai": {
                    "id": model,
                    "name": model,
                    "owned_by": "openai",
                    "openai": {"id": model},
                    "urlIdx": 0
                },
                "urlIdx": 0,
                "info": {
                    "id": model,
                    "user_id": user_id,
                    "base_model_id": None,
                    "name": display_name,
                    "params": {"max_tokens": 4096, "top_p": 0.95, "temperature": 0.6, "top_k": 40},
                    "meta": {
                        "profile_image_url": "/static/favicon.png",
                        "description": description,
                        "capabilities": {
                            "vision": False,
                            "citations": True,
                            "preview_mode": False,
                            "web_search": True,
                            "language_detection": True,
                            "restore_n_source": True
                        },
                        "suggestion_prompts": None,
                        "tags": []
                    },
                    "access_control": None,
                    "is_active": True,
                    "updated_at": 1744522361,
                    "created_at": 1744522361
                },
                "actions": [],
                "tags": []
            },
            "chat_id": chat_id,
            "id": request_id
        }

        async with ClientSession() as session:
            async with session.post(
                cls.api_endpoint,
                headers=headers,
                cookies=cookies,
                json=payload,
                proxy=proxy
            ) as resp:
                await raise_for_status(resp)
                async for line in resp.content:
                    yield line.decode("utf-8")