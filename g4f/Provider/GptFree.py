import json
from aiohttp import ClientSession
from typing import AsyncGenerator

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import to_data_uri

class GptFree(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://gptfree.com"
    api_endpoint = "https://us-central1-gptfree-2.cloudfunctions.net/agent_stream"
    working = True
    supports_message_history = True
    supports_system_message = False

    default_model = ""

    _id_token: str = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://gptfree.com",
            "pragma": "no-cache",
            "referer": "https://gptfree.com/",
            "sec-ch-ua": '"Not;A=Brand";v="8", "Chromium";v="150", "Google Chrome";v="150"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
        }

        history = []
        current_message = ""
        images = []

        for msg in messages:
            content = msg["content"]
            
            # Handle multimodal messages (where content is a list of dicts)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        img_url = part.get("image_url", {}).get("url")
                        if img_url:
                            images.append(img_url)
                content = "\n".join(text_parts)
                
            if msg["role"] == "system":
                history.append({"type": "user", "content": content})
            elif msg["role"] == "user":
                current_message = content
                history.append({"type": "user", "content": content})
            elif msg["role"] == "assistant":
                history.append({"type": "agent", "content": content})

        # Process images from kwargs
        if "image" in kwargs and kwargs["image"]:
            images.append(to_data_uri(kwargs["image"]))
        if "images" in kwargs and kwargs["images"]:
            for img in kwargs["images"]:
                images.append(to_data_uri(img[0] if isinstance(img, tuple) else img))
        if "media" in kwargs and kwargs["media"]:
            for m in kwargs["media"]:
                images.append(to_data_uri(m[0]))

        # The last message is the current one
        if history and history[-1]["type"] == "user":
            current_message = history.pop()["content"]
        
        if not current_message.strip():
            if history and history[-1]["type"] == "user":
                current_message = history.pop()["content"]
            else:
                current_message = "Analyze this image" if images else "Hello"

        payload = {
            "message": current_message,
            "images": images,
            "history": history
        }
        firebase_api_key = "AIzaSyBdU-Np8RSh1tPSsPOWg3qIm6PnVK5PQb4"

        async with ClientSession() as session:
            for attempt in range(2):
                if not cls._id_token:
                    auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={firebase_api_key}"
                    async with session.post(auth_url, json={"returnSecureToken": True}, proxy=proxy) as auth_resp:
                        auth_resp.raise_for_status()
                        auth_data = await auth_resp.json()
                        cls._id_token = auth_data["idToken"]

                headers["Authorization"] = f"Bearer {cls._id_token}"

                async with session.post(
                    cls.api_endpoint,
                    headers=headers,
                    json=payload,
                    proxy=proxy
                ) as response:
                    if response.status == 401:
                        # Token expired or invalid, clear it and retry
                        cls._id_token = None
                        continue

                    response.raise_for_status()

                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '{}':
                                continue
                            try:
                                data = json.loads(data_str)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                pass
                    break
