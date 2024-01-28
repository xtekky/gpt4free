from __future__ import annotations

from datetime import datetime

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from ..requests import StreamSession

class Phind(AsyncGeneratorProvider):
    url = "https://www.phind.com"
    working = True
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
            impersonate="chrome110",
            proxies={"https": proxy},
            timeout=timeout
        ) as session:
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
                    "answerModel": "GPT-4" if model.startswith("gpt-4") else "Phind Model",
                    "creativeMode": creative_mode,
                    "customLinks": []
                },
                "context": "\n".join([message["content"] for message in messages if message["role"] == "system"]),
                "rewrittenQuestion": prompt,
                "challenge": 0.21132115912208504
            }
            async with session.post(f"https://https.api.phind.com/infer/", headers=headers, json=data) as response:
                new_line = False
                async for line in response.iter_lines():
                    if line.startswith(b"data: "):
                        chunk = line[6:]
                        if chunk.startswith(b'<PHIND_DONE/>'):
                            break
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
