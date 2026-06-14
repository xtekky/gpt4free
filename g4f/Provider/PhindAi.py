from __future__ import annotations

import re
import json

from .base_provider import AsyncProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from ..requests import StreamSession

class PhindAi(AsyncProvider, ProviderModelMixin):
    """
    Provider for phindai.org.
    """
    label = "PhindAi"
    url = "https://phindai.org"
    working = True
    needs_auth = False
    supports_stream = False
    supports_system_message = False
    supports_message_history = False
    use_stream_timeout = False

    default_model = "deepseek-v3"
    models = [default_model, "deepseek"]

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> str:
        # Extract the last user message as the query
        prompt = messages[-1]["content"]

        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
        }

        async with StreamSession(
            headers=headers,
            impersonate="chrome",
            proxy=proxy,
            timeout=120
        ) as session:
            # 1. Fetch main page to get nonce
            async with session.get(cls.url) as response:
                await raise_for_status(response)
                html = await response.text()
                
                match = re.search(r'"nonce":"([a-f0-9]+)"', html)
                if not match:
                    match = re.search(r"'nonce':'([a-f0-9]+)'", html)
                if not match:
                    raise RuntimeError("Failed to extract nonce from PhindAi response")
                nonce = match.group(1)

            # 2. Fetch the response
            ajax_url = f"{cls.url}/wp-admin/admin-ajax.php"
            payload = {
                "action": "phind_ai_send",
                "nonce": nonce,
                "message": prompt
            }

            async with session.post(ajax_url, data=payload) as response:
                await raise_for_status(response)
                try:
                    data = await response.json()
                except json.JSONDecodeError:
                    text = await response.text()
                    raise RuntimeError(f"Failed to decode JSON from PhindAi response: {text}")

                if not data.get("success"):
                    raise RuntimeError(f"PhindAi API returned success=False: {data}")

                response_text = data.get("data", {}).get("response", "")
                return response_text
