from __future__ import annotations

import uuid, json, time

from ..base_provider import AsyncGeneratorProvider
from ..helper import get_browser, get_cookies, format_prompt
from ...typing import AsyncResult, Messages
from ...requests import StreamSession

class OpenaiChat(AsyncGeneratorProvider):
    url                   = "https://chat.openai.com"
    needs_auth            = True
    working               = True
    supports_gpt_35_turbo = True
    _access_token         = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        access_token: str = None,
        cookies: dict = None,
        **kwargs
    ) -> AsyncResult:
        proxies = {"https": proxy}
        if not access_token:
            access_token = await cls.get_access_token(cookies, proxies)
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {access_token}",
        }
        async with StreamSession(
            proxies=proxies,
            headers=headers,
            impersonate="chrome107",
            timeout=timeout
        ) as session:
            messages = [
                {
                    "id": str(uuid.uuid4()),
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [format_prompt(messages)]},
                },
            ]
            data = {
                "action": "next",
                "messages": messages,
                "conversation_id": None,
                "parent_message_id": str(uuid.uuid4()),
                "model": "text-davinci-002-render-sha",
                "history_and_training_disabled": True,
            }
            async with session.post(f"{cls.url}/backend-api/conversation", json=data) as response:
                response.raise_for_status()
                last_message = ""
                async for line in response.iter_lines():
                    if line.startswith(b"data: "):
                        line = line[6:]
                        if line == b"[DONE]":
                            break
                        try:
                            line = json.loads(line)
                        except:
                            continue
                        if "message" not in line or "message_type" not in line["message"]["metadata"]:
                            continue
                        if line["message"]["metadata"]["message_type"] == "next":
                            new_message = line["message"]["content"]["parts"][0]
                            yield new_message[len(last_message):]
                            last_message = new_message

    @classmethod
    def browse_access_token(cls) -> str:
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            driver = get_browser()
        except ImportError:
            return

        driver.get(f"{cls.url}/")
        try:
            WebDriverWait(driver, 1200).until(
                EC.presence_of_element_located((By.ID, "prompt-textarea"))
            )
            javascript = "return (await (await fetch('/api/auth/session')).json())['accessToken']"
            return driver.execute_script(javascript)
        finally:
            time.sleep(1)
            driver.quit()

    @classmethod
    async def fetch_access_token(cls, cookies: dict, proxies: dict = None) -> str:
        async with StreamSession(proxies=proxies, cookies=cookies, impersonate="chrome107") as session:
            async with session.get(f"{cls.url}/api/auth/session") as response:
                response.raise_for_status()
                auth = await response.json()
                if "accessToken" in auth:
                    return auth["accessToken"]

    @classmethod
    async def get_access_token(cls, cookies: dict = None, proxies: dict = None) -> str:
        if not cls._access_token:
            cookies = cookies if cookies else get_cookies("chat.openai.com")
            if cookies:
                cls._access_token = await cls.fetch_access_token(cookies, proxies)
        if not cls._access_token:
            cls._access_token = cls.browse_access_token()
        if not cls._access_token:
            raise RuntimeError("Read access token failed")
        return cls._access_token

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
            ("access_token", "str"),
            ("cookies", "dict[str, str]")
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"