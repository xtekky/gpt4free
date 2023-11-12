from __future__ import annotations

import uuid, json, time, asyncio
from py_arkose_generator.arkose import get_values_for_request

from ..base_provider import AsyncGeneratorProvider
from ..helper import get_browser, get_cookies, format_prompt, get_event_loop
from ...typing import AsyncResult, Messages
from ...requests import StreamSession
from ... import debug

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
        auto_continue: bool = False,
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
        messages = [
            {
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [format_prompt(messages)]},
            },
        ]
        message_id = str(uuid.uuid4())
        data = {
            "action": "next",
            "arkose_token": await get_arkose_token(proxy),
            "messages": messages,
            "conversation_id": None,
            "parent_message_id": message_id,
            "model": "text-davinci-002-render-sha",
            "history_and_training_disabled": not auto_continue,
        }
        conversation_id = None
        while not end_turn:
            if not auto_continue:
                end_turn = True
            async with StreamSession(
                proxies=proxies,
                headers=headers,
                impersonate="chrome107",
                timeout=timeout
            ) as session:
                async with session.post(f"{cls.url}/backend-api/conversation", json=data) as response:
                    try:
                        response.raise_for_status()
                    except:
                        raise RuntimeError(f"Response: {await response.text()}")
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
                            if "message" not in line:
                                continue
                            if "error" in line and line["error"]:
                                raise RuntimeError(line["error"])
                            end_turn = line["message"]["end_turn"]
                            message_id = line["message"]["id"]
                            if line["conversation_id"]:
                                conversation_id = line["conversation_id"]
                            if "message_type" not in line["message"]["metadata"]:
                                continue
                            if line["message"]["metadata"]["message_type"] in ("next", "continue"):
                                new_message = line["message"]["content"]["parts"][0]
                                yield new_message[len(last_message):]
                                last_message = new_message
                            if end_turn:
                                return
                data = {
                    "action": "continue",
                    "arkose_token": await get_arkose_token(proxy),
                    "conversation_id": conversation_id,
                    "parent_message_id": message_id,
                    "model": "text-davinci-002-render-sha",
                    "history_and_training_disabled": False,
                }
                await asyncio.sleep(5)

    @classmethod
    async def browse_access_token(cls) -> str:
        def browse() -> str:
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
                driver.close()
                time.sleep(0.1)
                driver.quit()
        loop = get_event_loop()
        return await loop.run_in_executor(
            None,
            browse
        )

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
            cls._access_token = await cls.browse_access_token()
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
    
async def get_arkose_token(proxy: str = None) -> str:
    config = {
        "pkey": "3D86FBBA-9D22-402A-B512-3420086BA6CC",
        "surl": "https://tcr9i.chat.openai.com",
        "headers": {
            "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
        },
        "site": "https://chat.openai.com",
    }
    args_for_request = get_values_for_request(config)
    async with StreamSession(
        proxies={"https": proxy},
        impersonate="chrome107",
    ) as session:
        async with session.post(**args_for_request) as response:
            response.raise_for_status()
            decoded_json = await response.json()
            if "token" in decoded_json:
                return decoded_json["token"]
            raise RuntimeError(f"Response: {decoded_json}")