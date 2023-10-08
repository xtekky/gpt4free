from __future__ import annotations

import uuid, json, time

from ..base_provider import AsyncGeneratorProvider
from ..helper import get_browser, get_cookies, format_prompt
from ...typing import AsyncGenerator
from ...requests import StreamSession
import browser_cookie3

import requests

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
        messages: list[dict[str, str]],
        proxy: str = None,
        access_token: str = None,
        cookies: dict = None,
        **kwargs: dict
    ) -> AsyncGenerator:
        proxies = {"https": proxy}
        cookie=browser_cookie3.chrome(domain_name='chat.openai.com')
        cookies=requests.utils.dict_from_cookiejar(cookie)
        cookies_str=''
        for k,v in cookies.items():
            cookies_str+=f"{k}={v}; "
            
        if not access_token:
            access_token = await cls.get_access_token(cookies, proxies)            
        
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {access_token}",
            "cookie":cookies_str,
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
        }
            
        async with StreamSession(proxies=proxies, headers=headers, impersonate="chrome107") as session:
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
                first_message = True  # track if it's the first message
                async for line in response.iter_lines():
                    if line.startswith(b"data: "):
                        line = line[6:]
                        if line == b"[DONE]":
                            break
                        line = json.loads(line)
                        if "message" in line and not line["message"]["end_turn"]:
                            new_message = line["message"]["content"]["parts"][0]
                            if first_message:
                                first_message = False  #set to False after the first message
                            else:
                                yield new_message[len(last_message):]
                            last_message = new_message


    @classmethod
    def fetch_access_token(cls,proxies=None) -> str:
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium import webdriver
        except ImportError:
            return

        # Define proxy settings
        proxy = proxies['https']
        # print(proxy)
        options = webdriver.ChromeOptions()
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        options.add_argument(f'--proxy-server={proxy}')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36")
        driver = webdriver.Chrome(options=options)  # Initialize the driver with options

        if not driver:
            return

        driver.get(f"{cls.url}/")
        try:
            WebDriverWait(driver, 1200).until(
                EC.presence_of_element_located((By.ID, "prompt-textarea"))
            )
            javascript = "return (await (await fetch('/api/auth/session')).json())['accessToken']"
            return driver.execute_script(javascript)
        finally:
            driver.quit()

    @classmethod
    async def get_access_token(cls, cookies: dict = None, proxies: dict = None) -> str:
        if not cls._access_token:
            cookies = cookies if cookies else get_cookies("chat.openai.com")
            async with StreamSession(proxies=proxies, cookies=cookies, impersonate="chrome107") as session:
                async with session.get(f"{cls.url}/api/auth/session") as response:
                    response.raise_for_status()
                    auth = await response.json()
                    if "accessToken" in auth:
                        cls._access_token = auth["accessToken"]
            cls._access_token = cls.fetch_access_token(proxies=proxies)
            if not cls._access_token:
                raise RuntimeError("Missing access token")
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
