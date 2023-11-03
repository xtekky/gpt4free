from __future__ import annotations

import uuid, json, time, os
import tempfile, shutil, asyncio
import sys, subprocess

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
                "arkose_token": await get_arkose_token(proxy),
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
                        if "message" not in line:
                            continue
                        if "error" in line and line["error"]:
                            raise RuntimeError(line["error"])
                        if "message_type" not in line["message"]["metadata"]:
                            continue
                        if line["message"]["metadata"]["message_type"] == "next":
                            new_message = line["message"]["content"]["parts"][0]
                            yield new_message[len(last_message):]
                            last_message = new_message

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
    dir = os.path.dirname(os.path.dirname(__file__))
    include = f'{dir}/npm/node_modules/funcaptcha'
    config = {
        "pkey": "3D86FBBA-9D22-402A-B512-3420086BA6CC",
        "surl": "https://tcr9i.chat.openai.com",
        "data": {},
        "headers": {
            "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
        },
        "site": "https://chat.openai.com",
        "proxy": proxy
    }
    source = """
fun = require({include})
config = {config}
fun.getToken(config).then(token => {
    console.log(token.token)
})
"""
    source = source.replace('{include}', json.dumps(include))
    source = source.replace('{config}', json.dumps(config))
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(source.encode())
    tmp.close()
    try:
        return await exec_js(tmp.name)
    finally:
        os.unlink(tmp.name)

async def exec_js(file: str) -> str:
    node = shutil.which("node")
    if not node:
        if debug.logging:
            print('OpenaiChat: "node" not found')
        return
    if sys.platform == 'win32':
        p = subprocess.Popen(
            [node, file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate()
        if p.returncode == 0:
            return stdout.decode()
        raise RuntimeError(f"Exec Error: {stderr.decode()}")
    p = await asyncio.create_subprocess_exec(
        node, file,
        stderr=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE
    )
    stdout, stderr = await p.communicate()
    if p.returncode == 0:
        return stdout.decode()
    raise RuntimeError(f"Exec Error: {stderr.decode()}")