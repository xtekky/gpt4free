from __future__ import annotations

import uuid, json, asyncio, os
from py_arkose_generator.arkose import get_values_for_request
from asyncstdlib.itertools import tee
from async_property import async_cached_property
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from ..base_provider import AsyncGeneratorProvider
from ..helper import get_event_loop, format_prompt, get_cookies
from ...webdriver import get_browser
from ...typing import AsyncResult, Messages
from ...requests import StreamSession

models = {
    "gpt-3.5":       "text-davinci-002-render-sha",
    "gpt-3.5-turbo": "text-davinci-002-render-sha",
    "gpt-4":         "gpt-4",
    "gpt-4-gizmo":   "gpt-4-gizmo"
}

class OpenaiChat(AsyncGeneratorProvider):
    url                   = "https://chat.openai.com"
    working               = True
    needs_auth            = True
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    _cookies: dict        = {}

    @classmethod
    async def create(
        cls,
        prompt: str = None,
        model: str = "",
        messages: Messages = [],
        history_disabled: bool = False,
        action: str = "next",
        conversation_id: str = None,
        parent_id: str = None,
        **kwargs
    ) -> Response:
        if prompt:
            messages.append({
                "role": "user",
                "content": prompt
            })
        generator = cls.create_async_generator(
            model,
            messages,
            history_disabled=history_disabled,
            action=action,
            conversation_id=conversation_id,
            parent_id=parent_id,
            response_fields=True,
            **kwargs
        )
        return Response(
            generator,
            await anext(generator),
            action,
            messages,
            kwargs
        )

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        access_token: str = None,
        cookies: dict = None,
        auto_continue: bool = False,
        history_disabled: bool = True,
        action: str = "next",
        conversation_id: str = None,
        parent_id: str = None,
        response_fields: bool = False,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "gpt-3.5"
        elif model not in models:
            raise ValueError(f"Model are not supported: {model}")
        if not parent_id:
            parent_id = str(uuid.uuid4())
        if not cookies:
            cookies = cls._cookies
        if not access_token:
            if not cookies:
                cls._cookies = cookies = get_cookies("chat.openai.com")
            if "access_token" in cookies:
                access_token = cookies["access_token"]
        if not access_token:
            login_url = os.environ.get("G4F_LOGIN_URL")
            if login_url:
                yield f"Please login: [ChatGPT]({login_url})\n\n"
            cls._cookies["access_token"] = access_token = await cls.browse_access_token(proxy)
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {access_token}",
        }
        async with StreamSession(
            proxies={"https": proxy},
            impersonate="chrome110",
            headers=headers,
            timeout=timeout,
            cookies=dict([(name, value) for name, value in cookies.items() if name == "_puid"])
        ) as session:
            end_turn = EndTurn()
            while not end_turn.is_end:
                data = {
                    "action": action,
                    "arkose_token": await get_arkose_token(proxy, timeout),
                    "conversation_id": conversation_id,
                    "parent_message_id": parent_id,
                    "model": models[model],
                    "history_and_training_disabled": history_disabled and not auto_continue,
                }
                if action != "continue":
                    prompt = format_prompt(messages) if not conversation_id else messages[-1]["content"]
                    data["messages"] = [{
                        "id": str(uuid.uuid4()),
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": [prompt]},
                    }]
                async with session.post(f"{cls.url}/backend-api/conversation", json=data) as response:
                    try:
                        response.raise_for_status()
                    except:
                        raise RuntimeError(f"Error {response.status_code}: {await response.text()}")
                    last_message = 0
                    async for line in response.iter_lines():
                        if not line.startswith(b"data: "):
                            continue
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
                        if line["message"]["author"]["role"] != "assistant":
                            continue
                        if line["message"]["metadata"]["message_type"] in ("next", "continue", "variant"):
                            conversation_id = line["conversation_id"]
                            parent_id = line["message"]["id"]
                            if response_fields:
                                response_fields = False
                                yield ResponseFields(conversation_id, parent_id, end_turn)
                            new_message = line["message"]["content"]["parts"][0]
                            yield new_message[last_message:]
                            last_message = len(new_message)
                        if "finish_details" in line["message"]["metadata"]:
                            if line["message"]["metadata"]["finish_details"]["type"] == "stop":
                                end_turn.end()
                if not auto_continue:
                    break
                action = "continue"
                await asyncio.sleep(5)

    @classmethod
    async def browse_access_token(cls, proxy: str = None) -> str:
        def browse() -> str:
            driver = get_browser(proxy=proxy)
            try:
                driver.get(f"{cls.url}/")
                WebDriverWait(driver, 1200).until(
                    EC.presence_of_element_located((By.ID, "prompt-textarea"))
                )
                javascript = """
access_token = (await (await fetch('/api/auth/session')).json())['accessToken'];
expires = new Date(); expires.setTime(expires.getTime() + 60 * 60 * 24 * 7); // One week
document.cookie = 'access_token=' + access_token + ';expires=' + expires.toUTCString() + ';path=/';
return access_token;
"""
                return driver.execute_script(javascript)
            finally:
                driver.quit()
        loop = get_event_loop()
        return await loop.run_in_executor(
            None,
            browse
        )
    
async def get_arkose_token(proxy: str = None, timeout: int = None) -> str:
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
        timeout=timeout
    ) as session:
        async with session.post(**args_for_request) as response:
            response.raise_for_status()
            decoded_json = await response.json()
            if "token" in decoded_json:
                return decoded_json["token"]
            raise RuntimeError(f"Response: {decoded_json}")
        
class EndTurn():
    def __init__(self):
        self.is_end = False

    def end(self):
        self.is_end = True

class ResponseFields():
    def __init__(
        self,
        conversation_id: str,
        message_id: str,
        end_turn: EndTurn
    ):
        self.conversation_id = conversation_id
        self.message_id = message_id
        self._end_turn = end_turn
        
class Response():
    def __init__(
        self,
        generator: AsyncResult,
        fields: ResponseFields,
        action: str,
        messages: Messages,
        options: dict
    ):
        self.aiter, self.copy = tee(generator)
        self.fields = fields
        self.action = action
        self._messages = messages
        self._options = options

    def __aiter__(self):
        return self.aiter
    
    @async_cached_property
    async def message(self) -> str:
        return "".join([chunk async for chunk in self.copy])
    
    async def next(self, prompt: str, **kwargs) -> Response:
        return await OpenaiChat.create(
            **self._options,
            prompt=prompt,
            messages=await self.messages,
            action="next",
            conversation_id=self.fields.conversation_id,
            parent_id=self.fields.message_id,
            **kwargs
        )
    
    async def do_continue(self, **kwargs) -> Response:
        if self.end_turn:
            raise RuntimeError("Can't continue message. Message already finished.")
        return await OpenaiChat.create(
            **self._options,
            messages=await self.messages,
            action="continue",
            conversation_id=self.fields.conversation_id,
            parent_id=self.fields.message_id,
            **kwargs
        )
    
    async def variant(self, **kwargs) -> Response:
        if self.action != "next":
            raise RuntimeError("Can't create variant from continue or variant request.")
        return await OpenaiChat.create(
            **self._options,
            messages=self._messages,
            action="variant",
            conversation_id=self.fields.conversation_id,
            parent_id=self.fields.message_id,
            **kwargs
        )
    
    @async_cached_property
    async def messages(self):
        messages = self._messages
        messages.append({
            "role": "assistant", "content": await self.message
        })
        return messages
    
    @property
    def end_turn(self):
        return self.fields._end_turn.is_end