from __future__ import annotations

import asyncio
import json
try:
    import nodriver
except ImportError:
    pass

from ..typing import AsyncResult, Messages
from ..config import DEFAULT_MODEL
from ..requests import get_args_from_nodriver
from ..providers.base_provider import AuthFileMixin
from .template import OpenaiTemplate
from .. import debug

class EasyChat(OpenaiTemplate, AuthFileMixin):
    url = "https://chat3.eqing.tech"
    api_base = f"{url}/api/openai/v1"
    api_endpoint = f"{api_base}/chat/completions"
    working = True
    active_by_default = True
    
    default_model = "gpt-oss-120b-free"
    model_aliases = {
        DEFAULT_MODEL: default_model,
    }

    captchaToken: dict = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        extra_body: dict = None,
        **kwargs
    ) -> AsyncResult:
        args = None
        auth_file = cls.get_cache_file()
        if auth_file.exists():
            with auth_file.open("r") as f:
                args = json.load(f)
            cls.captchaToken = args.pop("captchaToken")
            if cls.captchaToken:
                debug.log("EasyChat: Using cached captchaToken.")
        async def callback(page):
            def on_request(event: nodriver.cdp.network.RequestWillBeSent, page=None):
                if event.request.url != cls.api_endpoint:
                    return
                if not event.request.post_data:
                    return
                cls.captchaToken = json.loads(event.request.post_data).get("captchaToken")
            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            button = await page.find("我已知晓")
            if button:
                await button.click()
            else:
                debug.error("No 'Agree' button found.")
            for _ in range(3):
                for _ in range(300):
                    modal = await page.find("Verifying...")
                    if not modal:
                        break
                    debug.log("EasyChaat: Waiting for captcha verification...")
                    await asyncio.sleep(1)
                if cls.captchaToken:
                    debug.log("EasyChat: Captcha token found, proceeding.")
                    break
                textarea = await page.select("textarea", 180)
                await textarea.send_keys("Hello")
                await asyncio.sleep(1)
                button = await page.select("button[class*='chat_chat-input-send']")
                if button:
                    await button.click()
            for _ in range(300):
                await asyncio.sleep(1)
                if cls.captchaToken:
                    break
            await asyncio.sleep(3)
        if not args:
            args = await get_args_from_nodriver(cls.url, proxy=proxy, callback=callback)
        if extra_body is None:
            extra_body = {}
        extra_body.setdefault("captchaToken", cls.captchaToken)
        try:
            last_chunk = None
            async for chunk in super().create_async_generator(
                model=model,
                messages=messages,
                extra_body=extra_body,
                **args
            ):
                # Remove provided by
                if last_chunk == "\n" and chunk == "\n":
                    break
                last_chunk = chunk
                yield chunk
        except Exception as e:
            if "CLEAR-CAPTCHA-TOKEN" in str(e):
                auth_file.unlink(missing_ok=True)
                cls.captchaToken = None
                debug.log("EasyChat: Captcha token cleared, please try again.")
            raise e
        with auth_file.open("w") as f:
            json.dump({**args, "captchaToken": cls.captchaToken}, f)
        