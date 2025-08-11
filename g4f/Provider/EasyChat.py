from __future__ import annotations

import os
import asyncio
import requests
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
from .helper import get_last_user_message
from .. import debug

class EasyChat(OpenaiTemplate, AuthFileMixin):
    url = "https://chat3.eqing.tech"
    api_base = f"{url}/api/openai/v1"
    api_endpoint = f"{api_base}/chat/completions"
    working = True
    active_by_default = True
    use_model_names = True
    
    default_model = DEFAULT_MODEL.split("/")[-1]
    model_aliases = {
        DEFAULT_MODEL: f"{default_model}-free",
    }

    captchaToken: str = None
    share_url: str = None
    looked: bool = False

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls.models:
            models = super().get_models(**kwargs)
            models = {m.replace("-free", ""): m for m in models if m.endswith("-free")}
            cls.model_aliases.update(models)
            cls.models = list(models)
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        extra_body: dict = None,
        **kwargs
    ) -> AsyncResult:
        cls.share_url = os.getenv("G4F_SHARE_URL")
        model = cls.get_model(model)
        args = None
        cache_file = cls.get_cache_file()
        async def callback(page):
            cls.captchaToken = None
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
                    debug.log("EasyChat: Waiting for captcha verification...")
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
        if cache_file.exists():
            with cache_file.open("r") as f:
                args = json.load(f)
            cls.captchaToken = args.pop("captchaToken")
            if cls.captchaToken:
                debug.log("EasyChat: Using cached captchaToken.")
        elif not cls.looked and cls.share_url:
            cls.looked = True
            try:
                debug.log("No cache file found, trying to fetch from share URL.")
                response = requests.get(cls.share_url, params={
                    "prompt": get_last_user_message(messages),
                    "model": model,
                    "provider": cls.__name__
                })
                response.raise_for_status()
                text, *sub = response.text.split("\n" * 10 + "<!--", 1)
                if sub:
                    debug.log("Save args to cache file:", str(cache_file))
                    with cache_file.open("w") as f:
                        f.write(sub[0].strip())
                yield text
            finally:
                cls.looked = False
            return
        for _ in range(2):
            if not args:
                args = await get_args_from_nodriver(cls.url, proxy=proxy, callback=callback, user_data_dir=None)
            if extra_body is None:
                extra_body = {}
            extra_body.setdefault("captchaToken", cls.captchaToken)
            try:
                last_chunk = None
                async for chunk in super().create_async_generator(
                    model=model,
                    messages=messages,
                    stream=True,
                    extra_body=extra_body,
                    **args,
                    **kwargs
                ):
                    # Remove provided by
                    if last_chunk == "\n" and chunk == "\n":
                        break
                    last_chunk = chunk
                    yield chunk
            except Exception as e:
                if "CLEAR-CAPTCHA-TOKEN" in str(e):
                    debug.log("EasyChat: Captcha token expired, clearing cache file.")
                    cache_file.unlink(missing_ok=True)
                    args = None
                    continue
                raise e
            break
        if not args:
            raise ValueError("Failed to retrieve arguments for EasyChat.")
        if os.getenv("G4F_SHARE_AUTH"):
            yield "\n" * 10
            yield "<!--"
            yield json.dumps({**args, "captchaToken": cls.captchaToken})
        with cache_file.open("w") as f:
            json.dump({**args, "captchaToken": cls.captchaToken}, f)
            