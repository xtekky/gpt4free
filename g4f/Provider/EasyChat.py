from __future__ import annotations

import json
import base64
import hashlib
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..config import DEFAULT_MODEL
from ..providers.base_provider import ProviderModelMixin
from .template import OpenaiTemplate
from .. import debug

class EasyChat(OpenaiTemplate, ProviderModelMixin):
    url = "https://chat3.eqing.tech"
    base_url = f"{url}/api/openai/v1"
    api_endpoint = f"{base_url}/chat/completions"
    working = True
    active_by_default = True
    use_model_names = True
    
    default_model = DEFAULT_MODEL.split("/")[-1]
    model_aliases = {
        DEFAULT_MODEL: f"{default_model}-free",
    }

    captchaToken: str = None

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls.models:
            models = super().get_models(**kwargs)
            models = {m.replace("-free", ""): m for m in models if m.endswith("-free")}
            cls.model_aliases.update(models)
            cls.models = list(models)
        return cls.models

    @classmethod
    async def _solve_altcha(cls, proxy: str = None) -> str:
        debug.log("EasyChat: Solving Altcha...")
        async with ClientSession() as session:
            async with session.get(f"{cls.url}/api/altcaptcha/challenge", proxy=proxy, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
                "Accept": "application/json"
            }) as response:
                response.raise_for_status()
                data = await response.json()
                
            salt = data['salt']
            challenge = data['challenge']
            maxnumber = data['maxnumber']
            algorithm = data['algorithm']
            signature = data['signature']
            
            for n in range(maxnumber + 1):
                text = f"{salt}{n}".encode('utf-8')
                if algorithm == "SHA-512":
                    h = hashlib.sha512(text).hexdigest()
                elif algorithm == "SHA-256":
                    h = hashlib.sha256(text).hexdigest()
                else:
                    raise ValueError(f"Unknown Altcha algorithm: {algorithm}")
                    
                if h == challenge:
                    payload = {
                        "algorithm": algorithm,
                        "challenge": challenge,
                        "number": n,
                        "salt": salt,
                        "signature": signature
                    }
                    token = base64.b64encode(json.dumps(payload).encode()).decode()
                    debug.log(f"EasyChat: Altcha solved (n={n})")
                    return token
            raise ValueError("Failed to solve Altcha")

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
        model = cls.get_model(model.replace("-free", ""))
        
        # Always solve Altcha fresh to avoid expiration
        cls.captchaToken = await cls._solve_altcha(proxy=proxy)
        
        if extra_body is None:
            extra_body = {}
        extra_body["captchaToken"] = cls.captchaToken

        try:
            last_chunk = None
            async for chunk in super().create_async_generator(
                model=model,
                messages=messages,
                stream=stream,
                extra_body=extra_body,
                proxy=proxy,
                **kwargs
            ):
                # Remove provided by
                if last_chunk == "\n" and chunk == "\n":
                    break
                last_chunk = chunk
                yield chunk
        except Exception as e:
            raise e