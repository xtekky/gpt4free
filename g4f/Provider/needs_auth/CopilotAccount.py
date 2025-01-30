from __future__ import annotations

import os
from typing import AsyncIterator

from ..base_provider import AsyncAuthedProvider
from ..Copilot import Copilot, readHAR, has_nodriver, get_access_token_and_cookies
from ...providers.response import AuthResult, RequestLogin
from ...typing import AsyncResult, Messages
from ...errors import NoValidHarFileError
from ... import debug

def cookies_to_dict():
    return Copilot._cookies if isinstance(Copilot._cookies, dict) else {c.name: c.value for c in Copilot._cookies}

class CopilotAccount(AsyncAuthedProvider, Copilot):
    needs_auth = True
    use_nodriver = True
    parent = "Copilot"
    default_model = "Copilot"
    default_vision_model = default_model
    models = [default_model]
    image_models = models
    model_aliases = {
        "dall-e-3": default_model
    }

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        try:
            Copilot._access_token, Copilot._cookies = readHAR(cls.url)
        except NoValidHarFileError as h:
            debug.log(f"Copilot: {h}")
            if has_nodriver:
                login_url = os.environ.get("G4F_LOGIN_URL")
                if login_url:
                    yield RequestLogin(cls.label, login_url)
                Copilot._access_token, Copilot._cookies = await get_access_token_and_cookies(cls.url, proxy)
            else:
                raise h
        yield AuthResult(
            api_key=Copilot._access_token,
            cookies=cookies_to_dict()
        )

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        **kwargs
    ) -> AsyncResult:
        Copilot._access_token = getattr(auth_result, "api_key")
        Copilot._cookies = getattr(auth_result, "cookies")
        Copilot.needs_auth = cls.needs_auth
        for chunk in Copilot.create_completion(model, messages, **kwargs):
            yield chunk
        auth_result.cookies = cookies_to_dict()