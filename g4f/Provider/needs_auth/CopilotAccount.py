from __future__ import annotations

import os
from typing import AsyncIterator

from ..Copilot import Copilot, readHAR, has_nodriver, get_access_token_and_cookies
from ...providers.response import AuthResult, RequestLogin
from ...errors import NoValidHarFileError
from ... import debug

class CopilotAccount(Copilot):
    needs_auth = True
    use_nodriver = True
    parent = "Copilot"
    default_model = "Copilot"
    default_vision_model = default_model
    model_aliases = {
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "o1": "Think Deeper",
        "dall-e-3": default_model
    }

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        try:
            cls._access_token, cls._cookies = readHAR(cls.url)
        except NoValidHarFileError as h:
            debug.log(f"Copilot: {h}")
            if has_nodriver:
                yield RequestLogin(cls.label, os.environ.get("G4F_LOGIN_URL", ""))
                cls._access_token, cls._cookies = await get_access_token_and_cookies(cls.url, proxy)
            else:
                raise h
        yield AuthResult(
            api_key=cls._access_token,
            cookies=cls.cookies_to_dict()
        )
