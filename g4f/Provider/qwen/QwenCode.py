from __future__ import annotations

from ...typing import Messages, AsyncResult
from ...errors import MissingAuthError
from ..template import OpenaiTemplate
from .qwenContentGenerator import QwenContentGenerator
from .qwenOAuth2 import QwenOAuth2Client
from .sharedTokenManager import TokenManagerError

class QwenCode(OpenaiTemplate):
    label = "Qwen Code 🤖"
    url = "https://qwen.ai"
    login_url = "https://github.com/QwenLM/qwen-code"
    working = True
    needs_auth = True
    active_by_default = True
    default_model = "qwen3-coder-plus"
    models = [default_model]
    client = QwenContentGenerator(QwenOAuth2Client())

    @classmethod
    def get_models(cls, **kwargs):
        if cls.live == 0:
            cls.client.shared_manager.checkAndReloadIfNeeded()
            creds = cls.client.shared_manager.getCurrentCredentials()
            if creds:
                cls.client.shared_manager.isTokenValid(creds)
            cls.live += 1
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        api_base: str = None,
        **kwargs
    ) -> AsyncResult:
        try:
            creds = await cls.client.get_valid_token()
            last_chunk = None
            async for chunk in super().create_async_generator(
                model,
                messages,
                api_key=creds.get("token", api_key),
                api_base=creds.get("endpoint", api_base),
                **kwargs
            ):
                if chunk != last_chunk:
                    yield chunk
                last_chunk = chunk
        except TokenManagerError:
            await cls.client.shared_manager.getValidCredentials(cls.client.qwen_client, True)
            creds = await cls.client.get_valid_token()
            last_chunk = None
            async for chunk in super().create_async_generator(
                model,
                messages,
                api_key=creds.get("token"),
                api_base=creds.get("endpoint"),
                **kwargs
            ):
                if chunk != last_chunk:
                    yield chunk
                last_chunk = chunk
        except:
            raise