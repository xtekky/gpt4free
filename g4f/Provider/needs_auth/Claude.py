from __future__ import annotations

import os

from ...typing import Messages, AsyncResult
from ...errors import MissingAuthError
from ..template import OpenaiTemplate

class Claude(OpenaiTemplate):
    label = "Claude 💥"
    url = "https://claude.ai"
    api_base = "https://g4f.dev/api/claude"
    working = True
    active_by_default = True
    login_url = "https://discord.gg/qXA4Wf4Fsm"
    orginization_id = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        api_base: str = api_base,
        **kwargs
    ) -> AsyncResult:
        api_key = os.environ.get("CLAUDE_COOKIE", api_key)
        if not api_key:
            raise MissingAuthError("Claude cookie not found. Please set the 'CLAUDE_COOKIE' environment variable.")
        if not cls.organization_id:
            cls.organization_id = os.environ.get("CLAUDE_ORGANIZATION_ID")
            if not cls.organization_id:
                raise MissingAuthError("Claude organization ID not found. Please set the 'CLAUDE_ORGANIZATION_ID' environment variable.")
        async for chunk in super().create_async_generator(
            model=model,
            messages=messages,
            api_base=f"{api_base}/{cls.organization_id}",
            headers={"cookie": api_key},
            **kwargs
        ):
            yield chunk