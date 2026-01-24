from __future__ import annotations

import os

from ...typing import Messages, AsyncResult
from ...errors import MissingAuthError
from ..template import OpenaiTemplate

class Claude(OpenaiTemplate):
    label = "Claude 💥"
    url = "https://claude.ai"
    base_url = "https://g4f.space/api/claude"
    working = True
    active_by_default = True
    login_url = "https://discord.gg/qXA4Wf4Fsm"
    organization_id = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        base_url: str = base_url,
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
            base_url=f"{base_url}/{cls.organization_id}",
            headers={"cookie": api_key},
            **kwargs
        ):
            yield chunk