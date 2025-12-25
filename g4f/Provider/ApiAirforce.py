from __future__ import annotations

from ..typing import Messages, AsyncResult
from .template import OpenaiTemplate
from ..errors import RateLimitError

class ApiAirforce(OpenaiTemplate):
    label = "Api.Airforce"
    url = "https://api.airforce"
    login_url = "https://panel.api.airforce/dashboard"
    base_url = "https://api.airforce/v1"
    working = True
    active_by_default = True
    use_image_size = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages = None,
        **kwargs
    ) -> AsyncResult:
        ratelimit_message = "Ratelimit Exceeded!"
        buffer = ""
        async for chunk in super().create_async_generator(
            model=model,
            messages=messages,
            **kwargs
        ):
            if not isinstance(chunk, str):
                yield chunk
                continue
            buffer += chunk
            if ratelimit_message in buffer:
                raise RateLimitError(ratelimit_message)
            if ratelimit_message.startswith(buffer):
                continue
            yield buffer
            buffer = ""