from __future__ import annotations

from typing import Union
from aiohttp import ClientResponse

from ...errors import ResponseStatusError
from ...requests import StreamResponse

async def raise_for_status(response: Union[StreamResponse, ClientResponse], message: str = None):
    if response.ok:
        return
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            data = await response.json()
            message = data.get("error", data.get("message", message))
            message = message.split(" <a ")[0]
        except Exception:
            pass
    if not message:
        text = await response.text()
        is_html = response.headers.get("content-type", "").startswith("text/html") or text.startswith("<!DOCTYPE")
        message = "HTML content" if is_html else text
    raise ResponseStatusError(f"Response {response.status}: {message}")