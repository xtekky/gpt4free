from __future__ import annotations

from typing import Union
from aiohttp import ClientResponse
from requests import Response as RequestsResponse

from ..errors import ResponseStatusError, RateLimitError
from . import Response, StreamResponse

class CloudflareError(ResponseStatusError):
    ...

def is_cloudflare(text: str) -> bool:
    if "<title>Attention Required! | Cloudflare</title>" in text:
        return True
    return '<div id="cf-please-wait">' in text or "<title>Just a moment...</title>" in text

def is_openai(text: str) -> bool:
    return "<p>Unable to load site</p>" in text

async def raise_for_status_async(response: Union[StreamResponse, ClientResponse], message: str = None):
    if response.status in (429, 402):
        raise RateLimitError(f"Response {response.status}: Rate limit reached")
    message = await response.text() if not response.ok and message is None else message
    if response.status == 403 and is_cloudflare(message):
        raise CloudflareError(f"Response {response.status}: Cloudflare detected")
    elif response.status == 403 and is_openai(message):
        raise ResponseStatusError(f"Response {response.status}: Bot are detected")
    elif not response.ok:
        raise ResponseStatusError(f"Response {response.status}: {message}")

def raise_for_status(response: Union[Response, StreamResponse, ClientResponse, RequestsResponse], message: str = None):
    if hasattr(response, "status"):
        return raise_for_status_async(response, message)

    if response.status_code in (429, 402):
        raise RateLimitError(f"Response {response.status_code}: Rate limit reached")
    elif response.status_code == 403 and is_cloudflare(response.text):
        raise CloudflareError(f"Response {response.status_code}: Cloudflare detected")
    elif not response.ok:
        raise ResponseStatusError(f"Response {response.status_code}: {response.text if message is None else message}")