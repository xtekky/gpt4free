"""Direct ChatGPT access-token authentication without a shared login cache."""

import asyncio
import base64
import json
import math
import time

from ...errors import MissingAuthError
from ...providers.response import AuthResult


def parse_access_token(value):
    """Read a JWT's expiry; server-side authentication validates its signature."""
    if not isinstance(value, str):
        raise MissingAuthError('ChatGPT access token is required')
    token = value.strip()
    if token.lower().startswith('bearer '):
        token = token[7:].strip()
    try:
        parts = token.split('.')
        if len(parts) != 3 or not all(parts):
            raise ValueError
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=' * (-len(parts[1]) % 4)))
        expires = payload.get('exp')
        if isinstance(expires, bool) or not isinstance(expires, (int, float)) or not math.isfinite(expires):
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise MissingAuthError('Invalid ChatGPT access token or expiry') from None
    if expires <= time.time():
        raise MissingAuthError('ChatGPT access token is expired; provide a fresh token')
    return token, expires


class AccessTokenAuthMixin:
    @classmethod
    def _explicit_auth(cls, kwargs):
        value = kwargs.get('api_key')
        if value is None:
            return None
        token, expires = parse_access_token(value)
        return AuthResult(
            api_key=token, expires=expires,
            cookies=dict(kwargs.get('cookies') or {}),
            headers={**cls.get_default_headers(),
                     **{k.lower(): v for k, v in (kwargs.get('headers') or {}).items()},
                     'authorization': f'Bearer {token}'},
            proof_token=kwargs.get('proof_token'),
            turnstile_token=kwargs.get('turnstile_token'),
            requirements_mode='classic',
        )

    @classmethod
    async def create_async_generator(cls, model, messages, **kwargs):
        auth = cls._explicit_auth(kwargs)
        if auth is None:
            async for chunk in super().create_async_generator(model, messages, **kwargs):
                yield chunk
            return
        # Caller-supplied credentials are used only for this request.
        response = cls.create_authed(model, messages, auth_result=auth, **kwargs)
        timeout = kwargs.get('stream_timeout') if cls.use_stream_timeout else kwargs.get('timeout')
        try:
            while True:
                try:
                    yield await asyncio.wait_for(response.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    break
        finally:
            await response.aclose()
