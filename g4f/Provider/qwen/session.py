"""In-memory sessions for Qwen Chat (separate from Qwen Code OAuth)."""

import asyncio
import base64
import hashlib
import json
import math
import datetime
import uuid
from http.cookiejar import Cookie, CookieJar
from http.cookies import SimpleCookie
from time import time
from urllib.parse import urlparse
from urllib.request import Request
from collections import OrderedDict
from concurrent.futures import Future
from threading import Lock

from ... import debug
from ...errors import CloudflareError, MissingAuthError, ResponseError, ResponseStatusError
from ...requests import StreamSession, raise_for_status


AUTH_URL = "https://auth.qwen.ai/api/v2/auths/refresh"
CHAT_URL = "https://chat.qwen.ai"
_sessions = OrderedDict()
_sessions_lock = Lock()


def token_expiry(token):
    """Read scheduling metadata, not a JWT authenticity check."""
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        expiry = float(claims["exp"])
        return expiry if math.isfinite(expiry) else None
    except (AttributeError, IndexError, KeyError, ValueError, TypeError):
        return None


def cookie_expiry(value):
    """Accept CDP epoch seconds and HAR ISO timestamps; -1 is a session cookie."""
    if value is None:
        return None
    try:
        expiry = float(value)
    except (ValueError, TypeError):
        try:
            stamp = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=datetime.timezone.utc)
            expiry = stamp.timestamp()
        except (ValueError, TypeError, OverflowError):
            return 0  # An invalid declared expiry must not become a permanent cookie.
    if not math.isfinite(expiry):
        return 0
    return int(expiry) if expiry >= 0 else None


class QwenAuth:
    """Reusable credentials; no credentials are written to disk or included in repr."""

    def __init__(self, token=None, refresh_token=None, cookies=None, headers=None):
        if (token is not None and not isinstance(token, str)) or (refresh_token is not None and not isinstance(refresh_token, str)):
            raise TypeError("Qwen tokens must be strings")
        self.access_token = token
        self.expires_at = token_expiry(token)
        self.fingerprint = {
            k: v for k, v in (headers or {}).items()
            if k.lower() not in ("authorization", "cookie")
        }
        self.jar = CookieJar()
        self._guard = Lock()
        self._refresh_future = None
        self._refresh_task = None
        self._epoch = 0
        self._closed = False
        self._has_refresh = False
        self.on_update = None
        if isinstance(cookies, dict):
            cookies = [dict(name=k, value=v) for k, v in cookies.items()]
        for cookie in cookies or []:
            self._set_cookie(cookie)
        if refresh_token is not None:
            self._set_cookie(dict(name="refresh_token", value=refresh_token, domain=".qwen.ai"))
        self.cache_id = hashlib.sha256((self.refresh_token or token or str(id(self))).encode()).hexdigest()

    def _set_cookie(self, data):
        name, value = data.get("name"), data.get("value")
        if not name or not isinstance(value, str):
            return
        domain = data.get("domain") or (".qwen.ai" if name == "refresh_token" else "chat.qwen.ai")
        if domain.lstrip(".") not in ("qwen.ai", "chat.qwen.ai", "auth.qwen.ai"):
            return
        if name == "refresh_token":
            self._has_refresh = True
        expires = cookie_expiry(data.get("expires"))
        self.jar.set_cookie(Cookie(
            version=0, name=name, value=value, port=None, port_specified=False,
            domain=domain, domain_specified=domain.startswith("."),
            domain_initial_dot=domain.startswith("."), path=data.get("path") or "/",
            path_specified=True, secure=data.get("secure", True), expires=expires,
            discard=expires is None, comment=None, comment_url=None,
            rest={"HttpOnly": None} if data.get("httpOnly", True) else {}, rfc2109=False,
        ))

    @property
    def refresh_token(self):
        """Read the current refresh cookie, respecting its scope and expiry."""
        if self._closed:
            return None
        request = Request(AUTH_URL)
        self.jar.add_cookie_header(request)
        parsed = SimpleCookie()
        parsed.load(request.get_header("Cookie", ""))
        cookie = parsed.get("refresh_token")
        return cookie.value if cookie is not None and cookie.value else None

    @property
    def can_refresh(self):
        return bool(self.refresh_token)

    @property
    def authenticated(self):
        # CookieJar prunes expired cookies when building a request. Remember
        # that credentials were supplied so expiration cannot enable guest mode.
        return not self._closed and bool(self.access_token or self._has_refresh)

    def get_cookies(self):
        """Return current cookie records for an explicitly requested export."""
        return [dict(name=c.name, value=c.value, domain=c.domain, path=c.path,
                     secure=c.secure, expires=c.expires, httpOnly="HttpOnly" in c._rest)
                for c in self.jar if not c.is_expired()]

    def request_headers(self, headers, url=CHAT_URL, authorization=True):
        result = {k: v for k, v in headers.items() if k.lower() not in ("authorization", "cookie")}
        target = urlparse(url)
        if target.scheme != "https" or target.hostname not in ("chat.qwen.ai", "auth.qwen.ai"):
            return result
        request = Request(url)
        self.jar.add_cookie_header(request)
        cookie = request.get_header("Cookie")
        if cookie:
            result["Cookie"] = cookie
        if authorization and self.access_token:
            result["Authorization"] = f"Bearer {self.access_token}"
        return result

    def update_cookies(self, response, url):
        # Extract Set-Cookie with the stdlib policy, retaining domain/path/deletions.
        from email.message import Message
        message = Message()
        headers = response.headers
        if hasattr(headers, "getall"):
            values = headers.getall("Set-Cookie", [])
        elif hasattr(headers, "get_list"):
            values = headers.get_list("Set-Cookie")
        else:
            value = headers.get("set-cookie") or headers.get("Set-Cookie")
            values = [value] if isinstance(value, str) else []
        for value in values:
            message.add_header("Set-Cookie", value)
        class CookieResponse:
            def info(self):
                return message
        self.jar.extract_cookies(CookieResponse(), Request(url))
        if any(cookie.name == "refresh_token" for cookie in self.jar):
            self._has_refresh = True

    def clear(self):
        with self._guard:
            self._epoch += 1
            self._closed = True
            self.access_token = self.expires_at = None
            self.jar.clear()

    def reset_guest(self, cookies):
        """Discard an exhausted guest's cookies and session-scoped upload cache identity."""
        with self._guard:
            if self._closed or self.authenticated:
                raise ValueError("Only an active guest session can be reset")
            self.jar.clear()
            self.cache_id = uuid.uuid4().hex
            for name, value in cookies.items():
                self._set_cookie(dict(name=name, value=value, domain="chat.qwen.ai"))

    async def ensure_valid(self, session, proxy=None, rejected_token=None, recover=False):
        with self._guard:
            if self._closed:
                raise MissingAuthError("Qwen session has ended; sign in again.")
            if not self.access_token and self._has_refresh and not self.can_refresh:
                raise MissingAuthError("Qwen refresh cookie is empty or expired; sign in again.")
            if recover and self.access_token and self.access_token != rejected_token:
                return self.access_token
            needs_refresh = self.can_refresh and (
                not self.access_token or self.expires_at is not None and self.expires_at - time() < 180
            )
            if not recover and not needs_refresh:
                if self.expires_at is not None and self.expires_at <= time():
                    raise MissingAuthError("Qwen access token has expired; provide a refresh_token or sign in again.")
                return self.access_token
            if not self.can_refresh:
                raise MissingAuthError("Qwen authentication failed; no usable refresh_token is available.")
            future = self._refresh_future
            if future is None:
                future = Future()
                self._refresh_future = future
                self._refresh_task = asyncio.create_task(self._run_refresh(session, proxy, future))
        # One future works across event loops; cancellation of a waiter does not
        # cancel the refresh another request is already relying on.
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(wrapped)

    async def _run_refresh(self, session, proxy, future):
        try:
            # Refresh owns its transport: closing/cancelling the requesting chat
            # must not terminate a refresh that other requests are waiting for.
            async with StreamSession(headers=self.fingerprint) as refresh_session:
                result = await self._refresh(refresh_session, proxy)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)
        finally:
            with self._guard:
                if self._refresh_future is future:
                    self._refresh_future = self._refresh_task = None

    async def _refresh(self, session, proxy):
        epoch = self._epoch
        headers = {
            "accept": "application/json", "content-type": "application/json",
            "version": "0.3.12", "source": "web",
            "timezone": datetime.datetime.now().astimezone().strftime("%a %b %d %Y %H:%M:%S GMT%z"),
            **{k.lower(): v for k, v in self.request_headers(self.fingerprint, AUTH_URL, authorization=False).items()},
            "x-request-id": str(uuid.uuid4()),
        }
        headers.update({"origin": CHAT_URL, "referer": CHAT_URL + "/", "x-request-origin": CHAT_URL})
        debug.log("[Qwen] Refreshing the authenticated session.")
        async with session.get(AUTH_URL, headers=headers, proxy=proxy, timeout=30, allow_redirects=False) as response:
            try:
                await raise_for_status(response)
            except MissingAuthError:
                self.clear()
                raise MissingAuthError("Qwen refresh session has expired; sign in again.") from None
            except ResponseStatusError as error:
                if response.status == 403 and not isinstance(error, CloudflareError) and "application/json" in response.headers.get("content-type", ""):
                    rejected = await response.json()
                    rejection = (rejected.get("data") or rejected) if isinstance(rejected, dict) else {}
                    if isinstance(rejection, dict) and str(rejection.get("code", "")).lower() in ("forbidden", "unauthorized", "invalid token"):
                        self.clear()
                        raise MissingAuthError("Qwen refresh session was rejected; sign in again.") from None
                raise
            if "text/html" in response.headers.get("content-type", ""):
                raise CloudflareError("Qwen authentication requires browser verification.")
            payload = await response.json()
            if not isinstance(payload, dict):
                raise ResponseError("Unexpected Qwen token refresh response.")
            data = payload.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            token = data.get("access_token")
            if not isinstance(token, str) or not token or payload.get("success") is not True:
                code = str(data.get("code") or "refresh_failed")
                if code.lower() in ("unauthorized", "invalid token", "forbidden", "401"):
                    self.clear()
                    raise MissingAuthError("Qwen refresh session has expired; sign in again.")
                details = data.get("details") or data.get("message") or "no access_token was returned"
                message = f"Qwen token refresh failed ({code}): {details}"
                for secret in [self.access_token, *[c.value for c in self.jar]]:
                    if secret:
                        message = message.replace(secret, "[redacted]")
                raise ResponseError(message)
            expiry = data.get("expires_at")
            try:
                expiry = float(expiry) if expiry is not None else token_expiry(token)
            except (ValueError, TypeError):
                expiry = token_expiry(token)
            if expiry is None or not math.isfinite(expiry) or expiry <= time():
                raise ResponseError("Qwen refresh returned a token without a valid expiry.")
            with self._guard:
                if self._epoch != epoch or self._closed:
                    raise MissingAuthError("Qwen session changed while refreshing; the result was discarded.")
                self.update_cookies(response, AUTH_URL)
                self.access_token, self.expires_at = token, expiry
            debug.log("[Qwen] Session refreshed successfully.")
            if self.on_update is not None:
                try:
                    self.on_update(self)
                except Exception:
                    debug.log("[Qwen] Session refreshed, but the requested credential export failed.")
            return token


def cached_auth(token=None, refresh_token=None, cookies=None, headers=None, proxy=None):
    """Bounded in-memory credential cache, independent of event-loop lifetime."""
    candidate = QwenAuth(token, refresh_token, cookies, headers)
    if not candidate.authenticated:
        return candidate
    credential = candidate.refresh_token or token
    if not credential:
        return candidate
    identity = hashlib.sha256(json.dumps([credential, candidate.fingerprint, proxy], sort_keys=True).encode()).hexdigest()
    with _sessions_lock:
        if identity not in _sessions:
            if len(_sessions) >= 32:
                _sessions.popitem(last=False)
            _sessions[identity] = candidate
        _sessions.move_to_end(identity)
        return _sessions[identity]
