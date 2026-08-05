from __future__ import annotations

import asyncio
import logging
import uuid
from typing import AsyncGenerator, Optional, Any

import aiohttp
try:
    import socketio
    has_socketio = True
except ImportError:
    has_socketio = False

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...cookies import get_cookies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://api.miniapps.ai"
WS_URL = "wss://api.miniapps.ai"
APP_URL = "https://miniapps.ai"
COOKIE_DOMAIN = "api.miniapps.ai"
COOKIE_DOMAIN2 = ".api.miniapps.ai"

_DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": APP_URL,
    "Referer": APP_URL + "/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

class MiniApps(AsyncGeneratorProvider, ProviderModelMixin):
    url = APP_URL
    working = has_socketio  # Requires socketio for streaming
    supports_stream = True

    default_model = "claude-37"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: Optional[dict] = None,
        **kwargs
    ) -> AsyncResult:
        """
        Async generator that yields tokens from MiniApps.ai streaming chat.

        Expects either:
        - a valid session cookie stored in a file (cookie_file=...)
        - or a Google ID token (google_id_token=...) for login
        - or pre-authenticated cookies from environment (MINIAPPS_COOKIES)
        """
        if not model:
            model = cls.default_model
        if not cookies:
            cookies = {** get_cookies(COOKIE_DOMAIN), ** get_cookies(COOKIE_DOMAIN2)}
        print(f"Using cookies: {cookies}")
        async with aiohttp.ClientSession(headers=_DEFAULT_HEADERS, cookies=cookies) as session:
            # ---------- Step 1: CSRF token ----------
            csrf_token = await cls._get_csrf_token(session)

            # ---------- Step 2: Authenticate if needed ----------
            # Try to use existing cookies - check if we need to login
            # For simplicity, we assume the user provides either:
            #  - "google_id_token" in kwargs, or
            #  - "cookie_file" path to a cookies file, or
            #  - already has valid cookies from environment
            # If none, we raise.
            ws_token = await cls._authenticate(session, csrf_token, **kwargs)

            # ---------- Step 3: Get tool info ----------
            tool = await cls._get_tool_info(session, model, csrf_token)
            tool_id = tool["id"]
            revision = tool.get("revision", 1)
            model_id = tool.get("modelId", "")

            # ---------- Step 4: Send message ----------
            # For simplicity, we only send the last message; the API may support full history via conversation_id
            conversation_id = kwargs.get("conversation_id")
            request_id = str(uuid.uuid4())

            send_result = await cls._send_message(
                session,
                csrf_token,
                tool_id,
                revision,
                model_id,
                messages,
                conversation_id=conversation_id,
                request_id=request_id,
            )
            actual_conversation_id = send_result.get("conversationId", conversation_id)
            if not conversation_id:
                # Store for continuation if needed (optional)
                cls.last_conversation_id = actual_conversation_id

            # ---------- Step 5: Stream via Socket.IO ----------
            # Prepare cookie string for Socket.IO authentication (from session cookies)
            cookies = session.cookie_jar.filter_cookies(BASE_URL)
            cookie_str = "; ".join(f"{c.key}={c.value}" for c in cookies.values())

            async for token in cls._stream_response(
                ws_token,
                actual_conversation_id,
                request_id,
                cookie_str=cookie_str,
                timeout=kwargs.get("timeout", 120)
            ):
                yield token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_csrf_token(session: aiohttp.ClientSession) -> str:
        """GET /auth/csrf and return the csrf token."""
        url = f"{BASE_URL}/auth/csrf"
        async with session.get(url) as resp:
            data = await resp.json()
            token = data.get("csrfToken", "")
            if not token:
                raise RuntimeError("Failed to obtain CSRF token")
            return token

    @staticmethod
    async def _authenticate(
        session: aiohttp.ClientSession,
        csrf_token: str,
        **kwargs
    ) -> str:
        """Authenticate if needed and return the WebSocket token (w)."""
        # First, try to use existing cookies by calling /auth/me
        url = f"{BASE_URL}/auth/me"
        headers = {"x-csrf-token": csrf_token}
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("w"):
                    return data["w"]
                # else: not fully authenticated, need login

        # If we have a google_id_token, perform Google login
        google_id_token = kwargs.get("google_id_token")
        if google_id_token:
            login_url = f"{BASE_URL}/auth/google/login"
            payload = {"idToken": google_id_token}
            headers = {"x-csrf-token": csrf_token, "Content-Type": "application/json"}
            async with session.post(login_url, json=payload, headers=headers) as resp:
                data = await resp.json()
                if not resp.ok:
                    raise RuntimeError(f"Google login failed: {data}")
                login_hash = data.get("hash")
                if not login_hash:
                    raise RuntimeError("No login hash returned")

                # Setup user (if new account) – simplified; assumes the account exists or we just need to complete
                # For existing accounts, we might not need setup. We'll try /auth/me again.
                # Actually, after google_login, we should have a session. Let's try /auth/me again.
                async with session.get(url, headers={"x-csrf-token": csrf_token}) as resp2:
                    if resp2.ok:
                        data2 = await resp2.json()
                        if data2.get("w"):
                            return data2["w"]
                    # If still not, try setup_user (if we have login_hash)
                    if login_hash:
                        setup_url = f"{BASE_URL}/auth/setup/user"
                        setup_payload = {
                            "username": kwargs.get("username", "g4f_user"),
                            "password": kwargs.get("password", "TempPass123!"),
                            "hash": login_hash,
                        }
                        async with session.post(setup_url, json=setup_payload, headers=headers) as resp3:
                            if resp3.ok:
                                data3 = await resp3.json()
                                if data3.get("w"):
                                    return data3["w"]
            raise RuntimeError("Authentication failed")

        # If all fails, raise
        raise RuntimeError(
            "Authentication required. Provide google_id_token or valid session cookies."
        )

    @staticmethod
    async def _get_tool_info(
        session: aiohttp.ClientSession,
        slug: str,
        csrf_token: str
    ) -> dict:
        """Fetch tool metadata by slug (model name)."""
        url = f"{BASE_URL}/tools/s/{slug}"
        headers = {"x-csrf-token": csrf_token}
        params = {"lang": "en"}
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Tool info failed for {slug}: {await resp.text()}")
            return await resp.json()

    @staticmethod
    async def _send_message(
        session: aiohttp.ClientSession,
        csrf_token: str,
        tool_id: str,
        revision: int,
        model_id: str,
        messages: list,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        language: str = "en"
    ) -> dict:
        """POST /chat to send a message and get a conversation."""
        if not request_id:
            request_id = str(uuid.uuid4())
        elements = []
        for m in messages:
            if m.get("role") != "user":
                elements = []
            if isinstance(m.get("content"), str):
                elements.append({
                    "type": "text",
                    "text": m["content"]
                })
            else:
                for part in m.get("content", []):
                    elements.append(part)
        body = {
            "toolId": tool_id,
            "revision": revision,
            "modelId": model_id,
            "requestId": request_id,
            "elements": elements,
            "language": language,
        }
        if conversation_id:
            body["conversationId"] = conversation_id

        url = f"{BASE_URL}/chat"
        headers = {
            "x-csrf-token": csrf_token,
            "Content-Type": "application/json",
        }
        async with session.post(url, json=body, headers=headers) as resp:
            if not resp.ok:
                raise RuntimeError(f"Send message failed: {await resp.text()}")
            data = await resp.json()
            return data

    @staticmethod
    async def _stream_response(
        ws_token: str,
        conversation_id: str,
        request_id: str,
        cookie_str: str = "",
        timeout: int = 120
    ) -> AsyncGenerator[str, None]:
        """
        Connect to Socket.IO and yield tokens from the AI stream.

        Uses an asyncio.Queue to transfer data from event callbacks to
        the generator.
        """
        queue: asyncio.Queue[str] = asyncio.Queue()
        done = asyncio.Event()
        error: Optional[Exception] = None

        sio = socketio.AsyncClient(
            logger=False,
            engineio_logger=False,
            reconnection=False
        )

        @sio.on("chat-token")
        async def on_token(data):
            print("Received token event:", data)
            token = ""
            if isinstance(data, str):
                token = data
            elif isinstance(data, dict):
                token = data.get("text") or data.get("token") or data.get("content", "")
            if token:
                await queue.put(token)

        @sio.on("chat-message")
        async def on_message(data):
            # Complete message – signal end of stream
            done.set()

        @sio.on("chat-error")
        async def on_error(data):
            nonlocal error
            msg = data if isinstance(data, str) else data.get("message", str(data))
            error = Exception(f"Stream error: {msg}")
            done.set()

        @sio.on("chat-done:{conversation_id}")
        async def on_chat_done(data):
            done.set()

        # Connect
        headers = {
            "Origin": APP_URL,
            "Cookie": cookie_str,
        }
        try:
            await sio.connect(
                WS_URL,
                socketio_path="/socket.io/",
                transports=["websocket"],
                auth={"token": ws_token},
                headers=headers,
                wait_timeout=10
            )
        except Exception as e:
            logger.error("Socket.IO connection failed: %s", e)
            yield f"\n[Connection error: {e}]"
            return

        # We also need to register the conversation-specific done event
        # after we know the conversation_id
        await sio.emit("join", {"conversationId": conversation_id})

        try:
            # Loop until done or timeout
            while True:
                # Wait for either a token or the done event
                token_task = asyncio.create_task(queue.get())
                done_task = asyncio.create_task(
                    asyncio.wait_for(done.wait(), timeout=timeout)
                )
                # Cancel the other task when one finishes
                token_task.add_done_callback(lambda _: done_task.cancel())
                done_task.add_done_callback(lambda _: token_task.cancel())

                try:
                    token = await token_task
                    yield token
                except asyncio.CancelledError:
                    # Either done event fired or timeout
                    break
        except asyncio.TimeoutError:
            # Fallback: try to fetch conversation via REST
            # This is optional; can yield a fallback message
            yield f"\n[Stream timed out after {timeout}s]"
        except Exception as exc:
            yield f"\n[Stream error: {exc}]"
        finally:
            await sio.disconnect()
            if error:
                raise error  # Re-raise the error after disconnecting
