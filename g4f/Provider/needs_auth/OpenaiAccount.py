from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
import urllib.parse
import uuid

from typing import AsyncIterator

from ...typing import AsyncResult, Messages
from ...requests import StreamSession, has_cdp
from ...requests.raise_for_status import raise_for_status
from ...errors import MissingAuthError, NoValidHarFileError
from ...cookies import get_cookies_dir
from ..base_provider import AsyncAuthedProvider, ProviderModelMixin
from ..helper import to_string
from ..openai.har_file import get_har_files
from ..openai.new import get_config, get_requirements_token
from ..openai.proofofwork import generate_proof_token
from ...providers.response import (
    JsonConversation,
    FinishReason,
    Reasoning,
    TitleGeneration,
    RequestLogin,
    AuthResult,
)
from ... import debug

class Conversation(JsonConversation):
    """Tracks the authenticated chat session state.

    Passing the conversation back into the provider continues that chat:
    ``conversation_id`` addresses the server-side chat and ``message_id``
    is the last assistant message, used as the parent of the next turn.
    """

    def __init__(self, model: str):
        self.model = model
        self.conversation_id: str = None
        self.message_id: str = None
        self.finish_reason: str = None
        self.recipient: str = "all"
        self.thoughts_summary: str = ""

class OpenaiAccount(AsyncAuthedProvider, ProviderModelMixin):
    """ChatGPT account chat driven by a captured browser session (HAR file).

    Implements the authenticated "f/conversation" flow observed in a real
    logged-in session: sentinel chat-requirements prepare/finalize (with a
    freshly minted proof-of-work), a conversation prepare call for the
    conduit token, and the SSE-streamed conversation request.

    Credentials (cookies, access token, fingerprint headers, turnstile
    token) are read from a HAR capture of chatgpt.com in the cookies dir
    (``~/.g4f/cookies/chatgpt.com.har``): log in on chatgpt.com, open the
    browser DevTools network tab and "Save all as HAR with content" while
    the chat is open. When no valid HAR capture is available, the provider
    opens chatgpt.com in a real browser via CDP instead, waits for the
    login and captures the same credentials from the browser's own
    conversation request. The proof-of-work and requirements tokens are
    minted fresh for every request; only the long-lived session
    credentials come from the capture. Media upload and image generation
    are not implemented.
    """

    label = "OpenAI ChatGPT"
    url = "https://chatgpt.com"
    working = True
    active_by_default = True
    needs_auth = True
    supports_stream = True
    supports_message_history = True
    supports_system_message = True

    default_model = "auto"
    # The backend routes the chat to the account's available model; the
    # names are advertised so model/provider validation passes (see the
    # ChatGPT provider). Unknown slugs are passed through untouched.
    models = [default_model]
    model_aliases = {
        alias: alias for alias in (
            "gpt-5-2", "gpt-5-1", "gpt-5", "gpt-4.5", "gpt-4.1", "gpt-4.1-mini",
            "gpt-4o", "gpt-4o-mini", "gpt-4", "o1", "o1-mini", "o3-mini",
            "o3-mini-high", "o4-mini", "o4-mini-high",
        )
    }

    requirements_prepare_url = "https://chatgpt.com/backend-api/sentinel/chat-requirements/prepare"
    requirements_finalize_url = "https://chatgpt.com/backend-api/sentinel/chat-requirements/finalize"
    conversation_prepare_url = "https://chatgpt.com/backend-api/f/conversation/prepare"
    conversation_url = "https://chatgpt.com/backend-api/f/conversation"

    # Static fingerprint headers worth copying from the captured session.
    # Session-bound headers are deliberately NOT copied:
    # - "authorization" is stored separately and expiry-checked;
    # - the sentinel tokens are single-use and minted fresh per turn;
    # - "x-conduit-token" is returned by the prepare call per turn;
    # - "oai-session-id" is session-bound, a fresh one is generated;
    # - "x-oai-turn-trace-id" / "x-oai-is-client-observation" are per-turn
    #   telemetry traces;
    # - "x-openai-web-sse-compression" is an opt-in stream compression that
    #   is not requested (the client handles transfer compression itself).
    _HAR_HEADERS_WHITELIST = (
        "user-agent",
        "accept-language",
        "oai-language",
        "oai-client-version",
        "oai-client-build-number",
        "oai-telemetry",
        "sec-ch-ua",
        "sec-ch-ua-arch",
        "sec-ch-ua-bitness",
        "sec-ch-ua-full-version",
        "sec-ch-ua-full-version-list",
        "sec-ch-ua-mobile",
        "sec-ch-ua-model",
        "sec-ch-ua-platform",
        "sec-ch-ua-platform-version",
    )

    # Cached HAR auth state (reparsed when older than the TTL).
    _auth_state = None
    _auth_state_loaded_at = 0.0
    _AUTH_STATE_TTL = 600.0

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        """Yield the auth state from the HAR capture or a live browser session.

        Without a valid HAR capture (see ``_read_har``), a browser is opened
        via CDP and the login is awaited (see ``_read_cdp``).
        """
        try:
            if cls._auth_state is None or time.time() - cls._auth_state_loaded_at > cls._AUTH_STATE_TTL:
                cls._auth_state = cls._read_har()
                cls._auth_state_loaded_at = time.time()
        except NoValidHarFileError as e:
            yield RequestLogin(cls.label, os.environ.get("G4F_LOGIN_URL", ""))
            if not has_cdp:
                raise MissingAuthError(
                    f"OpenaiAccount: {e} (browser automation unavailable — install aiohttp)"
                ) from e
            debug.log(f"OpenaiAccount: {e} — waiting for login in a CDP browser")
            try:
                cls._auth_state = await cls._read_cdp(proxy)
                cls._auth_state_loaded_at = time.time()
            except MissingAuthError:
                raise
            except Exception as e:
                raise MissingAuthError(f"OpenaiAccount: browser login failed: {e}") from e
        yield cls._auth_state

    @classmethod
    def reset_auth(cls):
        # Drop the cached HAR state along with the persisted cache file, so
        # the next login re-reads the capture (it may have been re-exported).
        cls._auth_state = None
        cls.delete_cache_file()

    @classmethod
    async def get_quota(cls, **kwargs):
        auth = cls.get_auth_result()
        async with StreamSession(
            cookies=auth.cookies, headers=auth.headers, impersonate="chrome"
        ) as session:
            async with session.get(f"{cls.url}/backend-api/me") as response:
                await raise_for_status(response)
                user = await response.json()
                return {"id": user.get("id"), "name": user.get("name")}

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        proxy: str = None,
        timeout: int = 360,
        conversation: Conversation = None,
        return_conversation: bool = True,
        **kwargs,
    ) -> AsyncResult:
        model = cls.get_model(model)
        if conversation is None:
            conversation = Conversation(model)
        expires = getattr(auth_result, "expires", None)
        if expires is not None and time.time() > expires:
            raise MissingAuthError("OpenaiAccount: access token is expired")
        headers = cls._create_headers(auth_result)
        async with StreamSession(
            proxy=proxy, impersonate="chrome", timeout=timeout,
            cookies=getattr(auth_result, "cookies", None),
        ) as session:
            # Sentinel: mint a fresh chat-requirements token for this turn.
            chat_token, proof_token = await cls._get_requirements(session, headers, auth_result)
            # Conversation prepare: registers the turn, returns the conduit token.
            conduit_token = await cls._prepare_conversation(session, headers, model, conversation)
            data = {
                "action": "next",
                "messages": cls._create_messages(messages, conversation.conversation_id),
                "parent_message_id": conversation.message_id or "client-created-root",
                "model": model or "auto",
                "client_prepare_state": "success",
                "timezone_offset_min": -120,
                "timezone": "Europe/Berlin",
                "conversation_mode": {"kind": "primary_assistant"},
                "enable_message_followups": True,
                "system_hints": [],
                "model_response_contracts": [{
                    "id": "photo_upload_action.v1",
                    "protocol_version": 1,
                    "presets": ["cap:image", "cap:file", "placement:end"],
                }],
                "supports_buffering": True,
                "supported_encodings": ["v1"],
                "client_contextual_info": {
                    "is_dark_mode": False,
                    "time_since_loaded": random.randint(2, 60),
                    "page_height": 714,
                    "page_width": 1060,
                    "pixel_ratio": 1,
                    "screen_height": 900,
                    "screen_width": 1600,
                    "app_name": "chatgpt.com",
                    "has_web_push_capabilities": True,
                    "web_push_notification_permission": "default",
                },
                "paragen_cot_summary_display_override": "allow",
                "force_parallel_switch": "auto",
                "local_function_names": ["local.continue_in_work"],
            }
            if conversation.conversation_id is not None:
                data["conversation_id"] = conversation.conversation_id
                debug.log(f"OpenaiAccount: Use conversation: {conversation.conversation_id}")
            request_headers = {
                **headers,
                **cls._target_headers(cls.conversation_url),
                "accept": "text/event-stream",
                "openai-sentinel-chat-requirements-token": chat_token,
                "x-conduit-token": conduit_token,
            }
            if proof_token is not None:
                request_headers["openai-sentinel-proof-token"] = proof_token
            turnstile_token = getattr(auth_result, "turnstile_token", None)
            if turnstile_token:
                request_headers["openai-sentinel-turnstile-token"] = turnstile_token
            async with session.post(cls.conversation_url, json=data, headers=request_headers) as response:
                if response.status in (401, 403):
                    raise MissingAuthError(f"OpenaiAccount: response status: {response.status}")
                await raise_for_status(response)
                async for line in response.iter_lines():
                    for chunk in cls._iter_line(line, conversation):
                        yield chunk
        if conversation.finish_reason is None:
            conversation.finish_reason = "stop"
        if return_conversation:
            yield conversation
        yield FinishReason(conversation.finish_reason)

    @classmethod
    async def _get_requirements(cls, session: StreamSession, headers: dict, auth_result: AuthResult) -> tuple[str, str]:
        """Mint a fresh chat-requirements token (sentinel prepare/finalize)."""
        user_agent = headers.get("user-agent")
        async with session.post(
            cls.requirements_prepare_url,
            json={"p": get_requirements_token(get_config(user_agent))},
            headers={**headers, **cls._target_headers(cls.requirements_prepare_url)},
        ) as response:
            if response.status in (401, 403):
                raise MissingAuthError(f"OpenaiAccount: requirements prepare status: {response.status}")
            await raise_for_status(response)
            requirements = await response.json()
        prepare_token = requirements.get("prepare_token")
        if not prepare_token:
            raise RuntimeError(
                f"OpenaiAccount: requirements prepare returned no token: {list(requirements)}"
            )
        # Solve the proof-of-work challenge for this turn.
        proof_token = None
        meta = requirements.get("proofofwork") or {}
        if meta.get("required"):
            proof_token = generate_proof_token(
                required=True,
                seed=meta.get("seed", ""),
                difficulty=meta.get("difficulty", ""),
                user_agent=user_agent,
                proof_token=getattr(auth_result, "proof_token", None),
            )
        turnstile_token = getattr(auth_result, "turnstile_token", None)
        if (requirements.get("turnstile") or {}).get("required") and not turnstile_token:
            raise MissingAuthError(
                "OpenaiAccount: turnstile token required"
                " — re-capture the HAR file from a logged-in browser session"
            )
        finalize = {"prepare_token": prepare_token}
        if proof_token is not None:
            finalize["proofofwork"] = proof_token
        if turnstile_token:
            finalize["turnstile"] = turnstile_token
        async with session.post(
            cls.requirements_finalize_url,
            json=finalize,
            headers={**headers, **cls._target_headers(cls.requirements_finalize_url)},
        ) as response:
            if response.status in (401, 403):
                raise MissingAuthError(f"OpenaiAccount: requirements finalize status: {response.status}")
            await raise_for_status(response)
            data = await response.json()
        chat_token = data.get("token")
        if not chat_token:
            raise RuntimeError(f"OpenaiAccount: requirements finalize returned no token: {list(data)}")
        return chat_token, proof_token

    @classmethod
    async def _prepare_conversation(
        cls, session: StreamSession, headers: dict, model: str, conversation: Conversation
    ) -> str:
        """Register the turn with the conversation prepare endpoint."""
        data = {
            "action": "next",
            "parent_message_id": conversation.message_id or "client-created-root",
            "model": model or "auto",
            "client_prepare_state": "none",
            "client_prepare_dispatch": "immediate",
            "client_prepare_source": "context_change",
            "timezone_offset_min": -120,
            "timezone": "Europe/Berlin",
            "conversation_mode": {"kind": "primary_assistant"},
            "system_hints": [],
            "model_response_contracts": [{
                "id": "photo_upload_action.v1",
                "protocol_version": 1,
                "presets": ["cap:image", "cap:file", "placement:end"],
            }],
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {
                "app_name": "chatgpt.com",
                "has_web_push_capabilities": True,
                "web_push_notification_permission": "default",
            },
            "local_function_names": ["local.continue_in_work"],
        }
        if conversation.conversation_id is not None:
            data["conversation_id"] = conversation.conversation_id
        async with session.post(
            cls.conversation_prepare_url,
            json=data,
            headers={**headers, **cls._target_headers(cls.conversation_prepare_url)},
        ) as response:
            if response.status in (401, 403):
                raise MissingAuthError(f"OpenaiAccount: conversation prepare status: {response.status}")
            await raise_for_status(response)
            conduit_token = (await response.json()).get("conduit_token")
        if not conduit_token:
            raise RuntimeError("OpenaiAccount: conversation prepare returned no conduit token")
        return conduit_token

    @classmethod
    def _create_messages(cls, messages: Messages, conversation_id: str = None) -> list:
        """Build the f/conversation message objects.

        When continuing a chat, only messages the server does not know yet
        (those after the last assistant reply) are sent.
        """
        if conversation_id is not None:
            pending = []
            for message in messages:
                if message.get("role") == "assistant":
                    pending = []
                else:
                    pending.append(message)
            messages = pending
        return [
            {
                "id": str(uuid.uuid4()),
                "author": {"role": message["role"]},
                "create_time": time.time(),
                "content": {
                    "content_type": "text",
                    "parts": [to_string(message["content"])],
                },
                "metadata": {
                    "automation_creation_attribution": {
                        "origin": "conversation",
                        "flow_id": str(uuid.uuid4()),
                    },
                    "serialization_metadata": {"custom_symbol_offsets": []},
                    "submission_mode": "manual_send",
                },
            }
            for message in messages
        ]

    @classmethod
    def _iter_line(cls, line: bytes, conversation: Conversation):
        """Parse one SSE line of the conversation stream (patch protocol)."""
        if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
            return
        try:
            event = json.loads(line[6:])
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        if event.get("type") == "title_generation":
            yield TitleGeneration(event["title"])
        if event.get("error"):
            raise RuntimeError(event["error"])
        path = event.get("p")
        value = event.get("v")
        # Reasoning streams live under /message/content/thoughts.
        if path is not None and path.startswith("/message/content/thoughts"):
            if path.endswith("/summary"):
                conversation.thoughts_summary += value or ""
            elif path.endswith("/content"):
                if conversation.thoughts_summary:
                    yield Reasoning(token="", status=conversation.thoughts_summary)
                    conversation.thoughts_summary = ""
                yield Reasoning(token=value)
            return
        if "v" not in event:
            return
        if isinstance(value, str):
            # Streaming text patch for the assistant message.
            if (not path or path == "/message/content/parts/0") and conversation.recipient == "all":
                yield value
        elif isinstance(value, list):
            # A batch of patches; concatenate the text ones.
            buffer = ""
            for patch in value:
                if not isinstance(patch, dict):
                    continue
                patch_path = patch.get("p")
                if patch_path == "/message/content/parts/0" and conversation.recipient == "all":
                    buffer += patch.get("v") or ""
                elif patch_path == "/message/metadata":
                    finish = (patch.get("v") or {}).get("finish_details", {}).get("type")
                    if finish:
                        conversation.finish_reason = finish
            if buffer:
                yield buffer
        elif isinstance(value, dict):
            # Message snapshot: carries the conversation and message ids.
            if conversation.conversation_id is None and value.get("conversation_id"):
                conversation.conversation_id = value["conversation_id"]
                debug.log(f"OpenaiAccount: New conversation: {conversation.conversation_id}")
            message = value.get("message") or {}
            conversation.recipient = message.get("recipient", conversation.recipient)
            if message.get("author", {}).get("role") == "assistant":
                conversation.message_id = message.get("id")
                if message.get("status") == "finished_successfully":
                    finish = (message.get("metadata") or {}).get("finish_details", {}).get("type")
                    if finish:
                        conversation.finish_reason = finish

    @classmethod
    def _create_headers(cls, auth_result: AuthResult) -> dict:
        """Base headers: HAR fingerprint headers plus fresh session state."""
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            **(getattr(auth_result, "headers", None) or {}),
        }
        device_id = headers.get("oai-device-id") or (getattr(auth_result, "cookies", None) or {}).get("oai-did")
        if device_id:
            headers["oai-device-id"] = device_id
        headers["oai-session-id"] = str(uuid.uuid4())
        headers["authorization"] = f"Bearer {auth_result.api_key}"
        return headers

    @staticmethod
    def _target_headers(url: str) -> dict:
        """Routing headers the web frontend adds for its backend paths."""
        path = urllib.parse.urlparse(url).path
        return {
            "x-openai-target-path": path,
            "x-openai-target-route": path,
            "x-openai-web-frontend": "core_web",
        }

    @classmethod
    def _read_har(cls) -> AuthResult:
        """Extract the session credentials from a captured chatgpt.com HAR file.

        Prefers "f/conversation" entries (richest header set) and skips
        entries without an authorization header or cookies.
        """
        captured = None
        for path in get_har_files():
            with open(path, "rb") as file:
                try:
                    har = json.load(file)
                except json.JSONDecodeError:
                    continue
            for entry in har.get("log", {}).get("entries", []):
                request = entry.get("request", {})
                url = request.get("url", "")
                if "chatgpt.com" not in url or "/backend-api/" not in url:
                    continue
                headers = {
                    h["name"].lower(): h["value"]
                    for h in request.get("headers", [])
                    if h.get("name") and not h["name"].startswith(":")
                }
                if "authorization" not in headers:
                    continue
                cookies = {
                    c["name"]: c["value"]
                    for c in request.get("cookies", [])
                    if c.get("name")
                }
                if not cookies:
                    continue
                score = 2 if "/f/conversation" in url else 1
                if captured is None or score > captured[0]:
                    captured = (score, headers, cookies)
        if captured is None:
            raise NoValidHarFileError(
                f"no chatgpt.com session found"
                f" — save a HAR file from a logged-in browser to {get_cookies_dir()}"
            )
        _, headers, cookies = captured
        return cls._create_auth_result(headers, cookies)

    @classmethod
    async def _read_cdp(cls, proxy: str = None) -> AuthResult:
        """Capture the session credentials from a live browser via CDP.

        Opens chatgpt.com in a visible browser window (a login the user has
        to complete cannot happen headless), waits for the profile button
        that only exists for authenticated sessions, sends a seed message
        and intercepts the browser's own "f/conversation" request to copy
        its authorization header and cookies. Intercepted requests are
        continued, so the chat the user sees completes normally.
        """
        from ...requests.cdp import CDPSession

        async with CDPSession(proxy=proxy, headless=False) as session:
            await session.call("Network.enable")
            await session.call(
                "Fetch.enable",
                patterns=[{"urlPattern": "*backend-api/f/conversation*", "requestStage": "Request"}],
            )
            await session.navigate(cls.url)
            # Wait for the login (up to 5 minutes, like the nodriver flow):
            # the profile button only exists for authenticated sessions.
            deadline = time.time() + 300
            while not await session.evaluate_js(
                "!!document.querySelector('[data-testid=\"accounts-profile-button\"]')"
            ):
                if time.time() > deadline:
                    raise MissingAuthError(
                        "OpenaiAccount: login was not completed in the browser window in time"
                    )
                await asyncio.sleep(2)
            debug.log("OpenaiAccount: Login detected — waiting for the composer")
            deadline = time.time() + 30
            reloaded = False
            while not await session.evaluate_js("!!document.querySelector('#prompt-textarea')"):
                if time.time() > deadline:
                    if reloaded:
                        raise MissingAuthError("OpenaiAccount: chat composer not found in the browser")
                    # The SPA can get stuck after the login redirect — reload once.
                    reloaded = True
                    deadline = time.time() + 60
                    debug.log("OpenaiAccount: composer not found — reloading the page")
                    await session.navigate(cls.url)
                await asyncio.sleep(1)
            # Trigger a chat so the browser mints the authenticated request.
            await session.evaluate_js("""
                (() => {
                    const editor = document.querySelector('#prompt-textarea');
                    editor.focus();
                    document.execCommand('insertText', false, 'Hello');
                })()
            """)
            await asyncio.sleep(1)
            await session.evaluate_js("""
                (() => {
                    const button = document.querySelector('[data-testid="send-button"], [data-composer-submit]');
                    if (button) button.click();
                })()
            """)
            # Capture the browser's own conversation request (up to 2 minutes).
            # A queue (not wait_for_event) so requests paused while another
            # one is processed are never lost.
            queue: asyncio.Queue = asyncio.Queue()
            session.add_event_handler("Fetch.requestPaused", queue)
            headers = None
            deadline = time.time() + 120
            while headers is None:
                try:
                    paused = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    if time.time() > deadline:
                        raise MissingAuthError(
                            "OpenaiAccount: no conversation request captured in the browser"
                        )
                    continue
                request = paused.get("request") or {}
                found = {
                    key.lower(): value for key, value in (request.get("headers") or {}).items()
                    if not key.startswith(":")
                }
                # The sentinel tokens are single-use, but only the long-lived
                # credentials are captured — let the browser request pass.
                await session.call("Fetch.continueRequest", requestId=paused.get("requestId"))
                if "authorization" in found:
                    headers = found
            # Resume requests paused while no one was listening anymore.
            await session.call("Fetch.disable")
            cookies = await session.get_cookies([f"{cls.url}/"])
        return cls._create_auth_result(headers, cookies)

    @classmethod
    def _create_auth_result(cls, headers: dict, cookies: dict) -> AuthResult:
        """Build the auth state from captured request headers and cookies."""
        access_token = headers["authorization"].split(" ")[-1]
        expires = cls._get_expires(access_token)
        if expires is not None and time.time() > expires:
            raise NoValidHarFileError("access token is expired — re-capture it")
        proof_token = None
        raw = headers.get("openai-sentinel-proof-token")
        if raw:
            try:
                proof_token = json.loads(
                    base64.b64decode(raw.split("gAAAAAB", 1)[-1].encode()).decode()
                )
            except (ValueError, IndexError):
                debug.log("OpenaiAccount: Could not decode proof token from capture")
        auth_headers = {name: headers[name] for name in cls._HAR_HEADERS_WHITELIST if name in headers}
        return AuthResult(
            api_key=access_token,
            cookies=cookies,
            headers=auth_headers,
            expires=expires,
            proof_token=proof_token,
            turnstile_token=headers.get("openai-sentinel-turnstile-token"),
        )

    @staticmethod
    def _get_expires(access_token: str) -> float:
        """Read the expiry claim from the access token JWT payload."""
        try:
            claim = access_token.split(".")[1]
            claim = (claim + "=" * (4 - len(claim) % 4)).encode()
            return json.loads(base64.b64decode(claim)).get("exp")
        except (ValueError, IndexError, TypeError):
            return None
