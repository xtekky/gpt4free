from __future__ import annotations

import os
import json
import time
import asyncio
import uuid
from typing import AsyncIterator

from ...requests.cdp_browser import cdp
from ...typing import Messages, AsyncResult
from ...providers.response import (
    JsonConversation,
    Reasoning,
    TitleGeneration,
    AuthResult,
    RequestLogin,
)
from ...requests import StreamSession, get_nodriver_session, DEFAULT_HEADERS
from ...errors import MissingAuthError
from ... import debug
from ..base_provider import AsyncAuthedProvider, ProviderModelMixin
from ..helper import format_prompt, get_last_user_message

class Conversation(JsonConversation):
    def __init__(
        self,
        conversation_id: str,
        initial_load_id: str = None,
        initial_leaf_id: str = None,
        active_response_id: str = None,
    ) -> None:
        self.conversation_id = conversation_id
        # ids tracked from the stream, needed to resume the conversation
        self.initial_load_id = initial_load_id
        self.initial_leaf_id = initial_leaf_id
        self.active_response_id = active_response_id

class Grok(AsyncAuthedProvider, ProviderModelMixin):
    label = "Grok AI"
    url = "https://grok.com"
    cookie_domain = ".grok.com"
    assets_url = "https://assets.grok.com"
    conversation_url = "https://grok.com/rest/app-chat/conversations"
    ws_url = "wss://grok.com/ws/mgw/"

    needs_auth = True
    working = True

    # Updated to Grok 4 as default
    default_model = "fast"

    # Updated model list with latest Grok 4 and 3 models
    models = [
        default_model,
        "auto",
        "heavy",
        "expert"
    ]

    model_aliases = {
        # Grok 3 aliases
        "grok-3-thinking": "reasoning",
        "grok-3-r1": "reasoning",
        "grok-3-mini-thinking": "reasoning",
        # Latest alias
        "grok": "auto",
        # Grok 4 models
        "grok-4": "auto",
        "grok-4-heavy": "heavy",
        "grok-4-reasoning": "reasoning",
        # Grok 3 models
        "grok-3": "auto",
        "grok-3-reasoning": "reasoning",
        "grok-3-mini": "auto",
        "grok-3-mini-reasoning": "reasoning",
        # Legacy Grok 2 (still supported)
        "grok-2": "auto",
        "grok-2-image": "image",
        # Latest aliases
        "grok-latest": "auto"
    }

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        auth_result = AuthResult(headers=DEFAULT_HEADERS, impersonate="chrome")
        auth_result.headers["referer"] = cls.url + "/"
        async with get_nodriver_session(proxy=proxy) as browser:
            yield RequestLogin(cls.__name__, os.environ.get("G4F_LOGIN_URL") or "")
            page = await browser.get(cls.url)
            auth_result.headers["user-agent"] = await page.evaluate(
                "window.navigator.userAgent", return_by_value=True
            )
            # Wait for the login cookies (allows the user to log in manually).
            # The mgw WebSocket is authenticated with these cookies only.
            deadline = time.time() + 300
            while time.time() < deadline:
                result = await page.send(cdp.network.get_cookies([cls.url]))
                auth_result.cookies = {
                    c["name"]: c["value"] for c in result.get("cookies", [])
                }
                if "sso" in auth_result.cookies or "sso_rw" in auth_result.cookies:
                    break
                await asyncio.sleep(2)
        yield auth_result

    @classmethod
    def _get_mode_id(cls, model: str) -> str:
        # Map model names to the mgw session mode id
        if "auto" in model:
            return "auto"
        if "heavy" in model or "big-brain" in model:
            return "heavy"
        if "expert" in model:
            return "expert"
        if "reasoning" in model or "thinking" in model or "r1" in model:
            return "reasoning"
        if "deepsearch" in model:
            return "deepsearch"
        return "fast"

    @classmethod
    def _create_session_event(cls, mode_id: str, conversation: Conversation = None) -> dict:
        x_grok = {
            "protocol_capabilities": [
                "conversation_attached",
                "custom_methods_v1",
                "workspace_servers_v1",
            ],
            "use_chunk": True,
            "client_side_toolsets": ["connectors-v2", "connectors"],
            "enable_side_by_side": True,
            "force_side_by_side": False,
            "enable_image_generation": True,
            "image_generation_count": 2,
            "disable_text_follow_ups": False,
            "disable_artifact": True,
            "force_concise": False,
        }
        event = {
            "type": "session.create",
            "event_id": f"evt_init_{uuid.uuid4()}",
            "session": {
                "model": mode_id,
                "x_grok": x_grok,
            },
        }
        message = {"event": event}
        if conversation is not None and conversation.conversation_id:
            # Resume an existing conversation (matches the captured client trace)
            conversation_id = conversation.conversation_id
            message["session_id"] = conversation_id
            x_grok["conversation_id"] = conversation_id
            x_grok["load_existing"] = True
            if conversation.initial_load_id:
                x_grok["initial_load_id"] = conversation.initial_load_id
            if conversation.initial_leaf_id:
                x_grok["initial_leaf_id"] = conversation.initial_leaf_id
            if conversation.active_response_id:
                x_grok["active_response_id"] = conversation.active_response_id
        return message

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        conversation: Conversation = None,
        **kwargs,
    ) -> AsyncResult:
        conversation_id = None if conversation is None else conversation.conversation_id
        prompt = (
            format_prompt(messages)
            if conversation_id is None
            else get_last_user_message(messages)
        )
        mode_id = cls._get_mode_id(model)

        auth_result.headers.setdefault("origin", cls.url)

        async with StreamSession(**auth_result.get_dict()) as session:
            uid = auth_result.cookies.get("x-userid")
            try:
                ws = await session.ws_connect(f"{cls.ws_url}?uid={uid}", timeout=30)
            except Exception as e:
                if "401" in str(e) or "403" in str(e):
                    raise MissingAuthError(f"Grok authentication required: {e}") from e
                raise

            session_id = None
            response_sent = False
            received_chunk = False
            thinking_duration = None
            # ids tracked for conversation resuming
            last_user_message_id = None
            last_response_id = None

            # 1. Create the realtime session (resumes the conversation if given)
            await ws.send_str(json.dumps(cls._create_session_event(mode_id, conversation)))

            # 2. Keep the gateway connection alive (server replies with pong, which is ignored)
            async def heartbeat():
                while True:
                    await asyncio.sleep(20)
                    try:
                        await ws.send_str(json.dumps({
                            "event": {
                                "type": "ping",
                                "event_id": f"evt_hb_{int(time.time() * 1000)}",
                            }
                        }))
                    except Exception:
                        return

            hb_task = asyncio.create_task(heartbeat())

            try:
                while True:
                    # Safety net: if the gateway goes silent, end the stream
                    try:
                        raw = await ws.receive_str(timeout=120)
                    except Exception as e:
                        debug.error(f"Error receiving from websocket:", e)
                        break
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(data, dict):
                        continue

                    if session_id is None:
                        session_id = data.get("session_id") or (
                            (data.get("event") or {}).get("session") or {}
                        ).get("id")

                    event = data.get("event") or {}
                    event_type = event.get("type", "")

                    # Heartbeat replies from the server are dropped
                    if event_type == "pong":
                        continue
                    # Answer server-initiated pings
                    if event_type == "ping":
                        await ws.send_str(json.dumps({
                            "event": {"type": "pong", "event_id": event.get("event_id")}
                        }))
                        continue

                    # 3. Send the prompt only after the server confirmed the session
                    if event_type in ("session.created", "conversation.attached") and session_id is not None and not response_sent:
                        response_sent = True
                        await ws.send_str(json.dumps({
                            "session_id": session_id,
                            "event": {
                                "type": "response.create",
                                "event_id": f"evt_resp_{int(time.time() * 1000)}",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "x_grok": {
                                        "client_message_id": str(uuid.uuid4()),
                                        "input_chunks": [{"text": {"text": prompt}}],
                                    },
                                },
                            },
                        }))
                        await ws.send_str(json.dumps({
                            "session_id": session_id,
                            "event": {"type": "presence.update", "visible": True},
                        }))

                    if event_type == "conversation.attached":
                        attached_id = (event.get("conversation") or {}).get("id")
                        if attached_id and conversation_id is None:
                            conversation_id = attached_id
                        continue

                    # Track ids from the stream for later resuming
                    item = event.get("item") or {}
                    if isinstance(item, dict) and item.get("role") == "user" and item.get("id"):
                        last_user_message_id = item["id"]
                    response_data = event.get("response") or {}
                    if isinstance(response_data, dict) and response_data.get("id"):
                        last_response_id = response_data["id"]

                    if event_type == "response.chunk":
                        chunk_text = (event.get("chunk") or {}).get("text") or {}
                        text = chunk_text.get("text")
                        if not text:
                            continue
                        received_chunk = True
                        channel = chunk_text.get("channel", "")
                        if channel == "CHANNEL_ASSISTANT_THINKING":
                            if thinking_duration is None:
                                thinking_duration = time.time()
                                if "heavy" in model or "big-brain" in model:
                                    status = "🧠 Big Brain mode active..."
                                else:
                                    status = "🤔 Is thinking..."
                                yield Reasoning(status=status)
                            yield Reasoning(token=text)
                        elif channel == "CHANNEL_ASSISTANT_RESPONSE":
                            if thinking_duration is not None:
                                thinking_duration = time.time() - thinking_duration
                                status = (
                                    f"Thought for {thinking_duration:.2f}s"
                                    if thinking_duration > 1
                                    else ""
                                )
                                thinking_duration = None
                                yield Reasoning(status=status)
                            yield text
                        continue

                    # Fallback if the gateway streams aggregated deltas instead of chunks
                    if event_type == "response.output_text.delta" and not received_chunk:
                        delta = event.get("delta") or ""
                        if delta:
                            is_thinking = (event.get("x_grok") or {}).get("is_thinking", False)
                            if is_thinking:
                                if thinking_duration is None:
                                    thinking_duration = time.time()
                                    yield Reasoning(status="🤔 Is thinking...")
                                yield Reasoning(token=delta)
                            else:
                                if thinking_duration is not None:
                                    thinking_duration = time.time() - thinking_duration
                                    status = (
                                        f"Thought for {thinking_duration:.2f}s"
                                        if thinking_duration > 1
                                        else ""
                                    )
                                    thinking_duration = None
                                    yield Reasoning(status=status)
                                yield delta
                        continue

                    if event_type == "conversation.title.updated":
                        title = event.get("title") or (event.get("conversation") or {}).get("title")
                        if title:
                            yield TitleGeneration(title)
                        continue

                    if event_type == "response.grok.output":
                        output = event.get("output") or {}
                        stream_error = output.get("stream_error")
                        if stream_error:
                            raise RuntimeError(f"Grok stream error: {stream_error}")
                        continue

                    if event_type in ("response.done", "response.completed"):
                        break

                    if event_type == "error":
                        error_data = event.get("error", event)
                        if isinstance(error_data, dict):
                            message = error_data.get("message", "")
                        else:
                            message = str(error_data)
                        raise RuntimeError(f"Grok error: {message or str(event)[:200]}")
            finally:
                hb_task.cancel()
                await ws.close()

        # Return conversation for continuation, with ids needed to resume
        if conversation_id is not None and kwargs.get("return_conversation", True):
            yield Conversation(
                conversation_id,
                initial_load_id=(f"{last_user_message_id}:0" if last_user_message_id else None),
                initial_leaf_id=last_response_id,
                active_response_id=last_response_id,
            )
