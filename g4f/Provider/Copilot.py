from __future__ import annotations

import json
import asyncio
import base64
from urllib.parse import parse_qs, urlparse

from ..requests.cdp import CDPSession

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages
from ..errors import MissingAuthError
from ..providers.response import *
from ..image import is_accepted_format
from .helper import get_last_user_message
from .. import debug


class Conversation(JsonConversation):
    conversation_id: str

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id


def extract_bucket_items(messages: Messages) -> list[dict]:
    """Extract bucket items from messages content."""
    bucket_items = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            for content_item in message["content"]:
                if (
                    isinstance(content_item, dict)
                    and "bucket_id" in content_item
                    and "name" not in content_item
                ):
                    bucket_items.append(content_item)
        if message.get("role") == "assistant":
            bucket_items = []
    return bucket_items


def iter_stream_events(payload: str, allow_plain_text: bool = False):
    """Normalize websocket frames into legacy stream events.

    The classic copilot.microsoft.com socket emits ``{"event": ...}`` dicts.
    The newer substrate.office.com/m365Copilot/Chathub socket used by the
    office web client speaks SignalR: ``{"type": 1, "target": "update",
    "arguments": [...]}`` invocation frames carry cumulative message
    snapshots and ``writeAtCursor`` deltas, a ``{"type": 2}`` frame ends
    the turn.
    """
    try:
        msg = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        if allow_plain_text and isinstance(payload, str) and payload:
            yield {"event": "appendText", "text": payload}
        return
    yield from _normalize_stream_message(msg)

# Message types that carry no response content
_SKIPPED_MESSAGE_TYPES = {
    "Progress", "HintInvocation", "ReferencesListComplete", "Suggestion",
    "SearchQuery", "GeneratedCode", "TaskComplete", "Disengaged",
}

def _adaptive_card_text(message: dict) -> str:
    """Extract the visible text from a bot message's adaptive cards."""
    parts = []
    for card in message.get("adaptiveCards") or []:
        if not isinstance(card, dict):
            continue
        for block in card.get("body") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(parts)

def _normalize_stream_message(msg):
    if isinstance(msg, list):
        for item in msg:
            yield from _normalize_stream_message(item)
    elif isinstance(msg, dict):
        if "event" in msg:
            yield msg
        elif msg.get("type") == 1 and isinstance(msg.get("arguments"), list):
            # SignalR invocation frame (Chathub socket)
            for argument in msg["arguments"]:
                if not isinstance(argument, dict):
                    continue
                if argument.get("conversationId"):
                    yield {
                        "event": "startMessage",
                        "conversationId": argument["conversationId"],
                    }
                delta = argument.get("writeAtCursor")
                if isinstance(delta, str) and delta:
                    yield {"event": "appendText", "text": delta}
                if isinstance(argument.get("messages"), list):
                    for item in argument["messages"]:
                        yield from _normalize_stream_message(item)
        elif msg.get("type") == 2:
            # SignalR completion frame
            result = (msg.get("item") or {}).get("result") or {}
            if result.get("value") not in (None, "Success"):
                yield {"event": "error", **result}
            else:
                yield {"event": "done"}
        elif msg.get("type") in (3, 6):
            pass  # SignalR close/ping — handled by the browser itself
        elif msg.get("author") == "user":
            pass  # Never echo the user's own message
        elif msg.get("messageType") in _SKIPPED_MESSAGE_TYPES:
            pass
        elif isinstance(msg.get("adaptiveCards"), list) and msg.get("adaptiveCards"):
            # Bot message: a cumulative text snapshot
            text = _adaptive_card_text(msg) or msg.get("text") or ""
            if text:
                yield {"event": "replaceText", "text": text}
            for attribution in msg.get("sourceAttributions") or []:
                if isinstance(attribution, dict) and attribution.get("url"):
                    yield {
                        "event": "citation",
                        "url": attribution["url"],
                        "title": attribution.get("displayName"),
                    }
            suggestions = [
                item.get("text")
                for item in msg.get("suggestedResponses") or []
                if isinstance(item, dict) and item.get("text")
            ]
            if suggestions:
                yield {"event": "suggestedFollowups", "suggestions": suggestions}
        elif isinstance(msg.get("messages"), list):
            for item in msg["messages"]:
                yield from _normalize_stream_message(item)
        elif isinstance(msg.get("text"), str):
            yield {"event": "appendText", "text": msg["text"]}
        else:
            yield msg
    elif isinstance(msg, str):
        if msg:
            yield {"event": "appendText", "text": msg}
    else:
        yield msg


class Copilot(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Microsoft Copilot"
    url = "https://copilot.com"

    working = True
    active_by_default = True
    use_nodriver = True
    needs_auth = True
    use_stream_timeout = False

    default_model = "Copilot"
    models = [default_model, "Think Deeper", "Smart (GPT-5)", "Study"]
    model_aliases = {
        "o1": "Think Deeper",
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "gpt-5": "GPT-5",
        "study": "Study",
    }
    
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = None,
        prompt: str = None,
        conversation: BaseConversation = None,
        **kwargs,
    ) -> AsyncResult:
        if prompt is None:
            prompt = get_last_user_message(messages, False)
        if conversation is not None:
            url = f"{cls.url}/chats/{conversation.conversation_id}"
        else:
            url = cls.url
        async with CDPSession(proxy=proxy, headless=False) as session:
            # Listen for websocket events from the chat API. Register before
            # navigating, so the chat socket created during page load is seen.
            queue: asyncio.Queue = asyncio.Queue()
            ws_events = (
                "Network.webSocketCreated",
                "Network.webSocketFrameReceived",
                "Network.webSocketClosed",
            )
            # Only listen to websocket events — skip all other CDP traffic
            session.set_event_filter(list(ws_events))
            for ws_event in ws_events:
                session.add_event_handler(ws_event, queue)

            await session.navigate(url)

            # Wait for the chat UI. A login redirect flow (copilot.com/chat
            # ?...IdentityProvider=msa → login.microsoftonline.com) may run
            # first. Never clear cookies here — that would log the user out
            # of their Microsoft account. The browser is visible, so a manual
            # login also completes this wait. While a login page is open the
            # timeout keeps extending: raising here would only make the
            # retry loop open a fresh tab and reload the site again.
            loop = asyncio.get_running_loop()
            has_input = False
            deadline = loop.time() + 120
            while loop.time() < deadline:
                if await session.evaluate_js("!!document.querySelector('textarea, [contenteditable=\"true\"]')"):
                    has_input = True
                    break
                if await session.evaluate_js(
                    "/^login\\.(microsoftonline|live)\\.com$/.test(location.hostname)"
                ):
                    deadline = loop.time() + 30
                await asyncio.sleep(0.5)
            if not has_input:
                raise MissingAuthError("Copilot: No prompt input found on page")

            await session.insert_text_and_submit(prompt)

            done = False
            image_prompt: str = None
            last_msg = None
            sources = {}
            ws_request_id = None
            ws_is_chathub = False
            ws_conversation_id = None
            got_response = False
            # Text emitted so far — used to turn the Chathub protocol's
            # cumulative snapshots into append-only deltas.
            emitted_text = ""
            sent_conversation = False

            def handle_ws_message(event: dict):
                """Handle a single CDP event and yield response chunks."""
                nonlocal done, image_prompt, last_msg, got_response
                nonlocal ws_request_id, ws_is_chathub, ws_conversation_id, sources
                nonlocal emitted_text, sent_conversation
                method = event.get("_method")
                if method == "Network.webSocketCreated":
                    # Track the chat websocket, so only its frames are processed
                    ws_url = event.get("url", "")
                    if ws_url.startswith("wss://substrate.office.com/m365Copilot/Chathub"):
                        # The office web client streams via the M365 Chathub
                        ws_request_id = event.get("requestId")
                        ws_is_chathub = True
                        query = parse_qs(urlparse(ws_url).query)
                        ws_conversation_id = query.get("ConversationId", [None])[0]
                    return
                if method == "Network.webSocketClosed":
                    if ws_request_id is not None and event.get("requestId") == ws_request_id:
                        done = True
                    return
                if ws_request_id is not None and event.get("requestId") != ws_request_id:
                    return
                payload_data = (event.get("response") or {}).get("payloadData")
                if not payload_data:
                    return
                for msg in iter_stream_events(payload_data, ws_is_chathub):
                    last_msg = msg
                    event_name = msg.get("event") if isinstance(msg, dict) else None
                    if event_name in (
                        "appendText", "replaceText", "imageGenerated",
                        "partialImageGenerated", "chainOfThought",
                    ):
                        got_response = True
                    if event_name == "startMessage":
                        conversation_id = msg.get("conversationId") or ws_conversation_id
                        if conversation_id and not sent_conversation:
                            sent_conversation = True
                            yield Conversation(conversation_id)
                    elif event_name == "appendText":
                        text = msg.get("text") or ""
                        emitted_text += text
                        yield text
                    elif event_name == "generatingImage":
                        image_prompt = msg.get("prompt")
                    elif event_name == "imageGenerated":
                        yield ImageResponse(
                            msg.get("url"), image_prompt, {"preview": msg.get("thumbnailUrl")}
                        )
                    elif event_name == "done":
                        yield FinishReason("stop")
                        done = True
                    elif event_name == "suggestedFollowups":
                        yield SuggestedFollowups(msg.get("suggestions"))
                        done = True
                        return
                    elif event_name == "replaceText":
                        text = msg.get("text") or ""
                        if text.startswith(emitted_text):
                            # Cumulative snapshot: only forward the new part
                            delta = text[len(emitted_text):]
                            emitted_text = text
                            if delta:
                                yield delta
                        else:
                            emitted_text = text
                            yield text
                    elif event_name == "titleUpdate":
                        yield TitleGeneration(msg.get("title"))
                    elif event_name == "citation":
                        sources[msg.get("url")] = msg
                        yield SourceLink(
                            list(sources.keys()).index(msg.get("url")), msg.get("url")
                        )
                    elif event_name == "partialImageGenerated":
                        mime_type = is_accepted_format(
                            base64.b64decode(msg.get("content")[:12])
                        )
                        yield ImagePreview(
                            f"data:{mime_type};base64,{msg.get('content')}", image_prompt
                        )
                    elif event_name == "chainOfThought":
                        yield Reasoning(msg.get("text"))
                    elif event_name == "error":
                        raise RuntimeError(f"Error: {msg}")
                    elif event_name not in [
                        "received",
                        "startMessage",
                        "partCompleted",
                        "connected",
                    ]:
                        debug.log(f"Copilot Message: {payload_data[:100]}...")

            try:
                while not done:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout)
                    except asyncio.TimeoutError:
                        break
                    for chunk in handle_ws_message(event):
                        yield chunk
            finally:
                for event_name in (
                    "Network.webSocketCreated",
                    "Network.webSocketFrameReceived",
                    "Network.webSocketClosed",
                ):
                    session.remove_event_handler(event_name, queue)
            if not done:
                if got_response:
                    # The Chathub socket closes without sending a done event
                    debug.log("Copilot: Stream ended without done event")
                    yield FinishReason("stop")
                else:
                    raise MissingAuthError(f"Invalid response: {last_msg}")
            if sources:
                yield Sources(sources.values())