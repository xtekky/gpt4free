from __future__ import annotations

import html
import json
import re
import urllib.parse

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from ..requests.cdp import CDPSession
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..providers.response import JsonConversation, FinishReason
from .. import debug


class Conversation(JsonConversation):
    """Tracks the intercepted browser chat session.

    ``message_id`` is the last assistant message; passing the conversation
    back into the provider continues that chat instead of starting a new one.
    """

    def __init__(self, model: str):
        self.model = model
        self.conversation_id: str = None
        self.message_id: str = None
        self.user_message_count: int = 0


# The reply arrives as a DPU partial document with assistant stream blocks;
# prose lives in <p> tags, code in <pre>, headlines in <h1>-<h6>, quotes in
# <blockquote>, lists in <ul>/<ol> and writing documents in <section>; a
# gate rejection carries a failure status instead.
_RE_ASSISTANT_BLOCK = re.compile(
    r'<(p|pre|h[1-6]|blockquote|ul|ol|section)\b([^>]*\bdata-assistant-stream-block(?:="")?[^>]*)>(.*?)</\1>',
    re.DOTALL,
)
# Blocks nested inside sections carry no stream-block attributes.
_RE_INNER_BLOCK = re.compile(
    r"<(h[1-6]|p|pre|ul|ol|blockquote)\b[^>]*>(.*?)</\1>",
    re.DOTALL,
)
_RE_LIST_ITEM = re.compile(r"<li\b[^>]*>(.*?)</li>", re.DOTALL)
_RE_BLOCK_INDEX = re.compile(r'data-assistant-stream-block-index="(\d+)"')
_RE_FAILURE_STATUS = re.compile(r'data-failure-status="(\d+)"')
_RE_GATE_KIND = re.compile(r'data-gate-kind="(\w+)"')
_RE_XML_MARKER = re.compile(r"<\?[^>]*>")
_RE_CODE_LANGUAGE = re.compile(r'language-([\w+-]+)')


class ChatGPT(AsyncGeneratorProvider, ProviderModelMixin):
    """ChatGPT anonymous guest chat driven entirely by a real browser.

    Instead of simulating the anti-bot pipeline (proof-of-work, turnstile,
    requirements tokens) in Python, this provider lets the real browser mint
    every token naturally: it opens the guest page, triggers a chat with a
    seed prompt, intercepts the browser's own "conversation/updates" request
    via the CDP Fetch domain, aborts it, and replays it from Python with only
    the "prompt" form field swapped for the real prompt. The reply is read
    from the DPU partial document the endpoint returns.

    No proof tokens, no turnstile VM, no requirements simulation — the
    browser does all of that itself, so the request is byte-identical to a
    genuine one except for the prompt text. A fresh browser request
    replayed from Python passes the gate (verified), so no TLS/session
    binding blocks the replay.

    Long (pasted) prompts are supported: the prompt is swapped into the
    form body directly (nothing is typed into the page) and the reply
    timeout scales with the prompt length.
    """

    label = "ChatGPT"
    url = "https://chatgpt.com"
    working = True
    active_by_default = True
    needs_auth = False
    supports_stream = True
    supports_system_message = False
    supports_message_history = False

    default_model = "auto"
    # The guest surface routes to ChatGPT's default model; it is advertised
    # for the models it can serve so model/provider validation passes.
    models = [default_model, "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-5"]

    updates_url_pattern = "*conversation/updates*"
    seed_prompt = "Hello"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 180,
        conversation: Conversation = None,
        return_conversation: bool = True,
        **kwargs,
    ) -> AsyncResult:
        model = cls.get_model(model)
        prompt = format_prompt(messages)
        if conversation is None:
            conversation = Conversation(model)

        async with CDPSession(proxy=proxy) as session:
            # Pause the browser's own updates request before it leaves.
            await session.call("Network.enable")
            await session.call(
                "Fetch.enable",
                patterns=[{"urlPattern": cls.updates_url_pattern, "requestStage": "Request"}],
            )
            # The "#q=" fragment makes the guest UI auto-submit a chat; the
            # explicit submit is a fallback for when it does not fire.
            await session.navigate(f"{cls.url}/#q={cls.seed_prompt}")
            paused = None
            try:
                paused = await session.wait_for_event("Fetch.requestPaused", timeout=30)
            except TimeoutError:
                try:
                    await session.insert_text_and_submit()
                    paused = await session.wait_for_event("Fetch.requestPaused", timeout=45)
                except TimeoutError:
                    pass
            if paused is None:
                raise RuntimeError("ChatGPT: no updates request intercepted")

            request = paused.get("request") or {}
            post_data = request.get("postData") or ""
            request_id = paused.get("requestId")
            if not post_data:
                await session.call("Fetch.failRequest", requestId=request_id, errorReason="Aborted")
                raise RuntimeError("ChatGPT: intercepted request has no form body")

            # Capture everything the browser was about to send: url, headers
            # (HTTP/2 pseudo-headers are added by the client itself) and the
            # cookies the request was authenticated with.
            url = request["url"]
            headers = {
                key: value for key, value in (request.get("headers") or {}).items()
                if not key.startswith(":")
            }
            cookies = await session.get_cookies([f"{cls.url}/"])

            # Stop the browser's request: the tokens are single-use, so the
            # replay below is the only consumer.
            await session.call("Fetch.failRequest", requestId=request_id, errorReason="Aborted")

        # Swap the seed prompt for the real one. Everything else —
        # requirements token, proof token, turnstile token, cookies,
        # headers — stays exactly as the browser minted it.
        fields = urllib.parse.parse_qsl(post_data, keep_blank_values=True)
        fields = [(key, prompt if key == "prompt" else value) for key, value in fields]
        # Followup: point the form at the previous assistant message so the
        # server appends to the existing conversation instead of a new one.
        if conversation.message_id:
            fields = [
                (key, cls._continue_state(value, conversation) if key == "conversationState" else value)
                for key, value in fields
            ]
        body = urllib.parse.urlencode(fields).encode()

        # Long (pasted) prompts take proportionally longer to answer:
        # scale the reply timeout with the prompt length.
        reply_timeout = min(max(timeout, 60 + len(prompt) // 20), 1800)

        # Replay the captured request from Python and read the DPU partial
        # document from the response.
        async with StreamSession(
            impersonate="chrome", timeout=reply_timeout, proxy=proxy, cookies=cookies,
        ) as stream_session:
            async with stream_session.post(url, data=body, headers=headers) as response:
                text = await response.text()

        reply = cls._parse_dpu(text, conversation)
        if reply is None:
            raise RuntimeError("ChatGPT: no reply received (gate rejection)")
        conversation.user_message_count += 1
        yield reply
        if return_conversation:
            yield conversation
        yield FinishReason("stop")

    @staticmethod
    def _continue_state(state: str, conversation: Conversation) -> str:
        """Rewrite a conversationState form value to continue an existing chat.

        Captured followup format (browser ground truth):
        {"backendConversationId": "<id>", "messages": [],
         "parentMessageId": "<last message id>", "userMessageCount": 1}
        """
        try:
            state = json.loads(state)
        except ValueError:
            return state
        if isinstance(state, dict):
            state["backendConversationId"] = conversation.conversation_id
            state["parentMessageId"] = conversation.message_id
            state["userMessageCount"] = conversation.user_message_count
            state.pop("safety", None)
            return json.dumps(state)
        return state

    @classmethod
    def _parse_dpu(cls, text: str, conversation: Conversation) -> str:
        """Extract the assistant reply from the DPU partial document."""
        failure = _RE_FAILURE_STATUS.search(text)
        if failure is not None:
            gate = _RE_GATE_KIND.search(text)
            debug.log(
                f"ChatGPT: gate rejected chat "
                f"(status={failure.group(1)}, kind={gate.group(1) if gate else 'unknown'})"
            )
            return None
        conversation_id = re.search(r'data-conversation-id="([\w-]+)"', text)
        if conversation_id:
            conversation.conversation_id = conversation_id.group(1)
        # The reply DPU renders the newest message last; the followup form
        # needs that message id as parentMessageId.
        message_ids = re.findall(r'data-message-id="([\w-]+)"', text)
        if message_ids:
            conversation.message_id = message_ids[-1]
        # Later blocks with the same index are streaming updates that
        # supersede earlier (partial) ones — keep only the last per index.
        blocks = {}
        for match in _RE_ASSISTANT_BLOCK.finditer(text):
            index_match = _RE_BLOCK_INDEX.search(match.group(2))
            index = int(index_match.group(1)) if index_match else 0
            blocks[index] = cls._convert_block(match.group(1), match.group(0), match.group(3))
        if not blocks:
            debug.log(f"ChatGPT: no assistant block in response: {text[:300]!r}")
            return None
        text = "\n".join(blocks[index] for index in sorted(blocks))
        return html.unescape(text)

    @staticmethod
    def _convert_block(tag: str, full_tag: str, content: str) -> str:
        """Convert one DPU block to markdown (code fences, headlines, quotes, lists)."""
        content = _RE_XML_MARKER.sub("", content)
        if tag == "section":
            return ChatGPT._convert_section(content)
        if tag == "pre":
            # Code: strip syntax-highlight spans, keep newlines, add fences
            # with the language from the inner <code class="language-…"> tag.
            lang_match = _RE_CODE_LANGUAGE.search(full_tag)
            lang = lang_match.group(1) if lang_match else ""
            code = re.sub(r"<[^>]+>", "", content).strip()
            return f"```{lang}\n{code}\n```"
        if tag in ("ul", "ol"):
            # Lists: each <li> becomes a markdown list entry.
            items = []
            for i, item in enumerate(_RE_LIST_ITEM.findall(content), 1):
                item = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", item)).strip()
                if item:
                    marker = f"{i}." if tag == "ol" else "-"
                    items.append(f"{marker} {item}")
            return "\n".join(items)
        # <br> is a line break, not markup to delete.
        content = re.sub(r"<br\s*/?>", "\n", content)
        text = re.sub(r"<[^>]+>", "", content).strip()
        if tag.startswith("h"):
            # Headlines: <h1>-<h6> map to markdown #-levels.
            return f"{'#' * int(tag[1])} {text}"
        if tag == "blockquote":
            return "\n".join(f"> {line}" for line in text.splitlines())
        return text

    @staticmethod
    def _convert_section(content: str) -> str:
        """Convert a writing/document section to markdown.

        The section chrome (header with title and edit/copy buttons,
        metadata spans) is dropped; the document body is converted with
        the same block rules as a top-level reply.
        """
        content = re.sub(r"<header\b.*?</header>", "", content, flags=re.DOTALL)
        parts = []
        pos = 0
        for inner in _RE_INNER_BLOCK.finditer(content):
            before = re.sub(r"<br\s*/?>", "\n", content[pos:inner.start()])
            before = re.sub(r"[ \t]+", " ", re.sub(r"<[^>]+>", " ", before)).strip()
            if before:
                parts.append(before)
            parts.append(ChatGPT._convert_block(inner.group(1), inner.group(0), inner.group(2)))
            pos = inner.end()
        tail = re.sub(r"<br\s*/?>", "\n", content[pos:])
        tail = re.sub(r"[ \t]+", " ", re.sub(r"<[^>]+>", " ", tail)).strip()
        if tail:
            parts.append(tail)
        return "\n\n".join(part for part in parts if part)


if __name__ == "__main__":
    debug.logging = True
    import asyncio

    async def main():
        async for chunk in ChatGPT.create_async_generator(
            "auto", [{"role": "user", "content": "Guten Tag"}]
        ):
            if isinstance(chunk, (FinishReason, JsonConversation)):
                continue
            print(chunk)

    asyncio.run(main())
