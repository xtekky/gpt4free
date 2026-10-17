from __future__ import annotations

import asyncio
import json

from ..typing import AsyncResult, Messages
from ..requests.cdp import CDPSession
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message
from ..providers.response import JsonConversation, FinishReason
from .. import debug

class Conversation(JsonConversation):
    """Holds the ids of the browser chat session.

    Both ids are extracted from the page after the reply has finished;
    passing the conversation back into the provider reopens that chat
    (via its /c/<conversation_id> url) instead of starting a new one.
    """

    def __init__(self, model: str):
        self.model = model
        self.conversation_id: str = None
        self.message_id: str = None

class ChatGPT(AsyncGeneratorProvider, ProviderModelMixin):
    """ChatGPT chat driven entirely by a real browser.

    Instead of simulating the anti-bot pipeline (proof-of-work, turnstile,
    requirements tokens) in Python, this provider lets the real browser do
    the talking: it opens the chat page, types the prompt into the composer,
    clicks the send button and reads the reply from the page as it streams
    in. No request is intercepted or replayed, so every token the page mints
    stays genuine and the flow survives client-side changes.

    Long (pasted) prompts are supported: the reply timeout scales with the
    prompt length.
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
    models = [default_model]
    model_aliases = {
        "gpt-5-2": "",
        "gpt-5-1": "",
        "gpt-5": "",
        "gpt-4": "",
        "gpt-4.1": "",
        "gpt-4.1-mini": "",
        "gpt-4.5": "",
        "gpt-4o": "",
        "gpt-4o-mini": "",
        "o1": "",
        "o1-mini": "",
        "o3-mini": "",
        "o3-mini-high": "",
        "o4-mini": "",
        "o4-mini-high": "",
    }

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
        prompt = get_last_user_message(messages)
        if timeout is None:
            timeout = 180
        if conversation is None:
            conversation = Conversation(model)

        async with CDPSession(proxy=proxy, headless=False) as session:
            if getattr(conversation, "conversation_id", None):
                # Reopen the existing chat so the reply keeps its context.
                await session.navigate(f"{cls.url}/c/{conversation.conversation_id}")
            else:
                await session.navigate(cls.url)
            # Wait for the composer to appear, then replace its content
            # with the prompt and submit the form.
            for _ in range(20):
                if await session.evaluate_js("document.querySelector('.ProseMirror, [data-mobile-composer-prompt]') !== null"):
                    break
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError("ChatGPT: composer did not appear")
            await session.evaluate_js(f"""
                (() => {{
                    const editor = document.querySelector('.ProseMirror, [data-mobile-composer-prompt]');
                    editor.focus();
                    document.execCommand('selectAll', false, null);
                    document.execCommand('insertText', false, {json.dumps(prompt)});
                }})()
            """)
            await asyncio.sleep(1)
            if not await session.evaluate_js("""
                (() => {
                    const button = document.querySelector('form button[type="submit"]:not([disabled])');
                    if (!button) return false;
                    button.click();
                    return true;
                })()
            """):
                raise RuntimeError("ChatGPT: send button not found")

            # A single CDP call cannot outlast the session's 30 second
            # timeout, so a poller interval in the page collects the reply
            # while Python polls for it. The poller reads the last
            # assistant reply ([data-assistant-markdown]) as markdown
            # and flags itself done once the text stopped growing.
            await session.evaluate_js("""
                (() => {
                    const toMarkdown = (root) => {
                        const blocks = [];
                        const inline = (node) => {
                            let out = "";
                            for (const child of node.childNodes) {
                                if (child.nodeType === Node.TEXT_NODE) {
                                    out += child.textContent;
                                } else if (child.nodeType === Node.ELEMENT_NODE) {
                                    const tag = child.tagName;
                                    if (tag === "BR") {
                                        out += "\\n";
                                    } else if (tag === "A") {
                                        out += `[${inline(child)}](${child.getAttribute("href") || ""})`;
                                    } else if (tag === "STRONG" || tag === "B") {
                                        out += `**${inline(child)}**`;
                                    } else if (tag === "EM" || tag === "I") {
                                        out += `*${inline(child)}*`;
                                    } else if (tag === "CODE") {
                                        out += '\\\\`' + child.textContent + '\\\\`';
                                    } else if (tag === "DEL" || tag === "S") {
                                        out += `~~${inline(child)}~~`;
                                    } else {
                                        out += inline(child);
                                    }
                                }
                            }
                            return out;
                        };
                        const walk = (node) => {
                            for (const child of node.childNodes) {
                                if (child.nodeType !== Node.ELEMENT_NODE) continue;
                                const tag = child.tagName;
                                if (tag === "P") {
                                    blocks.push(inline(child));
                                } else if (/^H[1-6]$/.test(tag)) {
                                    if (child.textContent.trim().startsWith("ChatGPT")) continue;
                                    blocks.push(`${"#".repeat(+tag[1])} ${inline(child)}`);
                                } else if (tag === "UL" || tag === "OL") {
                                    let index = 1;
                                    for (const li of child.children) {
                                        if (li.tagName !== "LI") continue;
                                        const marker = tag === "OL" ? `${index++}. ` : "- ";
                                        const items = [];
                                        for (const part of li.childNodes) {
                                            if (part.nodeType === Node.TEXT_NODE) {
                                                if (part.textContent.trim()) items.push(part.textContent.trim());
                                            } else if (part.tagName === "P") {
                                                items.push(inline(part));
                                            } else if (part.tagName === "UL" || part.tagName === "OL") {
                                                items.push(null);
                                                walk({childNodes: [part]});
                                                items.push(null);
                                            } else {
                                                items.push(inline(part));
                                            }
                                        }
                                        const body = items.filter((item) => item !== null).join("\\n");
                                        blocks.push(marker + body);
                                    }
                                } else if (tag === "PRE") {
                                    const code = child.querySelector("code");
                                    const language = code ? (code.className.match(/language-([\\w+-]+)/) || [])[1] || "" : "";
                                    const fence = '\\\\`\\\\`\\\\`';
                                    blocks.push(fence + language + '\\n' + (code || child).textContent.replace(/\\n$/, "") + '\\n' + fence);
                                } else if (tag === "BLOCKQUOTE") {
                                    blocks.push(inline(child).split("\\n").map((line) => `> ${line}`).join("\\n"));
                                } else if (tag === "TABLE") {
                                    const rows = [...child.querySelectorAll("tr")].map((tr) =>
                                        [...tr.children].map((cell) => inline(cell).replace(/\\|/g, "\\\\|").trim())
                                    );
                                    if (rows.length) {
                                        blocks.push(`| ${rows[0].join(" | ")} |`);
                                        blocks.push(`| ${rows[0].map(() => "---").join(" | ")} |`);
                                        for (const row of rows.slice(1)) blocks.push(`| ${row.join(" | ")} |`);
                                    }
                                } else if (tag === "HR") {
                                    blocks.push("---");
                                } else {
                                    walk(child);
                                }
                            }
                        };
                        walk(root);
                        return blocks.filter((block) => block !== null && block.trim()).join("\\n\\n");
                    };
                    window.__g4fReply = null;
                    window.__g4fDone = false;
                    let lastText = "";
                    let stableSince = 0;
                    window.__g4fPoller = setInterval(() => {
                        const messages = document.querySelectorAll('div:has([data-conversation-role="assistant"]), [data-message-role="assistant"]');
                        const last = messages[messages.length - 1];
                        let text = "";
                        if (last) text = toMarkdown(last);
                        if (text && text === lastText) {
                            if (!stableSince) stableSince = Date.now();
                            if (Date.now() - stableSince > 1500) {
                                window.__g4fReply = text;
                                window.__g4fDone = true;
                                clearInterval(window.__g4fPoller);
                            }
                        } else {
                            stableSince = 0;
                            if (text) {
                                window.__g4fReply = text;
                                lastText = text;
                            }
                        }
                    }, 300);
                })()
            """)

            # Stream the reply as the page collects it.
            reply = ""
            deadline = asyncio.get_running_loop().time() + cls._reply_timeout(timeout, prompt)
            while True:
                state = await session.evaluate_js(
                    "window.__g4fReply ? {text: window.__g4fReply, done: window.__g4fDone} : null"
                )
                if state:
                    text = state["text"]
                    if len(text) > len(reply):
                        yield text[len(reply):]
                        reply = text
                    if state["done"]:
                        break
                if asyncio.get_running_loop().time() > deadline:
                    raise RuntimeError("ChatGPT: reply timed out")
                await asyncio.sleep(0.5)

            # Extract the ids needed to continue this conversation later.
            ids = await session.evaluate_js("""
                (() => {
                    const match = location.href.match(/\\/c\\/([0-9a-f-]{36})/);
                    const unit = document.querySelector('[data-chatgpt-search-unit-key$=":assistant"]');
                    let message_id = null;
                    if (unit) {
                        const ids = unit.getAttribute('data-chatgpt-search-message-ids') || '';
                        message_id = ids.split(' ')[0] || null;
                    }
                    return {conversation_id: match ? match[1] : null, message_id: message_id};
                })()
            """)
            if ids:
                conversation.conversation_id = ids.get("conversation_id") or conversation.conversation_id
                conversation.message_id = ids.get("message_id") or conversation.message_id

        if not reply:
            raise RuntimeError("ChatGPT: no reply received")
        if return_conversation:
            yield conversation
        yield FinishReason("stop")

    @staticmethod
    def _reply_timeout(timeout, prompt: str) -> int:
        """Scale the reply timeout with the prompt length.

        Long (pasted) prompts take proportionally longer to answer.
        """
        if timeout is None:
            timeout = 180
        return min(max(timeout, 60 + len(prompt) // 20), 1800)
