from __future__ import annotations

import json
import asyncio
import base64
from typing import AsyncIterator

try:
    import nodriver
    from nodriver import cdp
    has_nodriver = True
except ImportError:
    has_nodriver = False

from .base_provider import AsyncAuthedProvider, ProviderModelMixin
from .openai.har_file import get_headers, get_har_files
from ..typing import AsyncResult, Messages, MediaListType
from ..errors import NoValidHarFileError, MissingAuthError
from ..providers.response import *
from ..requests import get_nodriver_session
from ..image import to_bytes, is_accepted_format
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
                if isinstance(content_item, dict) and "bucket_id" in content_item and "name" not in content_item:
                    bucket_items.append(content_item)
        if message.get("role") == "assistant":
            bucket_items = []
    return bucket_items

class CopilotSession(AsyncAuthedProvider, ProviderModelMixin):
    parent = "Copilot"
    label = "Microsoft Copilot (Session)"
    url = "https://copilot.microsoft.com"
    
    working = has_nodriver
    use_nodriver = has_nodriver
    active_by_default = True
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
    lock = asyncio.Lock()

    @classmethod
    async def on_auth_async(cls, cookies: dict = None, proxy: str = None, **kwargs) -> AsyncIterator:
        yield AuthResult()

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        proxy: str = None,
        timeout: int = 30,
        prompt: str = None,
        media: MediaListType = None,
        conversation: BaseConversation = None,
        **kwargs
    ) -> AsyncResult:
        async with get_nodriver_session(proxy=proxy) as session:
            if prompt is None:
                prompt = get_last_user_message(messages, False)
            if conversation is not None:
                conversation_id = conversation.conversation_id
                url = f"{cls.url}/chats/{conversation_id}"
            else:
                url = cls.url
            page = await session.get(url)
            await page.send(cdp.network.enable())
            queue = asyncio.Queue()
            page.add_handler(
                cdp.network.WebSocketFrameReceived,
                lambda event: queue.put_nowait((event.request_id, event.response.payload_data)),
            )
            textarea = await page.select("textarea")
            if textarea is not None:
                await textarea.send_keys(prompt)
                await asyncio.sleep(1)
                button = await page.select("[data-testid=\"submit-button\"]")
                if button:
                    await button.click()
                    turnstile = await page.select('#cf-turnstile')
                    if turnstile:
                        debug.log("Found Element: 'cf-turnstile'")
                        await asyncio.sleep(3)
                        await click_trunstile(page)

        # uploaded_attachments = []
        # if auth_result.access_token:
        #     # Upload regular media (images)
        #     for media, _ in merge_media(media, messages):
        #         if not isinstance(media, str):
        #             data_bytes = to_bytes(media)
        #             response_json = await page.evaluate(f'''
        #             fetch('https://copilot.microsoft.com/c/api/attachments', {{
        #             method: 'POST',
        #             headers: {{
        #             'content-type': '{is_accepted_format(data_bytes)}',
        #             'content-length': '{len(data_bytes)}',
        #             "'x-useridentitytype': '{auth_result.useridentitytype}'," if getattr(auth_result, "useridentitytype", None) else ""
        #             }},
        #             body: new Uint8Array({list(data_bytes)})
        #             }}).then(r => r.json())
        #             ''')
        #             media = response_json.get("url")
        #         uploaded_attachments.append({{"type":"image", "url": media}})

        #     # Upload bucket files
        #     bucket_items = extract_bucket_items(messages)
        #     for item in bucket_items:
        #         try:
        #             # Handle plain text content from bucket
        #             bucket_path = Path(get_bucket_dir(item["bucket_id"]))
        #             for text_chunk in read_bucket(bucket_path):
        #                 if text_chunk.strip():
        #                     # Upload plain text as a text file
        #                     response_json = await page.evaluate(f'''
        #                     const formData = new FormData();
        #                     formData.append('file', new Blob(['{text_chunk.replace(chr(39), "\\'").replace(chr(10), "\\n").replace(chr(13), "\\r")}'], {{type: 'text/plain'}}), 'bucket_{item['bucket_id']}.txt');
        #                     fetch('https://copilot.microsoft.com/c/api/attachments', {{
        #                     method: 'POST',
        #                     headers: {{
        #                     "'x-useridentitytype': '{auth_result.useridentitytype}'," if auth_result.useridentitytype else ""
        #                     }},
        #                     body: formData
        #                     }}).then(r => r.json())
        #                     ''')
        #                     data = response_json
        #                     uploaded_attachments.append({{"type": "document", "attachmentId": data.get("id")}})
        #                     debug.log(f"Copilot: Uploaded bucket text content: {item['bucket_id']}")
        #                 else:
        #                     debug.log(f"Copilot: No text content found in bucket: {item['bucket_id']}")
        #         except Exception as e:
        #             debug.log(f"Copilot: Failed to upload bucket item: {item}")
        #             debug.error(e)

        done = False
        msg = None
        image_prompt: str = None
        last_msg = None
        sources = {}
        while not done:
            try:
                request_id, msg_txt = await asyncio.wait_for(queue.get(), 1 if done else timeout)
                msg = json.loads(msg_txt)
            except:
                break
            last_msg = msg
            if msg.get("event") == "startMessage":
                yield Conversation(msg.get("conversationId"))
            elif msg.get("event") == "appendText":
                yield msg.get("text")
            elif msg.get("event") == "generatingImage":
                image_prompt = msg.get("prompt")
            elif msg.get("event") == "imageGenerated":
                yield ImageResponse(msg.get("url"), image_prompt, {{"preview": msg.get("thumbnailUrl")}})
            elif msg.get("event") == "done":
                yield FinishReason("stop")
                done = True
            elif msg.get("event") == "suggestedFollowups":
                yield SuggestedFollowups(msg.get("suggestions"))
                break
            elif msg.get("event") == "replaceText":
                yield msg.get("text")
            elif msg.get("event") == "titleUpdate":
                yield TitleGeneration(msg.get("title"))
            elif msg.get("event") == "citation":
                sources[msg.get("url")] = msg
                yield SourceLink(list(sources.keys()).index(msg.get("url")), msg.get("url"))
            elif msg.get("event") == "partialImageGenerated":
                mime_type = is_accepted_format(base64.b64decode(msg.get("content")[:12]))
                yield ImagePreview(f"data:{mime_type};base64,{msg.get('content')}", image_prompt)
            elif msg.get("event") == "chainOfThought":
                yield Reasoning(msg.get("text"))
            elif msg.get("event") == "error":
                raise RuntimeError(f"Error: {msg}")
            elif msg.get("event") not in ["received", "startMessage", "partCompleted", "connected"]:
                debug.log(f"Copilot Message: {msg_txt[:100]}...")
        if not done:
            raise MissingAuthError(f"Invalid response: {last_msg}")
        if sources:
            yield Sources(sources.values())