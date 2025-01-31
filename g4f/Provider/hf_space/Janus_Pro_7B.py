from __future__ import annotations

import json
import uuid
import re
from datetime import datetime, timezone, timedelta
import urllib.parse

from ...typing import AsyncResult, Messages, Cookies
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, format_image_prompt
from ...providers.response import JsonConversation, ImageResponse, Notification
from ...requests.aiohttp import StreamSession, StreamResponse
from ...requests.raise_for_status import raise_for_status
from ...cookies import get_cookies
from ...errors import ResponseError
from ... import debug

class Janus_Pro_7B(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co/spaces/deepseek-ai/Janus-Pro-7B"
    api_url = "https://deepseek-ai-janus-pro-7b.hf.space"
    referer = f"{api_url}?__theme=light"

    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "janus-pro-7b"
    default_image_model = "janus-pro-7b-image"
    models = [default_model, default_image_model]
    image_models = [default_image_model]

    @classmethod
    def run(cls, method: str, session: StreamSession, prompt: str, conversation: JsonConversation):
            if method == "post":
                return session.post(f"{cls.api_url}/gradio_api/queue/join?__theme=light", **{
                    "headers": {
                        "content-type": "application/json",
                        "x-zerogpu-token": conversation.zerogpu_token,
                        "x-zerogpu-uuid": conversation.zerogpu_uuid,
                        "referer": cls.referer,
                    },
                    "json": {"data":[None,prompt,42,0.95,0.1],"event_data":None,"fn_index":2,"trigger_id":10,"session_hash":conversation.session_hash},
                })
            elif method == "image":
                return session.post(f"{cls.api_url}/gradio_api/queue/join?__theme=light", **{
                    "headers": {
                        "content-type": "application/json",
                        "x-zerogpu-token": conversation.zerogpu_token,
                        "x-zerogpu-uuid": conversation.zerogpu_uuid,
                        "referer": cls.referer,
                    },
                    "json": {"data":[prompt,1234,5,1],"event_data":None,"fn_index":3,"trigger_id":20,"session_hash":conversation.session_hash},
                })
            return session.get(f"{cls.api_url}/gradio_api/queue/data?session_hash={conversation.session_hash}", **{
                "headers": {
                    "accept": "text/event-stream",
                    "content-type": "application/json",
                    "referer": cls.referer,
                }
            })

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        cookies: Cookies = None,
        return_conversation: bool = False,
        conversation: JsonConversation = None,
        **kwargs
    ) -> AsyncResult:
        def generate_session_hash():
            """Generate a unique session hash."""
            return str(uuid.uuid4()).replace('-', '')[:12]

        method = "post"
        if model == cls.default_image_model or prompt is not None:
            method = "image"

        prompt = format_prompt(messages) if prompt is None and conversation is None else prompt
        prompt = format_image_prompt(messages, prompt)

        session_hash = generate_session_hash() if conversation is None else getattr(conversation, "session_hash")
        async with StreamSession(proxy=proxy, impersonate="chrome") as session:
            session_hash = generate_session_hash() if conversation is None else getattr(conversation, "session_hash")
            zerogpu_uuid, zerogpu_token = await get_zerogpu_token(session, conversation, cookies)
            if conversation is None or not hasattr(conversation, "session_hash"):
                conversation = JsonConversation(session_hash=session_hash, zerogpu_token=zerogpu_token, zerogpu_uuid=zerogpu_uuid)
            conversation.zerogpu_token = zerogpu_token
            if return_conversation:
                yield conversation

            async with cls.run(method, session, prompt, conversation) as response:
                await raise_for_status(response)

            async with cls.run("get", session, prompt, conversation) as response:
                response: StreamResponse = response
                async for line in response.iter_lines():
                    decoded_line = line.decode(errors="replace")
                    if decoded_line.startswith('data: '):
                        try:
                            json_data = json.loads(decoded_line[6:])
                            if json_data.get('msg') == 'log':
                                yield Notification(json_data["log"])

                            if json_data.get('msg') == 'process_generating':
                                if 'output' in json_data and 'data' in json_data['output']:
                                    yield f"data: {json.dumps(json_data['output']['data'])}"

                            if json_data.get('msg') == 'process_completed':
                                if 'output' in json_data and 'error' in json_data['output']:
                                    raise ResponseError("Text model is not working. Try out image model" if "AttributeError" in json_data['output']['error'] else json_data['output']['error'])
                                if 'output' in json_data and 'data' in json_data['output']:
                                    if "image" in json_data['output']['data'][0][0]:
                                        yield ImageResponse([image["image"]["url"] for image in json_data['output']['data'][0]], prompt)
                                    else:
                                        yield f"data: {json.dumps(json_data['output']['data'])}"
                                break

                        except json.JSONDecodeError:
                            debug.log("Could not parse JSON:", decoded_line)

async def get_zerogpu_token(session: StreamSession, conversation: JsonConversation, cookies: Cookies = None):
    zerogpu_uuid = None if conversation is None else getattr(conversation, "zerogpu_uuid", None)
    zerogpu_token = "[object Object]"

    cookies = get_cookies("huggingface.co", raise_requirements_error=False) if cookies is None else cookies
    if zerogpu_uuid is None:
        async with session.get(Janus_Pro_7B.url, cookies=cookies) as response:
            match = re.search(r"&quot;token&quot;:&quot;([^&]+?)&quot;", await response.text())
            if match:
                zerogpu_token = match.group(1)
            match = re.search(r"&quot;sessionUuid&quot;:&quot;([^&]+?)&quot;", await response.text())
            if match:
                zerogpu_uuid = match.group(1)
    if cookies:
        # Get current UTC time + 10 minutes
        dt = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(timespec='milliseconds')
        encoded_dt = urllib.parse.quote(dt)
        async with session.get(f"https://huggingface.co/api/spaces/deepseek-ai/Janus-Pro-7B/jwt?expiration={encoded_dt}&include_pro_status=true", cookies=cookies) as response:
            zerogpu_token = (await response.json())
            zerogpu_token = zerogpu_token["token"]
    
    return zerogpu_uuid, zerogpu_token