from __future__ import annotations

import json
import uuid
import re
import random
from datetime import datetime, timezone, timedelta
import urllib.parse

from ...typing import AsyncResult, Messages, Cookies, MediaListType
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, format_media_prompt
from ...providers.response import JsonConversation, ImageResponse, Reasoning
from ...requests.aiohttp import StreamSession, StreamResponse, FormData
from ...requests.raise_for_status import raise_for_status
from ...tools.media import merge_media
from ...image import to_bytes, is_accepted_format
from ...cookies import get_cookies
from ...errors import ResponseError, ModelNotFoundError
from ... import debug
from .raise_for_status import raise_for_status

class DeepseekAI_JanusPro7b(AsyncGeneratorProvider, ProviderModelMixin):
    label = "DeepseekAI Janus-Pro-7B"
    space = "deepseek-ai/Janus-Pro-7B"
    url = f"https://huggingface.co/spaces/{space}"
    api_url = "https://deepseek-ai-janus-pro-7b.hf.space"
    referer = f"{api_url}?__theme=light"

    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "janus-pro-7b"
    default_image_model = "janus-pro-7b-image"
    default_vision_model = default_model
    image_models = [default_image_model]
    vision_models = [default_vision_model]
    models = vision_models + image_models

    @classmethod
    def run(cls, method: str, session: StreamSession, prompt: str, conversation: JsonConversation, image: dict = None, seed: int = 0):
            headers = {
                "content-type": "application/json",
                "x-zerogpu-token": conversation.zerogpu_token,
                "x-zerogpu-uuid": conversation.zerogpu_uuid,
                "referer": cls.referer,
            }
            if method == "post":
                return session.post(f"{cls.api_url}/gradio_api/queue/join?__theme=light", **{
                    "headers": {k: v for k, v in headers.items() if v is not None},
                    "json": {"data":[image,prompt,seed,0.95,0.1],"event_data":None,"fn_index":2,"trigger_id":10,"session_hash":conversation.session_hash},
                })
            elif method == "image":
                return session.post(f"{cls.api_url}/gradio_api/queue/join?__theme=light", **{
                    "headers": {k: v for k, v in headers.items() if v is not None},
                    "json": {"data":[prompt,seed,5,1],"event_data":None,"fn_index":3,"trigger_id":20,"session_hash":conversation.session_hash},
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
        media: MediaListType = None,
        prompt: str = None,
        proxy: str = None,
        cookies: Cookies = None,
        api_key: str = None,
        zerogpu_uuid: str = "[object Object]",
        return_conversation: bool = True,
        conversation: JsonConversation = None,
        seed: int = None,
        **kwargs
    ) -> AsyncResult:
        if model and "janus" not in model:
            raise ModelNotFoundError(f"Model '{model}' not found. Available models: {', '.join(cls.models)}")
        method = "post"
        if model == cls.default_image_model or prompt is not None:
            method = "image"
        prompt = format_prompt(messages) if prompt is None and conversation is None else prompt
        prompt = format_media_prompt(messages, prompt)
        if seed is None:
            seed = random.randint(1000, 999999)

        session_hash = uuid.uuid4().hex if conversation is None else getattr(conversation, "session_hash", uuid.uuid4().hex)
        async with StreamSession(proxy=proxy, impersonate="chrome") as session:
            if api_key is None:
                zerogpu_uuid, api_key = await get_zerogpu_token(cls.space, session, conversation, cookies)
            if conversation is None or not hasattr(conversation, "session_hash"):
                conversation = JsonConversation(session_hash=session_hash, zerogpu_token=api_key, zerogpu_uuid=zerogpu_uuid)
            else:
                conversation.zerogpu_token = api_key
            if return_conversation:
                yield conversation

            media = list(merge_media(media, messages))
            if media:
                data = FormData()
                for i in range(len(media)):
                    media[i] = (to_bytes(media[i][0]), media[i][1])
                for image, image_name in media:
                    data.add_field(f"files", image, filename=image_name)
                async with session.post(f"{cls.api_url}/gradio_api/upload", params={"upload_id": session_hash}, data=data) as response:
                    await raise_for_status(response)
                    image_files = await response.json()
                media = [{
                    "path": image_file,
                    "url": f"{cls.api_url}/gradio_api/file={image_file}",
                    "orig_name": media[i][1],
                    "size": len(media[i][0]),
                    "mime_type": is_accepted_format(media[i][0]),
                    "meta": {
                        "_type": "gradio.FileData"
                    }
                } for i, image_file in enumerate(image_files)]

            async with cls.run(method, session, prompt, conversation, None if not media else media.pop(), seed) as response:
                await raise_for_status(response)

            async with cls.run("get", session, prompt, conversation, None, seed) as response:
                response: StreamResponse = response
                counter = 3
                async for line in response.iter_lines():
                    decoded_line = line.decode(errors="replace")
                    if decoded_line.startswith('data: '):
                        try:
                            json_data = json.loads(decoded_line[6:])
                            if json_data.get('msg') == 'log':
                                yield Reasoning(status=json_data["log"])

                            if json_data.get('msg') == 'progress':
                                if 'progress_data' in json_data:
                                    if json_data['progress_data']:
                                        progress = json_data['progress_data'][0]
                                        yield Reasoning(status=f"{progress['desc']} {progress['index']}/{progress['length']}")
                                    else:
                                        yield Reasoning(status=f"Generating")

                            elif json_data.get('msg') == 'heartbeat':
                                yield Reasoning(status=f"Generating{''.join(['.' for i in range(counter)])}")
                                counter  += 1

                            elif json_data.get('msg') == 'process_completed':
                                if 'output' in json_data and 'error' in json_data['output']:
                                    json_data['output']['error'] = json_data['output']['error'].split(" <a ")[0]
                                    raise ResponseError("Missing images input" if json_data['output']['error'] and "AttributeError" in json_data['output']['error'] else json_data['output']['error'])
                                if 'output' in json_data and 'data' in json_data['output']:
                                    yield Reasoning(status="")
                                    if "image" in json_data['output']['data'][0][0]:
                                        yield ImageResponse([image["image"]["url"] for image in json_data['output']['data'][0]], prompt)
                                    else:
                                        yield json_data['output']['data'][0]
                                break

                        except json.JSONDecodeError:
                            debug.log("Could not parse JSON:", decoded_line)

async def get_zerogpu_token(space: str, session: StreamSession, conversation: JsonConversation, cookies: Cookies = None):
    zerogpu_uuid = None if conversation is None else getattr(conversation, "zerogpu_uuid", None)
    zerogpu_token = "[object Object]"

    cookies = get_cookies("huggingface.co", raise_requirements_error=False) if cookies is None else cookies
    if zerogpu_uuid is None:
        async with session.get(f"https://huggingface.co/spaces/{space}", cookies=cookies) as response:
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
        async with session.get(f"https://huggingface.co/api/spaces/{space}/jwt?expiration={encoded_dt}&include_pro_status=true", cookies=cookies) as response:
            response_data = (await response.json())
            if "token" in response_data:
                zerogpu_token = response_data["token"]

    return zerogpu_uuid, zerogpu_token
