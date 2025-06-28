from __future__ import annotations

import json
import re
import os
import requests
import base64
import uuid
from typing import AsyncIterator

try:
    from curl_cffi.requests import Session
    from curl_cffi import CurlMime
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

from ...base_provider import ProviderModelMixin, AsyncAuthedProvider, AuthResult
from ...helper import format_prompt, format_media_prompt, get_last_user_message
from ....typing import AsyncResult, Messages, Cookies, MediaListType
from ....errors import MissingRequirementsError, MissingAuthError, ResponseError
from ....image import to_bytes
from ....requests import get_args_from_nodriver, DEFAULT_HEADERS
from ....requests.raise_for_status import raise_for_status
from ....providers.response import JsonConversation, ImageResponse, Sources, TitleGeneration, Reasoning, RequestLogin, FinishReason
from ....cookies import get_cookies
from ....tools.media import merge_media
from .models import default_model, default_vision_model, fallback_models, image_models, model_aliases, image_model_aliases
from .... import debug

class Conversation(JsonConversation):
    def __init__(self, models: dict):
        self.models: dict = models

class HuggingChat(AsyncAuthedProvider, ProviderModelMixin):
    domain = "huggingface.co"
    origin = f"https://{domain}"
    url = f"{origin}/chat"

    working = True
    use_nodriver = True
    supports_stream = True
    needs_auth = True
    default_model = default_model
    default_vision_model = default_vision_model
    model_aliases = {**model_aliases, **image_model_aliases}
    image_models = image_models
    text_models = fallback_models

    @classmethod
    def get_models(cls):
        if not cls.models:
            try:
                models = requests.get(f"{cls.url}/api/v2/models").json().get("json")
                cls.text_models = [model["id"] for model in models] 
                cls.models = cls.text_models + cls.image_models
                cls.vision_models = [model["id"] for model in models if model["multimodal"]]
            except Exception as e:
                debug.error(f"{cls.__name__}: Error reading models: {type(e).__name__}: {e}")
                cls.models = [*fallback_models]
        return cls.models

    @classmethod
    async def on_auth_async(cls, cookies: Cookies = None, proxy: str = None, **kwargs) -> AsyncIterator:
        if cookies is None:
            cookies = get_cookies(cls.domain, single_browser=True)
        if "hf-chat" in cookies:
            yield AuthResult(
                cookies=cookies,
                headers=DEFAULT_HEADERS,
                impersonate="chrome"
            )
            return
        if cls.needs_auth:
            yield RequestLogin(cls.__name__, os.environ.get("G4F_LOGIN_URL") or "")
            yield AuthResult(
                **await get_args_from_nodriver(
                    cls.url,
                    proxy=proxy,
                    wait_for='form[action$="/logout"]'
                )
            )
        else:
            yield AuthResult(
                cookies={
                    "hf-chat": str(uuid.uuid4())  # Generate a session ID
                },
                headers=DEFAULT_HEADERS,
                impersonate="chrome"
            )

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        prompt: str = None,
        media: MediaListType = None,
        return_conversation: bool = True,
        conversation: Conversation = None,
        web_search: bool = False,
        **kwargs
    ) -> AsyncResult:
        if not has_curl_cffi:
            raise MissingRequirementsError('Install "curl_cffi" package | pip install -U curl_cffi')
        if not model and media is not None:
            model = cls.default_vision_model
        model = cls.get_model(model)

        session = Session(**auth_result.get_dict())

        if conversation is None or not hasattr(conversation, "models"):
            conversation = Conversation({})

        if model not in conversation.models:
            conversationId = cls.create_conversation(session, model)
            debug.log(f"Conversation created: {json.dumps(conversationId[8:] + '...')}")
            messageId = cls.fetch_message_id(session, conversationId)
            conversation.models[model] = {"conversationId": conversationId, "messageId": messageId}
            if return_conversation:
                yield conversation
            inputs = format_prompt(messages)
        else:
            conversationId = conversation.models[model]["conversationId"]
            conversation.models[model]["messageId"] = cls.fetch_message_id(session, conversationId)
            inputs = get_last_user_message(messages)

        settings = {
            "inputs": inputs,
            "id": conversation.models[model]["messageId"],
            "is_retry": False,
            "is_continue": False,
            "web_search": web_search,
            "tools": ["000000000000000000000001"] if model in cls.image_models else [],
        }

        headers = {
            'accept': '*/*',
            'origin': cls.origin,
            'referer': f'{cls.url}/conversation/{conversationId}',
        }
        data = CurlMime()
        data.addpart('data', data=json.dumps(settings, separators=(',', ':')))
        for image, filename in merge_media(media, messages):
            data.addpart(
                "files",
                filename=f"base64;{filename}",
                data=base64.b64encode(to_bytes(image))
            )

        response = session.post(
            f'{cls.url}/conversation/{conversationId}',
            headers=headers,
            multipart=data,
            stream=True
        )
        raise_for_status(response)

        sources = None
        for line in response.iter_lines():
            if not line:
                continue
            try:
                line = json.loads(line)
            except json.JSONDecodeError as e:
                debug.error(f"Failed to decode JSON: {line}, error: {e}")
                continue
            if "type" not in line:
                raise RuntimeError(f"Response: {line}")
            elif line["type"] == "stream":
                yield line["token"].replace('\u0000', '')
            elif line["type"] == "finalAnswer":
                if sources is not None:
                    yield sources
                yield FinishReason("stop")
                break
            elif line["type"] == "file":
                url = f"{cls.url}/conversation/{conversationId}/output/{line['sha']}"
                yield ImageResponse(url, format_media_prompt(messages, prompt), options={"cookies": auth_result.cookies})
            elif line["type"] == "webSearch" and "sources" in line:
                sources = Sources(line["sources"])
            elif line["type"] == "title":
                yield TitleGeneration(line["title"])
            elif line["type"] == "reasoning":
                yield Reasoning(line.get("token"), status=line.get("status"))

    @classmethod
    def create_conversation(cls, session: Session, model: str):
        if model in cls.image_models:
            model = cls.default_model
        json_data = {
            'model': model,
        }
        response = session.post(f'{cls.url}/conversation', json=json_data)
        if response.status_code == 401:
            raise MissingAuthError(response.text)
        if response.status_code == 400:
            raise ResponseError(f"{response.text}: Model: {model}")
        raise_for_status(response)
        return response.json().get('conversationId')

    @classmethod
    def fetch_message_id(cls, session: Session, conversation_id: str):
        response = session.get(
            f"{cls.url}/api/v2/conversations/{conversation_id}"
        )
        raise_for_status(response)

        try:
            data = response.json()['json']
        except json.JSONDecodeError as e:
            debug.error(f"Failed to decode JSON: {e}")
            return None

        messages_data_list = data.get("messages", [])
        return messages_data_list[-1]['id'] if messages_data_list else None