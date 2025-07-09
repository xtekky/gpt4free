from __future__ import annotations

import json
import time
import random
import re
import os
import base64
import asyncio
from aiohttp import ClientSession, FormData

from ..typing import AsyncResult, Messages, MediaListType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..providers.response import JsonConversation, FinishReason, ImageResponse
from ..image import to_data_uri, is_data_an_media
from ..tools.media import merge_media


class Conversation(JsonConversation):
    """Manages all session-specific state for Opera Aria."""
    access_token: str = None
    refresh_token: str = None
    encryption_key: str = None
    expires_at: float = 0
    conversation_id: str = None
    is_first_request: bool = True
    
    def __init__(self, refresh_token: str = None):
        """Initializes a new session, generating a unique encryption key."""
        self.refresh_token = refresh_token
        self.encryption_key = self._generate_encryption_key()
        self.is_first_request = True
    
    def is_token_expired(self) -> bool:
        """Check if the current token has expired"""
        return time.time() >= self.expires_at
    
    def update_token(self, access_token: str, expires_in: int):
        """Update the access token and expiration time"""
        self.access_token = access_token
        self.expires_at = time.time() + expires_in - 60
    
    @staticmethod
    def _generate_encryption_key() -> str:
        """Generates a 32-byte, Base64-encoded key for the session."""
        random_bytes = os.urandom(32)
        return base64.b64encode(random_bytes).decode('utf-8')

    @staticmethod
    def generate_conversation_id() -> str:
        """Generate conversation ID in Opera Aria format"""
        parts = [
            ''.join(random.choices('0123456789abcdef', k=8)),
            ''.join(random.choices('0123456789abcdef', k=4)),
            '11f0',
            ''.join(random.choices('0123456789abcdef', k=4)),
            ''.join(random.choices('0123456789abcdef', k=12))
        ]
        return '-'.join(parts)


class OperaAria(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Opera Aria"
    url = "https://play.google.com/store/apps/details?id=com.opera.browser"
    api_endpoint = "https://composer.opera-api.com/api/v1/a-chat"
    token_endpoint = "https://oauth2.opera-api.com/oauth2/v1/token/"
    signup_endpoint = "https://auth.opera.com/account/v2/external/anonymous/signup"
    upload_endpoint = "https://composer.opera-api.com/api/v1/images/upload"
    check_status_endpoint = "https://composer.opera-api.com/api/v1/images/check-status/"
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'aria'
    default_image_model = 'aria'
    image_models = ['aria']
    default_vision_model = 'aria'
    vision_models = ['aria']
    models = ['aria']
        
    @classmethod
    async def _generate_refresh_token(cls, session: ClientSession) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 OPR/89.0.0.0",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "client_id": "ofa-client",
            "client_secret": "N9OscfA3KxlJASuIe29PGZ5RpWaMTBoy",
            "grant_type": "client_credentials",
            "scope": "anonymous_account"
        }
        async with session.post(cls.token_endpoint, headers=headers, data=data) as response:
            response.raise_for_status()
            anonymous_token_data = await response.json()
            anonymous_access_token = anonymous_token_data["access_token"]

        headers = {
            "User-Agent": "Mozilla 5.0 (Linux; Android 14) com.opera.browser OPR/89.5.4705.84314",
            "Authorization": f"Bearer {anonymous_access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        }
        data = {"client_id": "ofa", "service": "aria"}
        async with session.post(cls.signup_endpoint, headers=headers, json=data) as response:
            response.raise_for_status()
            signup_data = await response.json()
            auth_token = signup_data["token"]

        headers = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 OPR/89.0.0.0",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "auth_token": auth_token,
            "client_id": "ofa",
            "device_name": "GPT4FREE",
            "grant_type": "auth_token",
            "scope": "ALL"
        }
        async with session.post(cls.token_endpoint, headers=headers, data=data) as response:
            response.raise_for_status()
            final_token_data = await response.json()
            return final_token_data["refresh_token"]

    @classmethod
    def get_model(cls, model: str) -> str:
        return cls.model_aliases.get(model, cls.default_model)

    @classmethod
    async def get_access_token(cls, session: ClientSession, conversation: Conversation) -> str:
        if not conversation.refresh_token:
            conversation.refresh_token = await cls._generate_refresh_token(session)

        if conversation.access_token and not conversation.is_token_expired():
            return conversation.access_token
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 OPR/89.0.0.0"
        }
        data = {
            "client_id": "ofa",
            "grant_type": "refresh_token",
            "refresh_token": conversation.refresh_token,
            "scope": "shodan:aria user:read"
        }
        async with session.post(cls.token_endpoint, headers=headers, data=data) as response:
            response.raise_for_status()
            result = await response.json()
            conversation.update_token(
                access_token=result["access_token"],
                expires_in=result.get("expires_in", 3600)
            )
            return result["access_token"]

    @classmethod
    async def check_upload_status(cls, session: ClientSession, access_token: str, image_id: str, max_attempts: int = 30):
        headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 OPR/89.0.0.0",
        }
        url = f"{cls.check_status_endpoint}{image_id}"
        for _ in range(max_attempts):
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                if result.get("status") == "ok":
                    return
                if result.get("status") == "failed":
                    raise Exception(f"Image upload failed for {image_id}")
                await asyncio.sleep(0.5)
        raise Exception(f"Timeout waiting for image upload status for {image_id}")

    @classmethod
    async def upload_media(cls, session: ClientSession, access_token: str, media_data: bytes, filename: str) -> str:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Origin": "opera-aria://ui",
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 OPR/89.0.0.0",
        }
        form_data = FormData()
        if not filename:
            filename = str(int(time.time() * 1000))
        content_type = is_data_an_media(media_data, filename) or "application/octet-stream"
        form_data.add_field('image_file', media_data, filename=filename, content_type=content_type)
        
        async with session.post(cls.upload_endpoint, headers=headers, data=form_data) as response:
            response.raise_for_status()
            result = await response.json()
            image_id = result.get("image_id")
            if not image_id:
                raise Exception("No image_id returned from upload")
            await cls.check_upload_status(session, access_token, image_id)
            return image_id

    @classmethod
    def extract_image_urls(cls, text: str) -> list[str]:
        pattern = r'!\[\]\((https?://[^\)]+)\)'
        urls = re.findall(pattern, text)
        return [url.replace(r'\/', '/') for url in urls]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        refresh_token: str = None,
        conversation: Conversation = None,
        return_conversation: bool = False,
        stream: bool = True,
        media: MediaListType = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        if conversation is None:
            conversation = Conversation(refresh_token)
        elif refresh_token and not conversation.refresh_token:
            conversation.refresh_token = refresh_token
        
        async with ClientSession() as session:
            access_token = await cls.get_access_token(session, conversation)
            
            media_attachments = []
            merged_media = list(merge_media(media, messages))
            if merged_media:
                for media_data, media_name in merged_media:
                    try:
                        if isinstance(media_data, str) and media_data.startswith("data:"):
                            data_part = media_data.split(",", 1)[1]
                            media_bytes = base64.b64decode(data_part)
                        elif hasattr(media_data, 'read'):
                            media_bytes = media_data.read()
                        elif isinstance(media_data, (str, os.PathLike)):
                            with open(media_data, 'rb') as f:
                                media_bytes = f.read()
                        else:
                            media_bytes = media_data
                        
                        image_id = await cls.upload_media(session, access_token, media_bytes, media_name)
                        media_attachments.append(image_id)
                    except Exception:
                        continue
            
            headers = {
                "Accept": "text/event-stream" if stream else "application/json",
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Origin": "opera-aria://ui",
                "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 OPR/89.0.0.0",
                "X-Opera-Timezone": "+03:00",
                "X-Opera-UI-Language": "en"
            }
            
            data = {
                "query": format_prompt(messages), "stream": stream, "linkify": True,
                "linkify_version": 3, "sia": True, "media_attachments": media_attachments,
                "encryption": {"key": conversation.encryption_key}
            }
            
            if not conversation.is_first_request and conversation.conversation_id:
                data["conversation_id"] = conversation.conversation_id
            
            async with session.post(cls.api_endpoint, headers=headers, json=data, proxy=proxy) as response:
                response.raise_for_status()
                
                if stream:
                    text_buffer, image_urls, finish_reason = [], [], None
                    
                    async for line in response.content:
                        if not line: continue
                        decoded = line.decode('utf-8').strip()
                        if not decoded.startswith('data: '): continue
                        
                        content = decoded[6:]
                        if content == '[DONE]': break
                        
                        try:
                            json_data = json.loads(content)
                            if 'message' in json_data:
                                message_chunk = json_data['message']
                                found_urls = cls.extract_image_urls(message_chunk)
                                if found_urls:
                                    image_urls.extend(found_urls)
                                else:
                                    text_buffer.append(message_chunk)
                            
                            if 'conversation_id' in json_data and json_data['conversation_id']:
                                conversation.conversation_id = json_data['conversation_id']
                            
                            if 'finish_reason' in json_data and json_data.get('finish_reason'):
                                finish_reason = json_data['finish_reason']

                        except json.JSONDecodeError:
                            continue
                    
                    if image_urls:
                        yield ImageResponse(image_urls, format_prompt(messages))
                    elif text_buffer:
                        yield "".join(text_buffer)
                    
                    if finish_reason:
                        yield FinishReason(finish_reason)

                else: # Non-streaming
                    json_data = await response.json()
                    if 'message' in json_data:
                        message = json_data['message']
                        image_urls = cls.extract_image_urls(message)
                        if image_urls:
                            yield ImageResponse(image_urls, format_prompt(messages))
                        else:
                            yield message
                    
                    if 'conversation_id' in json_data and json_data['conversation_id']:
                        conversation.conversation_id = json_data['conversation_id']
                    
                    if 'finish_reason' in json_data and json_data['finish_reason']:
                        yield FinishReason(json_data['finish_reason'])
                
                conversation.is_first_request = False
                
                if return_conversation:
                    yield conversation
