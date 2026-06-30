from __future__ import annotations

import json
import time
import os
import base64
import asyncio
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages, MediaListType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..providers.response import JsonConversation, FinishReason, ImageResponse
from ..image import is_data_an_media
from ..tools.media import merge_media


class Conversation(JsonConversation):
    """Manages session state for Opera Aria."""
    access_token: str = None
    refresh_token: str = None
    encryption_key: str = None
    expires_at: float = 0
    conversation_id: str = None
    is_first_request: bool = True
    
    def __init__(self, refresh_token: str = None):
        self.refresh_token = refresh_token
        self.encryption_key = base64.b64encode(os.urandom(32)).decode('utf-8')
        self.is_first_request = True
    
    def is_token_expired(self) -> bool:
        return time.time() >= self.expires_at
    
    def update_token(self, access_token: str, expires_in: int):
        self.access_token = access_token
        self.expires_at = time.time() + expires_in - 60


class OperaAria(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Opera Aria"
    url = "https://play.google.com/store/apps/details?id=com.opera.browser"
    
    # Endpoints
    api_endpoint_v1 = "https://composer.opera-api.com/api/v1/a-chat"
    api_endpoint_v2 = "https://composer.opera-api.com/api/v2/a-chat"
    upload_endpoint = "https://composer.opera-api.com/api/v2/files/upload"
    files_endpoint = "https://composer.opera-api.com/api/v2/files/"
    token_endpoint = "https://oauth2.opera-api.com/oauth2/v1/token/"
    signup_endpoint = "https://auth.opera.com/account/v2/external/anonymous/signup"
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    # Models
    default_model = 'aria'
    default_image_model = default_model
    default_vision_model = default_model
    
    models = [default_model, 'aria-legacy']
    image_models = [default_model]
    vision_models = [default_model]
    
    _model_to_version = {
        'aria': 'v2',
        'aria-legacy': 'v1',
    }
    
    _user_agent_v1 = "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 OPR/89.0.0.0"
    _user_agent_v2 = "Mozilla/5.0 (Linux; U; Android 14; Pixel 8 Pro Build/UQ1A.240205.004; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/138.0.7204.179 Mobile Safari/537.36 OPR/99.0.2254.81922"

    @classmethod
    def get_model(cls, model: str) -> str:
        if not model:
            return cls.default_model
        return cls.model_aliases.get(model, model if model in cls.models else cls.default_model)

    @classmethod
    def _get_api_version(cls, model: str) -> str:
        return cls._model_to_version.get(model, 'v2')

    # ==================== Authentication ====================
    
    @classmethod
    async def _generate_refresh_token(cls, session: ClientSession) -> str:
        """Generate refresh token via anonymous signup."""
        # Step 1: Get anonymous access token
        async with session.post(
            cls.token_endpoint,
            headers={
                "User-Agent": "okhttp/5.3.2",
                "Content-Type": "application/x-www-form-urlencoded",
                "x-requested-with": "XMLHttpRequest",
                "x-opera-client-cache": "1"
            },
            data={
                "client_id": "mini-client",
                "client_secret": "Pcc5NvlCrxl02pMw32kO6WrnhpS0pUZ95YrDP8XNKJJQvFht4wQDkFJ7v9x5hn7C",
                "grant_type": "client_credentials",
                "scope": "anonymous_account"
            }
        ) as response:
            response.raise_for_status()
            anon_token = (await response.json())["access_token"]

        # Step 2: Anonymous signup
        async with session.post(
            cls.signup_endpoint,
            headers={
                "User-Agent": "okhttp/5.3.2",
                "Authorization": f"Bearer {anon_token}",
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "x-requested-with": "XMLHttpRequest",
                "x-opera-client-cache": "1"
            },
            json={"client_id": "mini"}
        ) as response:
            response.raise_for_status()
            auth_token = (await response.json())["token"]

        # Step 3: Exchange auth_token for refresh token
        async with session.post(
            cls.token_endpoint,
            headers={
                "User-Agent": "okhttp/5.3.2",
                "Content-Type": "application/x-www-form-urlencoded",
                "x-requested-with": "XMLHttpRequest",
                "x-opera-client-cache": "1"
            },
            data={
                "auth_token": auth_token,
                "client_id": "mini",
                "grant_type": "auth_token",
                "scope": "shodan:aria"
            }
        ) as response:
            response.raise_for_status()
            return (await response.json())["refresh_token"]

    @classmethod
    async def get_access_token(cls, session: ClientSession, conversation: Conversation) -> str:
        """Get valid access token."""
        if not conversation.refresh_token:
            conversation.refresh_token = await cls._generate_refresh_token(session)

        if conversation.access_token and not conversation.is_token_expired():
            return conversation.access_token
        
        data = {
            "client_id": "mini",
            "grant_type": "refresh_token",
            "refresh_token": conversation.refresh_token,
            "scope": "shodan:aria"
        }
        
        async with session.post(
            cls.token_endpoint,
            headers={
                "User-Agent": "okhttp/5.3.2",
                "Content-Type": "application/x-www-form-urlencoded",
                "x-requested-with": "XMLHttpRequest",
                "x-opera-client-cache": "1"
            },
            data=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            conversation.update_token(result["access_token"], result.get("expires_in", 3600))
            return result["access_token"]

    # ==================== V2 Media Upload ====================

    @classmethod
    def _detect_mimetype(cls, data: bytes, filename: str = None) -> str:
        """Detect mimetype from bytes."""
        mimetype = is_data_an_media(data, filename)
        if mimetype:
            return mimetype
        if data[:4] == b'\x89PNG':
            return "image/png"
        if data[:2] == b'\xff\xd8':
            return "image/jpeg"
        if data[:4] == b'RIFF' and len(data) > 12 and data[8:12] == b'WEBP':
            return "image/webp"
        if data[:3] == b'GIF':
            return "image/gif"
        return "application/octet-stream"

    @classmethod
    async def _upload_file(cls, session: ClientSession, access_token: str, media_bytes: bytes, filename: str = None) -> str:
        """Upload file via signed GCS URL (V2 only)."""
        mimetype = cls._detect_mimetype(media_bytes, filename)
        size = len(media_bytes)
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Origin": "https://composer.opera-api.com",
            "Referer": "https://composer.opera-api.com/assets/aria/index.html",
            "User-Agent": cls._user_agent_v2,
            "X-Opera-Timezone": "+02:00",
            "X-Opera-UI-Language": "en",
            "X-Requested-With": "com.opera.mini.native"
        }
        
        # Step 1: Get signed URL
        async with session.post(cls.upload_endpoint, headers=headers, json={"mimetype": mimetype, "size": size}) as response:
            response.raise_for_status()
            info = await response.json()
        
        file_id = info["file_id"]
        gcs_headers = info.get("headers", {})
        
        # Step 2: Upload to GCS
        async with session.put(
            info["upload_url"],
            headers={
                "Content-Type": gcs_headers.get("Content-Type", mimetype),
                "X-Goog-Content-Length-Range": gcs_headers.get("X-Goog-Content-Length-Range", f"0,{size}"),
                "Origin": "https://composer.opera-api.com",
                "User-Agent": cls._user_agent_v2
            },
            data=media_bytes
        ) as response:
            response.raise_for_status()
        
        # Step 3: Poll for completion
        for attempt in range(60):
            async with session.get(f"{cls.files_endpoint}{file_id}", headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                status = result.get("upload_status")
                if status == "finished":
                    return file_id
                if status in ("failed", "error"):
                    raise Exception(f"Upload failed: {result}")
            await asyncio.sleep(min(1 + attempt * 0.3, 2))
        
        raise Exception(f"Upload timeout for {file_id}")

    @classmethod
    async def _process_media(cls, session: ClientSession, access_token: str, media: MediaListType, messages: Messages) -> list:
        """Process and upload all media (V2 only)."""
        attachments = []
        for media_data, media_name in merge_media(media, messages):
            try:
                if isinstance(media_data, str) and media_data.startswith("data:"):
                    media_bytes = base64.b64decode(media_data.split(",", 1)[1])
                elif isinstance(media_data, str) and media_data.startswith(("http://", "https://")):
                    async with session.get(media_data) as response:
                        response.raise_for_status()
                        media_bytes = await response.read()
                elif hasattr(media_data, 'read'):
                    media_bytes = media_data.read()
                elif isinstance(media_data, (str, os.PathLike)):
                    with open(media_data, 'rb') as f:
                        media_bytes = f.read()
                elif isinstance(media_data, bytes):
                    media_bytes = media_data
                else:
                    continue
                
                file_id = await cls._upload_file(session, access_token, media_bytes, media_name)
                attachments.append(file_id)
            except Exception as e:
                import logging
                logging.warning(f"Failed to upload {media_name}: {e}")
        
        return attachments

    # ==================== Request Building ====================

    @classmethod
    def _build_headers(cls, access_token: str, version: str) -> dict:
        """Build request headers."""
        if version == 'v1':
            return {
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Origin": "opera-aria://ui",
                "User-Agent": cls._user_agent_v1,
                "X-Opera-Timezone": "+02:00",
                "X-Opera-UI-Language": "en"
            }
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Origin": "https://composer.opera-api.com",
            "Referer": "https://composer.opera-api.com/assets/aria/index.html",
            "User-Agent": cls._user_agent_v2,
            "X-Opera-Timezone": "+02:00",
            "X-Opera-UI-Language": "en",
            "X-Requested-With": "com.opera.mini.native"
        }

    @classmethod
    def _build_payload(cls, messages: Messages, conversation: Conversation, attachments: list, version: str, **kwargs) -> dict:
        """Build request payload."""
        query = format_prompt(messages)
        
        if version == 'v1':
            data = {
                "query": query,
                "stream": True,
                "linkify": True,
                "linkify_version": 3,
                "sia": True,
                "media_attachments": [],
                "encryption": {"key": conversation.encryption_key}
            }
        else:
            data = {
                "query": query,
                "sia": True,
                "think_harder": kwargs.get("think_harder", False),
                "supported_features": [],
                "file_attachments": attachments,
                "encryption": {"key": conversation.encryption_key}
            }
        
        if not conversation.is_first_request and conversation.conversation_id:
            data["conversation_id"] = conversation.conversation_id
        
        return data

    # ==================== SSE Parsing ====================

    @classmethod
    def _parse_sse(cls, line: str) -> tuple[str | None, dict | None]:
        """Parse SSE line into (event_type, data)."""
        line = line.strip()
        if line.startswith('event:'):
            return line[6:].strip(), None
        if line.startswith('data:'):
            content = line[5:].strip()
            if content in ('[DONE]', 'null', ''):
                return None, None
            try:
                return None, json.loads(content)
            except json.JSONDecodeError:
                return None, None
        return None, None

    @classmethod
    def _extract_content(cls, data: dict, version: str) -> tuple[str | None, str | None]:
        """
        Extract (text, image_url) from response.
        
        For image responses (content_type == 'image'):
          - Returns image_url, ignores accompanying text message
        For text responses (content_type == 'text'):
          - Returns text message only
        """
        if version == 'v1':
            msg = data.get('message')
            return (msg, None) if isinstance(msg, str) else (None, None)
        
        response = data.get('response', {})
        if not isinstance(response, dict):
            return None, None
        
        content_type = response.get('content_type')
        
        # Image response: extract URL only, skip text like
        # "I've put together the image; take a moment to see it!"
        if content_type == 'image':
            return None, response.get('image_url')
        
        # Text response: extract message only
        text = response.get('message') if isinstance(response.get('message'), str) else None
        return text, None

    @classmethod
    def _extract_conversation_id(cls, data: dict, version: str) -> str | None:
        """Extract conversation ID."""
        if version == 'v1':
            return data.get('conversation_id')
        metadata = data.get('metadata', {})
        return metadata.get('conversation_id') if isinstance(metadata, dict) else None

    # ==================== Main Generator ====================

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        refresh_token: str = None,
        conversation: Conversation = None,
        return_conversation: bool = False,
        media: MediaListType = None,
        prompt: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        version = cls._get_api_version(model)
        api_endpoint = cls.api_endpoint_v1 if version == 'v1' else cls.api_endpoint_v2
        
        if conversation is None:
            conversation = Conversation(refresh_token)
        elif refresh_token and not conversation.refresh_token:
            conversation.refresh_token = refresh_token
        
        async with ClientSession() as session:
            access_token = await cls.get_access_token(session, conversation)
            
            # Upload media (V2 only)
            attachments = []
            if version == 'v2' and (media or any(isinstance(m, dict) and m.get("media") for m in messages)):
                attachments = await cls._process_media(session, access_token, media, messages)
            
            headers = cls._build_headers(access_token, version)
            payload = cls._build_payload(messages, conversation, attachments, version, **kwargs)
            
            # Save original prompt for ImageResponse alt text
            original_prompt = prompt if prompt else (messages[-1]["content"] if messages else "")
            
            async with session.post(api_endpoint, headers=headers, json=payload, proxy=proxy) as response:
                response.raise_for_status()
                
                image_urls = []
                skip_next = False
                
                async for line in response.content:
                    decoded = line.decode('utf-8').strip()
                    if not decoded:
                        continue
                    
                    event_type, data = cls._parse_sse(decoded)
                    
                    if event_type == 'thinking_status':
                        skip_next = True
                        continue
                    
                    if data is None:
                        continue
                    
                    if skip_next:
                        skip_next = False
                        continue
                    
                    text, image_url = cls._extract_content(data, version)
                    
                    if image_url:
                        image_urls.append(image_url)
                    
                    if text:
                        yield text
                    
                    conv_id = cls._extract_conversation_id(data, version)
                    if conv_id:
                        conversation.conversation_id = conv_id
                
                # Yield generated images with original prompt as alt
                if image_urls:
                    yield ImageResponse(image_urls, original_prompt)
                
                conversation.is_first_request = False
                
                if return_conversation:
                    yield conversation
