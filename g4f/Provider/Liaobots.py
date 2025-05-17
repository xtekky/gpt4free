from __future__ import annotations

import uuid
import json
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector
from ..requests import raise_for_status
from ..errors import RateLimitError

models = {
    "claude-3-5-sonnet-20241022": {
        "id": "claude-3-5-sonnet-20241022",
        "name": "claude-3-5-sonnet-20241022",
        "model": "claude-3-5-sonnet-20241022",
        "provider": "Anthropic",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 25.366666666666667,
    },
    "claude-3-5-sonnet-20241022-t": {
        "id": "claude-3-5-sonnet-20241022-t",
        "name": "claude-3-5-sonnet-20241022-t",
        "model": "claude-3-5-sonnet-20241022-t",
        "provider": "Anthropic",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 39.820754716981135,
    },
    "claude-3-7-sonnet-20250219": {
        "id": "claude-3-7-sonnet-20250219",
        "name": "claude-3-7-sonnet-20250219",
        "model": "claude-3-7-sonnet-20250219",
        "provider": "Anthropic",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 47.02970297029703,
    },
    "claude-3-7-sonnet-20250219-t": {
        "id": "claude-3-7-sonnet-20250219-t",
        "name": "claude-3-7-sonnet-20250219-t",
        "model": "claude-3-7-sonnet-20250219-t",
        "provider": "Anthropic",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 39.04289693593315,
    },
    "deepseek-v3": {
        "id": "deepseek-v3",
        "name": "deepseek-v3",
        "model": "deepseek-v3",
        "provider": "DeepSeek",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 40.484657419083646,
    },
    "gemini-1.0-pro-latest-123": {
        "id": "gemini-1.0-pro-latest-123",
        "name": "gemini-1.0-pro-latest-123",
        "model": "gemini-1.0-pro-latest-123",
        "provider": "Google",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 10,
    },
    "gemini-2.0-flash": {
        "id": "gemini-2.0-flash",
        "name": "gemini-2.0-flash",
        "model": "gemini-2.0-flash",
        "provider": "Google",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 216.44162436548223,
    },
    "gemini-2.0-flash-exp": {
        "id": "gemini-2.0-flash-exp",
        "name": "gemini-2.0-flash-exp",
        "model": "gemini-2.0-flash-exp",
        "provider": "Google",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 0,
        "tps": 0,
    },
    "gemini-2.0-flash-thinking-exp": {
        "id": "gemini-2.0-flash-thinking-exp",
        "name": "gemini-2.0-flash-thinking-exp",
        "model": "gemini-2.0-flash-thinking-exp",
        "provider": "Google",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 0,
        "tps": 0,
    },
    "gemini-2.5-flash-preview-04-17": {
        "id": "gemini-2.5-flash-preview-04-17",
        "name": "gemini-2.5-flash-preview-04-17",
        "model": "gemini-2.5-flash-preview-04-17",
        "provider": "Google",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 189.84010840108402,
    },
    "gemini-2.5-pro-official": {
        "id": "gemini-2.5-pro-official",
        "name": "gemini-2.5-pro-official",
        "model": "gemini-2.5-pro-official",
        "provider": "Google",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 91.00613496932516,
    },
    "gemini-2.5-pro-preview-03-25": {
        "id": "gemini-2.5-pro-preview-03-25",
        "name": "gemini-2.5-pro-preview-03-25",
        "model": "gemini-2.5-pro-preview-03-25",
        "provider": "Google",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 99.05660377358491,
        "tps": 45.050511247443765,
    },
    "gemini-2.5-pro-preview-05-06": {
        "id": "gemini-2.5-pro-preview-05-06",
        "name": "gemini-2.5-pro-preview-05-06",
        "model": "gemini-2.5-pro-preview-05-06",
        "provider": "Google",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 99.29617834394904,
    },
    "gpt-4-turbo-2024-04-09": {
        "id": "gpt-4-turbo-2024-04-09",
        "name": "gpt-4-turbo-2024-04-09",
        "model": "gpt-4-turbo-2024-04-09",
        "provider": "OpenAI",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 1,
    },
    "gpt-4.1": {
        "id": "gpt-4.1",
        "name": "gpt-4.1",
        "model": "gpt-4.1",
        "provider": "OpenAI",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 42.857142857142854,
        "tps": 19.58032786885246,
    },
    "gpt-4.1-mini": {
        "id": "gpt-4.1-mini",
        "name": "gpt-4.1-mini",
        "model": "gpt-4.1-mini",
        "provider": "OpenAI",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 68.75,
        "tps": 12.677576601671309,
    },
    "gpt-4.1-mini-2025-04-14": {
        "id": "gpt-4.1-mini-2025-04-14",
        "name": "gpt-4.1-mini-2025-04-14",
        "model": "gpt-4.1-mini-2025-04-14",
        "provider": "OpenAI",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 94.23076923076923,
        "tps": 8.297687861271676,
    },
    "gpt-4o-2024-11-20": {
        "id": "gpt-4o-2024-11-20",
        "name": "gpt-4o-2024-11-20",
        "model": "gpt-4o-2024-11-20",
        "provider": "OpenAI",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 73.3955223880597,
    },
    "gpt-4o-mini-2024-07-18": {
        "id": "gpt-4o-mini-2024-07-18",
        "name": "gpt-4o-mini-2024-07-18",
        "model": "gpt-4o-mini-2024-07-18",
        "provider": "OpenAI",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 26.874455100261553,
    },
    "grok-3": {
        "id": "grok-3",
        "name": "grok-3",
        "model": "grok-3",
        "provider": "xAI",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 51.110652663165794,
    },
    "grok-3-reason": {
        "id": "grok-3-reason",
        "name": "grok-3-reason",
        "model": "grok-3-reason",
        "provider": "xAI",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 62.81976744186046,
    },
    "o3-mini-2025-01-31": {
        "id": "o3-mini-2025-01-31",
        "name": "o3-mini-2025-01-31",
        "model": "o3-mini-2025-01-31",
        "provider": "Unknown",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 125.31410256410257,
    },
    "qwen3-235b-a22b": {
        "id": "qwen3-235b-a22b",
        "name": "qwen3-235b-a22b",
        "model": "qwen3-235b-a22b",
        "provider": "Alibaba",
        "maxLength": 0,
        "tokenLimit": 0,
        "context": 0,
        "success_rate": 100,
        "tps": 25.846153846153847,
    },
}

class Liaobots(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://liaobots.work"
    working = True
    supports_message_history = True
    supports_system_message = True
    
    default_model = "grok-3"
    models = list(models.keys())
    model_aliases = {
        # Anthropic
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022-t",
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219-t",
        
        # DeepSeek
        #"deepseek-v3": "deepseek-v3",
        
        # Google
        "gemini-1.0-pro": "gemini-1.0-pro-latest-123",
        "gemini-2.0-flash": "gemini-2.0-flash-exp",
        "gemini-2.0-flash-thinking": "gemini-2.0-flash-thinking-exp",
        "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro": "gemini-2.5-pro-official",
        "gemini-2.5-pro": "gemini-2.5-pro-preview-03-25",
        "gemini-2.5-pro": "gemini-2.5-pro-preview-05-06",
        
        # OpenAI
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
        "gpt-4": "gpt-4o-2024-11-20",
        "gpt-4o": "gpt-4o-2024-11-20",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        
        # xAI
        "grok-3-reason": "grok-3-reason",
        "o3-mini": "o3-mini-2025-01-31",
        "qwen-3-235b": "qwen3-235b-a22b",
    }
    
    _auth_code = None
    _cookie_jar = None

    @classmethod
    def is_supported(cls, model: str) -> bool:
        """
        Check if the given model is supported.
        """
        return model in models or model in cls.model_aliases

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        connector: BaseConnector = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://liaobots.work",
            "priority": "u=1, i",
            "referer": "https://liaobots.work/en",
            "sec-ch-ua": "\"Chromium\";v=\"135\", \"Not-A.Brand\";v=\"8\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Linux\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        }
        
        async with ClientSession(
            headers=headers,
            cookie_jar=cls._cookie_jar,
            connector=get_connector(connector, proxy, True)
        ) as session:
            # First, get a valid auth code
            await cls.get_auth_code(session)
            
            # Create conversation ID
            conversation_id = str(uuid.uuid4())
            
            # Prepare request data
            data = {
                "conversationId": conversation_id,
                "models": [{
                    "modelId": model,
                    "provider": models[model]["provider"]
                }],
                "search": "false",
                "messages": messages,
                "key": "",
                "prompt": kwargs.get("system_message", "你是 {{model}}，一个由 {{provider}} 训练的大型语言模型，请仔细遵循用户的指示。")
            }
            
            # Try to make the chat request
            try:
                # Make the chat request with the current auth code
                async with session.post(
                    f"{cls.url}/api/chat",
                    json=data,
                    headers={"x-auth-code": cls._auth_code},
                    ssl=False
                ) as response:
                    # Check if we got a streaming response
                    content_type = response.headers.get("Content-Type", "")
                    if "text/event-stream" in content_type:
                        async for line in response.content:
                            if line.startswith(b"data: "):
                                try:
                                    response_data = json.loads(line[6:])
                                    
                                    # Check for error response
                                    if response_data.get("error") is True:
                                        # Raise RateLimitError for payment required or other errors
                                        if "402" in str(response_data.get("res_status", "")):
                                            raise RateLimitError("This model requires payment or credits")
                                        else:
                                            error_msg = response_data.get('message', 'Unknown error')
                                            raise RateLimitError(f"Error: {error_msg}")
                                    
                                    # Process normal response
                                    if response_data.get("role") == "assistant" and "content" in response_data:
                                        content = response_data.get("content")
                                        yield content
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Not a streaming response, might be an error or HTML
                        response_text = await response.text()
                        
                        # If we got HTML, we need to bypass CAPTCHA
                        if response_text.startswith("<!DOCTYPE html>"):
                            await cls.bypass_captcha(session)
                            
                            # Get a fresh auth code
                            await cls.get_auth_code(session)
                            
                            # Try the request again
                            async with session.post(
                                f"{cls.url}/api/chat",
                                json=data,
                                headers={"x-auth-code": cls._auth_code},
                                ssl=False
                            ) as response2:
                                # Check if we got a streaming response
                                content_type = response2.headers.get("Content-Type", "")
                                if "text/event-stream" in content_type:
                                    async for line in response2.content:
                                        if line.startswith(b"data: "):
                                            try:
                                                response_data = json.loads(line[6:])
                                                
                                                # Check for error response
                                                if response_data.get("error") is True:
                                                    # Raise RateLimitError for payment required or other errors
                                                    if "402" in str(response_data.get("res_status", "")):
                                                        raise RateLimitError("This model requires payment or credits")
                                                    else:
                                                        error_msg = response_data.get('message', 'Unknown error')
                                                        raise RateLimitError(f"Error: {error_msg}")
                                                
                                                # Process normal response
                                                if response_data.get("role") == "assistant" and "content" in response_data:
                                                    content = response_data.get("content")
                                                    yield content
                                            except json.JSONDecodeError:
                                                continue
                                else:
                                    raise RateLimitError("Failed to get streaming response")
                        else:
                            raise RateLimitError("Failed to connect to the service")
            except Exception as e:
                # If it's already a RateLimitError, re-raise it
                if isinstance(e, RateLimitError):
                    raise
                # Otherwise, wrap it in a RateLimitError
                raise RateLimitError(f"Error processing request: {str(e)}")

    @classmethod
    async def bypass_captcha(cls, session: ClientSession) -> None:
        """
        Bypass the CAPTCHA verification by directly making the recaptcha API request.
        """
        try:
            # First, try the direct recaptcha API request
            async with session.post(
                f"{cls.url}/recaptcha/api/login",
                json={"token": "abcdefghijklmnopqrst"},
                ssl=False
            ) as response:
                if response.status == 200:
                    try:
                        response_text = await response.text()
                        
                        # Try to parse as JSON
                        try:
                            response_data = json.loads(response_text)
                            
                            # Check if we got a successful response
                            if response_data.get("code") == 200:
                                cls._cookie_jar = session.cookie_jar
                        except json.JSONDecodeError:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass

    @classmethod
    async def get_auth_code(cls, session: ClientSession) -> None:
        """
        Get a valid auth code by sending a request with an empty authcode.
        """
        try:
            # Send request with empty authcode to get a new one
            auth_request_data = {
                "authcode": "",
                "recommendUrl": "https://liaobots.work/zh"
            }
            
            async with session.post(
                f"{cls.url}/api/user",
                json=auth_request_data,
                ssl=False
            ) as response:
                if response.status == 200:
                    response_text = await response.text()
                    
                    try:
                        response_data = json.loads(response_text)
                        
                        if "authCode" in response_data:
                            cls._auth_code = response_data["authCode"]
                            cls._cookie_jar = session.cookie_jar
                            return
                    except json.JSONDecodeError:
                        # If we got HTML, it might be the CAPTCHA page
                        if response_text.startswith("<!DOCTYPE html>"):
                            await cls.bypass_captcha(session)
                            
                            # Try again after bypassing CAPTCHA
                            async with session.post(
                                f"{cls.url}/api/user",
                                json=auth_request_data,
                                ssl=False
                            ) as response2:
                                if response2.status == 200:
                                    response_text2 = await response2.text()
                                    
                                    try:
                                        response_data2 = json.loads(response_text2)
                                        
                                        if "authCode" in response_data2:
                                            cls._auth_code = response_data2["authCode"]
                                            cls._cookie_jar = session.cookie_jar
                                            return
                                    except json.JSONDecodeError:
                                        pass
        except Exception:
            pass
            
        # If we're here, we couldn't get a valid auth code
        # Set a default one as a fallback
        cls._auth_code = "DvS3A5GTE9f0D"  # Fallback to one of the provided auth codes
