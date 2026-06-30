from __future__ import annotations

import os
import json
import time
import asyncio
import uuid
from typing import Dict, Any, AsyncIterator

try:
    import zendriver as nodriver
except ImportError:
    pass

from ...typing import Messages, AsyncResult
from ...providers.response import JsonConversation, Reasoning, ImagePreview, ImageResponse, TitleGeneration, AuthResult, RequestLogin
from ...requests import StreamSession, get_nodriver, DEFAULT_HEADERS, merge_cookies
from ...requests.raise_for_status import raise_for_status
from ...errors import MissingAuthError
from ..base_provider import AsyncAuthedProvider, ProviderModelMixin
from ..helper import format_prompt, get_last_user_message

class Conversation(JsonConversation):
    def __init__(self,
        conversation_id: str
    ) -> None:
        self.conversation_id = conversation_id

class Grok(AsyncAuthedProvider, ProviderModelMixin):
    label = "Grok AI"
    url = "https://grok.com"
    cookie_domain = ".grok.com"
    assets_url = "https://assets.grok.com"
    conversation_url = "https://grok.com/rest/app-chat/conversations"

    needs_auth = True
    working = True

    # Updated to Grok 4 as default
    default_model = "grok-4"
    
    # Updated model list with latest Grok 4 and 3 models
    models = [
        # Grok 4 models
        "grok-4",
        "grok-4-heavy",
        "grok-4-reasoning",
        
        # Grok 3 models  
        "grok-3",
        "grok-3-reasoning",
        "grok-3-mini",
        "grok-3-mini-reasoning",
        
        # Legacy Grok 2 (still supported)
        "grok-2",
        "grok-2-image",
        
        # Latest aliases
        "grok-latest",
    ]
    
    model_aliases = {
        # Grok 3 aliases
        "grok-3-thinking": "grok-3-reasoning",
        "grok-3-r1": "grok-3-reasoning", 
        "grok-3-mini-thinking": "grok-3-mini-reasoning",
        
        # Latest alias
        "grok": "grok-latest",
    }

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        auth_result = AuthResult(headers=DEFAULT_HEADERS, impersonate="chrome")
        auth_result.headers["referer"] = cls.url + "/"
        browser, stop_browser = await get_nodriver(proxy=proxy)
        yield RequestLogin(cls.__name__, os.environ.get("G4F_LOGIN_URL") or "")
        try:
            page = await browser.get(cls.url)
            has_headers = False
            def on_request(event: nodriver.cdp.network.RequestWillBeSent, page=None):
                nonlocal has_headers
                if hasattr(event, "request") and event.request.url.startswith(cls.conversation_url + "/new"):
                    for key, value in event.request.headers.items():
                        auth_result.headers[key.lower()] = value
                    has_headers = True
            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            await page.reload()
            auth_result.headers["user-agent"] = await page.evaluate("window.navigator.userAgent", return_by_value=True)
            while True:
                if has_headers:
                    break
                input_element = None
                try:
                    input_element = await page.select("div.ProseMirror", 2)
                except Exception:
                    pass
                if not input_element:
                    try:
                        input_element = await page.select("textarea", 180)
                    except Exception:
                        pass
                if input_element:
                    try:
                        await input_element.click()
                        await input_element.send_keys("Hello")
                        await asyncio.sleep(0.5)
                        submit_btn = await page.select("button[type='submit']", 2)
                        if submit_btn:
                            await submit_btn.click()
                    except Exception:
                        pass
                await asyncio.sleep(1)
            auth_result.cookies = {}
            for c in await page.send(nodriver.cdp.network.get_cookies([cls.url])):
                auth_result.cookies[c.name] = c.value
            await page.close()
        finally:
            await stop_browser()
        yield auth_result

    @classmethod
    async def _prepare_payload(cls, model: str, message: str) -> Dict[str, Any]:
        # Map model names to modeId based on new Grok API
        mode_id = "auto" if "auto" in model else "fast"
        
        if "heavy" in model or "big-brain" in model:
            mode_id = "heavy"
        elif "expert" in model:
            mode_id = "expert"
        elif "reasoning" in model or "thinking" in model or "r1" in model:
            mode_id = "reasoning"
        elif "deepsearch" in model:
            mode_id = "deepsearch"
            
        return {
            "temporary": True,
            "message": message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "disableTextFollowUps": False,
            "responseMetadata": {},
            "disableMemory": False,
            "forceSideBySide": False,
            "isAsyncChat": False,
            "disableSelfHarmShortCircuit": False,
            "collectionIds": [],
            "disabledConnectorIds": [],
            "deviceEnvInfo": {
                "darkModeEnabled": True,
                "devicePixelRatio": 1.0,
                "screenWidth": 1920,
                "screenHeight": 1080,
                "viewportWidth": 1920,
                "viewportHeight": 1080
            },
            "modeId": mode_id,
            "linkQuery": False
        }

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        conversation: Conversation = None,
        **kwargs
    ) -> AsyncResult:
        conversation_id = None if conversation is None else conversation.conversation_id
        prompt = format_prompt(messages) if conversation_id is None else get_last_user_message(messages)
        
        async with StreamSession(
            **auth_result.get_dict()
        ) as session:
            payload = await cls._prepare_payload(model, prompt)
            
            # Add voice mode support flag (for future use)
            if kwargs.get("enable_voice", False):
                payload["enableVoiceMode"] = True
            
            if conversation_id is None:
                url = f"{cls.conversation_url}/new"
            else:
                url = f"{cls.conversation_url}/{conversation_id}/responses"
                
            async with session.post(url, json=payload, headers={"x-xai-request-id": str(uuid.uuid4())}) as response:
                if response.status == 403:
                    raise MissingAuthError("Invalid secrets")
                auth_result.cookies = merge_cookies(auth_result.cookies, response)
                await raise_for_status(response)
                
                thinking_duration = None
                deep_search_active = False
                
                async for line in response.iter_lines():
                    if line:
                        try:
                            json_data = json.loads(line)
                            result = json_data.get("result", {})
                            
                            if conversation_id is None:
                                conversation_id = result.get("conversation", {}).get("conversationId")
                            
                            response_data = result.get("response", {})
                            
                            # Handle DeepSearch status
                            deep_search = response_data.get("deepSearchStatus")
                            if deep_search:
                                if not deep_search_active:
                                    deep_search_active = True
                                    yield Reasoning(status="🔍 Deep searching...")
                                if deep_search.get("completed"):
                                    deep_search_active = False
                                    yield Reasoning(status="Deep search completed")
                            
                            # Handle image generation (Aurora for Grok 3+)
                            image = response_data.get("streamingImageGenerationResponse", None)
                            if image is not None:
                                image_url = image.get("imageUrl")
                                if image_url:
                                    yield ImagePreview(
                                        f'{cls.assets_url}/{image_url}',
                                        "",
                                        {"cookies": auth_result.cookies, "headers": auth_result.headers}
                                    )
                            
                            # Handle text tokens
                            token = response_data.get("token", result.get("token"))
                            is_thinking = response_data.get("isThinking", result.get("isThinking"))
                            
                            if token:
                                if is_thinking:
                                    if thinking_duration is None:
                                        thinking_duration = time.time()
                                        # Different status for different models
                                        if "grok-4" in model:
                                            status = "🧠 Grok 4 is processing..."
                                        elif "big-brain" in payload and payload["enableBigBrain"]:
                                            status = "🧠 Big Brain mode active..."
                                        else:
                                            status = "🤔 Is thinking..."
                                        yield Reasoning(status=status)
                                    yield Reasoning(token)
                                else:
                                    if thinking_duration is not None:
                                        thinking_duration = time.time() - thinking_duration
                                        status = f"Thought for {thinking_duration:.2f}s" if thinking_duration > 1 else ""
                                        thinking_duration = None
                                        yield Reasoning(status=status)
                                    yield token
                            
                            # Handle generated images
                            generated_images = response_data.get("modelResponse", {}).get("generatedImageUrls", None)
                            if generated_images:
                                yield ImageResponse(
                                    [f'{cls.assets_url}/{image}' for image in generated_images],
                                    "",
                                    {"cookies": auth_result.cookies, "headers": auth_result.headers}
                                )
                            
                            # Handle title generation
                            title = result.get("title", {}).get("newTitle", "")
                            if title:
                                yield TitleGeneration(title)
                                
                            # Handle tool usage information (Grok 4)
                            tool_usage = response_data.get("toolUsage")
                            if tool_usage:
                                tools_used = tool_usage.get("toolsUsed", [])
                                if tools_used:
                                    yield Reasoning(status=f"Used tools: {', '.join(tools_used)}")
                                    
                        except json.JSONDecodeError:
                            continue
                            
                # Return conversation ID for continuation
                if conversation_id is not None and kwargs.get("return_conversation", False):
                    yield Conversation(conversation_id)
