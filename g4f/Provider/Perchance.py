from __future__ import annotations

import asyncio
import random
import json
from g4f.requests.cdp import CDPSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..requests import StreamSession

class Perchance(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://perchance.org/ai-chat"
    api_endpoint = "https://text-generation.perchance.org/api/generate"
    verify_url = "https://text-generation.perchance.org/embed?thread=0"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    default_model = "perchance"

    # Class-level cache for credentials to avoid restarting browser for every request
    _user_key: str | None = None
    _cookies: dict | None = None
    _headers: dict | None = None

    @classmethod
    async def _get_user_key(cls, proxy: str = None) -> tuple[str, dict, dict]:
        """Runs CDP to solve Turnstile and extract the userKey and session cookies."""
        session = CDPSession(headless=True)
        await session.start()
        try:
            await session.navigate(cls.verify_url)
            
            # Setup message capture in page context
            setup_js = """
            window.collectedMessages = [];
            window.addEventListener("message", (e) => {
                if (e.data && (e.data.type.startsWith("stream") || e.data.type === "verified")) {
                    window.collectedMessages.push(e.data);
                }
            });
            """
            await session.evaluate_js(setup_js)
            
            # Always trigger verifyUser to ensure Turnstile solving/checking is executed
            verify_js = """
            window.postMessage({type: "verifyUser"}, window.location.origin);
            """
            await session.evaluate_js(verify_js)
            
            # Poll for userKey-0
            user_key = None
            for _ in range(30):
                user_key = await session.evaluate_js("localStorage.getItem('userKey-0')")
                if user_key:
                    break
                await asyncio.sleep(1)
                
            if not user_key:
                raise RuntimeError("Failed to verify user/solve Turnstile on Perchance.")
                
            cookies = await session.get_cookies()
            user_agent = await session.get_user_agent()
            
            headers = {
                "user-agent": user_agent,
                "Accept": "*/*",
                "Origin": "https://text-generation.perchance.org",
                "Referer": "https://text-generation.perchance.org/embed",
            }
            
            return user_key, cookies, headers
        finally:
            await session.close()

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        # Check if we need to fetch credentials (cold start)
        if not cls._user_key or not cls._cookies or not cls._headers:
            cls._user_key, cls._cookies, cls._headers = await cls._get_user_key(proxy=proxy)

        # Approximate token count (chars divided by ~4)
        instruction = format_prompt(messages)
        instruction_token_count = max(1, len(instruction) // 4)
        
        payload = {
            "instruction": instruction,
            "startWith": "",
            "stopSequences": ["\n\n", "\nAnon:", "\nBot:"],
            "generatorName": "ai-chat",
            "startWithTokenCount": 0,
            "instructionTokenCount": instruction_token_count
        }

        try:
            async with StreamSession(
                headers=cls._headers,
                cookies=cls._cookies,
                proxies={"all": proxy} if proxy else None,
                impersonate="chrome"
            ) as session:
                req_id = f"aiTextCompletion{random.random()}"
                gen_url = f"{cls.api_endpoint}?userKey={cls._user_key}&thread=0&requestId={req_id}&__cacheBust={random.random()}"
                
                async with session.post(gen_url, json=payload, proxy=proxy) as response:
                    # If Cloudflare blocks or token has expired, raise auth_failed to trigger a refresh
                    if response.status in (401, 403):
                        raise RuntimeError("auth_failed")
                        
                    if response.status != 200:
                        text = await response.text()
                        if "invalid_key" in text or "failed_verification" in text:
                            raise RuntimeError("auth_failed")
                        raise RuntimeError(f"generate failed: {response.status} {text}")
                    
                    # Parse custom SSE format
                    async for chunk_bytes in response.iter_content():
                        if chunk_bytes:
                            chunk_str = chunk_bytes.decode(errors="ignore")
                            for line in chunk_str.split("\n"):
                                line = line.strip()
                                if line.startswith('t:'):
                                    try:
                                        # Parse the text chunk safely via JSON loads (handles escapes)
                                        text_val = json.loads(line[2:])
                                        if text_val:
                                            yield text_val
                                    except Exception:
                                        pass

        except RuntimeError as e:
            if str(e) == "auth_failed":
                # Clear cached credentials and retry once with fresh verification
                cls._user_key = cls._cookies = cls._headers = None
                async for chunk in cls.create_async_generator(model, messages, proxy, **kwargs):
                    yield chunk
            else:
                raise e
