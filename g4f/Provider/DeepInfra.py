from __future__ import annotations

import asyncio
import requests

from ..requests.cdp import SyncCDPSession
from .. import debug
from .template import OpenaiTemplate

def find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def _get_turnstile_token_sync(model: str) -> str:
    """
    Synchronous Turnstile token retrieval using SyncCDPSession with retries.
    Uses a blocking recv() loop — no async timeouts, waits as long as needed.
    Designed to be run via asyncio.run_in_executor() from async context.
    """
    import time
    
    for attempt in range(3):
        port = find_free_port()
        session = SyncCDPSession(port=port, headless=False)
        session.start_chrome()
        
        try:
            url = f"https://deepinfra.com/{model}"
            debug.log(f"[DeepInfra] Navigating to {url} (Attempt {attempt + 1}/3)...")
            session.navigate(url)

            # Inject completions request blocker
            fetch_blocker_js = """
            const origFetch = window.fetch;
            window.fetch = async function(...args) {
                let url = args[0];
                if (typeof url === 'string' && url.includes('/chat/completions')) {
                    return new Response('{}', {status: 200});
                }
                return origFetch.apply(this, args);
            };
            """
            session.evaluate_js(fetch_blocker_js)

            # Try to click "Accept" on cookies consent popup if present
            session.evaluate_js("""
            (() => {
                const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Accept');
                if (btn) btn.click();
            })()
            """)

            # Click on an empty page area to give window focus — signals Cloudflare that
            # a real user is present, which speeds up Turnstile token generation significantly.
            session.click(200, 400)

            # Wait for textarea readiness, then focus and input text
            debug.log("[DeepInfra] Waiting for active textarea...")
            text_entered = False
            for _ in range(80):  # Up to 40 seconds
                try:
                    ready = session.evaluate_js("""
                    (() => {
                        const ta = document.querySelector('textarea');
                        const ts = document.querySelector('[name=cf-turnstile-response]');
                        if (!ta) return 'no_textarea';
                        if (ta.disabled) return 'disabled';
                        if (!ts) return 'no_turnstile';
                        ta.click();
                        ta.focus();
                        ta.scrollIntoView({ block: 'center' });
                        return 'ready';
                    })()
                    """)

                    if ready == 'ready':
                        debug.log("[DeepInfra] Textarea and Turnstile found, focusing and entering text...")

                        # Retrieve textarea nodeId for native focusing
                        doc = session.call('DOM.getDocument')
                        root_id = doc['root']['nodeId']
                        textarea = session.call('DOM.querySelector', nodeId=root_id, selector='textarea')

                        # Native focus via CDP
                        session.call('DOM.focus', nodeId=textarea['nodeId'])

                        # Enter text via native CDP command
                        import random
                        test_prompt = random.choice(["Hello", "Hi", "Hey there", "Testing", "Ping", "What's up?", "Can you hear me?"])
                        session.call("Input.insertText", text=test_prompt)

                        time.sleep(0.5)

                        # Simulate Enter keypress
                        session.call("Input.dispatchKeyEvent",
                                     type="keyDown",
                                     windowsVirtualKeyCode=13,
                                     key="Enter", code="Enter",
                                     text="\r", unmodifiedText="\r")
                        session.call("Input.dispatchKeyEvent",
                                     type="keyUp",
                                     windowsVirtualKeyCode=13,
                                     key="Enter", code="Enter",
                                     text="\r", unmodifiedText="\r")

                        text_entered = True
                        break
                except Exception:
                    pass
                time.sleep(0.5)

            if not text_entered:
                debug.log("[DeepInfra] Textarea/Turnstile not ready or failed to submit, retrying attempt...")
                session.close()
                continue

            # Poll page for Turnstile token
            debug.log("[DeepInfra] Waiting for Cloudflare Turnstile solve...")
            token_js = "document.querySelector('[name=cf-turnstile-response]') ? document.querySelector('[name=cf-turnstile-response]').value : ''"
            token = ""
            for i in range(240):  # Up to 120 seconds per attempt
                try:
                    token = session.evaluate_js(token_js)
                    if token:
                        debug.log(f"[DeepInfra] Token generated on check {i+1}!")
                        return token
                except Exception:
                    pass
                time.sleep(0.5)

        except Exception as e:
            debug.log(f"[DeepInfra] Error on attempt {attempt + 1}: {e}")
        finally:
            session.close()

    return ""

async def get_turnstile_token_async(model: str = None) -> str:
    """Run the synchronous Turnstile solver in a thread pool executor."""
    if model is None:
        model = DeepInfra.default_model
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _get_turnstile_token_sync, model)

class DeepInfra(OpenaiTemplate):
    url = "https://deepinfra.com"
    login_url = "https://deepinfra.com/dash/api_keys"
    base_url = "https://api.deepinfra.com/v1/openai"
    
    working = True
    active_by_default = True
    
    default_model = "zai-org/GLM-5.2"

    @classmethod
    async def get_quota(cls, **kwargs):
        return {}

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.models:
            url = 'https://api.deepinfra.com/models/featured'
            response = requests.get(url, timeout=kwargs.get("timeout", 15))
            models = response.json()
            
            cls.models = {model["model_name"]: {"id": model["model_name"], **model} for model in models if model.get("type") == "text-generation" or model.get("reported_type") == "text-to-image"}
            cls.image_models = [model["model_name"] for model in models if model.get("reported_type") == "text-to-image"]
            if cls.live == 0 and cls.models:
                cls.live += 1

        return cls.models
    
    @classmethod
    async def create_async_generator(cls, model, messages, api_key=None, headers=None, **kwargs):
        if not api_key:
            # Generate a Turnstile token for each request (required without an API key)
            token = await get_turnstile_token_async(model)
            if token:
                if headers is None:
                    headers = {}
                headers["X-DeepInfra-Turnstile"] = token
            else:
                raise ValueError("Failed to obtain Turnstile token for DeepInfra request.")

        async for chunk in super().create_async_generator(model, messages, api_key=api_key, headers=headers, **kwargs):
            yield chunk

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        headers = super().get_headers(stream, api_key, headers)
        if not api_key:
            headers["X-Deepinfra-Source"] = "web-page"
            headers["Origin"] = "https://deepinfra.com"
            headers["Referer"] = "https://deepinfra.com/"
        return headers
