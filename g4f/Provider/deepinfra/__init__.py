from __future__ import annotations

import asyncio
import requests
import time

from ..template import OpenaiTemplate
from ...requests import BrowserConfig
from .turnstile import Turnstile

def find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def get_turnstile_token_sync(model: str) -> str:
    """Get Turnstile token using Turnstile with a fallback to DrissionPage."""
    # 1. Try our lightweight CDP client first
    first_error = None
    try:
        if not BrowserConfig.port:
            import os
            import tempfile
            port = find_free_port()
            # Use a persistent temp directory to preserve browser session cookies and Turnstile reputation
            user_data_dir = os.path.join(tempfile.gettempdir(), "g4f_chrome_profile_light")
            client = Turnstile(port=port, user_data_dir=user_data_dir)
        else:
            client = Turnstile(port=BrowserConfig.port)
            client.connect()  # Ensure connection if using a specified port
        try:
            token = client.get_token(model)
            if token:
                return token
            print("[DeepInfra] Turnstile failed to obtain token. Trying fallback option (DrissionPage)...")
        except:
            raise
        finally:
            client.close()
    except Exception as e:
        first_error = e

    # 2. Fallback option: try original DrissionPage if installed
    try:
        from DrissionPage import ChromiumPage, ChromiumOptions
        co = ChromiumOptions()
        co.set_argument('--window-position=-2000,-2000')
        co.set_argument('--window-size=1024,768')
        co.set_argument('--log-level=3')
        page = ChromiumPage(co)
        try:
            page.get(f'https://deepinfra.com/{model}')
            
            # Block completions requests
            js_block_fetch = """
            const origFetch = window.fetch;
            window.fetch = async function(...args) {
                let url = args[0];
                if (typeof url === 'string' && url.includes('/chat/completions')) {
                    return new Response('{}', {status: 200});
                }
                return origFetch.apply(this, args);
            };
            """
            page.run_js(js_block_fetch)
            
            textarea = page.ele('tag:textarea', timeout=15)
            if textarea:
                textarea.input('Test Prompt')
                textarea.input('\n')
                
                token_input = page.ele('@name=cf-turnstile-response', timeout=20)
                if token_input:
                    for _ in range(40):
                        token = token_input.attr('value')
                        if token:
                            return token
                        time.sleep(0.5)
        finally:
            try:
                page.quit()
            except:
                pass
    except:
        raise first_error if first_error else Exception("Failed to obtain Turnstile token using both methods.")
    return ""

async def get_turnstile_token_async() -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_turnstile_token_sync, DeepInfra.default_model)

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
            response = requests.get(url)
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
            token = await get_turnstile_token_async()
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
            headers["X-Deepinfra-Source"] = "model-embed"
            headers["Origin"] = "https://deepinfra.com"
            headers["Referer"] = "https://deepinfra.com/"
        return headers
