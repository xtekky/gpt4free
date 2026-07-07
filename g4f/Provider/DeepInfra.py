from __future__ import annotations

import time
import requests

from ..typing import Messages, AsyncResult
from .template import OpenaiTemplate

def get_turnstile_token() -> str:
    """
    Opens the DeepInfra page using DrissionPage to obtain a Turnstile token.
    Raises MissingRequirementsError if DrissionPage is not installed.
    """
    try:
        from DrissionPage import ChromiumPage, ChromiumOptions
    except ImportError:
        from ..errors import MissingRequirementsError
        raise MissingRequirementsError('Install "DrissionPage" package to use DeepInfra without an API key | pip install DrissionPage')

    co = ChromiumOptions()
    # Hide the window off-screen
    co.set_argument('--window-position=-2000,-2000')
    co.set_argument('--window-size=800,600')
    co.set_argument('--log-level=3')
    
    page = ChromiumPage(co)
    
    try:
        # Generate a token on any model's page; it is valid for the entire domain
        page.get('https://deepinfra.com/zai-org/GLM-5.2')
        
        # Inject JS to block the original request
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
        
        # Initiate Turnstile
        textarea = page.ele('tag:textarea', timeout=15)
        if not textarea:
            return ""
            
        textarea.input('Test')
        textarea.input('\n')
        
        # Wait for the challenge to be solved
        token_input = page.ele('@name=cf-turnstile-response', timeout=20)
        
        if not token_input:
            return ""
            
        token = ""
        for _ in range(40):
            token = token_input.attr('value')
            if token:
                break
            time.sleep(0.5)
            
        return token
    finally:
        try:
            page.quit()
        except:
            pass

class DeepInfra(OpenaiTemplate):
    url = "https://deepinfra.com"
    login_url = "https://deepinfra.com/dash/api_keys"
    base_url = "https://api.deepinfra.com/v1/openai"
    
    working = True
    active_by_default = True
    
    default_model = "MiniMaxAI/MiniMax-M2.5"

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
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        headers = super().get_headers(stream, api_key, headers)
        if not api_key:
            headers["X-Deepinfra-Source"] = "model-embed"
            headers["Origin"] = "https://deepinfra.com"
            headers["Referer"] = "https://deepinfra.com/"
            
            # Generate a Turnstile token for each request (required without an API key)
            token = get_turnstile_token()
            if token:
                headers["X-DeepInfra-Turnstile"] = token
                
        return headers
