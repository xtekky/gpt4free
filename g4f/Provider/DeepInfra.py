from __future__ import annotations

import asyncio
import requests

from ..requests import get_nodriver_session
from .template import OpenaiTemplate

async def get_turnstile_token_async() -> str:
    try:
        import zendriver as zd
    except ImportError:
        from ..errors import MissingRequirementsError
        raise MissingRequirementsError('Install "zendriver" package to use DeepInfra without an API key | pip install zendriver')

    async with get_nodriver_session() as session:
        # Generate a token on any model's page; it is valid for the entire domain
        tab = await session.get('https://deepinfra.com/' + DeepInfra.default_model)

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
        await tab.evaluate(js_block_fetch)

        # Initiate Turnstile
        textarea = await tab.find('textarea', timeout=15)
        if not textarea:
            return ""

        await textarea.send_keys('Test\n')

        # Wait for the challenge to be solved
        token = ""
        for _ in range(40):
            token = await tab.evaluate(
                "(document.querySelector('[name=cf-turnstile-response]') || {value: ''}).value"
            )
            if token:
                break
            await asyncio.sleep(0.5)

        return token

class DeepInfra(OpenaiTemplate):
    url = "https://deepinfra.com"
    login_url = "https://deepinfra.com/dash/api_keys"
    base_url = "https://api.deepinfra.com/v1/openai"
    
    working = True
    active_by_default = True
    
    default_model = "zai-org/GLM-5.2"

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
