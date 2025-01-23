import json
import random
import re
import requests
from aiohttp import ClientSession
from typing import List

from ...typing import AsyncResult, Messages
from ...image import ImageResponse
from ...providers.response import FinishReason, Usage
from ...requests.raise_for_status import raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin

from ... import debug
 
def split_message(message: str, max_length: int = 1000) -> List[str]:
    """Splits the message into parts up to (max_length)."""
    chunks = []
    while len(message) > max_length:
        split_point = message.rfind(' ', 0, max_length)
        if split_point == -1:
            split_point = max_length
        chunks.append(message[:split_point])
        message = message[split_point:].strip()
    if message:
        chunks.append(message)
    return chunks

class Airforce(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://api.airforce"
    api_endpoint_completions = "https://api.airforce/chat/completions"
    api_endpoint_imagine2 = "https://api.airforce/imagine2"

    working = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "llama-3.1-70b-chat"
    default_image_model = "flux"
    
    models = []
    image_models = []
    
    hidden_models = {"Flux-1.1-Pro"}
    additional_models_imagine = ["flux-1.1-pro", "midjourney", "dall-e-3"]
    model_aliases = {
        # Alias mappings for models
        "openchat-3.5": "openchat-3.5-0106",
        "deepseek-coder": "deepseek-coder-6.7b-instruct",
        "hermes-2-dpo": "Nous-Hermes-2-Mixtral-8x7B-DPO",
        "hermes-2-pro": "hermes-2-pro-mistral-7b",
        "openhermes-2.5": "openhermes-2.5-mistral-7b",
        "lfm-40b": "lfm-40b-moe",
        "german-7b": "discolm-german-7b-v1",
        "llama-2-7b": "llama-2-7b-chat-int8",
        "llama-3.1-70b": "llama-3.1-70b-chat",
        "llama-3.1-8b": "llama-3.1-8b-chat",
        "llama-3.1-70b": "llama-3.1-70b-turbo",
        "llama-3.1-8b": "llama-3.1-8b-turbo",
        "neural-7b": "neural-chat-7b-v3-1",
        "zephyr-7b": "zephyr-7b-beta",
        "evil": "any-uncensored",
        "sdxl": "stable-diffusion-xl-lightning",
        "sdxl": "stable-diffusion-xl-base",
        "flux-pro": "flux-1.1-pro",
        "llama-3.1-8b": "llama-3.1-8b-chat"
    }

    @classmethod
    def get_models(cls):
        """Get available models with error handling"""
        if not cls.image_models:
            try:
                response = requests.get(
                    f"{cls.url}/imagine2/models",
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    }
                )
                response.raise_for_status()
                cls.image_models = response.json()
                if isinstance(cls.image_models, list):
                    cls.image_models.extend(cls.additional_models_imagine)
                else:
                    cls.image_models = cls.additional_models_imagine.copy()
            except Exception as e:
                debug.log(f"Error fetching image models: {e}")
                cls.image_models = cls.additional_models_imagine.copy()

        if not cls.models:
            try:
                response = requests.get(
                    f"{cls.url}/models",
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    }
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and 'data' in data:
                    cls.models = [model['id'] for model in data['data']]
                    cls.models.extend(cls.image_models)
                    cls.models = [model for model in cls.models if model not in cls.hidden_models]
                else:
                    cls.models = list(cls.model_aliases.keys())
            except Exception as e:
                debug.log(f"Error fetching text models: {e}")
                cls.models = list(cls.model_aliases.keys())

        return cls.models or list(cls.model_aliases.keys())

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the actual model name from alias"""
        return cls.model_aliases.get(model, model or cls.default_model)

    @classmethod
    def _filter_content(cls, part_response: str) -> str:
        """
        Filters out unwanted content from the partial response.
        """
        part_response = re.sub(
            r"One message exceeds the \d+chars per message limit\..+https:\/\/discord\.com\/invite\/\S+",
            '',
            part_response
        )

        part_response = re.sub(
            r"Rate limit \(\d+\/minute\) exceeded\. Join our discord for more: .+https:\/\/discord\.com\/invite\/\S+",
            '',
            part_response
        )

        return part_response

    @classmethod
    def _filter_response(cls, response: str) -> str:
        """
        Filters the full response to remove system errors and other unwanted text.
        """
        if "Model not found or too long input. Or any other error (xD)" in response:
            raise ValueError(response)

        filtered_response = re.sub(r"\[ERROR\] '\w{8}-\w{4}-\w{4}-\w{4}-\w{12}'", '', response)  # any-uncensored
        filtered_response = re.sub(r'<\|im_end\|>', '', filtered_response)  # remove <|im_end|> token
        filtered_response = re.sub(r'</s>', '', filtered_response)  # neural-chat-7b-v3-1  
        filtered_response = re.sub(r'^(Assistant: |AI: |ANSWER: |Output: )', '', filtered_response)  # phi-2
        filtered_response = cls._filter_content(filtered_response)
        return filtered_response

    @classmethod
    async def generate_image(
        cls,
        model: str,
        prompt: str,
        size: str,
        seed: int,
        proxy: str = None
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Accept": "image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "application/json",
        }
        params = {"model": model, "prompt": prompt, "size": size, "seed": seed}

        async with ClientSession(headers=headers) as session:
            async with session.get(cls.api_endpoint_imagine2, params=params, proxy=proxy) as response:
                if response.status == 200:
                    image_url = str(response.url)
                    yield ImageResponse(images=image_url, alt=prompt)
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Image generation failed: {response.status} - {error_text}")

    @classmethod
    async def generate_text(
        cls,
        model: str,
        messages: Messages,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        proxy: str = None
    ) -> AsyncResult:
        """
        Generates text, buffers the response, filters it, and returns the final result.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Accept": "application/json, text/event-stream",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "application/json",
        }

        final_messages = []
        for message in messages:
            message_chunks = split_message(message["content"], max_length=1000)
            final_messages.extend([{"role": message["role"], "content": chunk} for chunk in message_chunks])
        data = {
            "messages": final_messages,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if max_tokens != 512:
            data["max_tokens"] = max_tokens

        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint_completions, json=data, proxy=proxy) as response:
                await raise_for_status(response)

                if stream:
                    idx = 0
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                json_str = line[6:]  # Remove 'data: ' prefix
                                chunk = json.loads(json_str)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        chunk = cls._filter_response(delta['content'])
                                        if chunk:
                                            yield chunk
                                            idx += 1
                            except json.JSONDecodeError:
                                continue
                    if idx == 512:
                        yield FinishReason("length")
                else:
                    # Non-streaming response
                    result = await response.json()
                    if "usage" in result:
                        yield Usage(**result["usage"])
                        if result["usage"]["completion_tokens"] == 512:
                            yield FinishReason("length")
                    if 'choices' in result and result['choices']:
                        message = result['choices'][0].get('message', {})
                        content = message.get('content', '')
                        filtered_response = cls._filter_response(content)
                        yield filtered_response

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        max_tokens: int = 512,
        temperature: float = 1,
        top_p: float = 1,
        stream: bool = True,
        size: str = "1:1",
        seed: int = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        if model in cls.image_models:
            if prompt is None:
                prompt = messages[-1]['content']
            if seed is None:
                seed = random.randint(0, 10000)
            async for result in cls.generate_image(model, prompt, size, seed, proxy):
                yield result
        else:
            async for result in cls.generate_text(model, messages, max_tokens, temperature, top_p, stream, proxy):
                yield result
