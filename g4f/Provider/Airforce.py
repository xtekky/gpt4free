from __future__ import annotations
import random
import json
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse

def split_long_message(message: str, max_length: int = 4000) -> list[str]:
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

class Airforce(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://api.airforce"
    image_api_endpoint = "https://api.airforce/imagine2"
    text_api_endpoint = "https://api.airforce/chat/completions"
    working = True
    
    default_model = 'llama-3-70b-chat'
    
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    text_models = [
        # anthorpic
        'claude-3-haiku-20240307', 
        'claude-3-sonnet-20240229', 
        'claude-3-5-sonnet-20240620', 
        'claude-3-opus-20240229', 
        
        # openai
        'chatgpt-4o-latest', 
        'gpt-4', 
        #'gpt-4-0613', 
        'gpt-4-turbo', 
        'gpt-4o-mini-2024-07-18', 
        'gpt-4o-mini', 
        'gpt-3.5-turbo', 
        'gpt-3.5-turbo-0125', 
        'gpt-3.5-turbo-1106', 
        #'gpt-3.5-turbo-16k', # No response from the API.
        #'gpt-3.5-turbo-0613', # No response from the API.
        #'gpt-3.5-turbo-16k-0613', # No response from the API.
        'gpt-4o', 
        #'o1-mini', # No response from the API.
        
        # meta-llama
        'llama-3-70b-chat', 
        'llama-3-70b-chat-turbo', 
        'llama-3-8b-chat', 
        'llama-3-8b-chat-turbo', 
        'llama-3-70b-chat-lite', 
        'llama-3-8b-chat-lite', 
        #'llama-2-70b-chat', # Failed to load response after multiple retries.
        'llama-2-13b-chat', 
        #'llama-2-7b-chat', # Failed to load response after multiple retries.
        'llama-3.1-405b-turbo', 
        'llama-3.1-70b-turbo', 
        'llama-3.1-8b-turbo', 
        'LlamaGuard-2-8b', 
        'Llama-Guard-7b', 
        'Llama-3.2-90B-Vision-Instruct-Turbo',
        
        # codellama
        #'CodeLlama-7b-Python-hf', # Failed to load response after multiple retries.
        #'CodeLlama-7b-Python', 
        #'CodeLlama-13b-Python-hf', # Failed to load response after multiple retries.
        #'CodeLlama-34b-Python-hf', # Failed to load response after multiple retries.
        #'CodeLlama-70b-Python-hf', # Failed to load response after multiple retries.
        
        # 01-ai
        #'Yi-34B-Chat', # Failed to load response after multiple retries.
        #'Yi-34B', # Failed to load response after multiple retries.
        #'Yi-6B', # Failed to load response after multiple retries.
        
        # mistral-ai
        #'Mixtral-8x7B-v0.1', 
        #'Mixtral-8x22B', # Failed to load response after multiple retries.
        'Mixtral-8x7B-Instruct-v0.1', 
        'Mixtral-8x22B-Instruct-v0.1', 
        'Mistral-7B-Instruct-v0.1', 
        'Mistral-7B-Instruct-v0.2', 
        'Mistral-7B-Instruct-v0.3', 
        
        # openchat
        #'openchat-3.5', # Failed to load response after multiple retries.
        
        # wizardlm
        #'WizardLM-13B-V1.2', # Failed to load response after multiple retries.
        #'WizardCoder-Python-34B-V1.0', # Failed to load response after multiple retries.
        
        # qwen
        #'Qwen1.5-0.5B-Chat', # Failed to load response after multiple retries.
        #'Qwen1.5-1.8B-Chat', # Failed to load response after multiple retries.
        #'Qwen1.5-4B-Chat', # Failed to load response after multiple retries.
        'Qwen1.5-7B-Chat', 
        'Qwen1.5-14B-Chat', 
        'Qwen1.5-72B-Chat', 
        'Qwen1.5-110B-Chat', 
        'Qwen2-72B-Instruct', 
        
        # google
        'gemma-2b-it', 
        #'gemma-7b-it', # Failed to load response after multiple retries.
        #'gemma-2b', # Failed to load response after multiple retries.
        #'gemma-7b', # Failed to load response after multiple retries.
        'gemma-2-9b-it', # fix bug
        'gemma-2-27b-it', 
        
        # gemini
        'gemini-1.5-flash', 
        'gemini-1.5-pro', 
        
        # databricks
        'dbrx-instruct', 
        
        # lmsys
        #'vicuna-7b-v1.5', # Failed to load response after multiple retries.
        #'vicuna-13b-v1.5', # Failed to load response after multiple retries.
        
        # cognitivecomputations
        #'dolphin-2.5-mixtral-8x7b', # Failed to load response after multiple retries.
        
        # deepseek-ai
        #'deepseek-coder-33b-instruct', # No response from the API.
        #'deepseek-coder-67b-instruct', # Failed to load response after multiple retries.
        'deepseek-llm-67b-chat', 
        
        # NousResearch
        #'Nous-Capybara-7B-V1p9', # Failed to load response after multiple retries.
        'Nous-Hermes-2-Mixtral-8x7B-DPO', 
        #'Nous-Hermes-2-Mixtral-8x7B-SFT', # Failed to load response after multiple retries.
        #'Nous-Hermes-llama-2-7b', # Failed to load response after multiple retries.
        #'Nous-Hermes-Llama2-13b', # Failed to load response after multiple retries.
        'Nous-Hermes-2-Yi-34B', 
        
        # Open-Orca
        #'Mistral-7B-OpenOrca', # Failed to load response after multiple retries.
        
        # togethercomputer
        #'alpaca-7b', # Failed to load response after multiple retries.
        
        # teknium
        #'OpenHermes-2-Mistral-7B', # Failed to load response after multiple retries.
        #'OpenHermes-2.5-Mistral-7B', # Failed to load response after multiple retries.
        
        # microsoft
        'WizardLM-2-8x22B', 
        
        # Nexusflow
        #'NexusRaven-V2-13B', # Failed to load response after multiple retries.
        
        # Phind
        #'Phind-CodeLlama-34B-v2', # Failed to load response after multiple retries.
        
        # Snoflake
        #'snowflake-arctic-instruct', # No response from the API.
        
        # upstage
        'SOLAR-10.7B-Instruct-v1.0', 
        
        # togethercomputer
        #'StripedHyena-Hessian-7B', # Failed to load response after multiple retries.
        #'StripedHyena-Nous-7B', # Failed to load response after multiple retries.
        #'Llama-2-7B-32K-Instruct', # Failed to load response after multiple retries.
        #'CodeLlama-13b-Instruct', # No response from the API.
        #'evo-1-131k-base', # Failed to load response after multiple retries.
        #'OLMo-7B-Instruct', # Failed to load response after multiple retries.
        
        # garage-bAInd
        #'Platypus2-70B-instruct', # Failed to load response after multiple retries.
        
        # snorkelai
        #'Snorkel-Mistral-PairRM-DPO', # Failed to load response after multiple retries.
        
        # Undi95
        #'ReMM-SLERP-L2-13B', # Failed to load response after multiple retries.
        
        # Gryphe
        'MythoMax-L2-13b', 
        
        # Autism
        #'chronos-hermes-13b', # Failed to load response after multiple retries.
        
        # Undi95
        #'Toppy-M-7B', # Failed to load response after multiple retries.
        
        # iFlytek
        #'sparkdesk', # Failed to load response after multiple retries.
        
        # pawan
        'cosmosrp', 
        
    ]
    image_models = [
        'flux',
        'flux-realism',
        'flux-anime',
        'flux-3d',
        'flux-disney',
        'flux-pixel',
        'flux-4o',
        'any-dark',
        'dall-e-3',
    ]
    
    models = [
        *text_models,
        *image_models,
    ]
    model_aliases = {
        # anthorpic
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-opus": "claude-3-opus-20240229",
        
        # openai
        "gpt-4o": "chatgpt-4o-latest",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo": "gpt-3.5-turbo-1106",
        
        # meta-llama
        "llama-3-70b": "llama-3-70b-chat",
        "llama-3-70b": "llama-3-70b-chat-turbo",
        "llama-3-8b": "llama-3-8b-chat",
        "llama-3-8b": "llama-3-8b-chat-turbo",
        "llama-3-70b": "llama-3-70b-chat-lite",
        "llama-3-8b": "llama-3-8b-chat-lite",
        "llama-2-13b": "llama-2-13b-chat",
        "llama-3.1-405b": "llama-3.1-405b-turbo",
        "llama-3.1-70b": "llama-3.1-70b-turbo",
        "llama-3.1-8b": "llama-3.1-8b-turbo",
        "llamaguard-2-8b": "LlamaGuard-2-8b",
        "llamaguard-7b": "Llama-Guard-7b",
        "llama-3.2-90b": "Llama-3.2-90B-Vision-Instruct-Turbo",
        
        # mistral-ai
        "mixtral-8x7b": "Mixtral-8x7B-Instruct-v0.1",
        "mixtral-8x22b": "Mixtral-8x22B-Instruct-v0.1",
        "mistral-7b": "Mistral-7B-Instruct-v0.1",
        "mistral-7b": "Mistral-7B-Instruct-v0.2",
        "mistral-7b": "Mistral-7B-Instruct-v0.3",
        
        # qwen
        "qwen-1.5-7b": "Qwen1.5-7B-Chat",
        "qwen-1.5-14b": "Qwen1.5-14B-Chat",
        "qwen-1.5-72b": "Qwen1.5-72B-Chat",
        "qwen-1.5-110b": "Qwen1.5-110B-Chat",
        "qwen-2-72b": "Qwen2-72B-Instruct",
        
        # google
        "gemma-2b": "gemma-2b-it",
        "gemma-2-9b": "gemma-2-9b-it",
        "gemma-2-27b": "gemma-2-27b-it",
        
        # gemini
        "gemini-flash": "gemini-1.5-flash",
        "gemini-pro": "gemini-1.5-pro",
        
        # deepseek-ai
        "deepseek": "deepseek-llm-67b-chat",
        
        # NousResearch
        "mixtral-8x7b-dpo": "Nous-Hermes-2-Mixtral-8x7B-DPO",
        "yi-34b": "Nous-Hermes-2-Yi-34B",
        
        # microsoft
        "wizardlm-2-8x22b": "WizardLM-2-8x22B",
        
        # upstage
        "solar-10.7b": "SOLAR-10.7B-Instruct-v1.0",
        
        # Gryphe
        "mythomax-l2-13b": "MythoMax-L2-13b",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases.get(model, cls.default_model)
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        seed: int = None,
        size: str = "1:1",
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        # If the model is an image model, use the image API
        if model in cls.image_models:
            async for result in cls._generate_image(model, messages, proxy, seed, size):
                yield result
        # If the model is a text model, use the text API
        elif model in cls.text_models:
            async for result in cls._generate_text(model, messages, proxy, stream):
                yield result
    
    @classmethod
    async def _generate_image(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        seed: int = None,
        size: str = "1:1",
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "origin": "https://llmplayground.net",
            "user-agent": "Mozilla/5.0"
        }

        if seed is None:
            seed = random.randint(0, 100000)

        # Assume the first message is the prompt for the image
        prompt = messages[0]['content']

        async with ClientSession(headers=headers) as session:
            params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "seed": seed
            }
            async with session.get(f"{cls.image_api_endpoint}", params=params, proxy=proxy) as response:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '').lower()

                if 'application/json' in content_type:
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            yield chunk.decode('utf-8')
                elif 'image' in content_type:
                    image_data = b""
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            image_data += chunk
                    image_url = f"{cls.image_api_endpoint}?model={model}&prompt={prompt}&size={size}&seed={seed}"
                    alt_text = f"Generated image for prompt: {prompt}"
                    yield ImageResponse(images=image_url, alt=alt_text)

    @classmethod
    async def _generate_text(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "authorization": "Bearer missing api key",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0"
        }

        async with ClientSession(headers=headers) as session:
            formatted_prompt = cls._format_messages(messages)
            prompt_parts = split_long_message(formatted_prompt)
            full_response = ""

            for part in prompt_parts:
                data = {
                    "messages": [{"role": "user", "content": part}],
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": 1,
                    "top_p": 1,
                    "stream": stream
                }
                async with session.post(cls.text_api_endpoint, json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    part_response = ""
                    if stream:
                        async for line in response.content:
                            if line:
                                line = line.decode('utf-8').strip()
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    json_data = json.loads(line[6:])
                                    content = json_data['choices'][0]['delta'].get('content', '')
                                    part_response += content
                    else:
                        json_data = await response.json()
                        content = json_data['choices'][0]['message']['content']
                        part_response = content

                    full_response += part_response
            yield full_response

    @classmethod
    def _format_messages(cls, messages: Messages) -> str:
        """Formats messages for text generation."""
        return " ".join([msg['content'] for msg in messages])
