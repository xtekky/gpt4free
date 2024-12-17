from __future__ import annotations

from aiohttp import ClientSession
import random
import string
import json
import re
import aiohttp

import json
from pathlib import Path

from ..typing import AsyncResult, Messages, ImagesType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse, to_data_uri
from ..cookies import get_cookies_dir
from .helper import format_prompt

class Blackbox(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Blackbox AI"
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/chat"

    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = 'blackboxai'
    default_vision_model = default_model
    default_image_model = 'flux' 
    image_models = ['ImageGeneration', 'repomap']
    vision_models = [default_model, 'gpt-4o', 'gemini-pro', 'gemini-1.5-flash', 'llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-405b']

    userSelectedModel = ['gpt-4o', 'gemini-pro', 'claude-sonnet-3.5', 'blackboxai-pro']

    agentMode = {
        'ImageGeneration': {'mode': True, 'id': "ImageGenerationLV45LJp", 'name': "Image Generation"},
        'meta-llama/Llama-3.3-70B-Instruct-Turbo': {'mode': True, 'id': "meta-llama/Llama-3.3-70B-Instruct-Turbo", 'name': "Meta-Llama-3.3-70B-Instruct-Turbo"},
        'mistralai/Mistral-7B-Instruct-v0.2': {'mode': True, 'id': "mistralai/Mistral-7B-Instruct-v0.2", 'name': "Mistral-(7B)-Instruct-v0.2"},
        'deepseek-ai/deepseek-llm-67b-chat': {'mode': True, 'id': "deepseek-ai/deepseek-llm-67b-chat", 'name': "DeepSeek-LLM-Chat-(67B)"},
        'databricks/dbrx-instruct': {'mode': True, 'id': "databricks/dbrx-instruct", 'name': "DBRX-Instruct"},
        'meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro': {'mode': True, 'id': "meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro", 'name': "Meta-Llama-3.1-405B-Instruct-Turbo"}, #
        'Qwen/QwQ-32B-Preview': {'mode': True, 'id': "Qwen/QwQ-32B-Preview", 'name': "Qwen-QwQ-32B-Preview"},
        'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO': {'mode': True, 'id': "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", 'name': "Nous-Hermes-2-Mixtral-8x7B-DPO"}
    }

    trendingAgentMode = {
        "gemini-1.5-flash": {'mode': True, 'id': 'Gemini'},
        "llama-3.1-8b": {'mode': True, 'id': "llama-3.1-8b"},
        'llama-3.1-70b': {'mode': True, 'id': "llama-3.1-70b"},
        'llama-3.1-405b': {'mode': True, 'id': "llama-3.1-405"},
        #
        'Python Agent': {'mode': True, 'id': "Python Agent"},
        'Java Agent': {'mode': True, 'id': "Java Agent"},
        'JavaScript Agent': {'mode': True, 'id': "JavaScript Agent"},
        'HTML Agent': {'mode': True, 'id': "HTML Agent"},
        'Google Cloud Agent': {'mode': True, 'id': "Google Cloud Agent"},
        'Android Developer': {'mode': True, 'id': "Android Developer"},
        'Swift Developer': {'mode': True, 'id': "Swift Developer"},
        'Next.js Agent': {'mode': True, 'id': "Next.js Agent"},
        'MongoDB Agent': {'mode': True, 'id': "MongoDB Agent"},
        'PyTorch Agent': {'mode': True, 'id': "PyTorch Agent"},
        'React Agent': {'mode': True, 'id': "React Agent"},
        'Xcode Agent': {'mode': True, 'id': "Xcode Agent"},
        'AngularJS Agent': {'mode': True, 'id': "AngularJS Agent"},
        #
        'blackboxai-pro': {'mode': True, 'id': "BLACKBOXAI-PRO"},
        #
        'repomap': {'mode': True, 'id': "repomap"},
        #
        'Heroku Agent': {'mode': True, 'id': "Heroku Agent"},
        'Godot Agent': {'mode': True, 'id': "Godot Agent"},
        'Go Agent': {'mode': True, 'id': "Go Agent"},
        'Gitlab Agent': {'mode': True, 'id': "Gitlab Agent"},
        'Git Agent': {'mode': True, 'id': "Git Agent"},
        'Flask Agent': {'mode': True, 'id': "Flask Agent"},
        'Firebase Agent': {'mode': True, 'id': "Firebase Agent"},
        'FastAPI Agent': {'mode': True, 'id': "FastAPI Agent"},
        'Erlang Agent': {'mode': True, 'id': "Erlang Agent"},
        'Electron Agent': {'mode': True, 'id': "Electron Agent"},
        'Docker Agent': {'mode': True, 'id': "Docker Agent"},
        'DigitalOcean Agent': {'mode': True, 'id': "DigitalOcean Agent"},
        'Bitbucket Agent': {'mode': True, 'id': "Bitbucket Agent"},
        'Azure Agent': {'mode': True, 'id': "Azure Agent"},
        'Flutter Agent': {'mode': True, 'id': "Flutter Agent"},
        'Youtube Agent': {'mode': True, 'id': "Youtube Agent"},
        'builder Agent': {'mode': True, 'id': "builder Agent"},
    }
    
    additional_prefixes = {
        'gpt-4o': '@GPT-4o',
        'gemini-pro': '@Gemini-PRO',
        'claude-sonnet-3.5': '@Claude-Sonnet-3.5'
    }
    
    model_prefixes = {
        **{
            mode: f"@{value['id']}" for mode, value in trendingAgentMode.items() 
            if mode not in ["gemini-1.5-flash", "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b", "repomap"]
        },
        **additional_prefixes
    }

    models = list(dict.fromkeys([default_model, *userSelectedModel, *list(agentMode.keys()), *list(trendingAgentMode.keys())]))

    model_aliases = {
        ### chat ###
        "gpt-4": "gpt-4o",
        "gemini-flash": "gemini-1.5-flash",
        "claude-3.5-sonnet": "claude-sonnet-3.5",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "mixtral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "deepseek-chat": "deepseek-ai/deepseek-llm-67b-chat",
        "dbrx-instruct": "databricks/dbrx-instruct",
        "llama-3.1-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro",
        "qwq-32b": "Qwen/QwQ-32B-Preview",
        "hermes-2-dpo": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        
        ### image ###
        "flux": "ImageGeneration",
    }

    @classmethod
    def _get_cache_file(cls) -> Path:
        dir = Path(get_cookies_dir())
        dir.mkdir(exist_ok=True)
        return dir / 'blackbox.json'

    @classmethod
    def _load_cached_value(cls) -> str | None:
        cache_file = cls._get_cache_file()
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('validated_value')
            except Exception as e:
                print(f"Error reading cache file: {e}")
        return None

    @classmethod
    def _save_cached_value(cls, value: str):
        cache_file = cls._get_cache_file()
        try:
            with open(cache_file, 'w') as f:
                json.dump({'validated_value': value}, f)
        except Exception as e:
            print(f"Error writing to cache file: {e}")

    @classmethod
    async def fetch_validated(cls):
        cached_value = cls._load_cached_value()
        if cached_value:
            return cached_value

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(cls.url) as response:
                    if response.status != 200:
                        print("Failed to load the page.")
                        return cached_value
                    
                    page_content = await response.text()
                    js_files = re.findall(r'static/chunks/\d{4}-[a-fA-F0-9]+\.js', page_content)

                    uuid_format = r'["\']([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})["\']'

                    def is_valid_context(text_around):
                        return any(char + '=' in text_around for char in 'abcdefghijklmnopqrstuvwxyz')

                    for js_file in js_files:
                        js_url = f"{cls.url}/_next/{js_file}"
                        async with session.get(js_url) as js_response:
                            if js_response.status == 200:
                                js_content = await js_response.text()
                                for match in re.finditer(uuid_format, js_content):
                                    start = max(0, match.start() - 10)
                                    end = min(len(js_content), match.end() + 10)
                                    context = js_content[start:end]
                                    
                                    if is_valid_context(context):
                                        validated_value = match.group(1)
                                        cls._save_cached_value(validated_value)
                                        return validated_value
            except Exception as e:
                print(f"Error fetching validated value: {e}")

        return cached_value

    @staticmethod
    def generate_id(length=7):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    @classmethod
    def add_prefix_to_messages(cls, messages: Messages, model: str) -> Messages:
        prefix = cls.model_prefixes.get(model, "")
        if not prefix:
            return messages

        new_messages = []
        for message in messages:
            new_message = message.copy()
            if message['role'] == 'user':
                new_message['content'] = (prefix + " " + message['content']).strip()
            new_messages.append(new_message)

        return new_messages

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        web_search: bool = False,
        images: ImagesType = None,
        top_p: float = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> AsyncResult:
        message_id = cls.generate_id()
        messages = cls.add_prefix_to_messages(messages, model)
        validated_value = await cls.fetch_validated()
        formatted_message = format_prompt(messages)
        model = cls.get_model(model)
        
        messages = [{"id": message_id, "content": formatted_message, "role": "user"}]

        if images is not None:
            messages[-1]['data'] = {
                "imagesData": [
                    {
                        "filePath": f"MultipleFiles/{image_name}",
                        "contents": to_data_uri(image)
                    }
                    for image, image_name in images
                ],
                "fileText": "",
                "title": ""
            }

        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': cls.url,
            'referer': f'{cls.url}/',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }

        data = {
            "agentMode": cls.agentMode.get(model, {}) if model in cls.agentMode else {},
            "clickedAnswer2": False,
            "clickedAnswer3": False,
            "clickedForceWebSearch": False,
            "codeModelMode": True,
            "deepSearchMode": False,
            "githubToken": None,
            "id": message_id,
            "imageGenerationMode": False,
            "isChromeExt": False,
            "isMicMode": False,
            "maxTokens": max_tokens,
            "messages": messages,
            "mobileClient": False,
            "playgroundTemperature": temperature,
            "playgroundTopP": top_p,
            "previewToken": None,
            "trendingAgentMode": cls.trendingAgentMode.get(model, {}) if model in cls.trendingAgentMode else {},
            "userId": None,
            "userSelectedModel": model if model in cls.userSelectedModel else None,
            "userSystemPrompt": None,
            "validated": validated_value,
            "visitFromDelta": False,
            "webSearchModePrompt": False,
            "webSearchMode": web_search
        }

        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_text = await response.text()

                if model in cls.image_models:
                    image_matches = re.findall(r'!\[.*?\]\((https?://[^\)]+)\)', response_text)
                    if image_matches:
                        image_url = image_matches[0]
                        yield ImageResponse(image_url, prompt)
                        return

                response_text = re.sub(r'Generated by BLACKBOX.AI, try unlimited chat https://www.blackbox.ai', '', response_text, flags=re.DOTALL)
                response_text = re.sub(r'and for API requests replace  https://www.blackbox.ai with https://api.blackbox.ai', '', response_text, flags=re.DOTALL)

                json_match = re.search(r'\$~~~\$(.*?)\$~~~\$', response_text, re.DOTALL)
                if json_match:
                    search_results = json.loads(json_match.group(1))
                    answer = response_text.split('$~~~$')[-1].strip()

                    formatted_response = f"{answer}\n\n**Source:**"
                    for i, result in enumerate(search_results, 1):
                        formatted_response += f"\n{i}. {result['title']}: {result['link']}"

                    yield formatted_response
                else:
                    yield response_text.strip()
