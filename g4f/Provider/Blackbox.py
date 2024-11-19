from __future__ import annotations

from aiohttp import ClientSession
import random
import string
import json
import re
import aiohttp

from ..typing import AsyncResult, Messages, ImageType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse, to_data_uri
from .helper import get_random_string

class Blackbox(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Blackbox AI"
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/chat"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    _last_validated_value = None

    default_model = 'blackboxai'
    default_vision_model = default_model
    default_image_model = 'generate_image'  
    image_models = [default_image_model, 'repomap']
    text_models = [default_model, 'gpt-4o', 'gemini-pro', 'claude-sonnet-3.5', 'blackboxai-pro']
    vision_models = [default_model, 'gpt-4o', 'gemini-pro', 'blackboxai-pro']
    agentMode = {
        default_image_model: {'mode': True, 'id': "ImageGenerationLV45LJp", 'name': "Image Generation"},
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
    model_prefixes = {mode: f"@{value['id']}" for mode, value in trendingAgentMode.items() if mode not in ["gemini-1.5-flash", "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b", "repomap"]}
    models = [*text_models, default_image_model, *list(trendingAgentMode.keys())]
    model_aliases = {
        "gemini-flash": "gemini-1.5-flash",
        "claude-3.5-sonnet": "claude-sonnet-3.5",
        "flux": "Image Generation",
    }

    @classmethod
    async def fetch_validated(cls):       
        # If the key is already stored in memory, return it
        if cls._last_validated_value:
            return cls._last_validated_value

        # If the key is not found, perform a search
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(cls.url) as response:
                    if response.status != 200:
                        print("Failed to load the page.")
                        return cls._last_validated_value
                    
                    page_content = await response.text()
                    js_files = re.findall(r'static/chunks/\d{4}-[a-fA-F0-9]+\.js', page_content)

                key_pattern = re.compile(r'w="([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"')

                for js_file in js_files:
                    js_url = f"{cls.url}/_next/{js_file}"
                    async with session.get(js_url) as js_response:
                        if js_response.status == 200:
                            js_content = await js_response.text()
                            match = key_pattern.search(js_content)
                            if match:
                                validated_value = match.group(1)
                                cls._last_validated_value = validated_value  # Keep in mind
                                return validated_value
            except Exception as e:
                print(f"Error fetching validated value: {e}")

        return cls._last_validated_value

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
        proxy: str = None,
        web_search: bool = False,
        image: ImageType = None,
        image_name: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        message_id = get_random_string(7)
        messages = cls.add_prefix_to_messages(messages, model)
        validated_value = await cls.fetch_validated()

        if image is not None:
            messages[-1]['data'] = {
                'fileText': '',
                'imageBase64': to_data_uri(image),
                'title': image_name
            }

        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'origin': cls.url,
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': f'{cls.url}/',
            'sec-ch-ua': '"Not?A_Brand";v="99", "Chromium";v="130"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
        }

        data = {
            "messages": messages,
            "id": message_id,
            "previewToken": None,
            "userId": None,
            "codeModelMode": True,
            "agentMode": cls.agentMode.get(model, {}) if model in cls.agentMode else {},
            "trendingAgentMode": cls.trendingAgentMode.get(model, {}) if model in cls.trendingAgentMode else {},
            "isMicMode": False,
            "userSystemPrompt": None,
            "maxTokens": 1024,
            "playgroundTopP": 0.9,
            "playgroundTemperature": 0.5,
            "isChromeExt": False,
            "githubToken": None,
            "clickedAnswer2": False,
            "clickedAnswer3": False,
            "clickedForceWebSearch": False,
            "visitFromDelta": False,
            "mobileClient": False,
            "userSelectedModel": model if model in cls.text_models else None,
            "webSearchMode": web_search,
            "validated": validated_value,
        }

        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    text_chunk = chunk.decode(errors="ignore")
                    if model in cls.image_models:
                        image_matches = re.findall(r'!\[.*?\]\((https?://[^\)]+)\)', text_chunk)
                        if image_matches:
                            image_url = image_matches[0]
                            image_response = ImageResponse(images=[image_url])
                            yield image_response
                            continue

                    text_chunk = re.sub(r'Generated by BLACKBOX.AI, try unlimited chat https://www.blackbox.ai', '', text_chunk, flags=re.DOTALL)
                    json_match = re.search(r'\$~~~\$(.*?)\$~~~\$', text_chunk, re.DOTALL)
                    if json_match:
                        search_results = json.loads(json_match.group(1))
                        answer = text_chunk.split('$~~~$')[-1].strip()
                        formatted_response = f"{answer}\n\n**Source:**"
                        for i, result in enumerate(search_results, 1):
                            formatted_response += f"\n{i}. {result['title']}: {result['link']}"
                        yield formatted_response
                    else:
                        yield text_chunk.strip()
