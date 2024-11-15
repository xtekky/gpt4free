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
    
    image_models = ['Image Generation', 'repomap']
   
    userSelectedModel = ['gpt-4o', 'gemini-pro', 'claude-sonnet-3.5', 'blackboxai-pro']
    
    agentMode = {
        'Image Generation': {'mode': True, 'id': "ImageGenerationLV45LJp", 'name': "Image Generation"},
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

    
    models = [default_model, *userSelectedModel, *list(agentMode.keys()), *list(trendingAgentMode.keys())]
    
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
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

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
        message_id = cls.generate_id()
        messages_with_prefix = cls.add_prefix_to_messages(messages, model)
        validated_value = await cls.fetch_validated()

        if image is not None:
            messages_with_prefix[-1]['data'] = {
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
            "messages": messages_with_prefix,
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
            "userSelectedModel": model if model in cls.userSelectedModel else None,
            "webSearchMode": web_search,
            "validated": validated_value,
        }

        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_text = await response.text()
                
                if model in cls.image_models:
                    image_matches = re.findall(r'!\[.*?\]\((https?://[^\)]+)\)', response_text)
                    if image_matches:
                        image_url = image_matches[0]
                        image_response = ImageResponse(images=[image_url], alt="Generated Image")
                        yield image_response
                        return

                response_text = re.sub(r'Generated by BLACKBOX.AI, try unlimited chat https://www.blackbox.ai', '', response_text, flags=re.DOTALL)
                
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
