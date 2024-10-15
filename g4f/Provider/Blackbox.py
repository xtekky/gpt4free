from __future__ import annotations

import re
import random
import string
import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages, ImageType
from ..image import ImageResponse, to_data_uri
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class Blackbox(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/chat"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = 'blackboxai'
    image_models = ['ImageGeneration']
    models = [
        default_model,
        'blackboxai-pro',

        "llama-3.1-8b",
        'llama-3.1-70b',
        'llama-3.1-405b',

        'gpt-4o',

        'gemini-pro',
        'gemini-1.5-flash',

        'claude-sonnet-3.5',

        'PythonAgent',
        'JavaAgent',
        'JavaScriptAgent',
        'HTMLAgent',
        'GoogleCloudAgent',
        'AndroidDeveloper',
        'SwiftDeveloper',
        'Next.jsAgent',
        'MongoDBAgent',
        'PyTorchAgent',
        'ReactAgent',
        'XcodeAgent',
        'AngularJSAgent',
        *image_models,
    ]

    agentMode = {
        'ImageGeneration': {'mode': True, 'id': "ImageGenerationLV45LJp", 'name': "Image Generation"},
    }

    trendingAgentMode = {
        "blackboxai": {},
        "gemini-1.5-flash": {'mode': True, 'id': 'Gemini'},
        "llama-3.1-8b": {'mode': True, 'id': "llama-3.1-8b"},
        'llama-3.1-70b': {'mode': True, 'id': "llama-3.1-70b"},
        'llama-3.1-405b': {'mode': True, 'id': "llama-3.1-405b"},
        'blackboxai-pro': {'mode': True, 'id': "BLACKBOXAI-PRO"},
        'PythonAgent': {'mode': True, 'id': "Python Agent"},
        'JavaAgent': {'mode': True, 'id': "Java Agent"},
        'JavaScriptAgent': {'mode': True, 'id': "JavaScript Agent"},
        'HTMLAgent': {'mode': True, 'id': "HTML Agent"},
        'GoogleCloudAgent': {'mode': True, 'id': "Google Cloud Agent"},
        'AndroidDeveloper': {'mode': True, 'id': "Android Developer"},
        'SwiftDeveloper': {'mode': True, 'id': "Swift Developer"},
        'Next.jsAgent': {'mode': True, 'id': "Next.js Agent"},
        'MongoDBAgent': {'mode': True, 'id': "MongoDB Agent"},
        'PyTorchAgent': {'mode': True, 'id': "PyTorch Agent"},
        'ReactAgent': {'mode': True, 'id': "React Agent"},
        'XcodeAgent': {'mode': True, 'id': "Xcode Agent"},
        'AngularJSAgent': {'mode': True, 'id': "AngularJS Agent"},
    }
    
    userSelectedModel = {
        "gpt-4o": "gpt-4o",
        "gemini-pro": "gemini-pro",
        'claude-sonnet-3.5': "claude-sonnet-3.5",
    }
    
    model_prefixes = {
        'gpt-4o': '@GPT-4o',
        'gemini-pro': '@Gemini-PRO',
        'claude-sonnet-3.5': '@Claude-Sonnet-3.5',

        'PythonAgent': '@Python Agent',
        'JavaAgent': '@Java Agent',
        'JavaScriptAgent': '@JavaScript Agent',
        'HTMLAgent': '@HTML Agent',
        'GoogleCloudAgent': '@Google Cloud Agent',
        'AndroidDeveloper': '@Android Developer',
        'SwiftDeveloper': '@Swift Developer',
        'Next.jsAgent': '@Next.js Agent',
        'MongoDBAgent': '@MongoDB Agent',
        'PyTorchAgent': '@PyTorch Agent',
        'ReactAgent': '@React Agent',
        'XcodeAgent': '@Xcode Agent',
        'AngularJSAgent': '@AngularJS Agent',
        'blackboxai-pro': '@BLACKBOXAI-PRO',
        'ImageGeneration': '@Image Generation',
    }
    
    model_referers = {
        "blackboxai": f"{url}/?model=blackboxai",
        "gpt-4o": f"{url}/?model=gpt-4o",
        "gemini-pro": f"{url}/?model=gemini-pro",
        "claude-sonnet-3.5": f"{url}/?model=claude-sonnet-3.5"
    }
    
    model_aliases = {
        "gemini-flash": "gemini-1.5-flash",
        "claude-3.5-sonnet": "claude-sonnet-3.5",
        "flux": "ImageGeneration",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.userSelectedModel:
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
        image: ImageType = None,
        image_name: str = None,
        webSearchMode: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": cls.url,
            "pragma": "no-cache",
            "referer": cls.model_referers.get(model, cls.url),
            "sec-ch-ua": '"Not;A=Brand";v="24", "Chromium";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }

        if model in cls.model_prefixes:
            prefix = cls.model_prefixes[model]
            if not messages[0]['content'].startswith(prefix):
                messages[0]['content'] = f"{prefix} {messages[0]['content']}"
        
        random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
        messages[-1]['id'] = random_id
        messages[-1]['role'] = 'user'

        if image is not None:
            messages[-1]['data'] = {
                'fileText': '',
                'imageBase64': to_data_uri(image),
                'title': image_name
            }
            messages[-1]['content'] = 'FILE:BB\n$#$\n\n$#$\n' + messages[-1]['content']
        
        data = {
            "messages": messages,
            "id": random_id,
            "previewToken": None,
            "userId": None,
            "codeModelMode": True,
            "agentMode": {},
            "trendingAgentMode": {},
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
            "userSelectedModel": None,
            "webSearchMode": webSearchMode,
        }

        if model in cls.agentMode:
            data["agentMode"] = cls.agentMode[model]
        elif model in cls.trendingAgentMode:
            data["trendingAgentMode"] = cls.trendingAgentMode[model]
        elif model in cls.userSelectedModel:
            data["userSelectedModel"] = cls.userSelectedModel[model]
        
        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                if model == 'ImageGeneration':
                    response_text = await response.text()
                    url_match = re.search(r'https://storage\.googleapis\.com/[^\s\)]+', response_text)
                    if url_match:
                        image_url = url_match.group(0)
                        yield ImageResponse(image_url, alt=messages[-1]['content'])
                    else:
                        raise Exception("Image URL not found in the response")
                else:
                    full_response = ""
                    search_results_json = ""
                    async for chunk in response.content.iter_any():
                        if chunk:
                            decoded_chunk = chunk.decode()
                            decoded_chunk = re.sub(r'\$@\$v=[^$]+\$@\$', '', decoded_chunk)
                            if decoded_chunk.strip():
                                if '$~~~$' in decoded_chunk:
                                    search_results_json += decoded_chunk
                                else:
                                    full_response += decoded_chunk
                                    yield decoded_chunk

                    if data["webSearchMode"] and search_results_json:
                        match = re.search(r'\$~~~\$(.*?)\$~~~\$', search_results_json, re.DOTALL)
                        if match:
                            search_results = json.loads(match.group(1))
                            formatted_results = "\n\n**Sources:**\n"
                            for i, result in enumerate(search_results[:5], 1):
                                formatted_results += f"{i}. [{result['title']}]({result['link']})\n"
                            yield formatted_results
