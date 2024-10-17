from __future__ import annotations

import asyncio
import aiohttp
import random
import string
import json
import uuid
import re
from typing import Optional, AsyncGenerator, Union

from aiohttp import ClientSession, ClientResponseError

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse


class Blackbox(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Blackbox AI"
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/chat"
    working = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = 'blackboxai'
    image_models = ['ImageGeneration']
    models = [
        default_model,
        'blackboxai-pro',
        *image_models,
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
        "blackboxai": "/?model=blackboxai",
        "gpt-4o": "/?model=gpt-4o",
        "gemini-pro": "/?model=gemini-pro",
        "claude-sonnet-3.5": "/?model=claude-sonnet-3.5"
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
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @staticmethod
    def generate_random_string(length: int = 7) -> str:
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=length))

    @staticmethod
    def generate_next_action() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def generate_next_router_state_tree() -> str:
        router_state = [
            "",
            {
                "children": [
                    "(chat)",
                    {
                        "children": [
                            "__PAGE__",
                            {}
                        ]
                    }
                ]
            },
            None,
            None,
            True
        ]
        return json.dumps(router_state)

    @staticmethod
    def clean_response(text: str) -> str:
        pattern = r'^\$\@\$v=undefined-rv1\$\@\$'
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: Optional[str] = None,
        websearch: bool = False,
        **kwargs
    ) -> AsyncGenerator[Union[str, ImageResponse], None]:
        """
        Creates an asynchronous generator for streaming responses from Blackbox AI.

        Parameters:
            model (str): Model to use for generating responses.
            messages (Messages): Message history.
            proxy (Optional[str]): Proxy URL, if needed.
            websearch (bool): Enables or disables web search mode.
            **kwargs: Additional keyword arguments.

        Yields:
            Union[str, ImageResponse]: Segments of the generated response or ImageResponse objects.
        """
        model = cls.get_model(model)

        chat_id = cls.generate_random_string()
        next_action = cls.generate_next_action()
        next_router_state_tree = cls.generate_next_router_state_tree()

        agent_mode = cls.agentMode.get(model, {})
        trending_agent_mode = cls.trendingAgentMode.get(model, {})

        prefix = cls.model_prefixes.get(model, "")
        
        formatted_prompt = ""
        for message in messages:
            role = message.get('role', '').capitalize()
            content = message.get('content', '')
            if role and content:
                formatted_prompt += f"{role}: {content}\n"
        
        if prefix:
            formatted_prompt = f"{prefix} {formatted_prompt}".strip()

        referer_path = cls.model_referers.get(model, f"/?model={model}")
        referer_url = f"{cls.url}{referer_path}"

        common_headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'origin': cls.url,
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'sec-ch-ua': '"Chromium";v="129", "Not=A?Brand";v="8"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/129.0.0.0 Safari/537.36'
        }

        headers_api_chat = {
            'Content-Type': 'application/json',
            'Referer': referer_url
        }
        headers_api_chat_combined = {**common_headers, **headers_api_chat}

        payload_api_chat = {
            "messages": [
                {
                    "id": chat_id,
                    "content": formatted_prompt,
                    "role": "user"
                }
            ],
            "id": chat_id,
            "previewToken": None,
            "userId": None,
            "codeModelMode": True,
            "agentMode": agent_mode,
            "trendingAgentMode": trending_agent_mode,
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
            "webSearchMode": websearch,
            "userSelectedModel": cls.userSelectedModel.get(model, model)
        }

        headers_chat = {
            'Accept': 'text/x-component',
            'Content-Type': 'text/plain;charset=UTF-8',
            'Referer': f'{cls.url}/chat/{chat_id}?model={model}',
            'next-action': next_action,
            'next-router-state-tree': next_router_state_tree,
            'next-url': '/'
        }
        headers_chat_combined = {**common_headers, **headers_chat}

        data_chat = '[]'

        async with ClientSession(headers=common_headers) as session:
            try:
                async with session.post(
                    cls.api_endpoint,
                    headers=headers_api_chat_combined,
                    json=payload_api_chat,
                    proxy=proxy
                ) as response_api_chat:
                    response_api_chat.raise_for_status()
                    text = await response_api_chat.text()
                    cleaned_response = cls.clean_response(text)

                    if model in cls.image_models:
                        match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', cleaned_response)
                        if match:
                            image_url = match.group(1)
                            image_response = ImageResponse(images=image_url, alt="Generated Image")
                            yield image_response
                        else:
                            yield cleaned_response
                    else:
                        if websearch:
                            match = re.search(r'\$~~~\$(.*?)\$~~~\$', cleaned_response, re.DOTALL)
                            if match:
                                source_part = match.group(1).strip()
                                answer_part = cleaned_response[match.end():].strip()
                                try:
                                    sources = json.loads(source_part)
                                    source_formatted = "**Source:**\n"
                                    for item in sources:
                                        title = item.get('title', 'No Title')
                                        link = item.get('link', '#')
                                        position = item.get('position', '')
                                        source_formatted += f"{position}. [{title}]({link})\n"
                                    final_response = f"{answer_part}\n\n{source_formatted}"
                                except json.JSONDecodeError:
                                    final_response = f"{answer_part}\n\nSource information is unavailable."
                            else:
                                final_response = cleaned_response
                        else:
                            if '$~~~$' in cleaned_response:
                                final_response = cleaned_response.split('$~~~$')[0].strip()
                            else:
                                final_response = cleaned_response

                        yield final_response
            except ClientResponseError as e:
                error_text = f"Error {e.status}: {e.message}"
                try:
                    error_response = await e.response.text()
                    cleaned_error = cls.clean_response(error_response)
                    error_text += f" - {cleaned_error}"
                except Exception:
                    pass
                yield error_text
            except Exception as e:
                yield f"Unexpected error during /api/chat request: {str(e)}"

            chat_url = f'{cls.url}/chat/{chat_id}?model={model}'

            try:
                async with session.post(
                    chat_url,
                    headers=headers_chat_combined,
                    data=data_chat,
                    proxy=proxy
                ) as response_chat:
                    response_chat.raise_for_status()
                    pass
            except ClientResponseError as e:
                error_text = f"Error {e.status}: {e.message}"
                try:
                    error_response = await e.response.text()
                    cleaned_error = cls.clean_response(error_response)
                    error_text += f" - {cleaned_error}"
                except Exception:
                    pass
                yield error_text
            except Exception as e:
                yield f"Unexpected error during /chat/{chat_id} request: {str(e)}"
