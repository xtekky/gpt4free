from __future__ import annotations

from aiohttp import ClientSession

import re
import json
import random
import string
from pathlib import Path

from ..typing import AsyncResult, Messages, ImagesType
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse, to_data_uri
from ..cookies import get_cookies_dir
from .helper import format_prompt
from ..providers.response import FinishReason, JsonConversation

class Conversation(JsonConversation):
    validated_value: str = None
    chat_id: str = None
    message_history: Messages = []

    def __init__(self, model: str):
        self.model = model

class Blackbox(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Blackbox AI"
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/chat"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "blackboxai"
    default_vision_model = default_model
    default_image_model = 'ImageGeneration' 
    image_models = [default_image_model, "ImageGeneration2"]
    vision_models = [default_vision_model, 'gpt-4o', 'gemini-pro', 'gemini-1.5-flash', 'llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-405b']
    
    userSelectedModel = ['gpt-4o', 'gemini-pro', 'claude-sonnet-3.5', 'blackboxai-pro']

    agentMode = {
        'ImageGeneration': {'mode': True, 'id': "ImageGenerationLV45LJp", 'name': "Image Generation"},
        #
        'Meta-Llama-3.3-70B-Instruct-Turbo': {'mode': True, 'id': "meta-llama/Llama-3.3-70B-Instruct-Turbo", 'name': "Meta-Llama-3.3-70B-Instruct-Turbo"},
        'Mistral-(7B)-Instruct-v0.': {'mode': True, 'id': "mistralai/Mistral-7B-Instruct-v0.2", 'name': "Mistral-(7B)-Instruct-v0.2"},
        'DeepSeek-LLM-Chat-(67B)': {'mode': True, 'id': "deepseek-ai/deepseek-llm-67b-chat", 'name': "DeepSeek-LLM-Chat-(67B)"},
        'DBRX-Instruct': {'mode': True, 'id': "databricks/dbrx-instruct", 'name': "DBRX-Instruct"},
        'Qwen-QwQ-32B-Preview': {'mode': True, 'id': "Qwen/QwQ-32B-Preview", 'name': "Qwen-QwQ-32B-Preview"},
        'Nous-Hermes-2-Mixtral-8x7B-DPO': {'mode': True, 'id': "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", 'name': "Nous-Hermes-2-Mixtral-8x7B-DPO"},
        'DeepSeek-R1': {'mode': True, 'id': "deepseek-reasoner", 'name': "DeepSeek-R1"}
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
    
    models = list(dict.fromkeys([default_model, *userSelectedModel, *image_models, *list(agentMode.keys()), *list(trendingAgentMode.keys())]))

    model_aliases = {
        ### chat ###
        "gpt-4": "gpt-4o",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-pro": "gemini-pro",
        "claude-3.5-sonnet": "claude-sonnet-3.5",
        "llama-3.3-70b": "Meta-Llama-3.3-70B-Instruct-Turbo",
        "mixtral-7b": "Mistral-(7B)-Instruct-v0.",
        "deepseek-chat": "DeepSeek-LLM-Chat-(67B)",
        "dbrx-instruct": "DBRX-Instruct",
        "qwq-32b": "Qwen-QwQ-32B-Preview",
        "hermes-2-dpo": "Nous-Hermes-2-Mixtral-8x7B-DPO",
        "deepseek-r1": "DeepSeek-R1",
        
        ### image ###
        "flux": "ImageGeneration",
        "flux": "ImageGeneration2",
    }

    @classmethod
    async def fetch_validated(
        cls,
        url: str = "https://www.blackbox.ai",
        force_refresh: bool = False
    ) -> Optional[str]:
        """
        Asynchronously retrieves the validated_value from the specified URL.
        """
        cache_file = Path(get_cookies_dir()) / 'blackbox.json'
        
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if data.get('validated_value'):
                        return data['validated_value']
            except Exception as e:
                print(f"Error reading cache: {e}")
        
        js_file_pattern = r'static/chunks/\d{4}-[a-fA-F0-9]+\.js'
        uuid_pattern = r'["\']([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})["\']'

        def is_valid_context(text: str) -> bool:
            """Checks if the context is valid."""
            return any(char + '=' in text for char in 'abcdefghijklmnopqrstuvwxyz')

        async with ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        print("Failed to load the page.")
                        return None

                    page_content = await response.text()
                    js_files = re.findall(js_file_pattern, page_content)

                for js_file in js_files:
                    js_url = f"{url}/_next/{js_file}"
                    async with session.get(js_url) as js_response:
                        if js_response.status == 200:
                            js_content = await js_response.text()
                            for match in re.finditer(uuid_pattern, js_content):
                                start = max(0, match.start() - 10)
                                end = min(len(js_content), match.end() + 10)
                                context = js_content[start:end]

                                if is_valid_context(context):
                                    validated_value = match.group(1)
                                    
                                    # Save to cache
                                    cache_file.parent.mkdir(exist_ok=True)
                                    try:
                                        with open(cache_file, 'w') as f:
                                            json.dump({'validated_value': validated_value}, f)
                                    except Exception as e:
                                        print(f"Error writing cache: {e}")
                                        
                                    return validated_value

            except Exception as e:
                print(f"Error retrieving validated_value: {e}")

        return None

    @classmethod
    def generate_chat_id(cls) -> str:
        """Generate a random chat ID"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(7))

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
        conversation: Conversation = None,
        return_conversation: bool = False,
        **kwargs
    ) -> AsyncResult:      
        model = cls.get_model(model)
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': 'https://www.blackbox.ai',
            'referer': 'https://www.blackbox.ai/',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }
        
        async with ClientSession(headers=headers) as session:
            if model == "ImageGeneration2":
                prompt = messages[-1]["content"]
                data = {
                    "query": prompt,
                    "agentMode": True
                }
                headers['content-type'] = 'text/plain;charset=UTF-8'
                
                async with session.post(
                    "https://www.blackbox.ai/api/image-generator",
                    json=data,
                    proxy=proxy,
                    headers=headers
                ) as response:
                    await raise_for_status(response)
                    response_json = await response.json()
                    
                    if "markdown" in response_json:
                        image_url_match = re.search(r'!\[.*?\]\((.*?)\)', response_json["markdown"])
                        if image_url_match:
                            image_url = image_url_match.group(1)
                            yield ImageResponse(images=[image_url], alt=prompt)
                            return

            if conversation is None:
                conversation = Conversation(model)
                conversation.validated_value = await cls.fetch_validated()
                conversation.chat_id = cls.generate_chat_id()
                conversation.message_history = []
            
            current_messages = [{"id": conversation.chat_id, "content": format_prompt(messages), "role": "user"}]
            conversation.message_history.extend(messages)

            if images is not None:
                current_messages[-1]['data'] = {
                    "imagesData": [
                        {
                            "filePath": f"/{image_name}",
                            "contents": to_data_uri(image)
                        }
                        for image, image_name in images
                    ],
                    "fileText": "",
                    "title": ""
                }

            data = {
                "messages": current_messages,
                "agentMode": cls.agentMode.get(model, {}) if model in cls.agentMode else {},
                "id": conversation.chat_id,
                "previewToken": None,
                "userId": None,
                "codeModelMode": True,
                "trendingAgentMode": cls.trendingAgentMode.get(model, {}) if model in cls.trendingAgentMode else {},
                "isMicMode": False,
                "userSystemPrompt": None,
                "maxTokens": max_tokens,
                "playgroundTopP": top_p,
                "playgroundTemperature": temperature,
                "isChromeExt": False,
                "githubToken": "",
                "clickedAnswer2": False,
                "clickedAnswer3": False,
                "clickedForceWebSearch": False,
                "visitFromDelta": False,
                "mobileClient": False,
                "userSelectedModel": model if model in cls.userSelectedModel else None,
                "validated": conversation.validated_value,
                "imageGenerationMode": False,
                "webSearchModePrompt": False,
                "deepSearchMode": False,
                "domains": None,
                "vscodeClient": False,
                "codeInterpreterMode": False,
                "webSearchMode": web_search
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                response_text = await response.text()
                parts = response_text.split('$~~~$')
                text_to_yield = parts[2] if len(parts) >= 3 else response_text
                
                if not text_to_yield or text_to_yield.isspace():
                    return

                full_response = ""
                
                if model in cls.image_models:
                    image_url_match = re.search(r'!\[.*?\]\((.*?)\)', text_to_yield)
                    if image_url_match:
                        image_url = image_url_match.group(1)
                        prompt = messages[-1]["content"]
                        yield ImageResponse(images=[image_url], alt=prompt)
                else:
                    if "Generated by BLACKBOX.AI" in text_to_yield:
                        conversation.validated_value = await cls.fetch_validated(force_refresh=True)
                        if conversation.validated_value:
                            data["validated"] = conversation.validated_value
                            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as new_response:
                                await raise_for_status(new_response)
                                new_response_text = await new_response.text()
                                new_parts = new_response_text.split('$~~~$')
                                new_text = new_parts[2] if len(new_parts) >= 3 else new_response_text
                                
                                if new_text and not new_text.isspace():
                                    yield new_text
                                    full_response = new_text
                        else:
                            if text_to_yield and not text_to_yield.isspace():
                                yield text_to_yield
                                full_response = text_to_yield
                    else:
                        if text_to_yield and not text_to_yield.isspace():
                            yield text_to_yield
                            full_response = text_to_yield

                    if full_response:
                        if max_tokens and len(full_response) >= max_tokens:
                            reason = "length"
                        else:
                            reason = "stop"
                        
                        if return_conversation:
                            conversation.message_history.append({"role": "assistant", "content": full_response})
                            yield conversation
                        
                        yield FinishReason(reason)
