from __future__ import annotations

from aiohttp import ClientSession
import re
import json
import random
import string
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from ..typing import AsyncResult, Messages, ImagesType
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse, to_data_uri
from ..cookies import get_cookies_dir
from .helper import format_prompt, format_image_prompt
from ..providers.response import JsonConversation, Reasoning

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
    vision_models = [default_vision_model, 'gpt-4o', 'gemini-pro', 'deepseek-v3', 'gemini-1.5-flash', 'llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-405b']
    reasoning_models = ['deepseek-r1']
    
    userSelectedModel = ['gpt-4o', 'gemini-pro', 'claude-sonnet-3.5', 'deepseek-r1', 'deepseek-v3', 'blackboxai-pro']

    agentMode = {
        'ImageGeneration': {'mode': True, 'id': "ImageGenerationLV45LJp", 'name': "Image Generation"},
        'Meta-Llama-3.3-70B-Instruct-Turbo': {'mode': True, 'id': "meta-llama/Llama-3.3-70B-Instruct-Turbo", 'name': "Meta-Llama-3.3-70B-Instruct-Turbo"},
        'Mistral-(7B)-Instruct-v0.2': {'mode': True, 'id': "mistralai/Mistral-7B-Instruct-v0.2", 'name': "Mistral-(7B)-Instruct-v0.2"},
        'DeepSeek-LLM-Chat-(67B)': {'mode': True, 'id': "deepseek-ai/deepseek-llm-67b-chat", 'name': "DeepSeek-LLM-Chat-(67B)"},
        'DBRX-Instruct': {'mode': True, 'id': "databricks/dbrx-instruct", 'name': "DBRX-Instruct"},
        'Qwen-QwQ-32B-Preview': {'mode': True, 'id': "Qwen/QwQ-32B-Preview", 'name': "Qwen-QwQ-32B-Preview"},
        'Nous-Hermes-2-Mixtral-8x7B-DPO': {'mode': True, 'id': "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", 'name': "Nous-Hermes-2-Mixtral-8x7B-DPO"},
    }

    trendingAgentMode = {
        "gemini-1.5-flash": {'mode': True, 'id': 'Gemini'},
        "llama-3.1-8b": {'mode': True, 'id': "llama-3.1-8b"},
        'llama-3.1-70b': {'mode': True, 'id': "llama-3.1-70b"},
        'llama-3.1-405b': {'mode': True, 'id': "llama-3.1-405"},
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
        'repomap': {'mode': True, 'id': "repomap"},
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
    
    models = list(dict.fromkeys([default_model, *userSelectedModel, *reasoning_models, *image_models, *list(agentMode.keys()), *list(trendingAgentMode.keys())]))

    model_aliases = {
        "gpt-4": "gpt-4o",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-pro": "gemini-pro",
        "claude-3.5-sonnet": "claude-sonnet-3.5",
        "llama-3.3-70b": "Meta-Llama-3.3-70B-Instruct-Turbo",
        "mixtral-7b": "Mistral-(7B)-Instruct-v0.2",
        "deepseek-chat": "DeepSeek-LLM-Chat-(67B)",
        "dbrx-instruct": "DBRX-Instruct",
        "qwq-32b": "Qwen-QwQ-32B-Preview",
        "hermes-2-dpo": "Nous-Hermes-2-Mixtral-8x7B-DPO",
        "flux": "ImageGeneration",
    }

    @classmethod
    async def fetch_validated(cls, url: str = "https://www.blackbox.ai", force_refresh: bool = False) -> Optional[str]:
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
            return any(char + '=' in text for char in 'abcdefghijklmnopqrstuvwxyz')

        async with ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
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
    def generate_id(cls, length: int = 7) -> str:
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

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
                prompt = format_image_prompt(messages, prompt)
                data = {
                    "query": format_image_prompt(messages, prompt),
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
                            yield ImageResponse(images=[image_url], alt=format_image_prompt(messages, prompt))
                            return

            if conversation is None or not hasattr(conversation, "chat_id"):
                conversation = Conversation(model)
                conversation.validated_value = await cls.fetch_validated()
                conversation.chat_id = cls.generate_id()
                conversation.message_history = []

            current_messages = []
            for i, msg in enumerate(messages):
                msg_id = conversation.chat_id if i == 0 and msg["role"] == "user" else cls.generate_id()
                current_msg = {
                    "id": msg_id,
                    "content": msg["content"],
                    "role": msg["role"]
                }
                if msg["role"] == "assistant" and i == len(messages)-1:
                    current_time = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                    current_msg["createdAt"] = current_time
                current_messages.append(current_msg)
            
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
                "isMemoryEnabled": False,
                "mobileClient": False,
                "userSelectedModel": model if model in cls.userSelectedModel else None,
                "validated": conversation.validated_value,
                "imageGenerationMode": False,
                "webSearchModePrompt": False,
                "deepSearchMode": False,
                "domains": None,
                "vscodeClient": False,
                "codeInterpreterMode": False,
                "customProfile": {"name": "", "occupation": "", "traits": [], "additionalInfo": "", "enableNewChats": False},
                "webSearchMode": web_search
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                response_text = await response.text()
                parts = response_text.split('$~~~$')
                text_to_yield = parts[2] if len(parts) >= 3 else response_text
                
                if not text_to_yield or text_to_yield.isspace():
                    return

                if model in cls.image_models:
                    image_url_match = re.search(r'!\[.*?\]\((.*?)\)', text_to_yield)
                    if image_url_match:
                        image_url = image_url_match.group(1)
                        prompt = format_image_prompt(messages, prompt)
                        yield ImageResponse(images=[image_url], alt=prompt)
                else:
                    if model in cls.reasoning_models and "\n\n\n" in text_to_yield:
                        think_split = text_to_yield.split("\n\n\n", 1)
                        if len(think_split) > 1:
                            think_content, answer = think_split[0].strip(), think_split[1].strip()
                            yield Reasoning(status=think_content)
                            yield answer
                        else:
                            yield text_to_yield
                    elif "<think>" in text_to_yield:
                        pre_think, rest = text_to_yield.split('<think>', 1)
                        think_content, post_think = rest.split('</think>', 1)
                        
                        pre_think = pre_think.strip()
                        think_content = think_content.strip()
                        post_think = post_think.strip()
                        
                        if pre_think:
                            yield pre_think
                        if think_content:
                            yield Reasoning(status=think_content)
                        if post_think:
                            yield post_think
                            
                    elif "Generated by BLACKBOX.AI" in text_to_yield:
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
                        else:
                            if text_to_yield and not text_to_yield.isspace():
                                yield text_to_yield
                    else:
                        if text_to_yield and not text_to_yield.isspace():
                            yield text_to_yield

                    if return_conversation:
                        conversation.message_history.append({"role": "assistant", "content": text_to_yield})
                        yield conversation
