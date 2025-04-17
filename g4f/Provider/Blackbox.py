from __future__ import annotations

from aiohttp import ClientSession
import os
import re
import json
import random
import string
import base64
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from ..typing import AsyncResult, Messages, MediaListType
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .openai.har_file import get_har_files
from ..image import to_data_uri
from ..cookies import get_cookies_dir
from .helper import format_image_prompt, render_messages
from ..providers.response import JsonConversation, ImageResponse
from ..tools.media import merge_media
from ..errors import RateLimitError
from .. import debug

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
    default_image_model = 'flux'
    
    # Free models (available without subscription)
    fallback_models = [
        default_model,
        "gpt-4o-mini",
        "DeepSeek-V3",
        "DeepSeek-R1",
        "Meta-Llama-3.3-70B-Instruct-Turbo",
        "Mistral-Small-24B-Instruct-2501",
        "DeepSeek-LLM-Chat-(67B)",
        "Qwen-QwQ-32B-Preview",
        # Image models
        "flux",
        # Trending agent modes
        'Python Agent',
        'HTML Agent',
        'Builder Agent',
        'Java Agent',
        'JavaScript Agent',
        'React Agent',
        'Android Agent',
        'Flutter Agent',
        'Next.js Agent',
        'AngularJS Agent',
        'Swift Agent',
        'MongoDB Agent',
        'PyTorch Agent',
        'Xcode Agent',
        'Azure Agent',
        'Bitbucket Agent',
        'DigitalOcean Agent',
        'Docker Agent',
        'Electron Agent',
        'Erlang Agent',
        'FastAPI Agent',
        'Firebase Agent',
        'Flask Agent',
        'Git Agent',
        'Gitlab Agent',
        'Go Agent',
        'Godot Agent',
        'Google Cloud Agent',
        'Heroku Agent'
    ]
    
    # Premium models (require subscription)
    premium_models = [
        "GPT-4o",
        "o1",
        "o3-mini",
        "Claude-sonnet-3.7",
        "Claude-sonnet-3.5",
        "Gemini-Flash-2.0",
        "DBRX-Instruct",
        "blackboxai-pro",
        "Gemini-PRO"
    ]
    
    # Models available in the demo account
    demo_models = [
        default_model, 
        "blackboxai-pro",
        "gpt-4o-mini", 
        "GPT-4o", 
        "o1", 
        "o3-mini", 
        "Claude-sonnet-3.7", 
        "Claude-sonnet-3.5", 
        "DeepSeek-V3", 
        "DeepSeek-R1", 
        "DeepSeek-LLM-Chat-(67B)",
        "Meta-Llama-3.3-70B-Instruct-Turbo",
        "Mistral-Small-24B-Instruct-2501",
        "Qwen-QwQ-32B-Preview",
        # Image models
        "flux",
        # Trending agent modes
        'Python Agent',
        'HTML Agent',
        'Builder Agent',
        'Java Agent',
        'JavaScript Agent',
        'React Agent',
        'Android Agent',
        'Flutter Agent',
        'Next.js Agent',
        'AngularJS Agent',
        'Swift Agent',
        'MongoDB Agent',
        'PyTorch Agent',
        'Xcode Agent',
        'Azure Agent',
        'Bitbucket Agent',
        'DigitalOcean Agent',
        'Docker Agent',
        'Electron Agent',
        'Erlang Agent',
        'FastAPI Agent',
        'Firebase Agent',
        'Flask Agent',
        'Git Agent',
        'Gitlab Agent',
        'Go Agent',
        'Godot Agent',
        'Google Cloud Agent',
        'Heroku Agent'
    ]
    
    image_models = [default_image_model]   
    vision_models = [default_vision_model, 'GPT-4o', 'o1', 'o3-mini', 'Gemini-PRO', 'Gemini Agent', 'llama-3.1-8b Agent', 'llama-3.1-70b Agent', 'llama-3.1-405 Agent', 'Gemini-Flash-2.0', 'DeepSeek-V3']

    userSelectedModel = ['GPT-4o', 'o1', 'o3-mini', 'Gemini-PRO', 'Claude-sonnet-3.7', 'Claude-sonnet-3.5', 'DeepSeek-V3', 'DeepSeek-R1', 'Meta-Llama-3.3-70B-Instruct-Turbo', 'Mistral-Small-24B-Instruct-2501', 'DeepSeek-LLM-Chat-(67B)', 'DBRX-Instruct', 'Qwen-QwQ-32B-Preview', 'Nous-Hermes-2-Mixtral-8x7B-DPO', 'Gemini-Flash-2.0']

    # Agent mode configurations
    agentMode = {
        'GPT-4o': {'mode': True, 'id': "GPT-4o", 'name': "GPT-4o"},
        'Gemini-PRO': {'mode': True, 'id': "Gemini-PRO", 'name': "Gemini-PRO"},
        'Claude-sonnet-3.7': {'mode': True, 'id': "Claude-sonnet-3.7", 'name': "Claude-sonnet-3.7"},
        'Claude-sonnet-3.5': {'mode': True, 'id': "Claude-sonnet-3.5", 'name': "Claude-sonnet-3.5"},
        'DeepSeek-V3': {'mode': True, 'id': "deepseek-chat", 'name': "DeepSeek-V3"},
        'DeepSeek-R1': {'mode': True, 'id': "deepseek-reasoner", 'name': "DeepSeek-R1"},
        'Meta-Llama-3.3-70B-Instruct-Turbo': {'mode': True, 'id': "meta-llama/Llama-3.3-70B-Instruct-Turbo", 'name': "Meta-Llama-3.3-70B-Instruct-Turbo"},
        'Gemini-Flash-2.0': {'mode': True, 'id': "Gemini/Gemini-Flash-2.0", 'name': "Gemini-Flash-2.0"},
        'Mistral-Small-24B-Instruct-2501': {'mode': True, 'id': "mistralai/Mistral-Small-24B-Instruct-2501", 'name': "Mistral-Small-24B-Instruct-2501"},
        'DeepSeek-LLM-Chat-(67B)': {'mode': True, 'id': "deepseek-ai/deepseek-llm-67b-chat", 'name': "DeepSeek-LLM-Chat-(67B)"},
        'DBRX-Instruct': {'mode': True, 'id': "databricks/dbrx-instruct", 'name': "DBRX-Instruct"},
        'Qwen-QwQ-32B-Preview': {'mode': True, 'id': "Qwen/QwQ-32B-Preview", 'name': "Qwen-QwQ-32B-Preview"},
        'Nous-Hermes-2-Mixtral-8x7B-DPO': {'mode': True, 'id': "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", 'name': "Nous-Hermes-2-Mixtral-8x7B-DPO"},
    }

    # Trending agent modes
    trendingAgentMode = {
        'blackboxai-pro': {'mode': True, 'id': "BLACKBOXAI-PRO"},
        "Gemini Agent": {'mode': True, 'id': 'gemini'},
        "llama-3.1-405 Agent": {'mode': True, 'id': "llama-3.1-405"},
        'llama-3.1-70b Agent': {'mode': True, 'id': "llama-3.1-70b"},
        'llama-3.1-8b Agent': {'mode': True, 'id': "llama-3.1-8b"},
        'Python Agent': {'mode': True, 'id': "python"},
        'HTML Agent': {'mode': True, 'id': "html"},
        'Builder Agent': {'mode': True, 'id': "builder"},
        'Java Agent': {'mode': True, 'id': "java"},
        'JavaScript Agent': {'mode': True, 'id': "javascript"},
        'React Agent': {'mode': True, 'id': "react"},
        'Android Agent': {'mode': True, 'id': "android"},
        'Flutter Agent': {'mode': True, 'id': "flutter"},
        'Next.js Agent': {'mode': True, 'id': "next.js"},
        'AngularJS Agent': {'mode': True, 'id': "angularjs"},
        'Swift Agent': {'mode': True, 'id': "swift"},
        'MongoDB Agent': {'mode': True, 'id': "mongodb"},
        'PyTorch Agent': {'mode': True, 'id': "pytorch"},
        'Xcode Agent': {'mode': True, 'id': "xcode"},
        'Azure Agent': {'mode': True, 'id': "azure"},
        'Bitbucket Agent': {'mode': True, 'id': "bitbucket"},
        'DigitalOcean Agent': {'mode': True, 'id': "digitalocean"},
        'Docker Agent': {'mode': True, 'id': "docker"},
        'Electron Agent': {'mode': True, 'id': "electron"},
        'Erlang Agent': {'mode': True, 'id': "erlang"},
        'FastAPI Agent': {'mode': True, 'id': "fastapi"},
        'Firebase Agent': {'mode': True, 'id': "firebase"},
        'Flask Agent': {'mode': True, 'id': "flask"},
        'Git Agent': {'mode': True, 'id': "git"},
        'Gitlab Agent': {'mode': True, 'id': "gitlab"},
        'Go Agent': {'mode': True, 'id': "go"},
        'Godot Agent': {'mode': True, 'id': "godot"},
        'Google Cloud Agent': {'mode': True, 'id': "googlecloud"},
        'Heroku Agent': {'mode': True, 'id': "heroku"},
    }
    
    # Complete list of all models (for authorized users)
    _all_models = list(dict.fromkeys([
        *fallback_models,  # Include all free models
        *premium_models,   # Include all premium models
        *image_models,
        *list(agentMode.keys()),
        *list(trendingAgentMode.keys())
    ]))
   
    # Initialize models with fallback_models
    models = fallback_models
    
    model_aliases = {
        "gpt-4o": "GPT-4o",
        "claude-3.7-sonnet": "Claude-sonnet-3.7",
        "claude-3.5-sonnet": "Claude-sonnet-3.5",
        "deepseek-v3": "DeepSeek-V3",
        "deepseek-r1": "DeepSeek-R1",
        "deepseek-chat": "DeepSeek-LLM-Chat-(67B)",
        "llama-3.3-70b": "Meta-Llama-3.3-70B-Instruct-Turbo",
        "mixtral-small-24b": "Mistral-Small-24B-Instruct-2501",
        "qwq-32b": "Qwen-QwQ-32B-Preview",
    }

    @classmethod
    async def get_models_async(cls) -> list:
        """
        Asynchronous version of get_models that checks subscription status.
        Returns a list of available models based on subscription status.
        Premium users get the full list of models.
        Free users get fallback_models.
        Demo accounts get demo_models.
        """
        # Check if there are valid session data in HAR files
        session_data = cls._find_session_in_har_files()
        
        if not session_data:
            # For demo accounts - return demo models
            debug.log(f"Blackbox: Returning demo model list with {len(cls.demo_models)} models")
            return cls.demo_models
            
        # Check if this is a demo session
        demo_session = cls.generate_session()
        is_demo = (session_data['user'].get('email') == demo_session['user'].get('email'))
        
        if is_demo:
            # For demo accounts - return demo models
            debug.log(f"Blackbox: Returning demo model list with {len(cls.demo_models)} models")
            return cls.demo_models
        
        # For non-demo accounts, check subscription status
        if 'user' in session_data and 'email' in session_data['user']:
            subscription = await cls.check_subscription(session_data['user']['email'])
            if subscription['status'] == "PREMIUM":
                debug.log(f"Blackbox: Returning premium model list with {len(cls._all_models)} models")
                return cls._all_models
        
        # For free accounts - return free models
        debug.log(f"Blackbox: Returning free model list with {len(cls.fallback_models)} models")
        return cls.fallback_models
        
    @classmethod
    def get_models(cls) -> list:
        """
        Returns a list of available models based on authorization status.
        Authorized users get the full list of models.
        Free users get fallback_models.
        Demo accounts get demo_models.
        
        Note: This is a synchronous method that can't check subscription status,
        so it falls back to the basic premium access check.
        For more accurate results, use get_models_async when possible.
        """
        # Check if there are valid session data in HAR files
        session_data = cls._find_session_in_har_files()
        
        if not session_data:
            # For demo accounts - return demo models
            debug.log(f"Blackbox: Returning demo model list with {len(cls.demo_models)} models")
            return cls.demo_models
            
        # Check if this is a demo session
        demo_session = cls.generate_session()
        is_demo = (session_data['user'].get('email') == demo_session['user'].get('email'))
        
        if is_demo:
            # For demo accounts - return demo models
            debug.log(f"Blackbox: Returning demo model list with {len(cls.demo_models)} models")
            return cls.demo_models
        
        # For non-demo accounts, check premium access
        has_premium_access = cls._check_premium_access()
        
        if has_premium_access:
            # For premium users - all models
            debug.log(f"Blackbox: Returning premium model list with {len(cls._all_models)} models")
            return cls._all_models
        
        # For free accounts - return free models
        debug.log(f"Blackbox: Returning free model list with {len(cls.fallback_models)} models")
        return cls.fallback_models
    
    @classmethod
    async def check_subscription(cls, email: str) -> dict:
        """
        Check subscription status for a given email using the Blackbox API.
        
        Args:
            email: The email to check subscription for
            
        Returns:
            dict: Subscription status information with keys:
                - status: "PREMIUM" or "FREE"
                - customerId: Customer ID if available
                - isTrialSubscription: Whether this is a trial subscription
        """
        if not email:
            return {"status": "FREE", "customerId": None, "isTrialSubscription": False}
            
        headers = {
            'accept': '*/*',
            'accept-language': 'en',
            'content-type': 'application/json',
            'origin': 'https://www.blackbox.ai',
            'referer': 'https://www.blackbox.ai/?ref=login-success',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
        }
        
        try:
            async with ClientSession(headers=headers) as session:
                async with session.post(
                    'https://www.blackbox.ai/api/check-subscription',
                    json={"email": email}
                ) as response:
                    if response.status != 200:
                        debug.log(f"Blackbox: Subscription check failed with status {response.status}")
                        return {"status": "FREE", "customerId": None, "isTrialSubscription": False}
                    
                    result = await response.json()
                    status = "PREMIUM" if result.get("hasActiveSubscription", False) else "FREE"
                    
                    return {
                        "status": status,
                        "customerId": result.get("customerId"),
                        "isTrialSubscription": result.get("isTrialSubscription", False)
                    }
        except Exception as e:
            debug.log(f"Blackbox: Error checking subscription: {e}")
            return {"status": "FREE", "customerId": None, "isTrialSubscription": False}
    
    @classmethod
    def _check_premium_access(cls) -> bool:
        """
        Checks for an authorized session in HAR files.
        Returns True if a valid session is found that differs from the demo.
        """
        try:
            session_data = cls._find_session_in_har_files()
            if not session_data:
                return False
                
            # Check if this is not a demo session
            demo_session = cls.generate_session()
            if (session_data['user'].get('email') != demo_session['user'].get('email')):
                return True
            return False
        except Exception as e:
            debug.log(f"Blackbox: Error checking premium access: {e}")
            return False

    @classmethod
    def generate_session(cls, id_length: int = 21, days_ahead: int = 365) -> dict:
        """
        Generate a dynamic session with proper ID and expiry format.
        
        Args:
            id_length: Length of the numeric ID (default: 21)
            days_ahead: Number of days ahead for expiry (default: 365)
        
        Returns:
            dict: A session dictionary with user information and expiry
        """
        # Generate numeric ID
        numeric_id = ''.join(random.choice('0123456789') for _ in range(id_length))
        
        # Generate future expiry date
        future_date = datetime.now() + timedelta(days=days_ahead)
        expiry = future_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
        # Decode the encoded email
        encoded_email = "Z2lzZWxlQGJsYWNrYm94LmFp"  # Base64 encoded email
        email = base64.b64decode(encoded_email).decode('utf-8')
        
        # Generate random image ID for the new URL format
        chars = string.ascii_letters + string.digits + "-"
        random_img_id = ''.join(random.choice(chars) for _ in range(48))
        image_url = f"https://lh3.googleusercontent.com/a/ACg8oc{random_img_id}=s96-c"
        
        return {
            "user": {
                "name": "BLACKBOX AI", 
                "email": email, 
                "image": image_url, 
                "id": numeric_id
            }, 
            "expires": expiry
        }

    @classmethod
    def _find_session_in_har_files(cls) -> Optional[dict]:
        """
        Search for valid session data in HAR files.
        
        Returns:
            Optional[dict]: Session data if found, None otherwise
        """
        try:
            for file in get_har_files():
                try:
                    with open(file, 'rb') as f:
                        har_data = json.load(f)

                    for entry in har_data['log']['entries']:
                        # Only look at blackbox API responses
                        if 'blackbox.ai/api' in entry['request']['url']:
                            # Look for a response that has the right structure
                            if 'response' in entry and 'content' in entry['response']:
                                content = entry['response']['content']
                                # Look for both regular and Google auth session formats
                                if ('text' in content and 
                                    isinstance(content['text'], str) and 
                                    '"user"' in content['text'] and 
                                    '"email"' in content['text'] and
                                    '"expires"' in content['text']):
                                    try:
                                        # Remove any HTML or other non-JSON content
                                        text = content['text'].strip()
                                        if text.startswith('{') and text.endswith('}'):
                                            # Replace escaped quotes
                                            text = text.replace('\\"', '"')
                                            har_session = json.loads(text)

                                            # Check if this is a valid session object
                                            if (isinstance(har_session, dict) and 
                                                'user' in har_session and 
                                                'email' in har_session['user'] and
                                                'expires' in har_session):

                                                debug.log(f"Blackbox: Found session in HAR file: {file}")
                                                return har_session
                                    except json.JSONDecodeError as e:
                                        # Only print error for entries that truly look like session data
                                        if ('"user"' in content['text'] and 
                                            '"email"' in content['text']):
                                            debug.log(f"Blackbox: Error parsing likely session data: {e}")
                except Exception as e:
                    debug.log(f"Blackbox: Error reading HAR file {file}: {e}")
            return None
        except Exception as e:
            debug.log(f"Blackbox: Error searching HAR files: {e}")
            return None

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
                debug.log(f"Blackbox: Error reading cache: {e}")
        
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
                                        debug.log(f"Blackbox: Error writing cache: {e}")
                                        
                                    return validated_value

            except Exception as e:
                debug.log(f"Blackbox: Error retrieving validated_value: {e}")

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
        media: MediaListType = None,
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
            if conversation is None or not hasattr(conversation, "chat_id"):
                conversation = Conversation(model)
                conversation.validated_value = await cls.fetch_validated()
                conversation.chat_id = cls.generate_id()
                conversation.message_history = []

            current_messages = []
            for i, msg in enumerate(render_messages(messages)):
                msg_id = conversation.chat_id if i == 0 and msg["role"] == "user" else cls.generate_id()
                current_msg = {
                    "id": msg_id,
                    "content": msg["content"],
                    "role": msg["role"]
                }
                current_messages.append(current_msg)

            media = list(merge_media(media, messages))
            if media:
                current_messages[-1]['data'] = {
                    "imagesData": [
                        {
                            "filePath": f"/{image_name}",
                            "contents": to_data_uri(image)
                        }
                        for image, image_name in media
                    ],
                    "fileText": "",
                    "title": ""
                }

            # Get session data - try HAR files first, fall back to generated session
            session_data = cls._find_session_in_har_files() or cls.generate_session()
            
            # Log which session type is being used
            demo_session = cls.generate_session()
            is_demo = (session_data['user'].get('email') == demo_session['user'].get('email'))
            
            if is_demo:
                debug.log("Blackbox: Using generated demo session")
                # For demo account, set default values without checking subscription
                subscription_status = {"status": "FREE", "customerId": None, "isTrialSubscription": False}
                # Check if the requested model is in demo_models
                is_premium = model in cls.demo_models
                if not is_premium:
                    debug.log(f"Blackbox: Model {model} not available in demo account, falling back to default model")
                    model = cls.default_model
                    is_premium = True
            else:
                debug.log(f"Blackbox: Using session from HAR file (email: {session_data['user'].get('email', 'unknown')})")
                # Only check subscription for non-demo accounts
                subscription_status = {"status": "FREE", "customerId": None, "isTrialSubscription": False}
                if session_data.get('user', {}).get('email'):
                    subscription_status = await cls.check_subscription(session_data['user']['email'])
                    debug.log(f"Blackbox: Subscription status for {session_data['user']['email']}: {subscription_status['status']}")
                
                # Determine if user has premium access based on subscription status
                if subscription_status['status'] == "PREMIUM":
                    is_premium = True
                else:
                    # For free accounts, check if the requested model is in fallback_models
                    is_premium = model in cls.fallback_models
                    if not is_premium:
                        debug.log(f"Blackbox: Model {model} not available in free account, falling back to default model")
                        model = cls.default_model
                        is_premium = True
            
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
                "imageGenerationMode": model == cls.default_image_model,
                "webSearchModePrompt": False,
                "deepSearchMode": False,
                "domains": None,
                "vscodeClient": False,
                "codeInterpreterMode": False,
                "customProfile": {
                    "name": "",
                    "occupation": "",
                    "traits": [],
                    "additionalInfo": "",
                    "enableNewChats": False
                },
                "session": session_data,
                "isPremium": is_premium, 
                "subscriptionCache": {
                    "status": subscription_status['status'],
                    "customerId": subscription_status['customerId'],
                    "isTrialSubscription": subscription_status['isTrialSubscription'],
                    "lastChecked": int(datetime.now().timestamp() * 1000)
                },
                "beastMode": False,
                "reasoningMode": False,
                "webSearchMode": False
            }

            # Continue with the API request and async generator behavior
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                
                # Collect the full response
                full_response = []
                async for chunk in response.content.iter_any():
                    if chunk:
                        chunk_text = chunk.decode()
                        if "You have reached your request limit for the hour" in chunk_text:
                            raise RateLimitError(chunk_text)
                        full_response.append(chunk_text)
                        # Only yield chunks for non-image models
                        if model != cls.default_image_model:
                            yield chunk_text
                
                full_response_text = ''.join(full_response)
                
                # For image models, check for image markdown
                if model == cls.default_image_model:
                    image_url_match = re.search(r'!\[.*?\]\((.*?)\)', full_response_text)
                    if image_url_match:
                        image_url = image_url_match.group(1)
                        yield ImageResponse(urls=[image_url], alt=format_image_prompt(messages, prompt))
                        return
                
                # Handle conversation history once, in one place
                if return_conversation:
                    conversation.message_history.append({"role": "assistant", "content": full_response_text})
                    yield conversation
                # For image models that didn't produce an image, fall back to text response
                elif model == cls.default_image_model:
                    yield full_response_text
