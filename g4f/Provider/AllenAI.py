from __future__ import annotations
import json
from uuid import uuid4
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests.raise_for_status import raise_for_status
from ..providers.response import FinishReason, JsonConversation
from .helper import format_prompt, get_last_user_message


class Conversation(JsonConversation):
    parent: str = None
    x_anonymous_user_id: str = None

    def __init__(self, model: str):
        super().__init__()  # Ensure parent class is initialized
        self.model = model
        self.messages = []  # Instance-specific list
        if not self.x_anonymous_user_id:
            self.x_anonymous_user_id = str(uuid4())


class AllenAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Ai2 Playground"
    url = "https://playground.allenai.org"
    login_url = None
    api_endpoint = "https://olmo-api.allen.ai/v4/message/stream"
    
    working = True
    needs_auth = False
    use_nodriver = False
    supports_stream = True
    supports_system_message = False
    supports_message_history = True

    default_model = 'tulu3-405b'
    models = [
        default_model,
        'OLMo-2-1124-13B-Instruct',
        'tulu-3-1-8b',
        'Llama-3-1-Tulu-3-70B',
        'olmoe-0125'
    ]
    
    model_aliases = {
        "tulu-3-405b": default_model,
        "olmo-2-13b": "OLMo-2-1124-13B-Instruct",
        "tulu-3-1-8b": "tulu-3-1-8b",
        "tulu-3-70b": "Llama-3-1-Tulu-3-70B",
        "llama-3.1-405b": "tulu3-405b",
        "llama-3.1-8b": "tulu-3-1-8b",
        "llama-3.1-70b": "Llama-3-1-Tulu-3-70B",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        host: str = "inferd",
        private: bool = True,
        top_p: float = None,
        temperature: float = None,
        conversation: Conversation = None,
        return_conversation: bool = False,
        **kwargs
    ) -> AsyncResult:
        prompt = format_prompt(messages) if conversation is None else get_last_user_message(messages)
        # Initialize or update conversation
        if conversation is None:
            conversation = Conversation(model)
        
        # Generate new boundary for each request
        boundary = f"----WebKitFormBoundary{uuid4().hex}"
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": f"multipart/form-data; boundary={boundary}",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "x-anonymous-user-id": conversation.x_anonymous_user_id,
        }
        
        # Build multipart form data
        form_data = [
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="model"\r\n\r\n{cls.get_model(model)}\r\n',
            
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="host"\r\n\r\n{host}\r\n',
            
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="content"\r\n\r\n{prompt}\r\n',
            
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="private"\r\n\r\n{str(private).lower()}\r\n'
        ]
        
        # Add parent if exists in conversation
        if conversation.parent:
            form_data.append(
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="parent"\r\n\r\n{conversation.parent}\r\n'
            )
        
        # Add optional parameters
        if temperature is not None:
            form_data.append(
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="temperature"\r\n\r\n{temperature}\r\n'
            )
        
        if top_p is not None:
            form_data.append(
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="top_p"\r\n\r\n{top_p}\r\n'
            )
        
        form_data.append(f'--{boundary}--\r\n')
        data = "".join(form_data).encode()

        async with ClientSession(headers=headers) as session:
            async with session.post(
                cls.api_endpoint,
                data=data,
                proxy=proxy,
            ) as response:
                await raise_for_status(response)
                current_parent = None
                
                async for chunk in response.content:
                    if not chunk:
                        continue
                    decoded = chunk.decode(errors="ignore")
                    for line in decoded.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        
                        if isinstance(data, dict):
                            # Update the parental ID
                            if data.get("children"):
                                for child in data["children"]:
                                    if child.get("role") == "assistant":
                                        current_parent = child.get("id")
                                        break
                            
                            # We process content only from the assistant
                            if "message" in data and data.get("content"):
                                content = data["content"]
                                # Skip empty content blocks
                                if content.strip():
                                    yield content
                            
                            # Processing the final response
                            if data.get("final") or data.get("finish_reason") == "stop":
                                if current_parent:
                                    conversation.parent = current_parent
                                
                                # Add a message to the story
                                conversation.messages.extend([
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": content}
                                ])
                                
                                if return_conversation:
                                    yield conversation
                                
                                yield FinishReason("stop")
                                return
