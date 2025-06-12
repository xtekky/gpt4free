from __future__ import annotations
import json
from uuid import uuid4
from aiohttp import ClientSession
from ...typing import AsyncResult, Messages, MediaListType
from ...image import to_bytes, is_accepted_format, to_data_uri
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...requests.raise_for_status import raise_for_status
from ...providers.response import FinishReason, JsonConversation
from ..helper import format_prompt, get_last_user_message, format_media_prompt
from ...tools.media import merge_media


class Conversation(JsonConversation):
    x_anonymous_user_id: str = None

    def __init__(self, model: str):
        super().__init__()  # Ensure parent class is initialized
        self.model = model
        self.messages = []  # Instance-specific list
        self.parent = None  # Initialize parent as instance attribute
        if not self.x_anonymous_user_id:
            self.x_anonymous_user_id = str(uuid4())


class AllenAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Ai2 Playground"
    url = "https://playground.allenai.org"
    login_url = None
    api_endpoint = "https://olmo-api.allen.ai/v4/message/stream"
    
    working = False
    needs_auth = False
    use_nodriver = False
    supports_stream = True
    supports_system_message = False
    supports_message_history = True

    default_model = 'tulu3-405b'
    default_vision_model = 'mm-olmo-uber-model-v4-synthetic'
    vision_models = [default_vision_model]
    # Map models to their required hosts
    model_hosts = {
        default_model: "inferd",
        "OLMo-2-1124-13B-Instruct": "modal",
        "tulu-3-1-8b": "modal",
        "Llama-3-1-Tulu-3-70B": "modal",
        "olmoe-0125": "modal",
        "olmo-2-0325-32b-instruct": "modal",
        "mm-olmo-uber-model-v4-synthetic": "modal",
    }

    models = list(model_hosts.keys())
    
    model_aliases = {
        "tulu-3-405b": default_model,
        "olmo-1-7b": "olmoe-0125",
        "olmo-2-13b": "OLMo-2-1124-13B-Instruct",
        "olmo-2-32b": "olmo-2-0325-32b-instruct",
        "tulu-3-1-8b": "tulu-3-1-8b",
        "tulu-3-70b": "Llama-3-1-Tulu-3-70B",
        "llama-3.1-405b": "tulu3-405b",
        "llama-3.1-8b": "tulu-3-1-8b",
        "llama-3.1-70b": "Llama-3-1-Tulu-3-70B",
        "olmo-4-synthetic": "mm-olmo-uber-model-v4-synthetic",
    }


    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        host: str = None,
        private: bool = True,
        top_p: float = None,
        temperature: float = None,
        conversation: Conversation = None,
        return_conversation: bool = True,
        media: MediaListType = None,
        **kwargs
    ) -> AsyncResult:
        actual_model = cls.get_model(model)
        
        prompt = format_prompt(messages) if conversation is None else get_last_user_message(messages)
        
        # Determine the correct host for the model
        if host is None:
            # Use model-specific host from model_hosts dictionary
            host = cls.model_hosts[actual_model]
        
        # Initialize or update conversation
        # For mm-olmo-uber-model-v4-synthetic, always create a new conversation
        if conversation is None or actual_model == 'mm-olmo-uber-model-v4-synthetic':
            conversation = Conversation(actual_model)
        
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
        if hasattr(conversation, 'parent') and conversation.parent:
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
        
        # Always create a new conversation when an image is attached to avoid 403 errors
        if media is not None and len(media) > 0:
            conversation = Conversation(actual_model)
        
        # For each image in the media list (using merge_media to handle different formats)
        for image, image_name in merge_media(media, messages):
            image_bytes = to_bytes(image)
            form_data.extend([
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="files"; filename="{image_name}"\r\n'
                f'Content-Type: {is_accepted_format(image_bytes)}\r\n\r\n'
            ])
            form_data.append(image_bytes.decode('latin1'))
            form_data.append('\r\n')

        form_data.append(f'--{boundary}--\r\n')
        data = "".join(form_data).encode('latin1')

        async with ClientSession(headers=headers) as session:
            async with session.post(
                cls.api_endpoint,
                data=data,
                proxy=proxy,
            ) as response:
                await raise_for_status(response)
                current_parent = None
                
                async for line in response.content:
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
                                    # Ensure the parent attribute exists before setting it
                                    if not hasattr(conversation, 'parent'):
                                        setattr(conversation, 'parent', None)
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
