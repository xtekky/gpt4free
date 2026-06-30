import json
import random
import uuid
import re
import hashlib
from aiohttp import ClientSession, FormData
from typing import AsyncGenerator

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..providers.response import ImageResponse, Sources
from ..image import to_bytes

class DeepAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "DeepAI"
    url = "https://deepai.org/chat"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "standard"
    models = ["standard", "online", "gemma-4", "gemini-2.5-flash-lite", "deepseek-v3.2", "image"]
    model_aliases = {"gpt-4": "standard"}
    
    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        return cls.default_model

    @classmethod
    async def upload_file(cls, session: ClientSession, headers: dict, file_data: bytes, filename: str, proxy: str = None) -> str:
        import mimetypes
        content_type, _ = mimetypes.guess_type(filename)
        content_type = content_type or "image/png"
        
        data = FormData()
        data.add_field("file", file_data, filename=filename, content_type=content_type)
        upload_headers = {k: v for k, v in headers.items() if k.lower() not in ["content-type", "api-key"]}
        async with session.post(
            "https://api.deepai.org/chat_attachments/upload",
            headers=upload_headers,
            data=data,
            proxy=proxy
        ) as response:
            if not response.ok:
                error_text = await response.text()
                raise RuntimeError(f"Failed to upload file: {response.status} {error_text}")
            res_json = await response.json()
            if res_json.get("success"):
                return res_json["attachment"]["uuid"]
            raise RuntimeError(f"Failed to upload file: {res_json}")

    @classmethod
    def generate_api_key(cls, user_agent: str) -> str:
        myrandomstr = str(round(random.random() * 100000000000))
        
        def myhashfunction(input_str: str) -> str:
            return hashlib.md5(input_str.encode('utf-8')).hexdigest()[::-1]
            
        hash1 = myhashfunction(user_agent + myrandomstr + 'hackers_become_a_little_stinkier_every_time_they_hack')
        hash2 = myhashfunction(user_agent + hash1)
        hash3 = myhashfunction(user_agent + hash2)
        
        return f"tryit-{myrandomstr}-{hash3}"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        api_key: str = None,
        **kwargs
    ) -> AsyncGenerator:
        model = cls.get_model(model)
        
        user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
        api_key = cls.generate_api_key(user_agent)
        
        headers = {
            "api-key": api_key,
            "user-agent": user_agent,
            "origin": "https://deepai.org",
            "referer": "https://deepai.org/chat",
        }
        
        # Check if we should directly generate an image if the user is just asking for it
        # Actually we can just rely on the LLM to trigger the tool.
        
        async with ClientSession() as session:
            # Direct image generation bypass
            if model == "image":
                prompt = messages[-1]["content"] if messages else "A beautiful image"
                
                img_data = FormData()
                img_data.add_field("text", prompt)
                img_data.add_field("generation_source", "chat")
                img_data.add_field("width", "640")
                img_data.add_field("height", "640")
                img_data.add_field("image_generator_version", "hd")
                img_data.add_field("quality", "true")
                
                async with session.post(
                    "https://api.deepai.org/api/text2img",
                    headers=headers,
                    data=img_data,
                    proxy=proxy
                ) as img_resp:
                    img_resp.raise_for_status()
                    img_res_json = await img_resp.json()
                    if "output_url" in img_res_json:
                        yield ImageResponse(img_res_json["output_url"], alt=prompt)
                return

            # Extract files from kwargs (g4f standard for image uploads)
            attachment_uuids = []
            images = kwargs.get("images", [])
            if images:
                for img, name in images:
                    file_data = to_bytes(img)
                    file_uuid = await cls.upload_file(session, headers, file_data, name, proxy=proxy)
                    attachment_uuids.append(file_uuid)
                    
            # Also extract any images hidden directly in messages (if structured that way)
            for msg in messages:
                if "image" in msg and msg.get("image"):
                    file_data = to_bytes(msg["image"])
                    name = msg.get("filename", "image.png")
                    file_uuid = await cls.upload_file(session, headers, file_data, name, proxy=proxy)
                    attachment_uuids.append(file_uuid)

            # Update messages to include attachment_uuids and strip non-serializable fields
            cleaned_messages = []
            for msg in messages:
                new_msg = dict(msg)
                new_msg.pop("image", None)
                new_msg.pop("filename", None)
                cleaned_messages.append(new_msg)

            if attachment_uuids:
                for i in range(len(cleaned_messages) - 1, -1, -1):
                    if cleaned_messages[i]["role"] == "user":
                        cleaned_messages[i]["attachment_uuids"] = attachment_uuids
                        break

            data_dict = {
                "chat_style": "chat",
                "chatHistory": json.dumps(cleaned_messages),
                "model": model,
                "session_uuid": str(uuid.uuid4()),
                "sensitivity_request_id": str(uuid.uuid4()),
                "hacker_is_stinky": "very_stinky",
                "enabled_tools": json.dumps(["image_generator", "image_editor"], separators=(",", ":"))
            }
            
            if attachment_uuids:
                data_dict["attachment_uuids"] = json.dumps(attachment_uuids)

            data = FormData(data_dict)

            async with session.post(
                "https://api.deepai.org/hacking_is_a_serious_crime",
                headers=headers,
                data=data,
                proxy=proxy
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to chat: {response.status} {error_text}")
                
                buffer = ""
                async for chunk in response.content.iter_any():
                    if chunk:
                        chunk_text = chunk.decode(errors="ignore")
                        if "\x1c" in chunk_text or "\x1c" in buffer:
                            buffer += chunk_text
                        else:
                            yield chunk_text
                
                if not buffer:
                    return

                parts = buffer.split("\x1c")
                if parts[0].strip():
                    yield parts[0]
                    
                for part in parts[1:]:
                    if not part.strip():
                        continue
                    try:
                        data = json.loads(part)
                        if isinstance(data, list):
                            # Web search sources
                            yield Sources(data)
                        elif isinstance(data, dict) and data.get("type") == "generated_image":
                            image_prompt = data.get("prompt")
                            try:
                                img_data = FormData()
                                img_data.add_field("text", image_prompt)
                                img_data.add_field("generation_source", "chat")
                                img_data.add_field("width", "640")
                                img_data.add_field("height", "640")
                                img_data.add_field("image_generator_version", "hd")
                                img_data.add_field("quality", "true")
                                
                                async with session.post(
                                    "https://api.deepai.org/api/text2img",
                                    headers=headers,
                                    data=img_data,
                                    proxy=proxy
                                ) as img_resp:
                                    img_resp.raise_for_status()
                                    img_res_json = await img_resp.json()
                                    if "output_url" in img_res_json:
                                        yield ImageResponse(img_res_json["output_url"], alt=image_prompt)
                            except Exception as e:
                                pass
                        else:
                            # Unrecognized JSON, yield as text
                            yield "\x1c" + part
                    except json.JSONDecodeError:
                        # Not valid JSON, just yield the text
                        yield "\x1c" + part
