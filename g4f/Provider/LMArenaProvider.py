from __future__ import annotations

import json
import uuid
import requests

from ..typing import AsyncResult, Messages, MediaListType
from ..requests import StreamSession, FormData, raise_for_status
from ..providers.response import FinishReason, JsonConversation
from ..tools.media import merge_media
from ..image import to_bytes, is_accepted_format
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin
from .helper import get_last_user_message
from .. import debug

class LMArenaProvider(AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin):
    label = "LM Arena"
    url = "https://lmarena.ai"
    api_endpoint = "/queue/join?"
    working = False

    default_model = "gpt-4o"
    model_aliases = {default_model: "chatgpt-4o-latest-20250326"}
    models = [
		default_model,
		"o3-2025-04-16",
		"o4-mini-2025-04-16",
		"gpt-4.1-2025-04-14",
		"gemini-2.5-pro-exp-03-25",
		"llama-4-maverick-03-26-experimental",
		"grok-3-preview-02-24",
		"claude-3-7-sonnet-20250219",
		"claude-3-7-sonnet-20250219-thinking-32k",
		"deepseek-v3-0324",
		"llama-4-maverick-17b-128e-instruct",
		"gpt-4.1-mini-2025-04-14",
		"gpt-4.1-nano-2025-04-14",
		"gemini-2.0-flash-thinking-exp-01-21",
		"gemini-2.0-flash-001",
		"gemini-2.0-flash-lite-preview-02-05",
		"gemma-3-27b-it",
		"gemma-3-12b-it",
		"gemma-3-4b-it",
		"deepseek-r1",
		"claude-3-5-sonnet-20241022",
		"o3-mini",
		"llama-3.3-70b-instruct",
		"gpt-4o-mini-2024-07-18",
		"gpt-4o-2024-11-20",
		"gpt-4o-2024-08-06",
		"gpt-4o-2024-05-13",
		"command-a-03-2025",
		"qwq-32b",
		"p2l-router-7b",
		"claude-3-5-haiku-20241022",
		"claude-3-5-sonnet-20240620",
		"doubao-1.5-pro-32k-250115",
		"doubao-1.5-vision-pro-32k-250115",
		"mistral-small-24b-instruct-2501",
		"phi-4",
		"amazon-nova-pro-v1.0",
		"amazon-nova-lite-v1.0",
		"amazon-nova-micro-v1.0",
		"cobalt-exp-beta-v3",
		"cobalt-exp-beta-v4",
		"qwen-max-2025-01-25",
		"qwen-plus-0125-exp",
		"qwen2.5-vl-32b-instruct",
		"qwen2.5-vl-72b-instruct",
		"gemini-1.5-pro-002",
		"gemini-1.5-flash-002",
		"gemini-1.5-flash-8b-001",
		"gemini-1.5-pro-001",
		"gemini-1.5-flash-001",
		"llama-3.1-405b-instruct-bf16",
		"llama-3.3-nemotron-49b-super-v1",
		"llama-3.1-nemotron-ultra-253b-v1",
		"llama-3.1-nemotron-70b-instruct",
		"llama-3.1-70b-instruct",
		"llama-3.1-8b-instruct",
		"hunyuan-standard-2025-02-10",
		"hunyuan-large-2025-02-10",
		"hunyuan-standard-vision-2024-12-31",
		"hunyuan-turbo-0110",
		"hunyuan-turbos-20250226",
		"mistral-large-2411",
		"pixtral-large-2411",
		"mistral-large-2407",
		"llama-3.1-nemotron-51b-instruct",
		"granite-3.1-8b-instruct",
		"granite-3.1-2b-instruct",
		"step-2-16k-exp-202412",
		"step-2-16k-202502",
		"step-1o-vision-32k-highres",
		"yi-lightning",
		"glm-4-plus",
		"glm-4-plus-0111",
		"jamba-1.5-large",
		"jamba-1.5-mini",
		"gemma-2-27b-it",
		"gemma-2-9b-it",
		"gemma-2-2b-it",
		"eureka-chatbot",
		"claude-3-haiku-20240307",
		"claude-3-sonnet-20240229",
		"claude-3-opus-20240229",
		"nemotron-4-340b",
		"llama-3-70b-instruct",
		"llama-3-8b-instruct",
		"qwen2.5-plus-1127",
		"qwen2.5-coder-32b-instruct",
		"qwen2.5-72b-instruct",
		"qwen-max-0919",
		"qwen-vl-max-1119",
		"qwen-vl-max-0809",
		"llama-3.1-tulu-3-70b",
		"olmo-2-0325-32b-instruct",
		"gpt-3.5-turbo-0125",
		"reka-core-20240904",
		"reka-flash-20240904",
		"c4ai-aya-expanse-32b",
		"c4ai-aya-expanse-8b",
		"c4ai-aya-vision-32b",
		"command-r-plus-08-2024",
		"command-r-08-2024",
		"codestral-2405",
		"mixtral-8x22b-instruct-v0.1",
		"mixtral-8x7b-instruct-v0.1",
		"pixtral-12b-2409",
		"ministral-8b-2410"
	]
    vision_models = [
		"o3-2025-04-16",
		"o4-mini-2025-04-16",
		"gpt-4.1-2025-04-14",
		"gemini-2.5-pro-exp-03-25",
		"claude-3-7-sonnet-20250219",
		"claude-3-7-sonnet-20250219-thinking-32k",
		"llama-4-maverick-17b-128e-instruct",
		"gpt-4.1-mini-2025-04-14",
		"gpt-4.1-nano-2025-04-14",
		"gemini-2.0-flash-thinking-exp-01-21",
		"gemini-2.0-flash-001",
		"gemini-2.0-flash-lite-preview-02-05",
		"claude-3-5-sonnet-20241022",
		"gpt-4o-mini-2024-07-18",
		"gpt-4o-2024-11-20",
		"gpt-4o-2024-08-06",
		"gpt-4o-2024-05-13",
		"claude-3-5-sonnet-20240620",
		"doubao-1.5-vision-pro-32k-250115",
		"amazon-nova-pro-v1.0",
		"amazon-nova-lite-v1.0",
		"qwen2.5-vl-32b-instruct",
		"qwen2.5-vl-72b-instruct",
		"gemini-1.5-pro-002",
		"gemini-1.5-flash-002",
		"gemini-1.5-flash-8b-001",
		"gemini-1.5-pro-001",
		"gemini-1.5-flash-001",
		"hunyuan-standard-vision-2024-12-31",
		"pixtral-large-2411",
		"step-1o-vision-32k-highres",
		"claude-3-haiku-20240307",
		"claude-3-sonnet-20240229",
		"claude-3-opus-20240229",
		"qwen-vl-max-1119",
		"qwen-vl-max-0809",
		"reka-core-20240904",
		"reka-flash-20240904",
		"c4ai-aya-vision-32b",
		"pixtral-12b-2409"
	]

    _args: dict = None

    @classmethod
    def get_models(cls) -> list[str]:
        if not cls.models:
            url = "https://storage.googleapis.com/public-arena-no-cors/p2l-explorer/data/overall/arena.json"
            data = requests.get(url).json()
            cls.models = [model[0] for model in data["leaderboard"]]
        return cls.models

    @classmethod
    def _build_payloads(cls, model_id: str, session_hash: str, text: str, files: list, max_tokens: int, temperature: float, top_p: float):
        first_payload = {
            "data": [
                None,
                model_id,
                {"text": text, "files": files},
                {
                    "text_models": [model_id],
                    "all_text_models": [model_id],
                    "vision_models": [],
                    "all_vision_models": [],
                    "image_gen_models": [],
                    "all_image_gen_models": [],
                    "search_models": [],
                    "all_search_models": [],
                    "models": [model_id],
                    "all_models": [model_id],
                    "arena_type": "text-arena"
                }
            ],
            "event_data": None,
            "fn_index": 117,
            "trigger_id": 159,
            "session_hash": session_hash
        }

        second_payload = {
            "data": [],
            "event_data": None,
            "fn_index": 118,
            "trigger_id": 159,
            "session_hash": session_hash
        }

        third_payload = {
            "data": [None, temperature, top_p, max_tokens],
            "event_data": None,
            "fn_index": 119,
            "trigger_id": 159,
            "session_hash": session_hash
        }

        return first_payload, second_payload, third_payload

    @classmethod
    def _build_second_payloads(cls, model_id: str, session_hash: str, text: str, max_tokens: int, temperature: float, top_p: float):
        first_payload = {
            "data":[None,model_id,text,{
                "text_models":[model_id],
                "all_text_models":[model_id],
                "vision_models":[],
                "image_gen_models":[],
                "all_image_gen_models":[],
                "search_models":[],
                "all_search_models":[],
                "models":[model_id],
                "all_models":[model_id],
                "arena_type":"text-arena"}],
            "event_data": None,
            "fn_index": 120,
            "trigger_id": 157,
            "session_hash": session_hash
        }

        second_payload = {
            "data": [],
            "event_data": None,
            "fn_index": 121,
            "trigger_id": 157,
            "session_hash": session_hash
        }

        third_payload = {
            "data": [None, temperature, top_p, max_tokens],
            "event_data": None,
            "fn_index": 122,
            "trigger_id": 157,
            "session_hash": session_hash
        }

        return first_payload, second_payload, third_payload

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages,
        media: MediaListType = None,
        conversation: JsonConversation = None,
        return_conversation: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = cls.default_model
        if model in cls.model_aliases:
            model = cls.model_aliases[model]
        prompt = get_last_user_message(messages)
        new_conversation = False
        if conversation is None:
            conversation = JsonConversation(session_hash=str(uuid.uuid4()).replace("-", ""))
            new_conversation = True
        async with StreamSession(impersonate="chrome") as session:
            if new_conversation:
                media = list(merge_media(media, messages))
                if media:
                    data = FormData()
                    for i in range(len(media)):
                        media[i] = (to_bytes(media[i][0]), media[i][1])
                    for image, image_name in media:
                        data.add_field(f"files", image, filename=image_name)
                    async with session.post(f"{cls.url}/upload", params={"upload_id": conversation.session_hash}, data=data) as response:
                        await raise_for_status(response)
                        image_files = await response.json()
                    media = [{
                        "path": image_file,
                        "url": f"{cls.url}/file={image_file}",
                        "orig_name": media[i][1],
                        "size": len(media[i][0]),
                        "mime_type": is_accepted_format(media[i][0]),
                        "meta": {
                            "_type": "gradio.FileData"
                        }
                    } for i, image_file in enumerate(image_files)]
                first_payload, second_payload, third_payload = cls._build_payloads(model, conversation.session_hash, prompt, media, max_tokens, temperature, top_p)
            else:
                first_payload, second_payload, third_payload = cls._build_second_payloads(model, conversation.session_hash, prompt, max_tokens, temperature, top_p)

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # POST 1
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=first_payload, proxy=proxy, headers=headers) as response:
                await raise_for_status(response)

            # POST 2
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=second_payload, proxy=proxy, headers=headers) as response:
                await raise_for_status(response)

            # POST 3
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=third_payload, proxy=proxy, headers=headers) as response:
                await raise_for_status(response)

            # Long stream GET
            async def sse_stream():
                stream_url = f"{cls.url}/queue/data?session_hash={conversation.session_hash}"
                async with session.get(stream_url, headers={"Accept": "text/event-stream"}, proxy=proxy) as response:
                    await raise_for_status(response)
                    text_position = 0
                    count = 0
                    async for line in response.iter_lines():
                        if line.startswith(b"data: "):
                            try:
                                msg = json.loads(line[6:])
                            except Exception as e:
                                raise RuntimeError(f"Failed to decode JSON from stream: {line}", e)
                            if msg.get("msg") == "process_generating":
                                data = msg["output"]["data"][1]
                                if data:
                                    data = data[0]
                                if len(data) > 2:
                                    if isinstance(data[2], list):
                                        data[2] = data[2][-1]
                                    content = data[2][text_position:]
                                    if content.endswith("▌"):
                                        content = content[:-2]
                                    if content:
                                        count += 1
                                        yield count, content
                                        text_position += len(content)
                            elif msg.get("msg") == "close_stream":
                                break
                            elif msg.get("msg") not in ("process_completed", "process_starts", "estimation"):
                                debug.log(f"Unexpected message: {msg}")
            count = 0         
            async for count, chunk in sse_stream():
                yield chunk
            if count == 0:
                raise RuntimeError("No response from server.")
            if return_conversation:
                yield conversation
            if count == max_tokens:
                yield FinishReason("length")