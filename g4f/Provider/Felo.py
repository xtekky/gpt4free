from __future__ import annotations

import json
import uuid
import re

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from ..requests import StreamSession

class Felo(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Provider for felo.ai.
    """
    label = "Felo"
    url = "https://felo.ai"
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = False
    supports_message_history = False

    default_model = "felo-chat"
    
    # Mapping of g4f model names to Felo categories
    model_aliases = {
        "felo-chat": "chat",
        "felo-search": "google",
        "felo-scholar": "scholar",
        "felo-social": "social",
        "felo-document": "document"
    }
    
    models = list(model_aliases.keys())

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        prompt = messages[-1]["content"]
        search_uuid = str(uuid.uuid4())
        
        # Determine the category based on the requested model
        category = cls.model_aliases.get(model, "chat")

        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Referer": f"{cls.url}/search?q=hello",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
        }

        payload = {
            "query": prompt,
            "search_uuid": search_uuid,
            "lang": "",
            "agent_lang": "en",
            "search_options": {"langcode": "en-US"},
            "search_video": True,
            "query_from": "default",
            "category": category,
            "model": "",
            "auto_routing": True,
            "mode": "concise",
            "device_id": str(uuid.uuid4().hex),
            "source_message_rid": "",
            "documents": [],
            "document_action": "",
            "slides_source": {"type": "ask_question", "files": {}},
            "slide_template_uid": "",
            "selected_resource_ids": [],
            "process_id": search_uuid,
            "stream_protocol": "message_center_v1",
            "enable_task_state": True
        }

        async with StreamSession(
            headers=headers,
            impersonate="chrome",
            proxy=proxy,
            timeout=120
        ) as session:
            # 1. Start the thread and get the stream key
            threads_url = f"{cls.url}/api-proxy/main/search/threads"
            async with session.post(threads_url, json=payload) as response:
                await raise_for_status(response)
                try:
                    res_json = await response.json()
                except Exception:
                    raise RuntimeError(f"Failed to parse JSON response from {threads_url}")
                
                stream_key = res_json.get("stream_key")
                if not stream_key:
                    raise RuntimeError("Failed to get stream_key from Felo response")

            # 2. Connect to the SSE stream
            stream_url = f"{cls.url}/api/message/v1/stream/{stream_key}?offset=0"
            async with session.get(stream_url) as stream_res:
                await raise_for_status(stream_res)
                
                previous_text = ""
                sources_to_yield = None
                async for line in stream_res.iter_lines():
                    line = line.decode("utf-8").strip()
                    if line.startswith("data:{"):
                        try:
                            # Safely load the JSON inside the data block
                            data = json.loads(line[5:])
                            if "content" in data:
                                content_str = data["content"]
                                if isinstance(content_str, str):
                                    content_json = json.loads(content_str)
                                    
                                    # Handle answer types
                                    if content_json.get("data", {}).get("type") == "answer":
                                        text = content_json.get("data", {}).get("data", {}).get("text", "")

                                        if text.startswith(previous_text):
                                            # Yield only the new part of the text
                                            new_part = text[len(previous_text):]
                                            if new_part:
                                                yield new_part
                                                previous_text = text
                                        else:
                                            # If it doesn't strictly start with previous text, yield the whole text and update
                                            yield text
                                            previous_text = text
                                            
                                    # Handle sources for Web UI formatting
                                    elif content_json.get("data", {}).get("type") == "final_contexts":
                                        sources_list = content_json.get("data", {}).get("data", {}).get("sources", [])
                                        if sources_list:
                                            # format to match g4f standard sources
                                            formatted_sources = [{"url": s.get("link"), "title": s.get("title")} for s in sources_list if s.get("link")]
                                            if formatted_sources:
                                                from ..providers.response import Sources
                                                sources_to_yield = Sources(formatted_sources)
                        except Exception:
                            # Ignore malformed JSON or other parsing errors in the stream
                            continue
                            
                # Yield sources at the very end so they appear at the bottom of the response
                if sources_to_yield:
                    yield sources_to_yield
