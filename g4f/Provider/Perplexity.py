from __future__ import annotations

import json
import uuid

from ..typing import AsyncResult, Messages
from ..requests import StreamSession, raise_for_status
from ..errors import ResponseError
from ..providers.response import FinishReason
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class Perplexity(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Perplexity"
    url = "https://www.perplexity.ai"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "turbo"
    models = [
        default_model,
        "sonar",
        "sonar-pro",
    ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        # Generate UUIDs for request tracking
        frontend_uuid = str(uuid.uuid4())
        frontend_context_uuid = str(uuid.uuid4())
        visitor_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        
        headers = {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
            "x-perplexity-request-reason": "perplexity-query-state-provider",
            "x-request-id": request_id,
        }
        
        # Extract the last user message as the query
        query = ""
        for message in reversed(messages):
            if message["role"] == "user":
                query = message["content"]
                break
        
        # Prepare the request payload
        data = {
            "params": {
                "attachments": [],
                "language": "en-US",
                "timezone": "America/New_York",
                "search_focus": "internet",
                "sources": ["web"],
                "search_recency_filter": None,
                "frontend_uuid": frontend_uuid,
                "mode": "concise",
                "model_preference": model,
                "is_related_query": False,
                "is_sponsored": False,
                "visitor_id": visitor_id,
                "frontend_context_uuid": frontend_context_uuid,
                "prompt_source": "user",
                "query_source": "home",
                "is_incognito": False,
                "time_from_first_type": 0,
                "local_search_enabled": False,
                "use_schematized_api": True,
                "send_back_text_in_streaming_api": False,
                "supported_block_use_cases": [
                    "answer_modes",
                    "media_items",
                    "knowledge_cards",
                    "inline_entity_cards",
                    "place_widgets",
                    "finance_widgets",
                    "sports_widgets",
                    "shopping_widgets",
                    "jobs_widgets",
                    "search_result_widgets",
                    "clarification_responses",
                    "inline_images",
                    "inline_assets",
                    "inline_finance_widgets",
                    "placeholder_cards",
                    "diff_blocks",
                    "inline_knowledge_cards",
                    "entity_group_v2",
                    "refinement_filters",
                    "canvas_mode"
                ],
                "client_coordinates": None,
                "mentions": [],
                "dsl_query": query,
                "skip_search_enabled": False,
                "is_nav_suggestions_disabled": False,
                "always_search_override": False,
                "override_no_search": False,
                "comet_max_assistant_enabled": False,
                "version": "2.18"
            },
            "query_str": query
        }
        
        async with StreamSession(headers=headers, proxy=proxy, impersonate="chrome") as session:
            async with session.post(
                f"{cls.url}/rest/sse/perplexity_ask",
                json=data,
            ) as response:
                await raise_for_status(response)
                
                # Parse SSE stream
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                json_data = json.loads(data_str)
                                
                                # Handle different response types
                                if "text" in json_data:
                                    text = json_data["text"]
                                    if text:
                                        yield text
                                
                                # Check if response is final
                                if json_data.get("final", False):
                                    yield FinishReason("stop")
                                    break
                                    
                            except json.JSONDecodeError:
                                # Skip malformed JSON
                                continue
