from __future__ import annotations

import random
import uuid

from ..typing import AsyncResult, Messages, Cookies
from ..requests import StreamSession, raise_for_status, sse_stream
from ..cookies import get_cookies
from ..providers.response import ProviderInfo
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .. import debug

class Perplexity(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Perplexity"
    url = "https://www.perplexity.ai"
    cookie_domain = ".perplexity.ai"
    working = True
    active_by_default = True

    default_model = "auto"
    models = [
        default_model,
        "turbo",
        "pplx_pro",
        "gpt-5",
    ]
    model_aliases = {
        "gpt-5": "gpt5",
    }

    _user_id = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        cookies: Cookies = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = cls.default_model
        if cookies is None:
            cookies = get_cookies(cls.cookie_domain, False)
        else:
            cls._user_id = None

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

        async with StreamSession(headers=headers, cookies=cookies, proxy=proxy, impersonate="chrome") as session:
            if cls._user_id is None:
                async with session.get(f"{cls.url}/api/auth/session") as response:
                    await raise_for_status(response)
                    user = await response.json()
                    cls._user_id = user.get("user", {}).get("id")
                    debug.log(f"Perplexity user id: {cls._user_id}")
            if model == "auto":
                model = "pplx_pro" if cls._user_id else "turbo"
            yield ProviderInfo(**cls.get_dict(), model=model)
            if model in cls.model_aliases:
                model = cls.model_aliases[model]
            if cls._user_id is None:
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
            else:
                data = {
                    "params": {
                        "last_backend_uuid": None,
                        "read_write_token": "457a2d3d-c53f-4065-8554-7645a36fc220",
                        "attachments": [],
                        "language": "en-US",
                        "timezone": "America/New_York",
                        "search_focus": "internet",
                        "sources": ["web"],
                        "frontend_uuid": frontend_uuid,
                        "mode": "copilot",
                        "model_preference": "gpt5",
                        "is_related_query": False,
                        "is_sponsored": False,
                        "visitor_id": visitor_id,
                        "user_nextauth_id": cls._user_id,
                        "prompt_source": "user",
                        "query_source":"followup",
                        "is_incognito": False,
                        "time_from_first_type": random.randint(0, 1000),
                        "local_search_enabled": False,
                        "use_schematized_api": True,
                        "send_back_text_in_streaming_api": False,
                        "supported_block_use_cases": ["answer_modes", "media_items", "knowledge_cards", "inline_entity_cards", "place_widgets", "finance_widgets", "sports_widgets", "shopping_widgets", "jobs_widgets", "search_result_widgets", "clarification_responses", "inline_images", "inline_assets", "inline_finance_widgets", "placeholder_cards", "diff_blocks", "inline_knowledge_cards", "entity_group_v2", "refinement_filters", "canvas_mode"],
                        "client_coordinates": None,
                        "mentions": [],
                        "skip_search_enabled": True,
                        "is_nav_suggestions_disabled": False,
                        "followup_source": "link",
                        "version": "2.18"
                    },
                    "query_str": query
                }
            async with session.post(
                f"{cls.url}/rest/sse/perplexity_ask",
                json=data,
            ) as response:
                await raise_for_status(response)
                full_response = ""
                last_response = ""
                async for json_data in sse_stream(response):
                    for block in json_data.get("blocks", []):
                        for patch in block.get("diff_block", {}).get("patches", []):
                            value = patch.get("value", "")
                            value = value.get("answer", "") if isinstance(value, dict) else value
                            if value:
                                if value.startswith(full_response):
                                    value = value[len(full_response):]
                                if value.startswith(last_response):
                                    value = value[len(last_response):]
                                last_response = value
                                full_response += value
                                yield value
