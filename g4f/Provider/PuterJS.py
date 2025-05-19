from __future__ import annotations

from aiohttp import ClientSession
import json
import time
import re
import random
import json # Ensure json is imported if not already
import uuid # Import the uuid module
import asyncio
from typing import Optional, Dict, Any, List, Union # Assuming these are used elsewhere or can be pruned if not

from ..typing import AsyncResult, Messages, MediaListType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..providers.response import FinishReason, JsonConversation
from ..tools.media import merge_media
from ..image import to_data_uri, is_data_uri_an_image
from ..errors import RateLimitError, ModelNotFoundError
from .. import debug

class AuthData:
    """
    Stores authentication data for a specific model.
    """
    def __init__(self):
        self.auth_token: Optional[str] = None
        self.app_token: Optional[str] = None
        self.created_at: float = time.time()
        self.tokens_valid: bool = False
        self.rate_limited_until: float = 0
        
    def is_valid(self, expiration_time: int) -> bool:
        """Check if the tokens are still valid based on expiration time."""
        return (self.auth_token and self.app_token and 
                self.tokens_valid and 
                time.time() - self.created_at < expiration_time)
    
    def invalidate(self):
        """Mark tokens as invalid."""
        self.tokens_valid = False
    
    def set_rate_limit(self, seconds: int = 60):
        """Set rate limit expiry time."""
        self.rate_limited_until = time.time() + seconds
    
    def is_rate_limited(self) -> bool:
        """Check if currently rate limited."""
        return time.time() < self.rate_limited_until


class Conversation(JsonConversation):
    """
    Stores conversation state and authentication tokens for PuterJS provider.
    Maintains separate authentication data for different models.
    """
    message_history: Messages = []
    
    def __init__(self, model: str):
        self.model = model
        self.message_history = []
        # Authentication tokens by specific model
        self._auth_data: Dict[str, AuthData] = {}
    
    def get_auth_for_model(self, model: str, provider: PuterJS) -> AuthData:
        """Get the authentication data for a specific model."""
        # Create auth data for this model if it doesn't exist
        if model not in self._auth_data:
            self._auth_data[model] = AuthData()
        
        return self._auth_data[model]
    
    # Override get_dict to exclude auth_data
    def get_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation for JSON serialization."""
        return {
            "model": self.model,
            "message_history": self.message_history
        }
    
    # Override __getstate__ for pickle serialization
    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization."""
        # Include auth_data for pickle serialization
        state = self.__dict__.copy()
        return state
    
    # Override __setstate__ for pickle deserialization
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        self.__dict__.update(state)


class PuterJS(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Puter.js"
    url = "https://docs.puter.com/playground"
    api_endpoint = "https://api.puter.com/drivers/call"
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o'
    default_vision_model = default_model
    # https://api.puter.com/puterai/chat/models/
    openai_models = [default_vision_model,"gpt-4o-mini", "o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.5-preview"]
    claude_models = ["claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"]
    mistral_models = ["ministral-3b-2410","ministral-3b-latest","ministral-8b-2410","ministral-8b-latest","open-mistral-7b","mistral-tiny","mistral-tiny-2312","open-mixtral-8x7b","mistral-small","mistral-small-2312","open-mixtral-8x22b","open-mixtral-8x22b-2404","mistral-large-2411","mistral-large-latest","pixtral-large-2411","pixtral-large-latest","mistral-large-pixtral-2411","codestral-2501","codestral-latest","codestral-2412","codestral-2411-rc5","pixtral-12b-2409","pixtral-12b","pixtral-12b-latest","mistral-small-2503","mistral-small-latest"]
    xai_models = ["grok-beta", "grok-vision-beta"]
    deepseek_models = ["deepseek-chat","deepseek-reasoner"]
    gemini_models = ["gemini-1.5-flash","gemini-2.0-flash"]
    openrouter_models = ["openrouter:meta-llama/llama-3.3-8b-instruct:free","openrouter:nousresearch/deephermes-3-mistral-24b-preview:free","openrouter:mistralai/mistral-medium-3","openrouter:google/gemini-2.5-pro-preview","openrouter:arcee-ai/caller-large","openrouter:arcee-ai/spotlight","openrouter:arcee-ai/maestro-reasoning","openrouter:arcee-ai/virtuoso-large","openrouter:arcee-ai/coder-large","openrouter:arcee-ai/virtuoso-medium-v2","openrouter:arcee-ai/arcee-blitz","openrouter:microsoft/phi-4-reasoning-plus:free","openrouter:microsoft/phi-4-reasoning-plus","openrouter:microsoft/phi-4-reasoning:free","openrouter:qwen/qwen3-0.6b-04-28:free","openrouter:inception/mercury-coder-small-beta","openrouter:qwen/qwen3-1.7b:free","openrouter:qwen/qwen3-4b:free","openrouter:opengvlab/internvl3-14b:free","openrouter:opengvlab/internvl3-2b:free","openrouter:deepseek/deepseek-prover-v2:free","openrouter:deepseek/deepseek-prover-v2","openrouter:meta-llama/llama-guard-4-12b","openrouter:qwen/qwen3-30b-a3b:free","openrouter:qwen/qwen3-30b-a3b","openrouter:qwen/qwen3-8b:free","openrouter:qwen/qwen3-8b","openrouter:qwen/qwen3-14b:free","openrouter:qwen/qwen3-14b","openrouter:qwen/qwen3-32b:free","openrouter:qwen/qwen3-32b","openrouter:qwen/qwen3-235b-a22b:free","openrouter:qwen/qwen3-235b-a22b","openrouter:tngtech/deepseek-r1t-chimera:free","openrouter:thudm/glm-z1-rumination-32b","openrouter:thudm/glm-z1-9b:free","openrouter:thudm/glm-4-9b:free","openrouter:microsoft/mai-ds-r1:free","openrouter:thudm/glm-z1-32b:free","openrouter:thudm/glm-z1-32b","openrouter:thudm/glm-4-32b:free","openrouter:thudm/glm-4-32b","openrouter:google/gemini-2.5-flash-preview","openrouter:google/gemini-2.5-flash-preview:thinking","openrouter:openai/o4-mini-high","openrouter:openai/o3","openrouter:openai/o4-mini","openrouter:shisa-ai/shisa-v2-llama3.3-70b:free","openrouter:qwen/qwen2.5-coder-7b-instruct","openrouter:openai/gpt-4.1","openrouter:openai/gpt-4.1-mini","openrouter:openai/gpt-4.1-nano","openrouter:eleutherai/llemma_7b","openrouter:alfredpros/codellama-7b-instruct-solidity","openrouter:arliai/qwq-32b-arliai-rpr-v1:free","openrouter:agentica-org/deepcoder-14b-preview:free","openrouter:moonshotai/kimi-vl-a3b-thinking:free","openrouter:x-ai/grok-3-mini-beta","openrouter:x-ai/grok-3-beta","openrouter:nvidia/llama-3.3-nemotron-super-49b-v1:free","openrouter:nvidia/llama-3.3-nemotron-super-49b-v1","openrouter:nvidia/llama-3.1-nemotron-ultra-253b-v1:free","openrouter:meta-llama/llama-4-maverick:free","openrouter:meta-llama/llama-4-maverick","openrouter:meta-llama/llama-4-scout:free","openrouter:meta-llama/llama-4-scout","openrouter:all-hands/openhands-lm-32b-v0.1","openrouter:mistral/ministral-8b","openrouter:deepseek/deepseek-v3-base:free","openrouter:scb10x/llama3.1-typhoon2-8b-instruct","openrouter:scb10x/llama3.1-typhoon2-70b-instruct","openrouter:bytedance-research/ui-tars-72b:free","openrouter:qwen/qwen2.5-vl-3b-instruct:free","openrouter:google/gemini-2.5-pro-exp-03-25","openrouter:qwen/qwen2.5-vl-32b-instruct:free","openrouter:qwen/qwen2.5-vl-32b-instruct","openrouter:deepseek/deepseek-chat-v3-0324:free","openrouter:deepseek/deepseek-chat-v3-0324","openrouter:featherless/qwerky-72b:free","openrouter:openai/o1-pro","openrouter:mistralai/mistral-small-3.1-24b-instruct:free","openrouter:mistralai/mistral-small-3.1-24b-instruct","openrouter:open-r1/olympiccoder-32b:free","openrouter:google/gemma-3-1b-it:free","openrouter:google/gemma-3-4b-it:free","openrouter:google/gemma-3-4b-it","openrouter:ai21/jamba-1.6-large","openrouter:ai21/jamba-1.6-mini","openrouter:google/gemma-3-12b-it:free","openrouter:google/gemma-3-12b-it","openrouter:cohere/command-a","openrouter:openai/gpt-4o-mini-search-preview","openrouter:openai/gpt-4o-search-preview","openrouter:rekaai/reka-flash-3:free","openrouter:google/gemma-3-27b-it:free","openrouter:google/gemma-3-27b-it","openrouter:thedrummer/anubis-pro-105b-v1","openrouter:thedrummer/skyfall-36b-v2","openrouter:microsoft/phi-4-multimodal-instruct","openrouter:perplexity/sonar-reasoning-pro","openrouter:perplexity/sonar-pro","openrouter:perplexity/sonar-deep-research","openrouter:deepseek/deepseek-r1-zero:free","openrouter:qwen/qwq-32b:free","openrouter:qwen/qwq-32b","openrouter:moonshotai/moonlight-16b-a3b-instruct:free","openrouter:nousresearch/deephermes-3-llama-3-8b-preview:free","openrouter:openai/gpt-4.5-preview","openrouter:google/gemini-2.0-flash-lite-001","openrouter:anthropic/claude-3.7-sonnet","openrouter:anthropic/claude-3.7-sonnet:thinking","openrouter:anthropic/claude-3.7-sonnet:beta","openrouter:perplexity/r1-1776","openrouter:mistralai/mistral-saba","openrouter:cognitivecomputations/dolphin3.0-r1-mistral-24b:free","openrouter:cognitivecomputations/dolphin3.0-mistral-24b:free","openrouter:meta-llama/llama-guard-3-8b","openrouter:openai/o3-mini-high","openrouter:deepseek/deepseek-r1-distill-llama-8b","openrouter:google/gemini-2.0-flash-001","openrouter:qwen/qwen-vl-plus","openrouter:aion-labs/aion-1.0","openrouter:aion-labs/aion-1.0-mini","openrouter:aion-labs/aion-rp-llama-3.1-8b","openrouter:qwen/qwen-vl-max","openrouter:qwen/qwen-turbo","openrouter:qwen/qwen2.5-vl-72b-instruct:free","openrouter:qwen/qwen2.5-vl-72b-instruct","openrouter:qwen/qwen-plus","openrouter:qwen/qwen-max","openrouter:openai/o3-mini","openrouter:deepseek/deepseek-r1-distill-qwen-1.5b","openrouter:mistralai/mistral-small-24b-instruct-2501:free","openrouter:mistralai/mistral-small-24b-instruct-2501","openrouter:deepseek/deepseek-r1-distill-qwen-32b:free","openrouter:deepseek/deepseek-r1-distill-qwen-32b","openrouter:deepseek/deepseek-r1-distill-qwen-14b:free","openrouter:deepseek/deepseek-r1-distill-qwen-14b","openrouter:perplexity/sonar-reasoning","openrouter:perplexity/sonar","openrouter:liquid/lfm-7b","openrouter:liquid/lfm-3b","openrouter:deepseek/deepseek-r1-distill-llama-70b:free","openrouter:deepseek/deepseek-r1-distill-llama-70b","openrouter:deepseek/deepseek-r1:free","openrouter:deepseek/deepseek-r1","openrouter:minimax/minimax-01","openrouter:mistralai/codestral-2501","openrouter:microsoft/phi-4","openrouter:deepseek/deepseek-chat:free","openrouter:deepseek/deepseek-chat","openrouter:sao10k/l3.3-euryale-70b","openrouter:openai/o1","openrouter:eva-unit-01/eva-llama-3.33-70b","openrouter:x-ai/grok-2-vision-1212","openrouter:x-ai/grok-2-1212","openrouter:cohere/command-r7b-12-2024","openrouter:google/gemini-2.0-flash-exp:free","openrouter:meta-llama/llama-3.3-70b-instruct:free","openrouter:meta-llama/llama-3.3-70b-instruct","openrouter:amazon/nova-lite-v1","openrouter:amazon/nova-micro-v1","openrouter:amazon/nova-pro-v1","openrouter:qwen/qwq-32b-preview","openrouter:eva-unit-01/eva-qwen-2.5-72b","openrouter:openai/gpt-4o-2024-11-20","openrouter:mistralai/mistral-large-2411","openrouter:mistralai/mistral-large-2407","openrouter:mistralai/pixtral-large-2411","openrouter:x-ai/grok-vision-beta","openrouter:infermatic/mn-inferor-12b","openrouter:qwen/qwen-2.5-coder-32b-instruct:free","openrouter:qwen/qwen-2.5-coder-32b-instruct","openrouter:raifle/sorcererlm-8x22b","openrouter:eva-unit-01/eva-qwen-2.5-32b","openrouter:thedrummer/unslopnemo-12b","openrouter:anthropic/claude-3.5-haiku:beta","openrouter:anthropic/claude-3.5-haiku","openrouter:anthropic/claude-3.5-haiku-20241022:beta","openrouter:anthropic/claude-3.5-haiku-20241022","openrouter:neversleep/llama-3.1-lumimaid-70b","openrouter:anthracite-org/magnum-v4-72b","openrouter:anthropic/claude-3.5-sonnet:beta","openrouter:anthropic/claude-3.5-sonnet","openrouter:x-ai/grok-beta","openrouter:mistralai/ministral-8b","openrouter:mistralai/ministral-3b","openrouter:qwen/qwen-2.5-7b-instruct:free","openrouter:qwen/qwen-2.5-7b-instruct","openrouter:nvidia/llama-3.1-nemotron-70b-instruct","openrouter:inflection/inflection-3-productivity","openrouter:inflection/inflection-3-pi","openrouter:google/gemini-flash-1.5-8b","openrouter:thedrummer/rocinante-12b","openrouter:anthracite-org/magnum-v2-72b","openrouter:liquid/lfm-40b","openrouter:meta-llama/llama-3.2-3b-instruct:free","openrouter:meta-llama/llama-3.2-3b-instruct","openrouter:meta-llama/llama-3.2-1b-instruct:free","openrouter:meta-llama/llama-3.2-1b-instruct","openrouter:meta-llama/llama-3.2-90b-vision-instruct","openrouter:meta-llama/llama-3.2-11b-vision-instruct:free","openrouter:meta-llama/llama-3.2-11b-vision-instruct","openrouter:qwen/qwen-2.5-72b-instruct:free","openrouter:qwen/qwen-2.5-72b-instruct","openrouter:neversleep/llama-3.1-lumimaid-8b","openrouter:openai/o1-preview","openrouter:openai/o1-preview-2024-09-12","openrouter:openai/o1-mini","openrouter:openai/o1-mini-2024-09-12","openrouter:mistralai/pixtral-12b","openrouter:cohere/command-r-plus-08-2024","openrouter:cohere/command-r-08-2024","openrouter:qwen/qwen-2.5-vl-7b-instruct:free","openrouter:qwen/qwen-2.5-vl-7b-instruct","openrouter:sao10k/l3.1-euryale-70b","openrouter:microsoft/phi-3.5-mini-128k-instruct","openrouter:nousresearch/hermes-3-llama-3.1-70b","openrouter:nousresearch/hermes-3-llama-3.1-405b","openrouter:openai/chatgpt-4o-latest","openrouter:sao10k/l3-lunaris-8b","openrouter:aetherwiing/mn-starcannon-12b","openrouter:openai/gpt-4o-2024-08-06","openrouter:meta-llama/llama-3.1-405b:free","openrouter:meta-llama/llama-3.1-405b","openrouter:nothingiisreal/mn-celeste-12b","openrouter:perplexity/llama-3.1-sonar-small-128k-online","openrouter:perplexity/llama-3.1-sonar-large-128k-online","openrouter:meta-llama/llama-3.1-8b-instruct:free","openrouter:meta-llama/llama-3.1-8b-instruct","openrouter:meta-llama/llama-3.1-405b-instruct","openrouter:meta-llama/llama-3.1-70b-instruct","openrouter:mistralai/codestral-mamba","openrouter:mistralai/mistral-nemo:free","openrouter:mistralai/mistral-nemo","openrouter:openai/gpt-4o-mini","openrouter:openai/gpt-4o-mini-2024-07-18","openrouter:google/gemma-2-27b-it","openrouter:alpindale/magnum-72b","openrouter:google/gemma-2-9b-it:free","openrouter:google/gemma-2-9b-it","openrouter:01-ai/yi-large","openrouter:ai21/jamba-instruct","openrouter:anthropic/claude-3.5-sonnet-20240620:beta","openrouter:anthropic/claude-3.5-sonnet-20240620","openrouter:sao10k/l3-euryale-70b","openrouter:cognitivecomputations/dolphin-mixtral-8x22b","openrouter:qwen/qwen-2-72b-instruct","openrouter:mistralai/mistral-7b-instruct:free","openrouter:mistralai/mistral-7b-instruct","openrouter:nousresearch/hermes-2-pro-llama-3-8b","openrouter:mistralai/mistral-7b-instruct-v0.3","openrouter:microsoft/phi-3-mini-128k-instruct","openrouter:microsoft/phi-3-medium-128k-instruct","openrouter:neversleep/llama-3-lumimaid-70b","openrouter:deepseek/deepseek-coder","openrouter:google/gemini-flash-1.5","openrouter:openai/gpt-4o","openrouter:openai/gpt-4o:extended","openrouter:meta-llama/llama-guard-2-8b","openrouter:openai/gpt-4o-2024-05-13","openrouter:allenai/olmo-7b-instruct","openrouter:neversleep/llama-3-lumimaid-8b:extended","openrouter:neversleep/llama-3-lumimaid-8b","openrouter:sao10k/fimbulvetr-11b-v2","openrouter:meta-llama/llama-3-8b-instruct","openrouter:meta-llama/llama-3-70b-instruct","openrouter:mistralai/mixtral-8x22b-instruct","openrouter:microsoft/wizardlm-2-8x22b","openrouter:google/gemini-pro-1.5","openrouter:openai/gpt-4-turbo","openrouter:cohere/command-r-plus","openrouter:cohere/command-r-plus-04-2024","openrouter:sophosympatheia/midnight-rose-70b","openrouter:cohere/command","openrouter:cohere/command-r","openrouter:anthropic/claude-3-haiku:beta","openrouter:anthropic/claude-3-haiku","openrouter:anthropic/claude-3-opus:beta","openrouter:anthropic/claude-3-opus","openrouter:anthropic/claude-3-sonnet:beta","openrouter:anthropic/claude-3-sonnet","openrouter:cohere/command-r-03-2024","openrouter:mistralai/mistral-large","openrouter:openai/gpt-3.5-turbo-0613","openrouter:openai/gpt-4-turbo-preview","openrouter:nousresearch/nous-hermes-2-mixtral-8x7b-dpo","openrouter:mistralai/mistral-medium","openrouter:mistralai/mistral-small","openrouter:mistralai/mistral-tiny","openrouter:mistralai/mistral-7b-instruct-v0.2","openrouter:mistralai/mixtral-8x7b-instruct","openrouter:neversleep/noromaid-20b","openrouter:anthropic/claude-2.1:beta","openrouter:anthropic/claude-2.1","openrouter:anthropic/claude-2:beta","openrouter:anthropic/claude-2","openrouter:undi95/toppy-m-7b","openrouter:alpindale/goliath-120b","openrouter:openrouter/auto","openrouter:openai/gpt-3.5-turbo-1106","openrouter:openai/gpt-4-1106-preview","openrouter:jondurbin/airoboros-l2-70b","openrouter:openai/gpt-3.5-turbo-instruct","openrouter:mistralai/mistral-7b-instruct-v0.1","openrouter:pygmalionai/mythalion-13b","openrouter:openai/gpt-3.5-turbo-16k","openrouter:openai/gpt-4-32k","openrouter:openai/gpt-4-32k-0314","openrouter:mancer/weaver","openrouter:anthropic/claude-2.0:beta","openrouter:anthropic/claude-2.0","openrouter:undi95/remm-slerp-l2-13b","openrouter:gryphe/mythomax-l2-13b","openrouter:meta-llama/llama-2-70b-chat","openrouter:openai/gpt-3.5-turbo","openrouter:openai/gpt-3.5-turbo-0125","openrouter:openai/gpt-4","openrouter:openai/gpt-4-0314"]
    
    vision_models = [*openai_models, *claude_models, *mistral_models, *xai_models, *deepseek_models, *gemini_models, *openrouter_models]
    
    models = vision_models
    
    model_aliases = {              
        ### mistral_models ###
        "mixtral-8x22b": ["open-mixtral-8x22b", "open-mixtral-8x22b-2404"],
        "pixtral-large": ["pixtral-large-2411","pixtral-large-latest", "mistral-large-pixtral-2411"],
        
        ### openrouter_models ###
        # llama
        "llama-2-70b": "openrouter:meta-llama/llama-2-70b-chat",
        "llama-3-8b": "openrouter:meta-llama/llama-3-8b-instruct",
        "llama-3-70b": "openrouter:meta-llama/llama-3-70b-instruct",
        "llama-3.1-8b": ["openrouter:meta-llama/llama-3.1-8b-instruct:free", "openrouter:meta-llama/llama-3.1-8b-instruct"],
        "llama-3.1-70b": "openrouter:meta-llama/llama-3.1-70b-instruct",
        "llama-3.1-405b": ["openrouter:meta-llama/llama-3.1-405b:free", "openrouter:meta-llama/llama-3.1-405b", "openrouter:meta-llama/llama-3.1-405b-instruct"],
        "llama-3.2-1b": ["openrouter:meta-llama/llama-3.2-1b-instruct:free", "openrouter:meta-llama/llama-3.2-1b-instruct"],
        "llama-3.2-3b": ["openrouter:meta-llama/llama-3.2-3b-instruct:free","openrouter:meta-llama/llama-3.2-3b-instruct"],
        "llama-3.2-11b": ["openrouter:meta-llama/llama-3.2-11b-vision-instruct:free", "openrouter:meta-llama/llama-3.2-11b-vision-instruct"],
        "llama-3.2-90b": "openrouter:meta-llama/llama-3.2-90b-vision-instruct",
        "llama-3.3-8b": "openrouter:meta-llama/llama-3.3-8b-instruct:free",
        "llama-3.3-70b": ["openrouter:meta-llama/llama-3.3-70b-instruct:free", "openrouter:meta-llama/llama-3.3-70b-instruct"],
        "llama-4-maverick": ["openrouter:meta-llama/llama-4-maverick:free", "openrouter:meta-llama/llama-4-maverick"],
        "llama-4-scout": ["openrouter:meta-llama/llama-4-scout:free", "openrouter:meta-llama/llama-4-scout"],
        #"": "openrouter:meta-llama/llama-guard-3-8b",
        #"": "openrouter:meta-llama/llama-guard-2-8b",
        #"": "openrouter:meta-llama/llama-guard-4-12b",
        
        # google (gemini)
        "gemini-1.5-flash": ["gemini-1.5-flash", "openrouter:google/gemini-flash-1.5", "gemini-flash-1.5-8b"],
        "gemini-1.5-8b-flash": "openrouter:google/gemini-flash-1.5-8b",
        "gemini-1.5-pro": "openrouter:google/gemini-pro-1.5",
        "gemini-2.0-flash": ["gemini-2.0-flash", "openrouter:google/gemini-2.0-flash-lite-001", "openrouter:google/gemini-2.0-flash-001", "openrouter:google/gemini-2.0-flash-exp:free"],
        "gemini-2.5-pro": ["openrouter:google/gemini-2.5-pro-preview", "openrouter:google/gemini-2.5-pro-exp-03-25"],
        "gemini-2.5-flash": "openrouter:google/gemini-2.5-flash-preview",
        "gemini-2.5-flash-thinking": "openrouter:google/gemini-2.5-flash-preview:thinking",
        
        # google (gemma)
        "gemma-2-9b": ["openrouter:google/gemma-2-9b-it:free","openrouter:google/gemma-2-9b-it"],
        "gemma-2-27b": "openrouter:google/gemma-2-27b-it",
        "gemma-3-1b": "openrouter:google/gemma-3-1b-it:free",
        "gemma-3-4b": ["openrouter:google/gemma-3-4b-it:free", "openrouter:google/gemma-3-4b-it"],
        "gemma-3-12b": ["openrouter:google/gemma-3-12b-it:free", "openrouter:google/gemma-3-12b-it"],
        "gemma-3-27b": ["openrouter:google/gemma-3-27b-it:free", "openrouter:google/gemma-3-27b-it"],
        
        # openai (gpt-3.5)
        "gpt-3.5-turbo": ["openrouter:openai/gpt-3.5-turbo-0613", "openrouter:openai/gpt-3.5-turbo-1106", "openrouter:openai/gpt-3.5-turbo-0125", "openrouter:openai/gpt-3.5-turbo", "openrouter:openai/gpt-3.5-turbo-instruct", "openrouter:openai/gpt-3.5-turbo-16k"],
        
        # openai (gpt-4)
        "gpt-4": ["openrouter:openai/gpt-4-1106-preview", "openrouter:openai/gpt-4-32k", "openrouter:openai/gpt-4-32k-0314", "openrouter:openai/gpt-4", "openrouter:openai/gpt-4-0314",],
        "gpt-4-turbo": ["openrouter:openai/gpt-4-turbo", "openrouter:openai/gpt-4-turbo-preview"],
        
        # openai (gpt-4o)
        "gpt-4o": ["gpt-4o", "openrouter:openai/gpt-4o-2024-08-06", "openrouter:openai/gpt-4o-2024-11-20", "openrouter:openai/chatgpt-4o-latest", "openrouter:openai/gpt-4o", "openrouter:openai/gpt-4o:extended", "openrouter:openai/gpt-4o-2024-05-13",],
        "gpt-4o-search": "openrouter:openai/gpt-4o-search-preview",
        "gpt-4o-mini": ["gpt-4o-mini", "openrouter:openai/gpt-4o-mini",  "openrouter:openai/gpt-4o-mini-2024-07-18"],
        "gpt-4o-mini-search": "openrouter:openai/gpt-4o-mini-search-preview",
        
        # openai (o1)
        "o1": ["o1", "openrouter:openai/o1", "openrouter:openai/o1-preview", "openrouter:openai/o1-preview-2024-09-12"],
        "o1-mini": ["o1-mini", "openrouter:openai/o1-mini", "openrouter:openai/o1-mini-2024-09-12"],
        "o1-pro": ["o1-pro", "openrouter:openai/o1-pro"],
        
        # openai (o3)
        "o3": ["o3", "openrouter:openai/o3"],
        "o3-mini": ["o3-mini", "openrouter:openai/o3-mini", "openrouter:openai/o3-mini-high"],
        "o3-mini-high": "openrouter:openai/o3-mini-high",
        
        # openai (o4)
        "o4-mini": ["o4-mini", "openrouter:openai/o4-mini"],
        "o4-mini-high": "openrouter:openai/o4-mini-high",
        
        # openai (gpt-4.1)
        "gpt-4.1": ["gpt-4.1", "openrouter:openai/gpt-4.1"],
        "gpt-4.1-mini": ["gpt-4.1-mini", "openrouter:openai/gpt-4.1-mini"],
        "gpt-4.1-nano": ["gpt-4.1-nano", "openrouter:openai/gpt-4.1-nano"],
        
        # openai (gpt-4.5)
        "gpt-4.5": ["gpt-4.5-preview", "openrouter:openai/gpt-4.5-preview"],
        
        # mistralai
        "mistral-large": ["openrouter:mistralai/mistral-large", "openrouter:mistralai/mistral-large-2411", "openrouter:mistralai/mistral-large-2407", "openrouter:mistralai/pixtral-large-2411"], 
        "mistral-medium": ["openrouter:mistralai/mistral-medium", "openrouter:mistralai/mistral-medium-3"], 
        "mistral-small": ["mistral-small", "mistral-small-2312", "mistral-small-2503","mistral-small-latest", "openrouter:mistralai/mistral-small", "openrouter:mistralai/mistral-small-3.1-24b-instruct:free", "openrouter:mistralai/mistral-small-3.1-24b-instruct", "openrouter:mistralai/mistral-small-24b-instruct-2501:free", "openrouter:mistralai/mistral-small-24b-instruct-2501"], 
        "mistral-tiny": ["mistral-tiny", "mistral-tiny-2312", "openrouter:mistralai/mistral-tiny"], 
        "mistral-7b": ["open-mistral-7b", "openrouter:mistralai/mistral-7b-instruct", "openrouter:mistralai/mistral-7b-instruct:free", "openrouter:mistralai/mistral-7b-instruct-v0.1", "openrouter:mistralai/mistral-7b-instruct-v0.2", "openrouter:mistralai/mistral-7b-instruct-v0.3",], 
        "mixtral-8x7b": ["open-mixtral-8x7b", "openrouter:mistralai/mixtral-8x7b-instruct"], 
        "mixtral-8x22b": ["open-mixtral-8x22b", "open-mixtral-8x22b-2404", "openrouter:mistralai/mixtral-8x7b-instruct", "openrouter:mistralai/mixtral-8x22b-instruct"], 
        "ministral-8b": ["ministral-8b-2410", "ministral-8b-latest", "openrouter:mistral/ministral-8b", "openrouter:mistralai/ministral-8b"],
        "mistral-nemo": ["openrouter:mistralai/mistral-nemo:free", "openrouter:mistralai/mistral-nemo"],
        "ministral-3b": ["ministral-3b-2410", "ministral-3b-latest", "openrouter:mistralai/ministral-3b"],
        "mistral-saba": "openrouter:mistralai/mistral-saba",
        "codestral": ["codestral-2501","codestral-latest","codestral-2412","codestral-2411-rc5", "openrouter:mistralai/codestral-2501", "openrouter:mistralai/codestral-mamba"],
        "pixtral-12b": ["pixtral-12b-2409","pixtral-12b","pixtral-12b-latest", "openrouter:mistralai/pixtral-12b"],
        
        # nousresearch
        "hermes-2-dpo": "openrouter:nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
        "hermes-2-pro": "openrouter:nousresearch/hermes-2-pro-llama-3-8b",
        "hermes-3-70b": "openrouter:nousresearch/hermes-3-llama-3.1-70b",
        "hermes-3-405b": "openrouter:nousresearch/hermes-3-llama-3.1-405b",
        "deephermes-3-8b": "openrouter:nousresearch/deephermes-3-llama-3-8b-preview:free",
        "deephermes-3-24b": "openrouter:nousresearch/deephermes-3-mistral-24b-preview:free",
        
        # microsoft
        "phi-3-mini": "openrouter:microsoft/phi-3-mini-128k-instruct",
        "phi-3-medium": "openrouter:microsoft/phi-3-medium-128k-instruct",
        "phi-3.5-mini": "openrouter:microsoft/phi-3.5-mini-128k-instruct",
        "phi-4": "openrouter:microsoft/phi-4",
        "phi-4-multimodal": "openrouter:microsoft/phi-4-multimodal-instruct",
        "phi-4-reasoning": "openrouter:microsoft/phi-4-reasoning:free",
        "phi-4-reasoning-plus": ["openrouter:microsoft/phi-4-reasoning-plus:free", "openrouter:microsoft/phi-4-reasoning-plus"],
        
        "wizardlm-2-8x22b": "openrouter:microsoft/wizardlm-2-8x22b",
        
        "mai-ds-r1": "openrouter:microsoft/mai-ds-r1:free",     
        
        # anthropic
        "claude-3.7-sonnet": ["claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest", "openrouter:anthropic/claude-3.7-sonnet", "openrouter:anthropic/claude-3.7-sonnet:beta",],
        "claude-3.7-sonnet-thinking": "openrouter:anthropic/claude-3.7-sonnet:thinking",
        "claude-3.5-haiku": ["openrouter:anthropic/claude-3.5-haiku:beta", "openrouter:anthropic/claude-3.5-haiku", "openrouter:anthropic/claude-3.5-haiku-20241022:beta", "openrouter:anthropic/claude-3.5-haiku-20241022"],
        "claude-3.5-sonnet": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20240620", "openrouter:anthropic/claude-3.5-sonnet-20240620:beta", "openrouter:anthropic/claude-3.5-sonnet-20240620", "openrouter:anthropic/claude-3.5-sonnet:beta", "openrouter:anthropic/claude-3.5-sonnet",],
        "claude-3-haiku": ["claude-3-haiku-20240307", "openrouter:anthropic/claude-3-haiku:beta", "openrouter:anthropic/claude-3-haiku"],
        "claude-3-opus": ["openrouter:anthropic/claude-3-opus:beta", "openrouter:anthropic/claude-3-opus"],
        "claude-3-sonnet": ["openrouter:anthropic/claude-3-sonnet:beta", "openrouter:anthropic/claude-3-sonnet"],
        "claude-2.1": ["openrouter:anthropic/claude-2.1:beta", "openrouter:anthropic/claude-2.1"],
        "claude-2": ["openrouter:anthropic/claude-2:beta", "openrouter:anthropic/claude-2",],
        "claude-2.0": ["openrouter:anthropic/claude-2.0:beta", "openrouter:anthropic/claude-2.0"],
        
        # rekaai
        "reka-flash": "openrouter:rekaai/reka-flash-3:free",

        # cohere
        "command-r7b": "openrouter:cohere/command-r7b-12-2024",
        "command-r-plus": ["openrouter:cohere/command-r-plus-08-2024", "openrouter:cohere/command-r-plus", "openrouter:cohere/command-r-plus-04-2024"],
        "command": "openrouter:cohere/command",
        "command-r": ["openrouter:cohere/command-r-08-2024", "openrouter:cohere/command-r", "openrouter:cohere/command-r-03-2024"],
        "command-a": "openrouter:cohere/command-a",
        
        # qwen
        "qwq-32b": ["openrouter:qwen/qwq-32b-preview", "openrouter:qwen/qwq-32b:free", "openrouter:qwen/qwq-32b"],
        "qwen-vl-plus": "openrouter:qwen/qwen-vl-plus",
        "qwen-vl-max": "openrouter:qwen/qwen-vl-max",
        "qwen-turbo": "openrouter:qwen/qwen-turbo",
        "qwen-2.5-vl-72b": ["openrouter:qwen/qwen2.5-vl-72b-instruct:free", "openrouter:qwen/qwen2.5-vl-72b-instruct"],
        "qwen-plus": "openrouter:qwen/qwen-plus",
        "qwen-max": "openrouter:qwen/qwen-max",
        "qwen-2.5-coder-32b": ["openrouter:qwen/qwen-2.5-coder-32b-instruct:free", "openrouter:qwen/qwen-2.5-coder-32b-instruct"],
        "qwen-2.5-7b": ["openrouter:qwen/qwen-2.5-7b-instruct:free", "openrouter:qwen/qwen-2.5-7b-instruct"],
        "qwen-2.5-72b": ["openrouter:qwen/qwen-2.5-72b-instruct:free", "openrouter:qwen/qwen-2.5-72b-instruct"],
        "qwen-2.5-vl-7b": ["openrouter:qwen/qwen-2.5-vl-7b-instruct:free", "openrouter:qwen/qwen-2.5-vl-7b-instruct"],
        "qwen-2-72b": "openrouter:qwen/qwen-2-72b-instruct",
        "qwen-3-0.6b": "openrouter:qwen/qwen3-0.6b-04-28:free",
        "qwen-3-1.7b": "openrouter:qwen/qwen3-1.7b:free",
        "qwen-3-4b": "openrouter:qwen/qwen3-4b:free",
        "qwen-3-30b": ["openrouter:qwen/qwen3-30b-a3b:free", "openrouter:qwen/qwen3-30b-a3b"],
        "qwen-3-8b": ["openrouter:qwen/qwen3-8b:free", "openrouter:qwen/qwen3-8b"],
        "qwen-3-14b": ["openrouter:qwen/qwen3-14b:free", "openrouter:qwen/qwen3-14b"],
        "qwen-3-32b": ["openrouter:qwen/qwen3-32b:free", "openrouter:qwen/qwen3-32b"],
        "qwen-3-235b": ["openrouter:qwen/qwen3-235b-a22b:free", "openrouter:qwen/qwen3-235b-a22b"],
        "qwen-2.5-coder-7b": "openrouter:qwen/qwen2.5-coder-7b-instruct",
        "qwen-2.5-vl-3b": "openrouter:qwen/qwen2.5-vl-3b-instruct:free",
        "qwen-2.5-vl-32b": ["openrouter:qwen/qwen2.5-vl-32b-instruct:free", "openrouter:qwen/qwen2.5-vl-32b-instruct"],
        
        # deepseek
        "deepseek-prover-v2": ["openrouter:deepseek/deepseek-prover-v2:free", "openrouter:deepseek/deepseek-prover-v2"],
        "deepseek-v3": "openrouter:deepseek/deepseek-v3-base:free",
        "deepseek-v3-0324": ["deepseek-chat", "openrouter:deepseek/deepseek-chat-v3-0324:free", "openrouter:deepseek/deepseek-chat-v3-0324"],
        "deepseek-r1-zero": "openrouter:deepseek/deepseek-r1-zero:free",
        "deepseek-r1-distill-llama-8b": "openrouter:deepseek/deepseek-r1-distill-llama-8b",
        "deepseek-r1-distill-qwen-1.5b": "openrouter:deepseek/deepseek-r1-distill-qwen-1.5b",
        "deepseek-r1-distill-qwen-32b": ["openrouter:deepseek/deepseek-r1-distill-qwen-32b:free", "openrouter:deepseek/deepseek-r1-distill-qwen-32b"],
        "deepseek-r1-distill-qwen-14b": ["openrouter:deepseek/deepseek-r1-distill-qwen-14b:free","openrouter:deepseek/deepseek-r1-distill-qwen-14b"],
        "deepseek-r1-distill-llama-70b": ["openrouter:deepseek/deepseek-r1-distill-llama-70b:free", "openrouter:deepseek/deepseek-r1-distill-llama-70b"],
        "deepseek-r1": ["deepseek-reasoner", "openrouter:deepseek/deepseek-r1:free", "openrouter:deepseek/deepseek-r1"],
        "deepseek-chat": ["deepseek-chat", "openrouter:deepseek/deepseek-chat:free", "openrouter:deepseek/deepseek-chat"],
        "deepseek-coder": ["openrouter:deepseek/deepseek-coder"],
        
        # inflection
        "inflection-3-productivity": "openrouter:inflection/inflection-3-productivity",
        "inflection-3-pi": "openrouter:inflection/inflection-3-pi",
        
        # x-ai
        "grok-3-mini": "openrouter:x-ai/grok-3-mini-beta",
        "grok-3-beta": "openrouter:x-ai/grok-3-beta",
        "grok-2": ["openrouter:x-ai/grok-2-vision-1212", "openrouter:x-ai/grok-2-1212"],
        "grok": ["openrouter:x-ai/grok-vision-beta", "openrouter:x-ai/grok-2-vision-1212", "openrouter:x-ai/grok-2-1212", "grok-beta","grok-vision-beta", "openrouter:x-ai/grok-beta", "openrouter:x-ai/grok-3-beta", "openrouter:x-ai/grok-3-mini-beta"],
        "grok-beta": ["grok-beta","grok-vision-beta", "openrouter:x-ai/grok-beta", "openrouter:x-ai/grok-3-beta"],
        
        # perplexity
        "sonar-reasoning-pro": "openrouter:perplexity/sonar-reasoning-pro",
        "sonar-pro": "openrouter:perplexity/sonar-pro",
        "sonar-deep-research": "openrouter:perplexity/sonar-deep-research",
        "r1-1776": "openrouter:perplexity/r1-1776",
        "sonar-reasoning": "openrouter:perplexity/sonar-reasoning",
        "sonar": "openrouter:perplexity/sonar",
        "llama-3.1-sonar-small-online": "openrouter:perplexity/llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-online": "openrouter:perplexity/llama-3.1-sonar-large-128k-online",
        
        # nvidia
        "nemotron-49b": ["openrouter:nvidia/llama-3.3-nemotron-super-49b-v1:free", "openrouter:nvidia/llama-3.3-nemotron-super-49b-v1"],
        "nemotron-70b": "openrouter:nvidia/llama-3.1-nemotron-70b-instruct",
        "nemotron-253b": "openrouter:nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
        
        # thudm
        "glm-4": ["openrouter:thudm/glm-4-32b:free", "openrouter:thudm/glm-4-32b", "openrouter:thudm/glm-4-9b:free",],
        "glm-4-32b": ["openrouter:thudm/glm-4-32b:free", "openrouter:thudm/glm-4-32b"],
        "glm-z1-32b": ["openrouter:thudm/glm-z1-32b:free", "openrouter:thudm/glm-z1-32b"],
        "glm-4-9b": "openrouter:thudm/glm-4-9b:free",
        "glm-z1-9b": "openrouter:thudm/glm-z1-9b:free",
        "glm-z1-rumination-32b": "openrouter:thudm/glm-z1-rumination-32b",
        
        # minimax
        "minimax": "openrouter:minimax/minimax-01",
        
        # cognitivecomputations
        "dolphin-3.0-r1-24b": "openrouter:cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "dolphin-3.0-24b": "openrouter:cognitivecomputations/dolphin3.0-mistral-24b:free",
        "dolphin-8x22b": "openrouter:cognitivecomputations/dolphin-mixtral-8x22b",
        
        # agentica-org
        "deepcoder-14b": "openrouter:agentica-org/deepcoder-14b-preview:free",
        
        # moonshotai
        "kimi-vl-thinking": "openrouter:moonshotai/kimi-vl-a3b-thinking:free",
        "moonlight-16b": "openrouter:moonshotai/moonlight-16b-a3b-instruct:free",
        
        # featherless
        "qwerky-72b": "openrouter:featherless/qwerky-72b:free",
        
        # liquid
        "lfm-7b": "openrouter:liquid/lfm-7b",
        "lfm-3b": "openrouter:liquid/lfm-3b",
        "lfm-40b": "openrouter:liquid/lfm-40b",
        
        # arcee-ai
        #"": "openrouter:arcee-ai/caller-large",
        #"": "openrouter:arcee-ai/spotlight",
        #"": "openrouter:arcee-ai/maestro-reasoning",
        #"": "openrouter:arcee-ai/virtuoso-large",
        #"": "openrouter:arcee-ai/coder-large",
        #"": "openrouter:arcee-ai/virtuoso-medium-v2",
        #"":  "openrouter:arcee-ai/arcee-blitz",
        
        # inception
        #"": "openrouter:inception/mercury-coder-small-beta",

        # opengvlab
        #"": "openrouter:opengvlab/internvl3-14b:free",
        #"": "openrouter:opengvlab/internvl3-2b:free",    
        
        # tngtech
        #"": "openrouter:tngtech/deepseek-r1t-chimera:free",
        
        # shisa-ai
        #"": "openrouter:shisa-ai/shisa-v2-llama3.3-70b:free",
        
        # eleutherai
        #"": "openrouter:eleutherai/llemma_7b",
        
        # shisa-ai
        #"": "openrouter:alfredpros/codellama-7b-instruct-solidity",
        
        # arliai
        #"": "openrouter:arliai/qwq-32b-arliai-rpr-v1:free",
        
        # all-hands
        #"": "openrouter:all-hands/openhands-lm-32b-v0.1",
        
        # scb10x
        #"": "openrouter:scb10x/llama3.1-typhoon2-8b-instruct",
        #"": "openrouter:scb10x/llama3.1-typhoon2-70b-instruct",
        
        # bytedance-research
        #"": "openrouter:bytedance-research/ui-tars-72b:free",
        
        # open-r1
        #"": "openrouter:open-r1/olympiccoder-32b:free",
        
        # ai21
        #"": "openrouter:ai21/jamba-1.6-large",
        #"": "openrouter:ai21/jamba-1.6-mini",
        #"": "openrouter:ai21/jamba-instruct",
        
        # thedrummer
        #"": "openrouter:thedrummer/anubis-pro-105b-v1",
        #"": "openrouter:thedrummer/skyfall-36b-v2",
        #"": "openrouter:thedrummer/unslopnemo-12b",
        #"": "openrouter:thedrummer/rocinante-12b",
        
        # aion-labs
        #"": "openrouter:aion-labs/aion-1.0",
        #"": "openrouter:aion-labs/aion-1.0-mini",
        #"": "openrouter:aion-labs/aion-rp-llama-3.1-8b",
        
        # sao10k
        #"": "openrouter:sao10k/l3.3-euryale-70b",
        #"": "openrouter:sao10k/l3.1-euryale-70b",
        #"": "openrouter:sao10k/l3-lunaris-8b",
        #"": "openrouter:sao10k/l3-euryale-70b",
        #"": "openrouter:sao10k/fimbulvetr-11b-v2",
        
        # eva-unit-01
        #"": "openrouter:eva-unit-01/eva-llama-3.33-70b",
        #"": "openrouter:eva-unit-01/eva-qwen-2.5-72b",
        #"": "openrouter:eva-unit-01/eva-qwen-2.5-32b",
        
        # amazon
        #"": "openrouter:amazon/nova-lite-v1",
        #"": "openrouter:amazon/nova-micro-v1",
        #"": "openrouter:amazon/nova-pro-v1",
        
        # infermatic
        #"": "openrouter:infermatic/mn-inferor-12b",
        
        # raifle
        #"": "openrouter:raifle/sorcererlm-8x22b",
        
        # neversleep
        #"": "openrouter:neversleep/llama-3.1-lumimaid-70b",
        #"": "openrouter:neversleep/llama-3.1-lumimaid-8b",
        #"": "openrouter:neversleep/llama-3-lumimaid-70b",
        #"": "openrouter:neversleep/llama-3-lumimaid-8b:extended",
        #"": "openrouter:neversleep/llama-3-lumimaid-8b",
        #"": "openrouter:neversleep/noromaid-20b",
        
        # anthracite-org
        #"": "openrouter:anthracite-org/magnum-v2-72b",
        #"": "openrouter:anthracite-org/magnum-v4-72b",
        
        # aetherwiing
        #"": "openrouter:aetherwiing/mn-starcannon-12b",
        
        # nothingiisreal
        #"": "openrouter:nothingiisreal/mn-celeste-12b",
        
        # alpindale
        #"": "openrouter:alpindale/magnum-72b",
        #"": "openrouter:alpindale/goliath-120b",
        
        # 01-ai
        #"": "openrouter:01-ai/yi-large",
        
        # allenai
        #"": "openrouter:allenai/olmo-7b-instruct",
        
        # sophosympatheia
        #"": "openrouter:sophosympatheia/midnight-rose-70b",
        
        # undi95
        #"": "openrouter:undi95/toppy-m-7b",
        
        # jondurbin
        #"": "openrouter:jondurbin/airoboros-l2-70b",
        
        # pygmalionai
        #"": "openrouter:pygmalionai/mythalion-13b",
        
        # mancer
        #"": "openrouter:mancer/weaver",
        
        # undi95
        #"": "openrouter:undi95/remm-slerp-l2-13b",
        
        # gryphe
        #"": "openrouter:gryphe/mythomax-l2-13b",
    }
    
    # Token expiration time in seconds (30 minutes)
    TOKEN_EXPIRATION = 30 * 60
    
    # Rate limit handling
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    RATE_LIMIT_DELAY = 60  # seconds
    
    # Make classes available at the class level for easier access
    Conversation = Conversation
    AuthData = AuthData
    
    # Class-level auth data cache to reduce signup requests
    _shared_auth_data = {}
    
    @staticmethod
    def get_driver_for_model(model: str) -> str:
        """Determine the appropriate driver based on the model name."""
        if model in PuterJS.openai_models:
            return "openai-completion"
        elif model in PuterJS.claude_models:
            return "claude"
        elif model in PuterJS.mistral_models:
            return "mistral"
        elif model in PuterJS.xai_models:
            return "xai"
        elif model in PuterJS.deepseek_models:
            return "deepseek"
        elif model in PuterJS.gemini_models:
            return "gemini"
        elif model in PuterJS.openrouter_models:
            return "openrouter"
        else:
            # Default to OpenAI for unknown models
            return "openai-completion"
    
    @staticmethod
    def format_messages_with_images(messages: Messages, media: MediaListType = None) -> Messages:
        """
        Format messages to include image data in the proper format for vision models.
        
        Args:
            messages: List of message dictionaries
            media: List of tuples containing (image_data, image_name)
            
        Returns:
            Formatted messages with image content
        """
        if not media:
            return messages
        
        # Create a copy of messages to avoid modifying the original
        formatted_messages = messages.copy()
        
        # Find the last user message to add images to
        for i in range(len(formatted_messages) - 1, -1, -1):
            if formatted_messages[i]["role"] == "user":
                user_msg = formatted_messages[i]
                
                # Convert to content list format if it's a string
                if isinstance(user_msg["content"], str):
                    text_content = user_msg["content"]
                    user_msg["content"] = [{"type": "text", "text": text_content}]
                elif not isinstance(user_msg["content"], list):
                    # Initialize as empty list if not already a list or string
                    user_msg["content"] = []
                
                # Add image content
                for image_data, image_name in media:
                    if isinstance(image_data, str) and (image_data.startswith("http://") or image_data.startswith("https://")):
                        # Direct URL
                        user_msg["content"].append({
                            "type": "image_url",
                            "image_url": {"url": image_data}
                        })
                    else:
                        # Convert to data URI
                        image_uri = to_data_uri(image_data, image_name)
                        user_msg["content"].append({
                            "type": "image_url",
                            "image_url": {"url": image_uri}
                        })
                
                formatted_messages[i] = user_msg
                break
        
        return formatted_messages
    
    @classmethod
    async def _create_temporary_account(cls, session: ClientSession, proxy: str = None) -> Dict[str, str]:
        """
        Create a temporary account with retry logic and rate limit handling.
        
        Args:
            session: The aiohttp ClientSession
            proxy: Optional proxy URL
            
        Returns:
            Dict containing auth_token
            
        Raises:
            RateLimitError: If rate limited after retries
        """
        signup_headers = {
            "Content-Type": "application/json",
            "host": "puter.com",  # Kept the previous fix for Host header
            "connection": "keep-alive",
            "sec-ch-ua-platform": "macOS",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
            "sec-ch-ua-mobile": "?0",
            "accept": "*/*",
            "origin": "https://puter.com",
            "sec-fetch-site": "same-site",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            "referer": "https://puter.com/",
            "accept-encoding": "gzip",
            "accept-language": "en-US,en;q=0.9"
        }
        
        signup_data = {
            "is_temp": True,
            "client_id": str(uuid.uuid4())  # Changed to generate a standard UUID
        }
        
        for attempt in range(cls.MAX_RETRIES):
            try:
                async with session.post(
                    "https://puter.com/signup", 
                    headers=signup_headers, 
                    json=signup_data,
                    proxy=proxy,
                    timeout=30
                ) as signup_response:
                    if signup_response.status == 429:
                        # Rate limited, wait and retry
                        retry_after = int(signup_response.headers.get('Retry-After', cls.RATE_LIMIT_DELAY))
                        if attempt < cls.MAX_RETRIES - 1:
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            # from ..errors import RateLimitError (ensure this import path is correct for your project)
                            raise Exception(f"Rate limited by Puter.js API. Try again after {retry_after} seconds.") # Placeholder if RateLimitError not accessible
                    
                    if signup_response.status != 200:
                        error_text = await signup_response.text()
                        if attempt < cls.MAX_RETRIES - 1:
                            # Wait before retrying
                            await asyncio.sleep(cls.RETRY_DELAY * (attempt + 1))
                            continue
                        else:
                            raise Exception(f"Failed to create temporary account. Status: {signup_response.status}, Details: {error_text}")
                    
                    try:
                        return await signup_response.json()
                    except Exception as e:
                        if attempt < cls.MAX_RETRIES - 1:
                            await asyncio.sleep(cls.RETRY_DELAY)
                            continue
                        else:
                            raise Exception(f"Failed to parse signup response as JSON: {e}")
            
            except Exception as e:
                if attempt < cls.MAX_RETRIES - 1:
                    # Exponential backoff
                    await asyncio.sleep(cls.RETRY_DELAY * (2 ** attempt))
                    continue
                else:
                    raise e
        
        # Should not reach here, but just in case
        raise Exception("Failed to create temporary account after multiple retries")
    
    @classmethod
    async def _get_app_token(cls, session: ClientSession, auth_token: str, proxy: str = None) -> Dict[str, str]:
        """
        Get app token with retry logic and rate limit handling.
        
        Args:
            session: The aiohttp ClientSession
            auth_token: The auth token from signup
            proxy: Optional proxy URL
            
        Returns:
            Dict containing app_token
            
        Raises:
            RateLimitError: If rate limited after retries
        """
        app_token_headers = {
            "host": "api.puter.com",
            "connection": "keep-alive",
            "authorization": f"Bearer {auth_token}",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "accept": "*/*",
            "origin": "https://puter.com",
            "sec-fetch-site": "same-site",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            "referer": "https://puter.com/",
            "accept-encoding": "gzip",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json"
        }
        
        # Randomize origin slightly to avoid detection
        origins = ["http://docs.puter.com", "https://docs.puter.com", "https://puter.com"]
        app_token_data = {"origin": random.choice(origins)}
        
        for attempt in range(cls.MAX_RETRIES):
            try:
                async with session.post(
                    "https://api.puter.com/auth/get-user-app-token", 
                    headers=app_token_headers, 
                    json=app_token_data,
                    proxy=proxy,
                    timeout=30
                ) as app_token_response:
                    if app_token_response.status == 429:
                        # Rate limited, wait and retry
                        retry_after = int(app_token_response.headers.get('Retry-After', cls.RATE_LIMIT_DELAY))
                        if attempt < cls.MAX_RETRIES - 1:
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            raise RateLimitError(f"Rate limited by Puter.js API. Try again after {retry_after} seconds.")
                    
                    if app_token_response.status != 200:
                        error_text = await app_token_response.text()
                        if attempt < cls.MAX_RETRIES - 1:
                            # Wait before retrying
                            await asyncio.sleep(cls.RETRY_DELAY * (attempt + 1))
                            continue
                        else:
                            raise Exception(f"Failed to get app token. Status: {app_token_response.status}, Details: {error_text}")
                    
                    try:
                        return await app_token_response.json()
                    except Exception as e:
                        if attempt < cls.MAX_RETRIES - 1:
                            await asyncio.sleep(cls.RETRY_DELAY)
                            continue
                        else:
                            raise Exception(f"Failed to parse app token response as JSON: {e}")
            
            except Exception as e:
                if attempt < cls.MAX_RETRIES - 1:
                    # Exponential backoff
                    await asyncio.sleep(cls.RETRY_DELAY * (2 ** attempt))
                    continue
                else:
                    raise e
        
        # Should not reach here, but just in case
        raise Exception("Failed to get app token after multiple retries")

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the internal model name from the user-provided model name."""
        
        if not model:
            return cls.default_model
        
        # Check if the model exists directly in our models list
        if model in cls.models:
            return model
        
        # Check if there's an alias for this model
        if model in cls.model_aliases:
            alias = cls.model_aliases[model]
            # If the alias is a list, randomly select one of the options
            if isinstance(alias, list):
                selected_model = random.choice(alias)
                debug.log(f"PuterJS: Selected model '{selected_model}' from alias '{model}'")
                return selected_model
            debug.log(f"PuterJS: Using model '{alias}' for alias '{model}'")
            return alias
        
        raise ModelNotFoundError(f"Model {model} not found")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        stream: bool = True,
        conversation: Optional[JsonConversation] = None,
        return_conversation: bool = False,
        media: MediaListType = None,  # Add media parameter for images
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        # Check if we need to use a vision model
        has_images = False
        if media is not None and len(media) > 0:
            has_images = True
            # If images are present and model doesn't support vision, switch to default vision model
            if model not in cls.vision_models:
                model = cls.default_vision_model
        
        # Check for image URLs in messages
        if not has_images:
            for msg in messages:
                if msg["role"] == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "image_url":
                                has_images = True
                                if model not in cls.vision_models:
                                    model = cls.default_vision_model
                                break
                    elif isinstance(content, str):
                        # Check for URLs in the text
                        urls = re.findall(r'https?://\S+\.(jpg|jpeg|png|gif|webp)', content, re.IGNORECASE)
                        if urls:
                            has_images = True
                            if model not in cls.vision_models:
                                model = cls.default_vision_model
                            break
        
        # Check if the conversation is of the correct type
        if conversation is not None and not isinstance(conversation, cls.Conversation):
            # Convert generic JsonConversation to our specific Conversation class
            new_conversation = cls.Conversation(model)
            new_conversation.message_history = conversation.message_history.copy() if hasattr(conversation, 'message_history') else messages.copy()
            conversation = new_conversation
        
        # Initialize or update conversation
        if conversation is None:
            conversation = cls.Conversation(model)
            # Format messages with images if needed
            if has_images and media:
                conversation.message_history = cls.format_messages_with_images(messages, media)
            else:
                conversation.message_history = messages.copy()
        else:
            # Update message history with new messages
            if has_images and media:
                formatted_messages = cls.format_messages_with_images(messages, media)
                for msg in formatted_messages:
                    if msg not in conversation.message_history:
                        conversation.message_history.append(msg)
            else:
                for msg in messages:
                    if msg not in conversation.message_history:
                        conversation.message_history.append(msg)
        
        # Get the authentication data for this specific model
        auth_data = conversation.get_auth_for_model(model, cls)
        
        # Check if we can use shared auth data
        if model in cls._shared_auth_data and cls._shared_auth_data[model].is_valid(cls.TOKEN_EXPIRATION):
            # Copy shared auth data to conversation
            shared_auth = cls._shared_auth_data[model]
            auth_data.auth_token = shared_auth.auth_token
            auth_data.app_token = shared_auth.app_token
            auth_data.created_at = shared_auth.created_at
            auth_data.tokens_valid = shared_auth.tokens_valid
        
        # Check if rate limited
        if auth_data.is_rate_limited():
            wait_time = auth_data.rate_limited_until - time.time()
            if wait_time > 0:
                yield f"Rate limited. Please try again in {int(wait_time)} seconds."
                return
        
        async with ClientSession() as session:
            # Step 1: Create a temporary account (if needed)
            if not auth_data.is_valid(cls.TOKEN_EXPIRATION):
                try:
                    # Try to authenticate
                    signup_data = await cls._create_temporary_account(session, proxy)
                    auth_data.auth_token = signup_data.get("token")
                    
                    if not auth_data.auth_token:
                        yield f"Error: No auth token in response for model {model}"
                        return
                    
                    # Get app token
                    app_token_data = await cls._get_app_token(session, auth_data.auth_token, proxy)
                    auth_data.app_token = app_token_data.get("token")
                    
                    if not auth_data.app_token:
                        yield f"Error: No app token in response for model {model}"
                        return
                    
                    # Mark tokens as valid
                    auth_data.created_at = time.time()
                    auth_data.tokens_valid = True
                    
                    # Update shared auth data
                    cls._shared_auth_data[model] = auth_data
                
                except RateLimitError as e:
                    # Set rate limit and inform user
                    auth_data.set_rate_limit(cls.RATE_LIMIT_DELAY)
                    yield str(e)
                    return
                except Exception as e:
                    yield f"Error during authentication for model {model}: {str(e)}"
                    return
            
            # Step 3: Make the chat request with proper image handling
            try:
                chat_headers = {
                    "host": "api.puter.com",
                    "connection": "keep-alive",
                    "authorization": f"Bearer {auth_data.app_token}",
                    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
                    "content-type": "application/json;charset=UTF-8",
                    # Set appropriate accept header based on stream mode
                    "accept": "text/event-stream" if stream else "*/*",
                    "origin": "http://docs.puter.com",
                    "sec-fetch-site": "cross-site",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-dest": "empty",
                    "referer": "http://docs.puter.com/",
                    "accept-encoding": "gzip",
                    "accept-language": "en-US,en;q=0.9"
                }
                
                # Determine the appropriate driver based on the model
                driver = cls.get_driver_for_model(model)
                
                # Prepare messages for the API request
                processed_messages = conversation.message_history
                
                # Special handling for direct image URLs in the HTML example
                if has_images and not any(isinstance(msg.get("content"), list) for msg in processed_messages):
                    # This handles the case where an image URL is passed directly to puter.ai.chat()
                    # as in the HTML example: puter.ai.chat("What do you see?", "https://assets.puter.site/doge.jpeg")
                    for i, msg in enumerate(processed_messages):
                        if msg["role"] == "user":
                            # Check if there are image URLs in the media parameter
                            if media and len(media) > 0:
                                # Format with the media parameter
                                processed_messages = cls.format_messages_with_images([msg], media)
                            else:
                                # Check for URLs in the text content
                                content = msg.get("content", "")
                                if isinstance(content, str):
                                    urls = re.findall(r'https?://\S+\.(jpg|jpeg|png|gif|webp)', content, re.IGNORECASE)
                                    if urls:
                                        # Extract URLs from the text
                                        text_parts = []
                                        image_urls = []
                                        
                                        # Simple URL extraction
                                        words = content.split()
                                        for word in words:
                                            if re.match(r'https?://\S+\.(jpg|jpeg|png|gif|webp)', word, re.IGNORECASE):
                                                image_urls.append(word)
                                            else:
                                                text_parts.append(word)
                                        
                                        # Create formatted message with text and images
                                        formatted_content = []
                                        if text_parts:
                                            formatted_content.append({
                                                "type": "text", 
                                                "text": " ".join(text_parts)
                                            })
                                        
                                        for url in image_urls:
                                            formatted_content.append({
                                                "type": "image_url",
                                                "image_url": {"url": url}
                                            })
                                        
                                        processed_messages[i]["content"] = formatted_content
                            break
                
                chat_data = {
                    "interface": "puter-chat-completion",
                    "driver": driver,
                    "test_mode": False,
                    "method": "complete",
                    "args": {
                        "messages": processed_messages,
                        "model": model,
                        "stream": stream,
                        "max_tokens": kwargs.get("max_tokens", 4096)
                    }
                }
                
                # Add any additional parameters from kwargs
                for key, value in kwargs.items():
                    if key not in ["messages", "model", "stream", "max_tokens"]:
                        chat_data["args"][key] = value
                
                # Try the chat request with retries
                for attempt in range(cls.MAX_RETRIES):
                    try:
                        async with session.post(
                            cls.api_endpoint, 
                            headers=chat_headers, 
                            json=chat_data,
                            proxy=proxy,
                            timeout=120  # Longer timeout for vision requests
                        ) as response:
                            if response.status == 429:
                                # Rate limited, set rate limit and inform user
                                retry_after = int(response.headers.get('Retry-After', cls.RATE_LIMIT_DELAY))
                                auth_data.set_rate_limit(retry_after)
                                if attempt < cls.MAX_RETRIES - 1:
                                    await asyncio.sleep(min(retry_after, 10))  # Wait but cap at 10 seconds for retries
                                    continue
                                else:
                                    raise RateLimitError(f"Rate limited by Puter.js API. Try again after {retry_after} seconds.")
                            
                            # Handle authentication errors
                            if response.status in [401, 403]:
                                error_text = await response.text()
                                auth_data.invalidate()  # Mark tokens as invalid
                                if attempt < cls.MAX_RETRIES - 1:
                                    # Try to get new tokens
                                    signup_data = await cls._create_temporary_account(session, proxy)
                                    auth_data.auth_token = signup_data.get("token")
                                    
                                    app_token_data = await cls._get_app_token(session, auth_data.auth_token, proxy)
                                    auth_data.app_token = app_token_data.get("token")
                                    
                                    auth_data.created_at = time.time()
                                    auth_data.tokens_valid = True
                                    
                                    # Update shared auth data
                                    cls._shared_auth_data[model] = auth_data
                                    
                                    # Retry with new token
                                    continue
                                else:
                                    raise Exception(f"Authentication failed after {cls.MAX_RETRIES} attempts: {error_text}")
                            
                            if response.status != 200:
                                error_text = await response.text()
                                if attempt < cls.MAX_RETRIES - 1:
                                    # Wait before retrying
                                    await asyncio.sleep(cls.RETRY_DELAY * (attempt + 1))
                                    continue
                                else:
                                    raise Exception(f"Chat request failed. Status: {response.status}, Details: {error_text}")
                            
                            # Process successful response
                            if stream:
                                # Process the streaming response
                                full_response = ""
                                buffer = ""
                                
                                # Use iter_any() to process chunks as they arrive
                                async for chunk in response.content.iter_any():
                                    if chunk:
                                        try:
                                            chunk_text = chunk.decode('utf-8')
                                            buffer += chunk_text
                                            
                                            # Process complete lines in the buffer
                                            lines = buffer.split('\n')
                                            # Keep the last potentially incomplete line in the buffer
                                            buffer = lines.pop() if lines else ""
                                            
                                            for line in lines:
                                                line = line.strip()
                                                if not line:
                                                    continue
                                                    
                                                # Handle different streaming formats
                                                if line.startswith("data: "):
                                                    line = line[6:]  # Remove "data: " prefix
                                                    
                                                    # Skip "[DONE]" marker
                                                    if line == "[DONE]":
                                                        continue
                                                    
                                                    try:
                                                        data = json.loads(line)
                                                        
                                                        # OpenAI format
                                                        if "choices" in data:
                                                            choice = data["choices"][0]
                                                            delta = choice.get("delta", {})
                                                            content = delta.get("content", "")
                                                            if content:
                                                                full_response += content
                                                                yield content
                                                        # Puter.js format
                                                        elif "type" in data and data["type"] == "text":
                                                            content = data.get("text", "")
                                                            if content:
                                                                full_response += content
                                                                yield content
                                                    except json.JSONDecodeError:
                                                        # Not valid JSON, might be a plain text response
                                                        if line and line != "[DONE]":
                                                            full_response += line
                                                            yield line
                                                else:
                                                    # Try to parse as JSON directly
                                                    try:
                                                        data = json.loads(line)
                                                        if "type" in data and data["type"] == "text":
                                                            content = data.get("text", "")
                                                            if content:
                                                                full_response += content
                                                                yield content
                                                    except json.JSONDecodeError:
                                                        # Not valid JSON, might be a plain text response
                                                        if line and line != "[DONE]":
                                                            full_response += line
                                                            yield line
                                        except UnicodeDecodeError:
                                            continue
                                
                                # Process any remaining data in the buffer
                                if buffer:
                                    if buffer.startswith("data: "):
                                        buffer = buffer[6:]  # Remove "data: " prefix
                                    
                                    if buffer != "[DONE]":
                                        try:
                                            data = json.loads(buffer)
                                            if "choices" in data:
                                                choice = data["choices"][0]
                                                delta = choice.get("delta", {})
                                                content = delta.get("content", "")
                                                if content:
                                                    full_response += content
                                                    yield content
                                            elif "type" in data and data["type"] == "text":
                                                content = data.get("text", "")
                                                if content:
                                                    full_response += content
                                                    yield content
                                        except json.JSONDecodeError:
                                            # Not valid JSON, might be a plain text response
                                            if buffer and buffer != "[DONE]":
                                                full_response += buffer
                                                yield buffer
                                
                                # Add the assistant's response to the conversation history
                                if full_response:
                                    conversation.message_history.append({
                                        "role": "assistant",
                                        "content": full_response
                                    })
                                
                                # Return the conversation object if requested
                                if return_conversation:
                                    yield conversation
                                
                                # Signal completion
                                yield FinishReason("stop")
                            else:
                                # Process non-streaming response
                                try:
                                    response_json = await response.json()
                                except Exception as e:
                                    error_text = await response.text()
                                    if attempt < cls.MAX_RETRIES - 1:
                                        await asyncio.sleep(cls.RETRY_DELAY)
                                        continue
                                    else:
                                        raise Exception(f"Failed to parse chat response as JSON: {error_text}")
                                
                                if response_json.get("success") is True:
                                    # Extract the content from the response
                                    content = response_json.get("result", {}).get("message", {}).get("content", "")
                                    
                                    # Add the assistant's response to the conversation history
                                    if content:
                                        conversation.message_history.append({
                                            "role": "assistant",
                                            "content": content
                                        })
                                    
                                    yield content
                                    
                                    # Return the conversation object if requested
                                    if return_conversation:
                                        yield conversation
                                    
                                    # Signal completion
                                    yield FinishReason("stop")
                                else:
                                    # Handle error in response
                                    error_msg = response_json.get("error", {}).get("message", "Unknown error")
                                    
                                    # Check for rate limiting or auth errors in the error message
                                    if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                                        auth_data.set_rate_limit()
                                        if attempt < cls.MAX_RETRIES - 1:
                                            await asyncio.sleep(cls.RETRY_DELAY)
                                            continue
                                        else:
                                            raise RateLimitError(f"Rate limited: {error_msg}")
                                    
                                    if "auth" in error_msg.lower() or "token" in error_msg.lower():
                                        auth_data.invalidate()
                                        if attempt < cls.MAX_RETRIES - 1:
                                            # Try to get new tokens
                                            signup_data = await cls._create_temporary_account(session, proxy)
                                            auth_data.auth_token = signup_data.get("token")
                                            
                                            app_token_data = await cls._get_app_token(session, auth_data.auth_token, proxy)
                                            auth_data.app_token = app_token_data.get("token")
                                            
                                            auth_data.created_at = time.time()
                                            auth_data.tokens_valid = True
                                            
                                            # Update headers with new token
                                            chat_headers["authorization"] = f"Bearer {auth_data.app_token}"
                                            
                                            # Update shared auth data
                                            cls._shared_auth_data[model] = auth_data
                                            
                                            # Retry with new token
                                            continue
                                    
                                    # For other errors, retry or raise
                                    if attempt < cls.MAX_RETRIES - 1:
                                        await asyncio.sleep(cls.RETRY_DELAY)
                                        continue
                                    else:
                                        yield f"Error: {error_msg}"
                            
                            # If we get here, we've successfully processed the response
                            break
                    
                    except RateLimitError as e:
                        # Set rate limit and inform user
                        auth_data.set_rate_limit(cls.RATE_LIMIT_DELAY)
                        if attempt < cls.MAX_RETRIES - 1:
                            await asyncio.sleep(min(cls.RATE_LIMIT_DELAY, 10))  # Wait but cap at 10 seconds for retries
                            continue
                        else:
                            yield str(e)
                            return
                    
                    except Exception as e:
                        # For network errors or other exceptions
                        if "token" in str(e).lower() or "auth" in str(e).lower():
                            auth_data.invalidate()
                        
                        if attempt < cls.MAX_RETRIES - 1:
                            await asyncio.sleep(cls.RETRY_DELAY * (attempt + 1))
                            continue
                        else:
                            yield f"Error: {str(e)}"
                            return
            
            except Exception as e:
                # If any error occurs outside the retry loop
                if "token" in str(e).lower() or "auth" in str(e).lower():
                    auth_data.invalidate()
                
                yield f"Error: {str(e)}"
                return
