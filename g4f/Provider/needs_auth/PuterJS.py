from __future__ import annotations

import random
import json
import requests
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages, MediaListType
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...providers.response import FinishReason, Usage, Reasoning, ToolCalls
from ...tools.media import render_messages
from ...requests import see_stream, raise_for_status
from ...errors import ResponseError, ModelNotFoundError, MissingAuthError
from ..helper import format_media_prompt
from .. import debug

class PuterJS(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Puter.js"
    url = "https://docs.puter.com/playground"
    login_url = "https://github.com/HeyPuter/puter-cli"
    api_endpoint = "https://api.puter.com/drivers/call"
    working = True
    active_by_default = True
    needs_auth = True

    default_model = 'gpt-4o'
    default_vision_model = default_model
    openai_models = [default_vision_model,"gpt-4o-mini", "o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.5-preview"]
    claude_models = ["claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"]
    mistral_models = ["ministral-3b-2410","ministral-3b-latest","ministral-8b-2410","ministral-8b-latest","open-mistral-7b","mistral-tiny","mistral-tiny-2312","open-mixtral-8x7b","mistral-small","mistral-small-2312","open-mixtral-8x22b","open-mixtral-8x22b-2404","mistral-large-2411","mistral-large-latest","pixtral-large-2411","pixtral-large-latest","mistral-large-pixtral-2411","codestral-2501","codestral-latest","codestral-2412","codestral-2411-rc5","pixtral-12b-2409","pixtral-12b","pixtral-12b-latest","mistral-small-2503","mistral-small-latest"]
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
    }

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls.models:
            try:
                url = "https://api.puter.com/puterai/chat/models/"
                cls.models = requests.get(url).json().get("models", [])
                cls.models = [model for model in cls.models if "/" not in model and model not in ["abuse", "costly", "fake", "model-fallback-test-1"]]
            except Exception as e:
                debug.log(f"PuterJS: Failed to fetch models from API: {e}")
                cls.models = []
            cls.models += [model for model in cls.model_aliases.keys() if model not in cls.models]
            openrouter_models = [model for model in cls.models if "openrouter:" in model]
            cls.models = [model for model in cls.models if model not in openrouter_models] + openrouter_models
            cls.vision_models = []
            for model in cls.models:
                for tag in ["vision", "multimodal", "gpt", "o1", "o3", "o4"]:
                    if tag in model:
                        cls.vision_models.append(model)
        return cls.models

    @staticmethod
    def get_driver_for_model(model: str) -> str:
        """Determine the appropriate driver based on the model name."""
        if "openrouter:" in model:
            return "openrouter"
        elif model in PuterJS.openai_models or model.startswith("gpt-"):
            return "openai-completion"
        elif model in PuterJS.mistral_models:
            return "mistral"
        elif "grok" in model:
            return "xai"
        elif "claude" in model:
            return "claude"
        elif "deepseek" in model:
            return "deepseek"
        elif "gemini" in model:
            return "gemini"
        else:
            raise ModelNotFoundError(f"Model {model} not found in known drivers")

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the internal model name from the user-provided model name."""

        if not model:
            return cls.default_model
        
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

        # Check if the model exists directly in our models list
        if model in cls.models:
            return model

        raise ModelNotFoundError(f"Model {model} not found")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        stream: bool = True,
        api_key: str = None,
        media: MediaListType = None,
        extra_parameters: list[str] = ["temperature", "presence_penalty", "top_p", "frequency_penalty", "response_format", "tools", "parallel_tool_calls", "tool_choice", "reasoning_effort", "logit_bias", "voice", "modalities", "audio"],
        **kwargs
    ) -> AsyncResult:
        if not api_key:
            raise MissingAuthError("API key is required for Puter.js API")

        if not cls.models:
            cls.get_models()

        # Check if we need to use a vision model
        if not model and media is not None and len(media) > 0:
            model = cls.default_vision_model

        # Check for image URLs in messages
        if not model:
            for msg in messages:
                if msg["role"] == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "image_url":
                                model = cls.default_vision_model
                                break

        # Get the model name from the user-provided model
        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            pass

        async with ClientSession() as session:
            headers = {
                "authorization": f"Bearer {api_key}",
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

            json_data = {
                "interface": "puter-image-generation" if model in cls.image_models else "puter-chat-completion",
                "driver": driver,
                "test_mode": messages[0]["content"] == "test",
                "method": "generate" if model in cls.image_models else "complete",
                "args": {"prompt": format_media_prompt(messages, prompt)} if model in cls.image_models else {
                    "messages": list(render_messages(messages, media)),
                    "model": model,
                    "stream": stream,
                    **{param: kwargs.get(param) for param in extra_parameters if param in kwargs}
                }
            }
            async with session.post(
                cls.api_endpoint, 
                headers=headers, 
                json=json_data,
                proxy=proxy
            ) as response:
                await raise_for_status(response)
                mime_type = response.headers.get("content-type", "")
                if mime_type.startswith("text/plain"):
                    yield await response.text()
                    return
                elif mime_type.startswith("text/event-stream"):
                    reasoning = False
                    async for result in see_stream(response.content):
                        if "error" in result:
                            raise ResponseError(result["error"].get("message", result["error"]))
                        choices = result.get("choices", [{}])
                        choice = choices.pop() if choices else {}
                        content = choice.get("delta", {}).get("content")
                        reasoning_content = choice.get("delta", {}).get("reasoning_content")
                        if reasoning_content:
                            reasoning = True
                            yield Reasoning(reasoning_content)
                        elif content:
                            if reasoning:
                                yield Reasoning(status="")
                                reasoning = False
                            yield content
                        if result.get("usage") is not None:
                            yield Usage(**result["usage"])
                        tool_calls = choice.get("delta", {}).get("tool_calls")
                        if tool_calls:
                            yield ToolCalls(choice["delta"]["tool_calls"])
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            yield FinishReason(finish_reason)
                elif mime_type.startswith("application/json"):
                    result = await response.json()
                    if "choices" in result:
                        choice = result["choices"][0]
                    elif "result" in result:
                        choice = result.get("result", {})
                    else:
                        raise ResponseError(result)
                    message = choice.get("message", {})
                    reasoning_content = message.get("reasoning_content")
                    if reasoning_content:
                        yield Reasoning(reasoning_content)
                    content = message.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                yield item.get("text", "")
                    elif content:
                        yield content
                    if "tool_calls" in message:
                        yield ToolCalls(message["tool_calls"])
                    if result.get("usage") is not None:
                        yield Usage(**result["usage"])
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        yield FinishReason(finish_reason)
                elif mime_type.startswith("application/x-ndjson"):
                    async for line in response.content:
                        data = json.loads(line)
                        if data.get("type") == "text":
                            yield data.get("text", "")
                else:
                    raise ResponseError(f"Unexpected content type: {mime_type}")
