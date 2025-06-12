from __future__ import annotations

from aiohttp import ClientSession
import os
import re
import json
import random
import string
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from ...typing import AsyncResult, Messages, MediaListType
from ...requests.raise_for_status import raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..openai.har_file import get_har_files
from ...image import to_data_uri
from ...cookies import get_cookies_dir
from ..helper import format_media_prompt, render_messages
from ...providers.response import JsonConversation, ImageResponse
from ...tools.media import merge_media
from ...errors import RateLimitError, NoValidHarFileError
from ... import debug

class Conversation(JsonConversation):
    validated_value: str = None
    chat_id: str = None
    message_history: Messages = []

    def __init__(self, model: str):
        self.model = model

class BlackboxPro(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Blackbox AI Pro"
    url = "https://www.blackbox.ai"
    login_url = None
    api_endpoint = "https://www.blackbox.ai/api/chat"
    
    working = True
    needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "blackboxai"
    default_vision_model = default_model
    default_image_model = 'flux'
    
    # OpenRouter Free
    openrouter_free_models = [
        "Deepcoder 14B Preview",
        "DeepHermes 3 Llama 3 8B Preview",
        "DeepSeek R1 Zero",
        "DeepSeek V3",
        "DeepSeek V3 0324",
        "DeepSeek V3 Base",
        "Dolphin3.0 Mistral 24B",
        "Dolphin3.0 R1 Mistral 24B",
        "Flash 3",
        "Gemini 2.0 Flash Experimental",
        "Gemini 2.0 Flash Thinking Experimental",
        "Gemini 2.0 Flash Thinking Experimental 01-21",
        "Gemma 2 9B",
        "Gemma 3 12B",
        "Gemma 3 1B",
        "Gemma 3 27B",
        "Gemma 3 4B",
        "Kimi VL A3B Thinking",
        "LearnLM 1.5 Pro Experimental",
        "Llama 3.1 8B Instruct",
        "Llama 3.1 Nemotron 70B Instruct",
        "Llama 3.1 Nemotron Nano 8B v1",
        "Llama 3.1 Nemotron Ultra 253B v1",
        "Llama 3.2 11B Vision Instruct",
        "Llama 3.2 1B Instruct",
        "Llama 3.2 3B Instruct",
        "Llama 3.3 70B Instruct",
        "Llama 3.3 Nemotron Super 49B v1",
        "Llama 4 Maverick",
        "Llama 4 Scout",
        "Mistral 7B Instruct",
        "Mistral Nemo",
        "Mistral Small 3",
        "Mistral Small 3.1 24B",
        "Molmo 7B D",
        "Moonlight 16B A3B Instruct",
        "OlympicCoder 32B",
        "OlympicCoder 7B",
        "Qwen2.5 72B Instruct",
        "Qwen2.5 7B Instruct",
        "Qwen2.5 Coder 32B Instruct",
        "Qwen2.5 VL 32B Instruct",
        "Qwen2.5 VL 3B Instruct",
        "Qwen2.5 VL 72B Instruct",
        "Qwen2.5-VL 7B Instruct",
        "Qwerky 72B",
        "QwQ 32B",
        "QwQ 32B Preview",
        "QwQ 32B RpR v1",
        "R1",
        "R1 Distill Llama 70B",
        "R1 Distill Qwen 14B",
        "R1 Distill Qwen 32B",
        "Rogue Rose 103B v0.2",
        "UI-TARS 72B",
        "Zephyr 7B",
    ]

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
        
        # OpenRouter Free
        *openrouter_free_models,
        
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
        'Heroku Agent',
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
        "Gemini-PRO",
    ]
    
    # Premium/Pro models (require subscription) (OpenRouter)
    openrouter_pro_models = [
        "Aion-1.0",
        "Aion-1.0-Mini",
        "Aion-RP 1.0 (8B)",
        "Airoboros 70B",
        "Anubis Pro 105B V1",
        "Arctic Instruct",
        "Auto Router",
        "Bagel 34B v0.2",
        "Capybara 34B",
        "Capybara 7B",
        "ChatGPT-4o",
        "Chronos Hermes 13B v2",
        "Cinematika 7B (alpha)",
        "Claude 3 Haiku",
        "Claude 3 Haiku (self-moderated)",
        "Claude 3 Opus",
        "Claude 3 Opus (self-moderated)",
        "Claude 3 Sonnet",
        "Claude 3 Sonnet (self-moderated)",
        "Claude 3.5 Haiku",
        "Claude 3.5 Haiku (2024-10-22)",
        "Claude 3.5 Haiku (2024-10-22) (self-moderated)",
        "Claude 3.5 Haiku (self-moderated)",
        "Claude 3.5 Sonnet",
        "Claude 3.5 Sonnet (2024-06-20)",
        "Claude 3.5 Sonnet (2024-06-20) (self-moderated)",
        "Claude 3.5 Sonnet (self-moderated)",
        "Claude 3.7 Sonnet",
        "Claude 3.7 Sonnet (self-moderated)",
        "Claude 3.7 Sonnet (thinking)",
        "Claude Instant v1",
        "Claude Instant v1.0",
        "Claude Instant v1.1",
        "Claude v1",
        "Claude v1.2",
        "Claude v2",
        "Claude v2 (self-moderated)",
        "Claude v2.0",
        "Claude v2.0 (self-moderated)",
        "Claude v2.1",
        "Claude v2.1 (self-moderated)",
        "CodeLlama 34B Instruct",
        "CodeLlama 34B v2",
        "CodeLlama 70B Instruct",
        "CodeLLaMa 7B Instruct Solidity",
        "Codestral 2501",
        "Codestral Mamba",
        "Command",
        "Command A",
        "Command R",
        "Command R (03-2024)",
        "Command R (08-2024)",
        "Command R+",
        "Command R+ (04-2024)",
        "Command R+ (08-2024)",
        "Command R7B (12-2024)",
        "DBRX 132B Instruct",
        "DeepSeek R1",
        "DeepSeek V2.5",
        "DeepSeek V3",
        "DeepSeek V3 0324",
        "DeepSeek-Coder-V2",
        "Dolphin 2.6 Mixtral 8x7B \uD83D\uDC2C",
        "Dolphin 2.9.2 Mixtral 8x22B \uD83D\uDC2C",
        "Dolphin Llama 3 70B \uD83D\uDC2C",
        "Eagle 7B",
        "EVA Llama 3.33 70B",
        "EVA Qwen2.5 14B",
        "EVA Qwen2.5 32B",
        "EVA Qwen2.5 72B",
        "Fimbulvetr 11B v2",
        "FireLLaVA 13B",
        "Gemini 1.5 Flash ",
        "Gemini 1.5 Flash 8B",
        "Gemini 1.5 Flash 8B Experimental",
        "Gemini 1.5 Flash Experimental",
        "Gemini 1.5 Pro",
        "Gemini 1.5 Pro Experimental",
        "Gemini 2.0 Flash",
        "Gemini 2.0 Flash Lite",
        "Gemini 2.5 Pro",
        "Gemini Experimental 1114",
        "Gemini Experimental 1121",
        "Gemini Pro 1.0",
        "Gemini Pro Vision 1.0",
        "Gemma 2 27B",
        "Gemma 2 9B",
        "Gemma 3 12B",
        "Gemma 3 27B",
        "Gemma 3 4B",
        "Gemma 7B",
        "Goliath 120B",
        "GPT-3.5 Turbo",
        "GPT-3.5 Turbo (older v0301)",
        "GPT-3.5 Turbo (older v0613)",
        "GPT-3.5 Turbo 16k",
        "GPT-3.5 Turbo 16k",
        "GPT-3.5 Turbo 16k (older v1106)",
        "GPT-3.5 Turbo Instruct",
        "GPT-4",
        "GPT-4 (older v0314)",
        "GPT-4 32k",
        "GPT-4 32k (older v0314)",
        "GPT-4 Turbo",
        "GPT-4 Turbo (older v1106)",
        "GPT-4 Turbo Preview",
        "GPT-4 Vision",
        "GPT-4.1",
        "GPT-4.1 Mini",
        "GPT-4.1 Nano",
        "GPT-4.5 (Preview)",
        "GPT-4o",
        "GPT-4o (2024-05-13)",
        "GPT-4o (2024-08-06)",
        "GPT-4o (2024-11-20)",
        "GPT-4o (extended)",
        "GPT-4o Search Preview",
        "GPT-4o-mini",
        "GPT-4o-mini (2024-07-18)",
        "GPT-4o-mini Search Preview",
        "Grok 2",
        "Grok 2 1212",
        "Grok 2 mini",
        "Grok 2 Vision 1212",
        "Grok 3",
        "Grok 3 Mini Beta",
        "Grok Beta",
        "Grok Vision Beta",
        "Hermes 13B",
        "Hermes 2 Mistral 7B DPO",
        "Hermes 2 Mixtral 8x7B DPO",
        "Hermes 2 Mixtral 8x7B SFT",
        "Hermes 2 Pro - Llama-3 8B",
        "Hermes 2 Theta 8B",
        "Hermes 2 Vision 7B (alpha)",
        "Hermes 2 Yi 34B",
        "Hermes 3 405B Instruct",
        "Hermes 3 70B Instruct",
        "Hermes 70B",
        "Inflection 3 Pi",
        "Inflection 3 Productivity",
        "Jamba 1.5 Large",
        "Jamba 1.5 Mini",
        "Jamba 1.6 Large",
        "Jamba Instruct",
        "Jamba Mini 1.6",
        "L3.3 Electra R1 70B",
        "LFM 3B",
        "LFM 40B MoE",
        "LFM 7B",
        "Llama 2 13B Chat",
        "Llama 2 70B Chat",
        "Llama 3 70B (Base)",
        "Llama 3 70B Instruct",
        "Llama 3 8B (Base)",
        "Llama 3 8B Instruct",
        "Llama 3 8B Lunaris",
        "Llama 3 Euryale 70B v2.1",
        "Llama 3 Lumimaid 70B",
        "Llama 3 Lumimaid 8B",
        "Llama 3 Lumimaid 8B (extended)",
        "Llama 3 Soliloquy 7B v3 32K",
        "Llama 3 Soliloquy 8B v2",
        "Llama 3 Stheno 8B v3.3 32K",
        "Llama 3.1 405B (base)",
        "Llama 3.1 405B Instruct",
        "Llama 3.1 70B Hanami x1",
        "Llama 3.1 70B Instruct",
        "Llama 3.1 8B Instruct",
        "Llama 3.1 Euryale 70B v2.2",
        "Llama 3.1 Nemotron 70B Instruct",
        "Llama 3.1 Swallow 70B Instruct V0.3",
        "Llama 3.1 Swallow 8B Instruct V0.3",
        "Llama 3.1 Tulu 3 405B",
        "Llama 3.2 11B Vision Instruct",
        "Llama 3.2 1B Instruct",
        "Llama 3.2 3B Instruct",
        "Llama 3.2 90B Vision Instruct",
        "Llama 3.3 70B Instruct",
        "Llama 3.3 Euryale 70B",
        "Llama 4 Maverick",
        "Llama 4 Scout",
        "Llama Guard 3 8B",
        "LlamaGuard 2 8B",
        "LLaVA 13B",
        "LLaVA v1.6 34B",
        "Llemma 7b",
        "Lumimaid v0.2 70B",
        "Lumimaid v0.2 8B",
        "lzlv 70B",
        "Mag Mell R1 12B",
        "Magnum 72B",
        "Magnum v2 72B",
        "Magnum v4 72B",
        "Midnight Rose 70B",
        "MiniMax-01",
        "Ministral 3B",
        "Ministral 8B",
        "Ministral 8B",
        "Mistral 7B Instruct",
        "Mistral 7B Instruct v0.1",
        "Mistral 7B Instruct v0.2",
        "Mistral 7B Instruct v0.3",
        "Mistral Large",
        "Mistral Large 2407",
        "Mistral Large 2411",
        "Mistral Medium",
        "Mistral Nemo",
        "Mistral Nemo 12B Celeste",
        "Mistral Nemo Inferor 12B",
        "Mistral OpenOrca 7B",
        "Mistral Small",
        "Mistral Small 3",
        "Mistral Small 3.1 24B",
        "Mistral Tiny",
        "Mixtral 8x22B (base)",
        "Mixtral 8x22B Instruct",
        "Mixtral 8x7B Instruct",
        "Mythalion 13B",
        "MythoMax 13B",
        "MythoMist 7B",
        "Nemotron-4 340B Instruct",
        "Neural Chat 7B v3.1",
        "Noromaid 20B",
        "Noromaid Mixtral 8x7B Instruct",
        "Nova Lite 1.0",
        "Nova Micro 1.0",
        "Nova Pro 1.0",
        "o1",
        "o1-mini",
        "o1-mini (2024-09-12)",
        "o1-preview",
        "o1-preview (2024-09-12)",
        "o1-pro",
        "o3 Mini",
        "o3 Mini High",
        "Olmo 2 32B Instruct",
        "OLMo 7B Instruct",
        "OpenChat 3.5 7B",
        "OpenChat 3.6 8B",
        "OpenHands LM 32B V0.1",
        "OpenHermes 2 Mistral 7B",
        "OpenHermes 2.5 Mistral 7B",
        "Optimus Alpha",
        "PaLM 2 Chat",
        "PaLM 2 Chat 32k",
        "PaLM 2 Code Chat",
        "PaLM 2 Code Chat 32k",
        "Phi 4",
        "Phi 4 Multimodal Instruct",
        "Phi-3 Medium 128K Instruct",
        "Phi-3 Medium 4K Instruct",
        "Phi-3 Mini 128K Instruct",
        "Phi-3.5 Mini 128K Instruct",
        "Pixtral 12B",
        "Pixtral Large 2411",
        "Psyfighter 13B",
        "Psyfighter v2 13B",
        "Quasar Alpha",
        "Qwen 1.5 110B Chat",
        "Qwen 1.5 14B Chat",
        "Qwen 1.5 32B Chat",
        "Qwen 1.5 4B Chat",
        "Qwen 1.5 72B Chat",
        "Qwen 1.5 7B Chat",
        "Qwen 2 72B Instruct",
        "Qwen 2 7B Instruct",
        "Qwen VL Max",
        "Qwen VL Plus",
        "Qwen-Max ",
        "Qwen-Plus",
        "Qwen-Turbo",
        "Qwen2.5 32B Instruct",
        "Qwen2.5 72B Instruct",
        "Qwen2.5 7B Instruct",
        "Qwen2.5 Coder 32B Instruct",
        "Qwen2.5 VL 32B Instruct",
        "Qwen2.5 VL 72B Instruct",
        "Qwen2.5-VL 72B Instruct",
        "Qwen2.5-VL 7B Instruct",
        "QwQ 32B",
        "QwQ 32B Preview",
        "R1",
        "R1 Distill Llama 70B",
        "R1 Distill Llama 8B",
        "R1 Distill Qwen 1.5B",
        "R1 Distill Qwen 14B",
        "R1 Distill Qwen 32B",
        "Reflection 70B",
        "ReMM SLERP 13B",
        "Rocinante 12B",
        "RWKV v5 3B AI Town",
        "RWKV v5 World 3B",
        "Saba",
        "Skyfall 36B V2",
        "SorcererLM 8x22B",
        "Starcannon 12B",
        "StarCoder2 15B Instruct",
        "StripedHyena Hessian 7B (base)",
        "StripedHyena Nous 7B",
        "Synthia 70B",
        "Toppy M 7B",
        "Typhoon2 70B Instruct",
        "Typhoon2 8B Instruct",
        "Unslopnemo 12B",
        "Wayfarer Large 70B Llama 3.3",
        "Weaver (alpha)",
        "WizardLM-2 7B",
        "WizardLM-2 8x22B",
        "Xwin 70B",
        "Yi 1.5 34B Chat",
        "Yi 34B (base)",
        "Yi 34B 200K",
        "Yi 34B Chat",
        "Yi 6B (base)",
        "Yi Large",
        "Yi Large FC",
        "Yi Large Turbo",
        "Yi Vision",
        "Zephyr 141B-A35B",
    ]
    
    image_models = [default_image_model]   
    vision_models = [default_vision_model, 'GPT-4o', 'o1', 'o3-mini', 'Gemini-PRO', 'Gemini Agent', 'llama-3.1-8b Agent', 'llama-3.1-70b Agent', 'llama-3.1-405 Agent', 'Gemini-Flash-2.0', 'DeepSeek-V3']

    userSelectedModel = ['GPT-4o', 'o1', 'o3-mini', 'Gemini-PRO', 'Claude-sonnet-3.7', 'Claude-sonnet-3.5', 'DeepSeek-V3', 'DeepSeek-R1', 'Meta-Llama-3.3-70B-Instruct-Turbo', 'Mistral-Small-24B-Instruct-2501', 'DeepSeek-LLM-Chat-(67B)', 'DBRX-Instruct', 'Qwen-QwQ-32B-Preview', 'Nous-Hermes-2-Mixtral-8x7B-DPO', 'Gemini-Flash-2.0'] + openrouter_pro_models

    # Agent mode configurations
    agentMode = {
        # Free (OpenRouter)
        'Deepcoder 14B Preview': {'mode': True, 'id': "agentica-org/deepcoder-14b-preview:free", 'name': "Deepcoder 14B Preview"},
        'DeepHermes 3 Llama 3 8B Preview': {'mode': True, 'id': "nousresearch/deephermes-3-llama-3-8b-preview:free", 'name': "DeepHermes 3 Llama 3 8B Preview"},
        'DeepSeek R1 Zero': {'mode': True, 'id': "deepseek/deepseek-r1-zero:free", 'name': "DeepSeek R1 Zero"},
        'DeepSeek V3': {'mode': True, 'id': "deepseek/deepseek-chat:free", 'name': "DeepSeek V3"},
        'DeepSeek V3 0324': {'mode': True, 'id': "deepseek/deepseek-chat-v3-0324:free", 'name': "DeepSeek V3 0324"},
        'DeepSeek V3 Base': {'mode': True, 'id': "deepseek/deepseek-v3-base:free", 'name': "DeepSeek V3 Base"},
        'Dolphin3.0 Mistral 24B': {'mode': True, 'id': "cognitivecomputations/dolphin3.0-mistral-24b:free", 'name': "Dolphin3.0 Mistral 24B"},
        'Dolphin3.0 R1 Mistral 24B': {'mode': True, 'id': "cognitivecomputations/dolphin3.0-r1-mistral-24b:free", 'name': "Dolphin3.0 R1 Mistral 24B"},
        'Flash 3': {'mode': True, 'id': "rekaai/reka-flash-3:free", 'name': "Flash 3"},
        'Gemini 2.0 Flash Experimental': {'mode': True, 'id': "google/gemini-2.0-flash-exp:free", 'name': "Gemini 2.0 Flash Experimental"},
        'Gemini 2.0 Flash Thinking Experimental': {'mode': True, 'id': "google/gemini-2.0-flash-thinking-exp-1219:free", 'name': "Gemini 2.0 Flash Thinking Experimental"},
        'Gemini 2.0 Flash Thinking Experimental 01-21': {'mode': True, 'id': "google/gemini-2.0-flash-thinking-exp:free", 'name': "Gemini 2.0 Flash Thinking Experimental 01-21"},
        'Gemma 2 9B': {'mode': True, 'id': "google/gemma-2-9b-it:free", 'name': "Gemma 2 9B"},
        'Gemma 3 12B': {'mode': True, 'id': "google/gemma-3-12b-it:free", 'name': "Gemma 3 12B"},
        'Gemma 3 1B': {'mode': True, 'id': "google/gemma-3-1b-it:free", 'name': "Gemma 3 1B"},
        'Gemma 3 27B': {'mode': True, 'id': "google/gemma-3-27b-it:free", 'name': "Gemma 3 27B"},
        'Gemma 3 4B': {'mode': True, 'id': "google/gemma-3-4b-it:free", 'name': "Gemma 3 4B"},
        'Kimi VL A3B Thinking': {'mode': True, 'id': "moonshotai/kimi-vl-a3b-thinking:free", 'name': "Kimi VL A3B Thinking"},
        'LearnLM 1.5 Pro Experimental': {'mode': True, 'id': "google/learnlm-1.5-pro-experimental:free", 'name': "LearnLM 1.5 Pro Experimental"},
        'Llama 3.1 8B Instruct': {'mode': True, 'id': "meta-llama/llama-3.1-8b-instruct:free", 'name': "Llama 3.1 8B Instruct"},
        'Llama 3.1 Nemotron 70B Instruct': {'mode': True, 'id': "nvidia/llama-3.1-nemotron-70b-instruct:free", 'name': "Llama 3.1 Nemotron 70B Instruct"},
        'Llama 3.1 Nemotron Nano 8B v1': {'mode': True, 'id': "nvidia/llama-3.1-nemotron-nano-8b-v1:free", 'name': "Llama 3.1 Nemotron Nano 8B v1"},
        'Llama 3.1 Nemotron Ultra 253B v1': {'mode': True, 'id': "nvidia/llama-3.1-nemotron-ultra-253b-v1:free", 'name': "Llama 3.1 Nemotron Ultra 253B v1"},
        'Llama 3.2 11B Vision Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-11b-vision-instruct:free", 'name': "Llama 3.2 11B Vision Instruct"},
        'Llama 3.2 1B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-1b-instruct:free", 'name': "Llama 3.2 1B Instruct"},
        'Llama 3.2 3B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-3b-instruct:free", 'name': "Llama 3.2 3B Instruct"},
        'Llama 3.3 70B Instruct': {'mode': True, 'id': "meta-llama/llama-3.3-70b-instruct:free", 'name': "Llama 3.3 70B Instruct"},
        'Llama 3.3 Nemotron Super 49B v1': {'mode': True, 'id': "nvidia/llama-3.3-nemotron-super-49b-v1:free", 'name': "Llama 3.3 Nemotron Super 49B v1"},
        'Llama 4 Maverick': {'mode': True, 'id': "meta-llama/llama-4-maverick:free", 'name': "Llama 4 Maverick"},
        'Llama 4 Scout': {'mode': True, 'id': "meta-llama/llama-4-scout:free", 'name': "Llama 4 Scout"},
        'Mistral 7B Instruct': {'mode': True, 'id': "mistralai/mistral-7b-instruct:free", 'name': "Mistral 7B Instruct"},
        'Mistral Nemo': {'mode': True, 'id': "mistralai/mistral-nemo:free", 'name': "Mistral Nemo"},
        'Mistral Small 3': {'mode': True, 'id': "mistralai/mistral-small-24b-instruct-2501:free", 'name': "Mistral Small 3"},
        'Mistral Small 3.1 24B': {'mode': True, 'id': "mistralai/mistral-small-3.1-24b-instruct:free", 'name': "Mistral Small 3.1 24B"},
        'Molmo 7B D': {'mode': True, 'id': "allenai/molmo-7b-d:free", 'name': "Molmo 7B D"},
        'Moonlight 16B A3B Instruct': {'mode': True, 'id': "moonshotai/moonlight-16b-a3b-instruct:free", 'name': "Moonlight 16B A3B Instruct"},
        'OlympicCoder 32B': {'mode': True, 'id': "open-r1/olympiccoder-32b:free", 'name': "OlympicCoder 32B"},
        'OlympicCoder 7B': {'mode': True, 'id': "open-r1/olympiccoder-7b:free", 'name': "OlympicCoder 7B"},
        'Qwen2.5 72B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-72b-instruct:free", 'name': "Qwen2.5 72B Instruct"},
        'Qwen2.5 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-7b-instruct:free", 'name': "Qwen2.5 7B Instruct"},
        'Qwen2.5 Coder 32B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-coder-32b-instruct:free", 'name': "Qwen2.5 Coder 32B Instruct"},
        'Qwen2.5 VL 32B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-32b-instruct:free", 'name': "Qwen2.5 VL 32B Instruct"},
        'Qwen2.5 VL 3B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-3b-instruct:free", 'name': "Qwen2.5 VL 3B Instruct"},
        'Qwen2.5 VL 72B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-72b-instruct:free", 'name': "Qwen2.5 VL 72B Instruct"},
        'Qwen2.5-VL 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-vl-7b-instruct:free", 'name': "Qwen2.5-VL 7B Instruct"},
        'Qwerky 72B': {'mode': True, 'id': "featherless/qwerky-72b:free", 'name': "Qwerky 72B"},
        'QwQ 32B': {'mode': True, 'id': "qwen/qwq-32b:free", 'name': "QwQ 32B"},
        'QwQ 32B Preview': {'mode': True, 'id': "qwen/qwq-32b-preview:free", 'name': "QwQ 32B Preview"},
        'QwQ 32B RpR v1': {'mode': True, 'id': "arliai/qwq-32b-arliai-rpr-v1:free", 'name': "QwQ 32B RpR v1"},
        'R1': {'mode': True, 'id': "deepseek/deepseek-r1:free", 'name': "R1"},
        'R1 Distill Llama 70B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-llama-70b:free", 'name': "R1 Distill Llama 70B"},
        'R1 Distill Qwen 14B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-14b:free", 'name': "R1 Distill Qwen 14B"},
        'R1 Distill Qwen 32B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-32b:free", 'name': "R1 Distill Qwen 32B"},
        'Rogue Rose 103B v0.2': {'mode': True, 'id': "sophosympatheia/rogue-rose-103b-v0.2:free", 'name': "Rogue Rose 103B v0.2"},
        'UI-TARS 72B ': {'mode': True, 'id': "bytedance-research/ui-tars-72b:free", 'name': "UI-TARS 72B "},
        'Zephyr 7B': {'mode': True, 'id': "huggingfaceh4/zephyr-7b-beta:free", 'name': "Zephyr 7B"},  
        
        # Pro
        'Aion-1.0': {'mode': True, 'id': "aion-labs/aion-1.0", 'name': "Aion-1.0"},
        'Aion-1.0-Mini': {'mode': True, 'id': "aion-labs/aion-1.0-mini", 'name': "Aion-1.0-Mini"},
        'Aion-RP 1.0 (8B)': {'mode': True, 'id': "aion-labs/aion-rp-llama-3.1-8b", 'name': "Aion-RP 1.0 (8B)"},
        'Airoboros 70B': {'mode': True, 'id': "jondurbin/airoboros-l2-70b", 'name': "Airoboros 70B"},
        'Anubis Pro 105B V1': {'mode': True, 'id': "thedrummer/anubis-pro-105b-v1", 'name': "Anubis Pro 105B V1"},
        'Arctic Instruct': {'mode': True, 'id': "snowflake/snowflake-arctic-instruct", 'name': "Arctic Instruct"},
        'Auto Router': {'mode': True, 'id': "openrouter/auto", 'name': "Auto Router"},
        'Bagel 34B v0.2': {'mode': True, 'id': "jondurbin/bagel-34b", 'name': "Bagel 34B v0.2"},
        'Capybara 34B': {'mode': True, 'id': "nousresearch/nous-capybara-34b", 'name': "Capybara 34B"},
        'Capybara 7B': {'mode': True, 'id': "nousresearch/nous-capybara-7b", 'name': "Capybara 7B"},
        'ChatGPT-4o': {'mode': True, 'id': "openai/chatgpt-4o-latest", 'name': "ChatGPT-4o"},
        'Chronos Hermes 13B v2': {'mode': True, 'id': "austism/chronos-hermes-13b", 'name': "Chronos Hermes 13B v2"},
        'Cinematika 7B (alpha)': {'mode': True, 'id': "openrouter/cinematika-7b", 'name': "Cinematika 7B (alpha)"},
        'Claude 3 Haiku': {'mode': True, 'id': "anthropic/claude-3-haiku", 'name': "Claude 3 Haiku"},
        'Claude 3 Haiku (self-moderated)': {'mode': True, 'id': "anthropic/claude-3-haiku:beta", 'name': "Claude 3 Haiku (self-moderated)"},
        'Claude 3 Opus': {'mode': True, 'id': "anthropic/claude-3-opus", 'name': "Claude 3 Opus"},
        'Claude 3 Opus (self-moderated)': {'mode': True, 'id': "anthropic/claude-3-opus:beta", 'name': "Claude 3 Opus (self-moderated)"},
        'Claude 3 Sonnet': {'mode': True, 'id': "anthropic/claude-3-sonnet", 'name': "Claude 3 Sonnet"},
        'Claude 3 Sonnet (self-moderated)': {'mode': True, 'id': "anthropic/claude-3-sonnet:beta", 'name': "Claude 3 Sonnet (self-moderated)"},
        'Claude 3.5 Haiku': {'mode': True, 'id': "anthropic/claude-3.5-haiku", 'name': "Claude 3.5 Haiku"},
        'Claude 3.5 Haiku (2024-10-22)': {'mode': True, 'id': "anthropic/claude-3.5-haiku-20241022", 'name': "Claude 3.5 Haiku (2024-10-22)"},
        'Claude 3.5 Haiku (2024-10-22) (self-moderated)': {'mode': True, 'id': "anthropic/claude-3.5-haiku-20241022:beta", 'name': "Claude 3.5 Haiku (2024-10-22) (self-moderated)"},
        'Claude 3.5 Haiku (self-moderated)': {'mode': True, 'id': "anthropic/claude-3.5-haiku:beta", 'name': "Claude 3.5 Haiku (self-moderated)"},
        'Claude 3.5 Sonnet': {'mode': True, 'id': "anthropic/claude-3.5-sonnet", 'name': "Claude 3.5 Sonnet"},
        'Claude 3.5 Sonnet (2024-06-20)': {'mode': True, 'id': "anthropic/claude-3.5-sonnet-20240620", 'name': "Claude 3.5 Sonnet (2024-06-20)"},
        'Claude 3.5 Sonnet (2024-06-20) (self-moderated)': {'mode': True, 'id': "anthropic/claude-3.5-sonnet-20240620:beta", 'name': "Claude 3.5 Sonnet (2024-06-20) (self-moderated)"},
        'Claude 3.5 Sonnet (self-moderated)': {'mode': True, 'id': "anthropic/claude-3.5-sonnet:beta", 'name': "Claude 3.5 Sonnet (self-moderated)"},
        'Claude 3.7 Sonnet': {'mode': True, 'id': "anthropic/claude-3.7-sonnet", 'name': "Claude 3.7 Sonnet"},
        'Claude 3.7 Sonnet (self-moderated)': {'mode': True, 'id': "anthropic/claude-3.7-sonnet:beta", 'name': "Claude 3.7 Sonnet (self-moderated)"},
        'Claude 3.7 Sonnet (thinking)': {'mode': True, 'id': "anthropic/claude-3.7-sonnet:thinking", 'name': "Claude 3.7 Sonnet (thinking)"},
        'Claude Instant v1': {'mode': True, 'id': "anthropic/claude-instant-1", 'name': "Claude Instant v1"},
        'Claude Instant v1.0': {'mode': True, 'id': "anthropic/claude-instant-1.0", 'name': "Claude Instant v1.0"},
        'Claude Instant v1.1': {'mode': True, 'id': "anthropic/claude-instant-1.1", 'name': "Claude Instant v1.1"},
        'Claude v1': {'mode': True, 'id': "anthropic/claude-1", 'name': "Claude v1"},
        'Claude v1.2': {'mode': True, 'id': "anthropic/claude-1.2", 'name': "Claude v1.2"},
        'Claude v2': {'mode': True, 'id': "anthropic/claude-2", 'name': "Claude v2"},
        'Claude v2 (self-moderated)': {'mode': True, 'id': "anthropic/claude-2:beta", 'name': "Claude v2 (self-moderated)"},
        'Claude v2.0': {'mode': True, 'id': "anthropic/claude-2.0", 'name': "Claude v2.0"},
        'Claude v2.0 (self-moderated)': {'mode': True, 'id': "anthropic/claude-2.0:beta", 'name': "Claude v2.0 (self-moderated)"},
        'Claude v2.1': {'mode': True, 'id': "anthropic/claude-2.1", 'name': "Claude v2.1"},
        'Claude v2.1 (self-moderated)': {'mode': True, 'id': "anthropic/claude-2.1:beta", 'name': "Claude v2.1 (self-moderated)"},
        'CodeLlama 34B Instruct': {'mode': True, 'id': "meta-llama/codellama-34b-instruct", 'name': "CodeLlama 34B Instruct"},
        'CodeLlama 34B v2': {'mode': True, 'id': "phind/phind-codellama-34b", 'name': "CodeLlama 34B v2"},
        'CodeLlama 70B Instruct': {'mode': True, 'id': "meta-llama/codellama-70b-instruct", 'name': "CodeLlama 70B Instruct"},
        'CodeLLaMa 7B Instruct Solidity': {'mode': True, 'id': "alfredpros/codellama-7b-instruct-solidity", 'name': "CodeLLaMa 7B Instruct Solidity"},
        'Codestral 2501': {'mode': True, 'id': "mistralai/codestral-2501", 'name': "Codestral 2501"},
        'Codestral Mamba': {'mode': True, 'id': "mistralai/codestral-mamba", 'name': "Codestral Mamba"},
        'Command': {'mode': True, 'id': "cohere/command", 'name': "Command"},
        'Command A': {'mode': True, 'id': "cohere/command-a", 'name': "Command A"},
        'Command R': {'mode': True, 'id': "cohere/command-r", 'name': "Command R"},
        'Command R (03-2024)': {'mode': True, 'id': "cohere/command-r-03-2024", 'name': "Command R (03-2024)"},
        'Command R (08-2024)': {'mode': True, 'id': "cohere/command-r-08-2024", 'name': "Command R (08-2024)"},
        'Command R+': {'mode': True, 'id': "cohere/command-r-plus", 'name': "Command R+"},
        'Command R+ (04-2024)': {'mode': True, 'id': "cohere/command-r-plus-04-2024", 'name': "Command R+ (04-2024)"},
        'Command R+ (08-2024)': {'mode': True, 'id': "cohere/command-r-plus-08-2024", 'name': "Command R+ (08-2024)"},
        'Command R7B (12-2024)': {'mode': True, 'id': "cohere/command-r7b-12-2024", 'name': "Command R7B (12-2024)"},
        'DBRX 132B Instruct': {'mode': True, 'id': "databricks/dbrx-instruct", 'name': "DBRX 132B Instruct"},
        'DeepSeek R1': {'mode': True, 'id': "deepseek/deepseek-r1", 'name': "DeepSeek R1"},
        'DeepSeek V2.5': {'mode': True, 'id': "deepseek/deepseek-chat-v2.5", 'name': "DeepSeek V2.5"},
        'DeepSeek V3': {'mode': True, 'id': "deepseek/deepseek-chat", 'name': "DeepSeek V3"},
        'DeepSeek V3 0324': {'mode': True, 'id': "deepseek/deepseek-chat-v3-0324", 'name': "DeepSeek V3 0324"},
        'DeepSeek-Coder-V2': {'mode': True, 'id': "deepseek/deepseek-coder", 'name': "DeepSeek-Coder-V2"},
        'Dolphin 2.6 Mixtral 8x7B \uD83D\uDC2C': {'mode': True, 'id': "cognitivecomputations/dolphin-mixtral-8x7b", 'name': "Dolphin 2.6 Mixtral 8x7B \uD83D\uDC2C"},
        'Dolphin 2.9.2 Mixtral 8x22B \uD83D\uDC2C': {'mode': True, 'id': "cognitivecomputations/dolphin-mixtral-8x22b", 'name': "Dolphin 2.9.2 Mixtral 8x22B \uD83D\uDC2C"},
        'Dolphin Llama 3 70B \uD83D\uDC2C': {'mode': True, 'id': "cognitivecomputations/dolphin-llama-3-70b", 'name': "Dolphin Llama 3 70B \uD83D\uDC2C"},
        'Eagle 7B': {'mode': True, 'id': "recursal/eagle-7b", 'name': "Eagle 7B"},
        'EVA Llama 3.33 70B': {'mode': True, 'id': "eva-unit-01/eva-llama-3.33-70b", 'name': "EVA Llama 3.33 70B"},
        'EVA Qwen2.5 14B': {'mode': True, 'id': "eva-unit-01/eva-qwen-2.5-14b", 'name': "EVA Qwen2.5 14B"},
        'EVA Qwen2.5 32B': {'mode': True, 'id': "eva-unit-01/eva-qwen-2.5-32b", 'name': "EVA Qwen2.5 32B"},
        'EVA Qwen2.5 72B': {'mode': True, 'id': "eva-unit-01/eva-qwen-2.5-72b", 'name': "EVA Qwen2.5 72B"},
        'Fimbulvetr 11B v2': {'mode': True, 'id': "sao10k/fimbulvetr-11b-v2", 'name': "Fimbulvetr 11B v2"},
        'FireLLaVA 13B': {'mode': True, 'id': "fireworks/firellava-13b", 'name': "FireLLaVA 13B"},
        'Gemini 1.5 Flash ': {'mode': True, 'id': "google/gemini-flash-1.5", 'name': "Gemini 1.5 Flash "},
        'Gemini 1.5 Flash 8B': {'mode': True, 'id': "google/gemini-flash-1.5-8b", 'name': "Gemini 1.5 Flash 8B"},
        'Gemini 1.5 Flash 8B Experimental': {'mode': True, 'id': "google/gemini-flash-1.5-8b-exp", 'name': "Gemini 1.5 Flash 8B Experimental"},
        'Gemini 1.5 Flash Experimental': {'mode': True, 'id': "google/gemini-flash-1.5-exp", 'name': "Gemini 1.5 Flash Experimental"},
        'Gemini 1.5 Pro': {'mode': True, 'id': "google/gemini-pro-1.5", 'name': "Gemini 1.5 Pro"},
        'Gemini 1.5 Pro Experimental': {'mode': True, 'id': "google/gemini-pro-1.5-exp", 'name': "Gemini 1.5 Pro Experimental"},
        'Gemini 2.0 Flash': {'mode': True, 'id': "google/gemini-2.0-flash-001", 'name': "Gemini 2.0 Flash"},
        'Gemini 2.0 Flash Lite': {'mode': True, 'id': "google/gemini-2.0-flash-lite-001", 'name': "Gemini 2.0 Flash Lite"},
        'Gemini 2.5 Pro': {'mode': True, 'id': "google/gemini-2.5-pro-preview-03-25", 'name': "Gemini 2.5 Pro"},
        'Gemini Experimental 1114': {'mode': True, 'id': "google/gemini-exp-1114", 'name': "Gemini Experimental 1114"},
        'Gemini Experimental 1121': {'mode': True, 'id': "google/gemini-exp-1121", 'name': "Gemini Experimental 1121"},
        'Gemini Pro 1.0': {'mode': True, 'id': "google/gemini-pro", 'name': "Gemini Pro 1.0"},
        'Gemini Pro Vision 1.0': {'mode': True, 'id': "google/gemini-pro-vision", 'name': "Gemini Pro Vision 1.0"},
        'Gemma 2 27B': {'mode': True, 'id': "google/gemma-2-27b-it", 'name': "Gemma 2 27B"},
        'Gemma 2 9B': {'mode': True, 'id': "google/gemma-2-9b-it", 'name': "Gemma 2 9B"},
        'Gemma 3 12B': {'mode': True, 'id': "google/gemma-3-12b-it", 'name': "Gemma 3 12B"},
        'Gemma 3 27B': {'mode': True, 'id': "google/gemma-3-27b-it", 'name': "Gemma 3 27B"},
        'Gemma 3 4B': {'mode': True, 'id': "google/gemma-3-4b-it", 'name': "Gemma 3 4B"},
        'Gemma 7B': {'mode': True, 'id': "google/gemma-7b-it", 'name': "Gemma 7B"},
        'Goliath 120B': {'mode': True, 'id': "alpindale/goliath-120b", 'name': "Goliath 120B"},
        'GPT-3.5 Turbo': {'mode': True, 'id': "openai/gpt-3.5-turbo", 'name': "GPT-3.5 Turbo"},
        'GPT-3.5 Turbo (older v0301)': {'mode': True, 'id': "openai/gpt-3.5-turbo-0301", 'name': "GPT-3.5 Turbo (older v0301)"},
        'GPT-3.5 Turbo (older v0613)': {'mode': True, 'id': "openai/gpt-3.5-turbo-0613", 'name': "GPT-3.5 Turbo (older v0613)"},
        'GPT-3.5 Turbo 16k': {'mode': True, 'id': "openai/gpt-3.5-turbo-16k", 'name': "GPT-3.5 Turbo 16k"},
        'GPT-3.5 Turbo 16k': {'mode': True, 'id': "openai/gpt-3.5-turbo-0125", 'name': "GPT-3.5 Turbo 16k"},
        'GPT-3.5 Turbo 16k (older v1106)': {'mode': True, 'id': "openai/gpt-3.5-turbo-1106", 'name': "GPT-3.5 Turbo 16k (older v1106)"},
        'GPT-3.5 Turbo Instruct': {'mode': True, 'id': "openai/gpt-3.5-turbo-instruct", 'name': "GPT-3.5 Turbo Instruct"},
        'GPT-4': {'mode': True, 'id': "openai/gpt-4", 'name': "GPT-4"},
        'GPT-4 (older v0314)': {'mode': True, 'id': "openai/gpt-4-0314", 'name': "GPT-4 (older v0314)"},
        'GPT-4 32k': {'mode': True, 'id': "openai/gpt-4-32k", 'name': "GPT-4 32k"},
        'GPT-4 32k (older v0314)': {'mode': True, 'id': "openai/gpt-4-32k-0314", 'name': "GPT-4 32k (older v0314)"},
        'GPT-4 Turbo': {'mode': True, 'id': "openai/gpt-4-turbo", 'name': "GPT-4 Turbo"},
        'GPT-4 Turbo (older v1106)': {'mode': True, 'id': "openai/gpt-4-1106-preview", 'name': "GPT-4 Turbo (older v1106)"},
        'GPT-4 Turbo Preview': {'mode': True, 'id': "openai/gpt-4-turbo-preview", 'name': "GPT-4 Turbo Preview"},
        'GPT-4 Vision': {'mode': True, 'id': "openai/gpt-4-vision-preview", 'name': "GPT-4 Vision"},
        'GPT-4.1': {'mode': True, 'id': "openai/gpt-4.1", 'name': "GPT-4.1"},
        'GPT-4.1 Mini': {'mode': True, 'id': "openai/gpt-4.1-mini", 'name': "GPT-4.1 Mini"},
        'GPT-4.1 Nano': {'mode': True, 'id': "openai/gpt-4.1-nano", 'name': "GPT-4.1 Nano"},
        'GPT-4.5 (Preview)': {'mode': True, 'id': "openai/gpt-4.5-preview", 'name': "GPT-4.5 (Preview)"},
        'GPT-4o': {'mode': True, 'id': "openai/gpt-4o", 'name': "GPT-4o"},
        'GPT-4o (2024-05-13)': {'mode': True, 'id': "openai/gpt-4o-2024-05-13", 'name': "GPT-4o (2024-05-13)"},
        'GPT-4o (2024-08-06)': {'mode': True, 'id': "openai/gpt-4o-2024-08-06", 'name': "GPT-4o (2024-08-06)"},
        'GPT-4o (2024-11-20)': {'mode': True, 'id': "openai/gpt-4o-2024-11-20", 'name': "GPT-4o (2024-11-20)"},
        'GPT-4o (extended)': {'mode': True, 'id': "openai/gpt-4o:extended", 'name': "GPT-4o (extended)"},
        'GPT-4o Search Preview': {'mode': True, 'id': "openai/gpt-4o-search-preview", 'name': "GPT-4o Search Preview"},
        'GPT-4o-mini': {'mode': True, 'id': "openai/gpt-4o-mini", 'name': "GPT-4o-mini"},
        'GPT-4o-mini (2024-07-18)': {'mode': True, 'id': "openai/gpt-4o-mini-2024-07-18", 'name': "GPT-4o-mini (2024-07-18)"},
        'GPT-4o-mini Search Preview': {'mode': True, 'id': "openai/gpt-4o-mini-search-preview", 'name': "GPT-4o-mini Search Preview"},
        'Grok 2': {'mode': True, 'id': "x-ai/grok-2", 'name': "Grok 2"},
        'Grok 2 1212': {'mode': True, 'id': "x-ai/grok-2-1212", 'name': "Grok 2 1212"},
        'Grok 2 mini': {'mode': True, 'id': "x-ai/grok-2-mini", 'name': "Grok 2 mini"},
        'Grok 2 Vision 1212': {'mode': True, 'id': "x-ai/grok-2-vision-1212", 'name': "Grok 2 Vision 1212"},
        'Grok 3': {'mode': True, 'id': "x-ai/grok-3-beta", 'name': "Grok 3"},
        'Grok 3 Mini Beta': {'mode': True, 'id': "x-ai/grok-3-mini-beta", 'name': "Grok 3 Mini Beta"},
        'Grok Beta': {'mode': True, 'id': "x-ai/grok-beta", 'name': "Grok Beta"},
        'Grok Vision Beta': {'mode': True, 'id': "x-ai/grok-vision-beta", 'name': "Grok Vision Beta"},
        'Hermes 13B': {'mode': True, 'id': "nousresearch/nous-hermes-llama2-13b", 'name': "Hermes 13B"},
        'Hermes 2 Mistral 7B DPO': {'mode': True, 'id': "nousresearch/nous-hermes-2-mistral-7b-dpo", 'name': "Hermes 2 Mistral 7B DPO"},
        'Hermes 2 Mixtral 8x7B DPO': {'mode': True, 'id': "nousresearch/nous-hermes-2-mixtral-8x7b-dpo", 'name': "Hermes 2 Mixtral 8x7B DPO"},
        'Hermes 2 Mixtral 8x7B SFT': {'mode': True, 'id': "nousresearch/nous-hermes-2-mixtral-8x7b-sft", 'name': "Hermes 2 Mixtral 8x7B SFT"},
        'Hermes 2 Pro - Llama-3 8B': {'mode': True, 'id': "nousresearch/hermes-2-pro-llama-3-8b", 'name': "Hermes 2 Pro - Llama-3 8B"},
        'Hermes 2 Theta 8B': {'mode': True, 'id': "nousresearch/hermes-2-theta-llama-3-8b", 'name': "Hermes 2 Theta 8B"},
        'Hermes 2 Vision 7B (alpha)': {'mode': True, 'id': "nousresearch/nous-hermes-2-vision-7b", 'name': "Hermes 2 Vision 7B (alpha)"},
        'Hermes 2 Yi 34B': {'mode': True, 'id': "nousresearch/nous-hermes-yi-34b", 'name': "Hermes 2 Yi 34B"},
        'Hermes 3 405B Instruct': {'mode': True, 'id': "nousresearch/hermes-3-llama-3.1-405b", 'name': "Hermes 3 405B Instruct"},
        'Hermes 3 70B Instruct': {'mode': True, 'id': "nousresearch/hermes-3-llama-3.1-70b", 'name': "Hermes 3 70B Instruct"},
        'Hermes 70B': {'mode': True, 'id': "nousresearch/nous-hermes-llama2-70b", 'name': "Hermes 70B"},
        'Inflection 3 Pi': {'mode': True, 'id': "inflection/inflection-3-pi", 'name': "Inflection 3 Pi"},
        'Inflection 3 Productivity': {'mode': True, 'id': "inflection/inflection-3-productivity", 'name': "Inflection 3 Productivity"},
        'Jamba 1.5 Large': {'mode': True, 'id': "ai21/jamba-1-5-large", 'name': "Jamba 1.5 Large"},
        'Jamba 1.5 Mini': {'mode': True, 'id': "ai21/jamba-1-5-mini", 'name': "Jamba 1.5 Mini"},
        'Jamba 1.6 Large': {'mode': True, 'id': "ai21/jamba-1.6-large", 'name': "Jamba 1.6 Large"},
        'Jamba Instruct': {'mode': True, 'id': "ai21/jamba-instruct", 'name': "Jamba Instruct"},
        'Jamba Mini 1.6': {'mode': True, 'id': "ai21/jamba-1.6-mini", 'name': "Jamba Mini 1.6"},
        'L3.3 Electra R1 70B': {'mode': True, 'id': "steelskull/l3.3-electra-r1-70b", 'name': "L3.3 Electra R1 70B"},
        'LFM 3B': {'mode': True, 'id': "liquid/lfm-3b", 'name': "LFM 3B"},
        'LFM 40B MoE': {'mode': True, 'id': "liquid/lfm-40b", 'name': "LFM 40B MoE"},
        'LFM 7B': {'mode': True, 'id': "liquid/lfm-7b", 'name': "LFM 7B"},
        'Llama 2 13B Chat': {'mode': True, 'id': "meta-llama/llama-2-13b-chat", 'name': "Llama 2 13B Chat"},
        'Llama 2 70B Chat': {'mode': True, 'id': "meta-llama/llama-2-70b-chat", 'name': "Llama 2 70B Chat"},
        'Llama 3 70B (Base)': {'mode': True, 'id': "meta-llama/llama-3-70b", 'name': "Llama 3 70B (Base)"},
        'Llama 3 70B Instruct': {'mode': True, 'id': "meta-llama/llama-3-70b-instruct", 'name': "Llama 3 70B Instruct"},
        'Llama 3 8B (Base)': {'mode': True, 'id': "meta-llama/llama-3-8b", 'name': "Llama 3 8B (Base)"},
        'Llama 3 8B Instruct': {'mode': True, 'id': "meta-llama/llama-3-8b-instruct", 'name': "Llama 3 8B Instruct"},
        'Llama 3 8B Lunaris': {'mode': True, 'id': "sao10k/l3-lunaris-8b", 'name': "Llama 3 8B Lunaris"},
        'Llama 3 Euryale 70B v2.1': {'mode': True, 'id': "sao10k/l3-euryale-70b", 'name': "Llama 3 Euryale 70B v2.1"},
        'Llama 3 Lumimaid 70B': {'mode': True, 'id': "neversleep/llama-3-lumimaid-70b", 'name': "Llama 3 Lumimaid 70B"},
        'Llama 3 Lumimaid 8B': {'mode': True, 'id': "neversleep/llama-3-lumimaid-8b", 'name': "Llama 3 Lumimaid 8B"},
        'Llama 3 Lumimaid 8B (extended)': {'mode': True, 'id': "neversleep/llama-3-lumimaid-8b:extended", 'name': "Llama 3 Lumimaid 8B (extended)"},
        'Llama 3 Soliloquy 7B v3 32K': {'mode': True, 'id': "lynn/soliloquy-v3", 'name': "Llama 3 Soliloquy 7B v3 32K"},
        'Llama 3 Soliloquy 8B v2': {'mode': True, 'id': "lynn/soliloquy-l3", 'name': "Llama 3 Soliloquy 8B v2"},
        'Llama 3 Stheno 8B v3.3 32K': {'mode': True, 'id': "sao10k/l3-stheno-8b", 'name': "Llama 3 Stheno 8B v3.3 32K"},
        'Llama 3.1 405B (base)': {'mode': True, 'id': "meta-llama/llama-3.1-405b", 'name': "Llama 3.1 405B (base)"},
        'Llama 3.1 405B Instruct': {'mode': True, 'id': "meta-llama/llama-3.1-405b-instruct", 'name': "Llama 3.1 405B Instruct"},
        'Llama 3.1 70B Hanami x1': {'mode': True, 'id': "sao10k/l3.1-70b-hanami-x1", 'name': "Llama 3.1 70B Hanami x1"},
        'Llama 3.1 70B Instruct': {'mode': True, 'id': "meta-llama/llama-3.1-70b-instruct", 'name': "Llama 3.1 70B Instruct"},
        'Llama 3.1 8B Instruct': {'mode': True, 'id': "meta-llama/llama-3.1-8b-instruct", 'name': "Llama 3.1 8B Instruct"},
        'Llama 3.1 Euryale 70B v2.2': {'mode': True, 'id': "sao10k/l3.1-euryale-70b", 'name': "Llama 3.1 Euryale 70B v2.2"},
        'Llama 3.1 Nemotron 70B Instruct': {'mode': True, 'id': "nvidia/llama-3.1-nemotron-70b-instruct", 'name': "Llama 3.1 Nemotron 70B Instruct"},
        'Llama 3.1 Swallow 70B Instruct V0.3': {'mode': True, 'id': "tokyotech-llm/llama-3.1-swallow-70b-instruct-v0.3", 'name': "Llama 3.1 Swallow 70B Instruct V0.3"},
        'Llama 3.1 Swallow 8B Instruct V0.3': {'mode': True, 'id': "tokyotech-llm/llama-3.1-swallow-8b-instruct-v0.3", 'name': "Llama 3.1 Swallow 8B Instruct V0.3"},
        'Llama 3.1 Tulu 3 405B': {'mode': True, 'id': "allenai/llama-3.1-tulu-3-405b", 'name': "Llama 3.1 Tulu 3 405B"},
        'Llama 3.2 11B Vision Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-11b-vision-instruct", 'name': "Llama 3.2 11B Vision Instruct"},
        'Llama 3.2 1B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-1b-instruct", 'name': "Llama 3.2 1B Instruct"},
        'Llama 3.2 3B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-3b-instruct", 'name': "Llama 3.2 3B Instruct"},
        'Llama 3.2 90B Vision Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-90b-vision-instruct", 'name': "Llama 3.2 90B Vision Instruct"},
        'Llama 3.3 70B Instruct': {'mode': True, 'id': "meta-llama/llama-3.3-70b-instruct", 'name': "Llama 3.3 70B Instruct"},
        'Llama 3.3 Euryale 70B': {'mode': True, 'id': "sao10k/l3.3-euryale-70b", 'name': "Llama 3.3 Euryale 70B"},
        'Llama 4 Maverick': {'mode': True, 'id': "meta-llama/llama-4-maverick", 'name': "Llama 4 Maverick"},
        'Llama 4 Scout': {'mode': True, 'id': "meta-llama/llama-4-scout", 'name': "Llama 4 Scout"},
        'Llama Guard 3 8B': {'mode': True, 'id': "meta-llama/llama-guard-3-8b", 'name': "Llama Guard 3 8B"},
        'LlamaGuard 2 8B': {'mode': True, 'id': "meta-llama/llama-guard-2-8b", 'name': "LlamaGuard 2 8B"},
        'LLaVA 13B': {'mode': True, 'id': "liuhaotian/llava-13b", 'name': "LLaVA 13B"},
        'LLaVA v1.6 34B': {'mode': True, 'id': "liuhaotian/llava-yi-34b", 'name': "LLaVA v1.6 34B"},
        'Llemma 7b': {'mode': True, 'id': "eleutherai/llemma_7b", 'name': "Llemma 7b"},
        'Lumimaid v0.2 70B': {'mode': True, 'id': "neversleep/llama-3.1-lumimaid-70b", 'name': "Lumimaid v0.2 70B"},
        'Lumimaid v0.2 8B': {'mode': True, 'id': "neversleep/llama-3.1-lumimaid-8b", 'name': "Lumimaid v0.2 8B"},
        'lzlv 70B': {'mode': True, 'id': "lizpreciatior/lzlv-70b-fp16-hf", 'name': "lzlv 70B"},
        'Mag Mell R1 12B': {'mode': True, 'id': "inflatebot/mn-mag-mell-r1", 'name': "Mag Mell R1 12B"},
        'Magnum 72B': {'mode': True, 'id': "alpindale/magnum-72b", 'name': "Magnum 72B"},
        'Magnum v2 72B': {'mode': True, 'id': "anthracite-org/magnum-v2-72b", 'name': "Magnum v2 72B"},
        'Magnum v4 72B': {'mode': True, 'id': "anthracite-org/magnum-v4-72b", 'name': "Magnum v4 72B"},
        'Midnight Rose 70B': {'mode': True, 'id': "sophosympatheia/midnight-rose-70b", 'name': "Midnight Rose 70B"},
        'MiniMax-01': {'mode': True, 'id': "minimax/minimax-01", 'name': "MiniMax-01"},
        'Ministral 3B': {'mode': True, 'id': "mistralai/ministral-3b", 'name': "Ministral 3B"},
        'Ministral 8B': {'mode': True, 'id': "mistral/ministral-8b", 'name': "Ministral 8B"},
        'Ministral 8B': {'mode': True, 'id': "mistralai/ministral-8b", 'name': "Ministral 8B"},
        'Mistral 7B Instruct': {'mode': True, 'id': "mistralai/mistral-7b-instruct", 'name': "Mistral 7B Instruct"},
        'Mistral 7B Instruct v0.1': {'mode': True, 'id': "mistralai/mistral-7b-instruct-v0.1", 'name': "Mistral 7B Instruct v0.1"},
        'Mistral 7B Instruct v0.2': {'mode': True, 'id': "mistralai/mistral-7b-instruct-v0.2", 'name': "Mistral 7B Instruct v0.2"},
        'Mistral 7B Instruct v0.3': {'mode': True, 'id': "mistralai/mistral-7b-instruct-v0.3", 'name': "Mistral 7B Instruct v0.3"},
        'Mistral Large': {'mode': True, 'id': "mistralai/mistral-large", 'name': "Mistral Large"},
        'Mistral Large 2407': {'mode': True, 'id': "mistralai/mistral-large-2407", 'name': "Mistral Large 2407"},
        'Mistral Large 2411': {'mode': True, 'id': "mistralai/mistral-large-2411", 'name': "Mistral Large 2411"},
        'Mistral Medium': {'mode': True, 'id': "mistralai/mistral-medium", 'name': "Mistral Medium"},
        'Mistral Nemo': {'mode': True, 'id': "mistralai/mistral-nemo", 'name': "Mistral Nemo"},
        'Mistral Nemo 12B Celeste': {'mode': True, 'id': "nothingiisreal/mn-celeste-12b", 'name': "Mistral Nemo 12B Celeste"},
        'Mistral Nemo Inferor 12B': {'mode': True, 'id': "infermatic/mn-inferor-12b", 'name': "Mistral Nemo Inferor 12B"},
        'Mistral OpenOrca 7B': {'mode': True, 'id': "open-orca/mistral-7b-openorca", 'name': "Mistral OpenOrca 7B"},
        'Mistral Small': {'mode': True, 'id': "mistralai/mistral-small", 'name': "Mistral Small"},
        'Mistral Small 3': {'mode': True, 'id': "mistralai/mistral-small-24b-instruct-2501", 'name': "Mistral Small 3"},
        'Mistral Small 3.1 24B': {'mode': True, 'id': "mistralai/mistral-small-3.1-24b-instruct", 'name': "Mistral Small 3.1 24B"},
        'Mistral Tiny': {'mode': True, 'id': "mistralai/mistral-tiny", 'name': "Mistral Tiny"},
        'Mixtral 8x22B (base)': {'mode': True, 'id': "mistralai/mixtral-8x22b", 'name': "Mixtral 8x22B (base)"},
        'Mixtral 8x22B Instruct': {'mode': True, 'id': "mistralai/mixtral-8x22b-instruct", 'name': "Mixtral 8x22B Instruct"},
        'Mixtral 8x7B Instruct': {'mode': True, 'id': "mistralai/mixtral-8x7b-instruct", 'name': "Mixtral 8x7B Instruct"},
        'Mythalion 13B': {'mode': True, 'id': "pygmalionai/mythalion-13b", 'name': "Mythalion 13B"},
        'MythoMax 13B': {'mode': True, 'id': "gryphe/mythomax-l2-13b", 'name': "MythoMax 13B"},
        'MythoMist 7B': {'mode': True, 'id': "gryphe/mythomist-7b", 'name': "MythoMist 7B"},
        'Nemotron-4 340B Instruct': {'mode': True, 'id': "nvidia/nemotron-4-340b-instruct", 'name': "Nemotron-4 340B Instruct"},
        'Neural Chat 7B v3.1': {'mode': True, 'id': "intel/neural-chat-7b", 'name': "Neural Chat 7B v3.1"},
        'Noromaid 20B': {'mode': True, 'id': "neversleep/noromaid-20b", 'name': "Noromaid 20B"},
        'Noromaid Mixtral 8x7B Instruct': {'mode': True, 'id': "neversleep/noromaid-mixtral-8x7b-instruct", 'name': "Noromaid Mixtral 8x7B Instruct"},
        'Nova Lite 1.0': {'mode': True, 'id': "amazon/nova-lite-v1", 'name': "Nova Lite 1.0"},
        'Nova Micro 1.0': {'mode': True, 'id': "amazon/nova-micro-v1", 'name': "Nova Micro 1.0"},
        'Nova Pro 1.0': {'mode': True, 'id': "amazon/nova-pro-v1", 'name': "Nova Pro 1.0"},
        'o1': {'mode': True, 'id': "openai/o1", 'name': "o1"},
        'o1-mini': {'mode': True, 'id': "openai/o1-mini", 'name': "o1-mini"},
        'o1-mini (2024-09-12)': {'mode': True, 'id': "openai/o1-mini-2024-09-12", 'name': "o1-mini (2024-09-12)"},
        'o1-preview': {'mode': True, 'id': "openai/o1-preview", 'name': "o1-preview"},
        'o1-preview (2024-09-12)': {'mode': True, 'id': "openai/o1-preview-2024-09-12", 'name': "o1-preview (2024-09-12)"},
        'o1-pro': {'mode': True, 'id': "openai/o1-pro", 'name': "o1-pro"},
        'o3 Mini': {'mode': True, 'id': "openai/o3-mini", 'name': "o3 Mini"},
        'o3 Mini High': {'mode': True, 'id': "openai/o3-mini-high", 'name': "o3 Mini High"},
        'Olmo 2 32B Instruct': {'mode': True, 'id': "allenai/olmo-2-0325-32b-instruct", 'name': "Olmo 2 32B Instruct"},
        'OLMo 7B Instruct': {'mode': True, 'id': "allenai/olmo-7b-instruct", 'name': "OLMo 7B Instruct"},
        'OpenChat 3.5 7B': {'mode': True, 'id': "openchat/openchat-7b", 'name': "OpenChat 3.5 7B"},
        'OpenChat 3.6 8B': {'mode': True, 'id': "openchat/openchat-8b", 'name': "OpenChat 3.6 8B"},
        'OpenHands LM 32B V0.1': {'mode': True, 'id': "all-hands/openhands-lm-32b-v0.1", 'name': "OpenHands LM 32B V0.1"},
        'OpenHermes 2 Mistral 7B': {'mode': True, 'id': "teknium/openhermes-2-mistral-7b", 'name': "OpenHermes 2 Mistral 7B"},
        'OpenHermes 2.5 Mistral 7B': {'mode': True, 'id': "teknium/openhermes-2.5-mistral-7b", 'name': "OpenHermes 2.5 Mistral 7B"},
        'Optimus Alpha': {'mode': True, 'id': "openrouter/optimus-alpha", 'name': "Optimus Alpha"},
        'PaLM 2 Chat': {'mode': True, 'id': "google/palm-2-chat-bison", 'name': "PaLM 2 Chat"},
        'PaLM 2 Chat 32k': {'mode': True, 'id': "google/palm-2-chat-bison-32k", 'name': "PaLM 2 Chat 32k"},
        'PaLM 2 Code Chat': {'mode': True, 'id': "google/palm-2-codechat-bison", 'name': "PaLM 2 Code Chat"},
        'PaLM 2 Code Chat 32k': {'mode': True, 'id': "google/palm-2-codechat-bison-32k", 'name': "PaLM 2 Code Chat 32k"},
        'Phi 4': {'mode': True, 'id': "microsoft/phi-4", 'name': "Phi 4"},
        'Phi 4 Multimodal Instruct': {'mode': True, 'id': "microsoft/phi-4-multimodal-instruct", 'name': "Phi 4 Multimodal Instruct"},
        'Phi-3 Medium 128K Instruct': {'mode': True, 'id': "microsoft/phi-3-medium-128k-instruct", 'name': "Phi-3 Medium 128K Instruct"},
        'Phi-3 Medium 4K Instruct': {'mode': True, 'id': "microsoft/phi-3-medium-4k-instruct", 'name': "Phi-3 Medium 4K Instruct"},
        'Phi-3 Mini 128K Instruct': {'mode': True, 'id': "microsoft/phi-3-mini-128k-instruct", 'name': "Phi-3 Mini 128K Instruct"},
        'Phi-3.5 Mini 128K Instruct': {'mode': True, 'id': "microsoft/phi-3.5-mini-128k-instruct", 'name': "Phi-3.5 Mini 128K Instruct"},
        'Pixtral 12B': {'mode': True, 'id': "mistralai/pixtral-12b", 'name': "Pixtral 12B"},
        'Pixtral Large 2411': {'mode': True, 'id': "mistralai/pixtral-large-2411", 'name': "Pixtral Large 2411"},
        'Psyfighter 13B': {'mode': True, 'id': "jebcarter/psyfighter-13b", 'name': "Psyfighter 13B"},
        'Psyfighter v2 13B': {'mode': True, 'id': "koboldai/psyfighter-13b-2", 'name': "Psyfighter v2 13B"},
        'Quasar Alpha': {'mode': True, 'id': "openrouter/quasar-alpha", 'name': "Quasar Alpha"},
        'Qwen 1.5 110B Chat': {'mode': True, 'id': "qwen/qwen-110b-chat", 'name': "Qwen 1.5 110B Chat"},
        'Qwen 1.5 14B Chat': {'mode': True, 'id': "qwen/qwen-14b-chat", 'name': "Qwen 1.5 14B Chat"},
        'Qwen 1.5 32B Chat': {'mode': True, 'id': "qwen/qwen-32b-chat", 'name': "Qwen 1.5 32B Chat"},
        'Qwen 1.5 4B Chat': {'mode': True, 'id': "qwen/qwen-4b-chat", 'name': "Qwen 1.5 4B Chat"},
        'Qwen 1.5 72B Chat': {'mode': True, 'id': "qwen/qwen-72b-chat", 'name': "Qwen 1.5 72B Chat"},
        'Qwen 1.5 7B Chat': {'mode': True, 'id': "qwen/qwen-7b-chat", 'name': "Qwen 1.5 7B Chat"},
        'Qwen 2 72B Instruct': {'mode': True, 'id': "qwen/qwen-2-72b-instruct", 'name': "Qwen 2 72B Instruct"},
        'Qwen 2 7B Instruct': {'mode': True, 'id': "qwen/qwen-2-7b-instruct", 'name': "Qwen 2 7B Instruct"},
        'Qwen VL Max': {'mode': True, 'id': "qwen/qwen-vl-max", 'name': "Qwen VL Max"},
        'Qwen VL Plus': {'mode': True, 'id': "qwen/qwen-vl-plus", 'name': "Qwen VL Plus"},
        'Qwen-Max ': {'mode': True, 'id': "qwen/qwen-max", 'name': "Qwen-Max "},
        'Qwen-Plus': {'mode': True, 'id': "qwen/qwen-plus", 'name': "Qwen-Plus"},
        'Qwen-Turbo': {'mode': True, 'id': "qwen/qwen-turbo", 'name': "Qwen-Turbo"},
        'Qwen2.5 32B Instruct': {'mode': True, 'id': "qwen/qwen2.5-32b-instruct", 'name': "Qwen2.5 32B Instruct"},
        'Qwen2.5 72B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-72b-instruct", 'name': "Qwen2.5 72B Instruct"},
        'Qwen2.5 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-7b-instruct", 'name': "Qwen2.5 7B Instruct"},
        'Qwen2.5 Coder 32B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-coder-32b-instruct", 'name': "Qwen2.5 Coder 32B Instruct"},
        'Qwen2.5 VL 32B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-32b-instruct", 'name': "Qwen2.5 VL 32B Instruct"},
        'Qwen2.5 VL 72B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-72b-instruct", 'name': "Qwen2.5 VL 72B Instruct"},
        'Qwen2.5-VL 72B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-vl-72b-instruct", 'name': "Qwen2.5-VL 72B Instruct"},
        'Qwen2.5-VL 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-vl-7b-instruct", 'name': "Qwen2.5-VL 7B Instruct"},
        'QwQ 32B': {'mode': True, 'id': "qwen/qwq-32b", 'name': "QwQ 32B"},
        'QwQ 32B Preview': {'mode': True, 'id': "qwen/qwq-32b-preview", 'name': "QwQ 32B Preview"},
        'R1': {'mode': True, 'id': "deepseek/deepseek-r1", 'name': "R1"},
        'R1 Distill Llama 70B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-llama-70b", 'name': "R1 Distill Llama 70B"},
        'R1 Distill Llama 8B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-llama-8b", 'name': "R1 Distill Llama 8B"},
        'R1 Distill Qwen 1.5B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-1.5b", 'name': "R1 Distill Qwen 1.5B"},
        'R1 Distill Qwen 14B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-14b", 'name': "R1 Distill Qwen 14B"},
        'R1 Distill Qwen 32B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-32b", 'name': "R1 Distill Qwen 32B"},
        'Reflection 70B': {'mode': True, 'id': "mattshumer/reflection-70b", 'name': "Reflection 70B"},
        'ReMM SLERP 13B': {'mode': True, 'id': "undi95/remm-slerp-l2-13b", 'name': "ReMM SLERP 13B"},
        'Rocinante 12B': {'mode': True, 'id': "thedrummer/rocinante-12b", 'name': "Rocinante 12B"},
        'RWKV v5 3B AI Town': {'mode': True, 'id': "recursal/rwkv-5-3b-ai-town", 'name': "RWKV v5 3B AI Town"},
        'RWKV v5 World 3B': {'mode': True, 'id': "rwkv/rwkv-5-world-3b", 'name': "RWKV v5 World 3B"},
        'Saba': {'mode': True, 'id': "mistralai/mistral-saba", 'name': "Saba"},
        'Skyfall 36B V2': {'mode': True, 'id': "thedrummer/skyfall-36b-v2", 'name': "Skyfall 36B V2"},
        'SorcererLM 8x22B': {'mode': True, 'id': "raifle/sorcererlm-8x22b", 'name': "SorcererLM 8x22B"},
        'Starcannon 12B': {'mode': True, 'id': "aetherwiing/mn-starcannon-12b", 'name': "Starcannon 12B"},
        'StarCoder2 15B Instruct': {'mode': True, 'id': "bigcode/starcoder2-15b-instruct", 'name': "StarCoder2 15B Instruct"},
        'StripedHyena Hessian 7B (base)': {'mode': True, 'id': "togethercomputer/stripedhyena-hessian-7b", 'name': "StripedHyena Hessian 7B (base)"},
        'StripedHyena Nous 7B': {'mode': True, 'id': "togethercomputer/stripedhyena-nous-7b", 'name': "StripedHyena Nous 7B"},
        'Synthia 70B': {'mode': True, 'id': "migtissera/synthia-70b", 'name': "Synthia 70B"},
        'Toppy M 7B': {'mode': True, 'id': "undi95/toppy-m-7b", 'name': "Toppy M 7B"},
        'Typhoon2 70B Instruct': {'mode': True, 'id': "scb10x/llama3.1-typhoon2-70b-instruct", 'name': "Typhoon2 70B Instruct"},
        'Typhoon2 8B Instruct': {'mode': True, 'id': "scb10x/llama3.1-typhoon2-8b-instruct", 'name': "Typhoon2 8B Instruct"},
        'Unslopnemo 12B': {'mode': True, 'id': "thedrummer/unslopnemo-12b", 'name': "Unslopnemo 12B"},
        'Wayfarer Large 70B Llama 3.3': {'mode': True, 'id': "latitudegames/wayfarer-large-70b-llama-3.3", 'name': "Wayfarer Large 70B Llama 3.3"},
        'Weaver (alpha)': {'mode': True, 'id': "mancer/weaver", 'name': "Weaver (alpha)"},
        'WizardLM-2 7B': {'mode': True, 'id': "microsoft/wizardlm-2-7b", 'name': "WizardLM-2 7B"},
        'WizardLM-2 8x22B': {'mode': True, 'id': "microsoft/wizardlm-2-8x22b", 'name': "WizardLM-2 8x22B"},
        'Xwin 70B': {'mode': True, 'id': "xwin-lm/xwin-lm-70b", 'name': "Xwin 70B"},
        'Yi 1.5 34B Chat': {'mode': True, 'id': "01-ai/yi-1.5-34b-chat", 'name': "Yi 1.5 34B Chat"},
        'Yi 34B (base)': {'mode': True, 'id': "01-ai/yi-34b", 'name': "Yi 34B (base)"},
        'Yi 34B 200K': {'mode': True, 'id': "01-ai/yi-34b-200k", 'name': "Yi 34B 200K"},
        'Yi 34B Chat': {'mode': True, 'id': "01-ai/yi-34b-chat", 'name': "Yi 34B Chat"},
        'Yi 6B (base)': {'mode': True, 'id': "01-ai/yi-6b", 'name': "Yi 6B (base)"},
        'Yi Large': {'mode': True, 'id': "01-ai/yi-large", 'name': "Yi Large"},
        'Yi Large FC': {'mode': True, 'id': "01-ai/yi-large-fc", 'name': "Yi Large FC"},
        'Yi Large Turbo': {'mode': True, 'id': "01-ai/yi-large-turbo", 'name': "Yi Large Turbo"},
        'Yi Vision': {'mode': True, 'id': "01-ai/yi-vision", 'name': "Yi Vision"},
        'Zephyr 141B-A35B': {'mode': True, 'id': "huggingfaceh4/zephyr-orpo-141b-a35b", 'name': "Zephyr 141B-A35B"},
        
        # Default
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
        *openrouter_pro_models,   # Include all OpenRouter Pro models
        *image_models,
        *list(agentMode.keys()),
        *list(trendingAgentMode.keys())
    ]))
   
    # Initialize models with fallback_models
    models = fallback_models
    
    @classmethod
    async def get_models_async(cls) -> list:
        """
        Asynchronous version of get_models that checks subscription status.
        Returns a list of available models based on subscription status.
        Premium users get the full list of models.
        Free users get fallback_models.
        """
        # Check if there are valid session data in HAR files
        session_data = cls._find_session_in_har_files()
        
        if not session_data:
            # For users without HAR files - return free models
            debug.log(f"BlackboxPro: Returning free model list with {len(cls.fallback_models)} models")
            return cls.fallback_models
        
        # For accounts with HAR files, check subscription status
        if 'user' in session_data and 'email' in session_data['user']:
            subscription = await cls.check_subscription(session_data['user']['email'])
            if subscription['status'] == "PREMIUM":
                debug.log(f"BlackboxPro: Returning premium model list with {len(cls._all_models)} models")
                return cls._all_models
        
        # For free accounts - return free models
        debug.log(f"BlackboxPro: Returning free model list with {len(cls.fallback_models)} models")
        return cls.fallback_models
        
    @classmethod
    def get_models(cls) -> list:
        """
        Returns a list of available models based on authorization status.
        Authorized users get the full list of models.
        Free users get fallback_models.
        
        Note: This is a synchronous method that can't check subscription status,
        so it falls back to the basic premium access check.
        For more accurate results, use get_models_async when possible.
        """
        # Check if there are valid session data in HAR files
        session_data = cls._find_session_in_har_files()
        
        if not session_data:
            # For users without HAR files - return free models
            debug.log(f"BlackboxPro: Returning free model list with {len(cls.fallback_models)} models")
            return cls.fallback_models
        
        # For accounts with HAR files, check premium access
        has_premium_access = cls._check_premium_access()
        
        if has_premium_access:
            # For premium users - all models
            debug.log(f"BlackboxPro: Returning premium model list with {len(cls._all_models)} models")
            return cls._all_models
        
        # For free accounts - return free models
        debug.log(f"BlackboxPro: Returning free model list with {len(cls.fallback_models)} models")
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
            return {"status": "FREE", "customerId": None, "isTrialSubscription": False, "lastChecked": None}
            
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
                        debug.log(f"BlackboxPro: Subscription check failed with status {response.status}")
                        return {"status": "FREE", "customerId": None, "isTrialSubscription": False, "lastChecked": None}
                    
                    result = await response.json()
                    status = "PREMIUM" if result.get("hasActiveSubscription", False) else "FREE"
                    
                    return {
                        "status": status,
                        "customerId": result.get("customerId"),
                        "isTrialSubscription": result.get("isTrialSubscription", False),
                        "lastChecked": result.get("lastChecked")
                    }
        except Exception as e:
            debug.log(f"BlackboxPro: Error checking subscription: {e}")
            return {"status": "FREE", "customerId": None, "isTrialSubscription": False, "lastChecked": None}
    
    @classmethod
    def _check_premium_access(cls) -> bool:
        """
        Checks for an authorized session in HAR files.
        Returns True if a valid session is found.
        """
        try:
            session_data = cls._find_session_in_har_files()
            if not session_data:
                return False
                
            # Check if this is a premium session
            return True
        except Exception as e:
            debug.log(f"BlackboxPro: Error checking premium access: {e}")
            return False

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

                                                debug.log(f"BlackboxPro: Found session in HAR file: {file}")
                                                return har_session
                                    except json.JSONDecodeError as e:
                                        # Only print error for entries that truly look like session data
                                        if ('"user"' in content['text'] and 
                                            '"email"' in content['text']):
                                            debug.log(f"BlackboxPro: Error parsing likely session data: {e}")
                except Exception as e:
                    debug.log(f"BlackboxPro: Error reading HAR file {file}: {e}")
            return None
        except NoValidHarFileError:
            pass
        except Exception as e:
            debug.log(f"BlackboxPro: Error searching HAR files: {e}")
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
                debug.log(f"BlackboxPro: Error reading cache: {e}")
        
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
                                        debug.log(f"BlackboxPro: Error writing cache: {e}")
                                        
                                    return validated_value

            except Exception as e:
                debug.log(f"BlackboxPro: Error retrieving validated_value: {e}")

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
        return_conversation: bool = True,
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

            # Get session data from HAR files
            session_data = cls._find_session_in_har_files()

            # Check if we have a valid session
            if not session_data:
                # No valid session found, raise an error
                debug.log("BlackboxPro: No valid session found in HAR files")
                raise NoValidHarFileError("No valid Blackbox session found. Please log in to Blackbox AI in your browser first.")

            debug.log(f"BlackboxPro: Using session from HAR file (email: {session_data['user'].get('email', 'unknown')})")

            # Check subscription status
            subscription_status = {"status": "FREE", "customerId": None, "isTrialSubscription": False, "lastChecked": None}
            if session_data.get('user', {}).get('email'):
                subscription_status = await cls.check_subscription(session_data['user']['email'])
                debug.log(f"BlackboxPro: Subscription status for {session_data['user']['email']}: {subscription_status['status']}")

            # Determine if user has premium access based on subscription status
            if subscription_status['status'] == "PREMIUM":
                is_premium = True
            else:
                # For free accounts, check if the requested model is in fallback_models
                is_premium = model in cls.fallback_models
                if not is_premium:
                    debug.log(f"BlackboxPro: Model {model} not available in free account, falling back to default model")
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
                "designerMode": False,
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
                        yield ImageResponse(urls=[image_url], alt=format_media_prompt(messages, prompt))
                        return
                
                # Handle conversation history once, in one place
                if return_conversation:
                    conversation.message_history.append({"role": "assistant", "content": full_response_text})
                    yield conversation
                # For image models that didn't produce an image, fall back to text response
                elif model == cls.default_image_model:
                    yield full_response_text
