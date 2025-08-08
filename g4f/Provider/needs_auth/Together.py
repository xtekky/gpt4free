from __future__ import annotations


from ..template import OpenaiTemplate
from ...config import DEFAULT_MODEL

class Together(OpenaiTemplate):
    label = "Together"
    url = "https://together.xyz"
    login_url = "https://api.together.ai/"
    api_base = "https://api.together.xyz/v1"
    models_endpoint = "https://api.together.xyz/v1/models"

    active_by_default = True
    working = True
    needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = DEFAULT_MODEL
    default_vision_model = default_model
    default_image_model = 'black-forest-labs/FLUX.1.1-pro'
    vision_models = [
        default_vision_model,
        'Qwen/Qwen2-VL-72B-Instruct',
        'Qwen/Qwen2.5-VL-72B-Instruct',
        'arcee-ai/virtuoso-medium-v2',
        'arcee_ai/arcee-spotlight',
        'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
        'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
        'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'meta-llama/Llama-Vision-Free',
    ]
    image_models = []
    models = []
    model_configs = {}  # Store model configurations including stop tokens
    _models_cached = False
    _api_key_cache = None

    model_aliases = {
        ### Models Chat/Language ###
        # meta-llama
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "llama-2-70b": ["meta-llama/Llama-2-70b-hf", "meta-llama/Llama-2-70b-hf",],
        "llama-3-70b": ["meta-llama/Meta-Llama-3-70B-Instruct-Turbo", "meta-llama/Llama-3-70b-chat-hf", "Rrrr/meta-llama/Llama-3-70b-chat-hf-6f9ad551", "roberizk@gmail.com/meta-llama/Llama-3-70b-chat-hf-26ee936b", "roberizk@gmail.com/meta-llama/Meta-Llama-3-70B-Instruct-6feb41f7"],
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "llama-3.3-70b": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",],
        "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "llama-3.1-8b": ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "blackbox/meta-llama-3-1-8b"],
        "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        #"llama-vision": "meta-llama/Llama-Vision-Free",
        "llama-3-8b": ["meta-llama/Llama-3-8b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct-Lite", "roberizk@gmail.com/meta-llama/Meta-Llama-3-8B-Instruct-8ced8839",],
        "llama-3.1-70b": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "Rrrr/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-03dc18e1", "Rrrr/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-6c92f39d"],
        "llama-3.1-405b": ["meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "eddiehou/meta-llama/Llama-3.1-405B"],
        "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        
        # arcee-ai
        #"arcee-spotlight": "arcee_ai/arcee-spotlight",
        #"arcee-blitz": "arcee-ai/arcee-blitz",
        #"arcee-caller": "arcee-ai/caller",
        #"arcee-virtuoso-large": "arcee-ai/virtuoso-large",
        #"arcee-virtuoso-medium": "arcee-ai/virtuoso-medium-v2",
        #"arcee-coder-large": "arcee-ai/coder-large",
        #"arcee-maestro": "arcee-ai/maestro-reasoning",
        #"afm-4.5b": "arcee-ai/AFM-4.5B-Preview", 
        
        # deepseek-ai
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "deepseek-r1-distill-llama-70b": ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"],
        "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        
        # Qwen
        "qwen-2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "qwen-2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
        "qwq-32b": "Qwen/QwQ-32B",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "qwen-3-235b": ["Qwen/Qwen3-235B-A22B-fp8", "Qwen/Qwen3-235B-A22B-fp8-tput"],
        "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
        "qwen-3-32b": "Qwen/Qwen3-32B-FP8",
        
        # mistralai
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistral-7b": ["mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3"],
        
        # google
        "gemma-2b": "google/gemma-2b-it",
        "gemma-2-27b": "google/gemma-2-27b-it",
        "gemma-3-27b": "google/gemma-3-27b-it",
        "gemma-3n-e4b": "google/gemma-3n-E4B-it",
        
        # nvidia
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        
        # togethercomputer
        #"refuel-v2": "togethercomputer/Refuel-Llm-V2",
        #"moa": "togethercomputer/MoA-1",
        #"refuel-v2-small": "togethercomputer/Refuel-Llm-V2-Small",
        #"moa-turbo": "togethercomputer/MoA-1-Turbo",
        
        # lgai
        #"exaone-3.5-32b": "lgai/exaone-3-5-32b-instruct",
        #"exaone-deep-32b": "lgai/exaone-deep-32b",
        
        # Gryphe
        #"mythomax-2-13b": "Gryphe/MythoMax-L2-13b",
        #"mythomax-2-13b-lite": "Gryphe/MythoMax-L2-13b-Lite",
        
        # NousResearch
        "hermes-2-dpo": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        
        # marin-community
        #"marin-8b": "marin-community/marin-8b-instruct",
        
        # scb10x
        #"typhoon-2-70b": "scb10x/scb10x-llama3-1-typhoon2-70b-instruct",
        #"typhoon-2.1": "scb10x/scb10x-typhoon-2-1-gemma3-12b",
        
        # perplexity-ai
        "r1-1776": "perplexity-ai/r1-1776",

        ### Models Image ###
        # black-forest-labs
        "flux": ["black-forest-labs/FLUX.1-schnell-Free", "black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1.1-pro", "black-forest-labs/FLUX.1-pro", "black-forest-labs/FLUX.1.1-pro", "black-forest-labs/FLUX.1-pro", "black-forest-labs/FLUX.1-redux", "black-forest-labs/FLUX.1-depth", "black-forest-labs/FLUX.1-canny", "black-forest-labs/FLUX.1-kontext-max", "black-forest-labs/FLUX.1-dev-lora", "black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-dev-lora", "black-forest-labs/FLUX.1-kontext-pro", "black-forest-labs/FLUX.1-kontext-dev"],
        
        "flux-schnell": ["black-forest-labs/FLUX.1-schnell-Free", "black-forest-labs/FLUX.1-schnell"],
        "flux-pro": ["black-forest-labs/FLUX.1.1-pro", "black-forest-labs/FLUX.1-pro"],
        "flux-redux": "black-forest-labs/FLUX.1-redux",
        "flux-depth": "black-forest-labs/FLUX.1-depth",
        "flux-canny": "black-forest-labs/FLUX.1-canny",
        "flux-kontext-max": "black-forest-labs/FLUX.1-kontext-max",
        "flux-dev-lora": "black-forest-labs/FLUX.1-dev-lora",
        "flux-dev": ["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-dev-lora"],
        "flux-kontext-pro": "black-forest-labs/FLUX.1-kontext-pro",
        "flux-kontext-dev": "black-forest-labs/FLUX.1-kontext-dev",
    }