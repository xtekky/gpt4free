from __future__ import annotations

from .template import OpenaiTemplate

class DeepInfraChat(OpenaiTemplate):
    url = "https://deepinfra.com/chat"
    api_base = "https://api.deepinfra.com/v1/openai"
    working = True

    default_model = 'deepseek-ai/DeepSeek-V3'
    default_vision_model = 'microsoft/Phi-4-multimodal-instruct'
    vision_models = [default_vision_model, 'meta-llama/Llama-3.2-90B-Vision-Instruct']
    models = [
        'deepseek-ai/DeepSeek-Prover-V2-671B',
        'Qwen/Qwen3-235B-A22B',
        'Qwen/Qwen3-30B-A3B',
        'Qwen/Qwen3-32B',
        'Qwen/Qwen3-14B',
        'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'microsoft/phi-4-reasoning-plus',
        'microsoft/meta-llama/Llama-Guard-4-12B',
        'Qwen/QwQ-32B',
        'deepseek-ai/DeepSeek-V3-0324',
        'google/gemma-3-27b-it',
        'google/gemma-3-12b-it',
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        default_model,
        'mistralai/Mistral-Small-24B-Instruct-2501',
        'deepseek-ai/DeepSeek-R1',
        'deepseek-ai/DeepSeek-R1-Turbo',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'microsoft/phi-4',
        'microsoft/WizardLM-2-8x22B',
        'Qwen/Qwen2.5-72B-Instruct',
        'Qwen/Qwen2-72B-Instruct',
        'cognitivecomputations/dolphin-2.6-mixtral-8x7b',
        'cognitivecomputations/dolphin-2.9.1-llama-3-70b',
        'deepinfra/airoboros-70b',
        'lizpreciatior/lzlv_70b_fp16_hf',
        'microsoft/WizardLM-2-7B',
        'mistralai/Mixtral-8x22B-Instruct-v0.1',
    ] + vision_models
    model_aliases = {
        "deepseek-prover-v2-671b": "deepseek-ai/DeepSeek-Prover-V2-671B",
        "deepseek-prover-v2": "deepseek-ai/DeepSeek-Prover-V2-671B",
        "qwen-3-235b": "Qwen/Qwen3-235B-A22B",
        "qwen-3-30b": "Qwen/Qwen3-30B-A3B",
        "qwen-3-32b": "Qwen/Qwen3-32B",
        "qwen-3-14b": "Qwen/Qwen3-14B",
        "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "llama-4-maverick-17b": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "llama-4-scout-17b": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "phi-4-reasoning-plus": "microsoft/phi-4-reasoning-plus",
        #"": "meta-llama/Llama-Guard-4-12B",
        "qwq-32b": "Qwen/QwQ-32B",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
        "gemma-3-27b": "google/gemma-3-27b-it",
        "gemma-3-12b": "google/gemma-3-12b-it",
        "phi-4-multimodal": "microsoft/Phi-4-multimodal-instruct",
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-v3": default_model,
        "mistral-small": "mistralai/Mistral-Small-24B-Instruct-2501",
        "mixtral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
        "deepseek-r1-turbo": "deepseek-ai/DeepSeek-R1-Turbo",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "phi-4": "microsoft/phi-4",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
        "dolphin-2.6": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        "dolphin-2.9": "cognitivecomputations/dolphin-2.9.1-llama-3-70b",
        "airoboros-70b": "deepinfra/airoboros-70b",
        "lzlv-70b": "lizpreciatior/lzlv_70b_fp16_hf",
        "wizardlm-2-7b": "microsoft/WizardLM-2-7B",
        "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1"
    }
