from __future__ import annotations

from .template import OpenaiTemplate

class DeepInfraChat(OpenaiTemplate):
    url = "https://deepinfra.com/chat"
    api_base = "https://api.deepinfra.com/v1/openai"
    working = True

    default_model = 'deepseek-ai/DeepSeek-V3'
    default_vision_model = 'openbmb/MiniCPM-Llama3-V-2_5'
    vision_models = [default_vision_model, 'meta-llama/Llama-3.2-90B-Vision-Instruct']
    models = [
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'meta-llama/Llama-3.3-70B-Instruct',
        default_model,
        'mistralai/Mistral-Small-24B-Instruct-2501',
        'deepseek-ai/DeepSeek-R1',
        'deepseek-ai/DeepSeek-R1-Turbo',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'microsoft/phi-4',
        'microsoft/WizardLM-2-8x22B',
        'Qwen/Qwen2.5-72B-Instruct',
        '01-ai/Yi-34B-Chat',
        'Qwen/Qwen2-72B-Instruct',
        'cognitivecomputations/dolphin-2.6-mixtral-8x7b',
        'cognitivecomputations/dolphin-2.9.1-llama-3-70b',
        'databricks/dbrx-instruct',
        'deepinfra/airoboros-70b',
        'lizpreciatior/lzlv_70b_fp16_hf',
        'microsoft/WizardLM-2-7B',
        'mistralai/Mixtral-8x22B-Instruct-v0.1',
    ] + vision_models
    model_aliases = {
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-v3": default_model,
        "mixtral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1-Turbo",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-r1-distill-llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-r1-distill-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "phi-4": "microsoft/phi-4",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "yi-34b": "01-ai/Yi-34B-Chat",
        "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
        "dolphin-2.6": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        "dolphin-2.9": "cognitivecomputations/dolphin-2.9.1-llama-3-70b",
        "dbrx-instruct": "databricks/dbrx-instruct",
        "airoboros-70b": "deepinfra/airoboros-70b",
        "lzlv-70b": "lizpreciatior/lzlv_70b_fp16_hf",
        "wizardlm-2-7b": "microsoft/WizardLM-2-7B",
        "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "minicpm-2.5": "openbmb/MiniCPM-Llama3-V-2_5",
    }
