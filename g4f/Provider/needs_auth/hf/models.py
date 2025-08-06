from ....config import DEFAULT_MODEL

default_model = DEFAULT_MODEL
default_image_model = "black-forest-labs/FLUX.1-dev"
image_models = [    
    default_image_model,
    "black-forest-labs/FLUX.1-schnell",
]
text_models = [
    default_model,
    'meta-llama/Llama-3.3-70B-Instruct',
    'CohereForAI/c4ai-command-r-plus-08-2024',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'Qwen/QwQ-32B',
    'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    'Qwen/Qwen2.5-Coder-32B-Instruct',
    'meta-llama/Llama-3.2-11B-Vision-Instruct',
    'mistralai/Mistral-Nemo-Instruct-2407',
    'microsoft/Phi-3.5-mini-instruct',
]
fallback_models = text_models + image_models
model_aliases = {
    ### Chat ###
    "qwen-2.5-72b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "llama-3": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "command-r-plus": "CohereForAI/c4ai-command-r-plus-08-2024",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "qwq-32b": "Qwen/QwQ-32B",
    "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    ### Used in other providers ###
    "qwen-2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
    "qvq-72b": "Qwen/QVQ-72B-Preview",
}
image_model_aliases = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "stable-diffusion-3.5-large": "stabilityai/stable-diffusion-3.5-large",
    "sdxl-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sd-3.5-large": "stabilityai/stable-diffusion-3.5-large",
}
extra_models = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "NousResearch/Hermes-3-Llama-3.1-8B",
]
default_vision_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
default_llama_model = "meta-llama/Llama-3.3-70B-Instruct"
vision_models = [default_vision_model, "Qwen/Qwen2-VL-7B-Instruct"]