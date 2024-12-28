from __future__ import annotations

from .OpenaiAPI import OpenaiAPI

class glhfChat(OpenaiAPI):
    label = "glhf.chat"
    url = "https://glhf.chat"
    api_base = "https://glhf.chat/api/openai/v1"
    working = True
    model_aliases = {
        'Qwen2.5-Coder-32B-Instruct': 'hf:Qwen/Qwen2.5-Coder-32B-Instruct',
        'Llama-3.1-405B-Instruct': 'hf:meta-llama/Llama-3.1-405B-Instruct',
        'Llama-3.1-70B-Instruct': 'hf:meta-llama/Llama-3.1-70B-Instruct',
        'Llama-3.1-8B-Instruct': 'hf:meta-llama/Llama-3.1-8B-Instruct',
        'Llama-3.2-3B-Instruct': 'hf:meta-llama/Llama-3.2-3B-Instruct',
        'Llama-3.2-11B-Vision-Instruct': 'hf:meta-llama/Llama-3.2-11B-Vision-Instruct',
        'Llama-3.2-90B-Vision-Instruct': 'hf:meta-llama/Llama-3.2-90B-Vision-Instruct',
        'Qwen2.5-72B-Instruct': 'hf:Qwen/Qwen2.5-72B-Instruct',
        'Llama-3.3-70B-Instruct': 'hf:meta-llama/Llama-3.3-70B-Instruct',
        'gemma-2-9b-it': 'hf:google/gemma-2-9b-it',
        'gemma-2-27b-it': 'hf:google/gemma-2-27b-it',
        'Mistral-7B-Instruct-v0.3': 'hf:mistralai/Mistral-7B-Instruct-v0.3',
        'Mixtral-8x7B-Instruct-v0.1': 'hf:mistralai/Mixtral-8x7B-Instruct-v0.1',
        'Mixtral-8x22B-Instruct-v0.1': 'hf:mistralai/Mixtral-8x22B-Instruct-v0.1',
        'Nous-Hermes-2-Mixtral-8x7B-DPO': 'hf:NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
        'Qwen2.5-7B-Instruct': 'hf:Qwen/Qwen2.5-7B-Instruct',
        'SOLAR-10.7B-Instruct-v1.0': 'hf:upstage/SOLAR-10.7B-Instruct-v1.0',
        'Llama-3.1-Nemotron-70B-Instruct-HF': 'hf:nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'
    }