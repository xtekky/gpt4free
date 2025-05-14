from __future__ import annotations

from ..template import OpenaiTemplate

class GlhfChat(OpenaiTemplate):
    url = "https://glhf.chat"
    login_url = "https://glhf.chat/user-settings/api"
    api_base = "https://glhf.chat/api/openai/v1"

    working = True
    needs_auth = True

    default_model = "hf:meta-llama/Llama-3.3-70B-Instruct"
    models = ["hf:meta-llama/Llama-3.1-405B-Instruct", default_model, "hf:deepseek-ai/DeepSeek-V3", "hf:Qwen/QwQ-32B-Preview", "hf:huihui-ai/Llama-3.3-70B-Instruct-abliterated", "hf:anthracite-org/magnum-v4-12b", "hf:meta-llama/Llama-3.1-70B-Instruct", "hf:meta-llama/Llama-3.1-8B-Instruct", "hf:meta-llama/Llama-3.2-3B-Instruct", "hf:meta-llama/Llama-3.2-11B-Vision-Instruct", "hf:meta-llama/Llama-3.2-90B-Vision-Instruct", "hf:Qwen/Qwen2.5-72B-Instruct", "hf:Qwen/Qwen2.5-Coder-32B-Instruct", "hf:google/gemma-2-9b-it", "hf:google/gemma-2-27b-it", "hf:mistralai/Mistral-7B-Instruct-v0.3", "hf:mistralai/Mixtral-8x7B-Instruct-v0.1", "hf:mistralai/Mixtral-8x22B-Instruct-v0.1", "hf:NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", "hf:Qwen/Qwen2.5-7B-Instruct", "hf:upstage/SOLAR-10.7B-Instruct-v1.0", "hf:nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"]
