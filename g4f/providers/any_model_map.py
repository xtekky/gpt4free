audio_models = []
image_models = ['dall-e-3', 'gpt-image', 'sdxl-turbo', 'sd-3.5-large', 'flux', 'flux-pro', 'flux-kontext', 'black-forest-labs/FLUX-2-klein-4b', 'black-forest-labs/FLUX-2-klein-9b', 'flux-2-klein-4b', 'flux-2-klein-9b']
vision_models = []
video_models = ['video']
model_map = {
  "auto": {
    "DeepInfra": "",
    "LLM7": "",
    "ChatGPT": "",
    "KiloCode": "",
    "TeachAnything": "",
    "Qwen": "",
    "GeminiPro": "",
    "OpenRouterFree": "",
    "Ollama": "",
    "OperaAria": "",
    "GLM": "",
    "Pollinations": ""
  },
  "agent-tools": {
    "AgentTools": "Agent Tools (MCP)"
  },
  "gpt-4": {
    "Yqcloud": "gpt-4",
    "ChatGPT": "gpt-4"
  },
  "gpt-4o": {
    "ChatGPT": "gpt-4o"
  },
  "gpt-4o-mini": {
    "ChatGPT": "gpt-4o-mini"
  },
  "gpt-4o-mini-tts": {
    "OpenAIFM": "gpt-4o-mini-tts"
  },
  "o1": {
    "OpenaiAccount": "o1"
  },
  "o1-mini": {
    "OpenaiAccount": "o1-mini"
  },
  "o3-mini": {
    "ChatGPT": "o3-mini"
  },
  "o3-mini-high": {
    "OpenaiAccount": "o3-mini-high"
  },
  "o4-mini": {
    "ChatGPT": "o4-mini"
  },
  "o4-mini-high": {
    "ChatGPT": "o4-mini-high"
  },
  "gpt-4.1": {
    "ChatGPT": "gpt-4.1"
  },
  "gpt-4.1-mini": {
    "ChatGPT": "gpt-4.1-mini"
  },
  "gpt-4.1-nano": {
    "Pollinations": "gpt-4.1-nano"
  },
  "gpt-4.5": {
    "ChatGPT": "gpt-4.5"
  },
  "gpt-oss-120b": {
    "Together": "openai/gpt-oss-120b",
    "OpenRouter": "openai/gpt-oss-120b",
    "Groq": "openai/gpt-oss-120b",
    "HuggingFace": "openai/gpt-oss-120b"
  },
  "dall-e-3": {
    "OpenaiAccount": "dall-e-3",
    "MicrosoftDesigner": "dall-e-3",
    "BingCreateImages": "dall-e-3"
  },
  "gpt-image": {
    "PollinationsImage": "gpt-image"
  },
  "meta-ai": {
    "MetaAI": "meta-ai"
  },
  "llama-2-70b": {
    "Together": "llama-2-70b"
  },
  "llama-3-8b": {
    "Together": "llama-3-8b"
  },
  "llama-3-70b": {
    "Together": "llama-3-70b"
  },
  "llama-3.1-8b": {
    "Together": "llama-3.1-8b",
    "HuggingFace": "meta-llama/Llama-3.1-8B-Instruct"
  },
  "llama-3.1-70b": {
    "Together": "llama-3.1-70b",
    "HuggingFace": "meta-llama/Llama-3.1-70B-Instruct"
  },
  "llama-3.1-405b": {
    "Together": "llama-3.1-405b"
  },
  "llama-3.2-3b": {
    "Together": "llama-3.2-3b",
    "HuggingFace": "meta-llama/Llama-3.2-3B-Instruct"
  },
  "llama-3.2-90b": {
    "Together": "llama-3.2-90b"
  },
  "llama-4-scout": {
    "Pollinations": "llama-4-scout",
    "Together": "llama-4-scout"
  },
  "llama-4-maverick": {
    "Together": "llama-4-maverick"
  },
  "mistral-7b": {
    "Together": "mistral-7b"
  },
  "mixtral-8x7b": {
    "Together": "mixtral-8x7b"
  },
  "mistral-small-24b": {
    "Together": "mistral-small-24b"
  },
  "mistral-small-3.1-24b": {
    "Pollinations": "mistral-small-3.1-24b"
  },
  "hermes-2-dpo": {
    "Together": "hermes-2-dpo"
  },
  "gemini-2.5-flash": {
    "Gemini": "gemini-2.5-flash",
    "GeminiPro": "gemini-2.5-flash",
    "GeminiCLI": "gemini-2.5-flash"
  },
  "gemini-2.5-pro": {
    "Gemini": "gemini-2.5-pro",
    "GeminiPro": "gemini-2.5-pro",
    "GeminiCLI": "gemini-2.5-pro"
  },
  "gemini-3-pro-preview": {
    "GeminiCLI": "gemini-3-pro-preview"
  },
  "gemini-3.1-pro": {
    "Gemini": "gemini-3.1-pro"
  },
  "gemini-3.1-flash-lite": {
    "Gemini": "gemini-3.1-flash-lite"
  },
  "gemini-3.6-flash": {
    "Gemini": "gemini-3.6-flash"
  },
  "gemini-3.7-flash": {
    "Gemini": "gemini-3.7-flash"
  },
  "gemini-3.8-flash": {
    "Gemini": "gemini-3.8-flash"
  },
  "gemini-3.8-pro": {
    "Gemini": "gemini-3.8-pro"
  },
  "gemini-3.5-flash-lite": {
    "Gemini": "gemini-3.5-flash-lite"
  },
  "gemini-3.5-flash": {
    "Gemini": "gemini-3.5-flash"
  },
  "gemini-3.5-flash-thinking": {
    "Gemini": "gemini-3.5-flash-thinking"
  },
  "gemini-auto": {
    "Gemini": "gemini-auto"
  },
  "gemini-3.5-flash-thinking-lite": {
    "Gemini": "gemini-3.5-flash-thinking-lite"
  },
  "gemini-flash-lite": {
    "Gemini": "gemini-flash-lite"
  },
  "command-r7b": {
    "HuggingSpace": "command-r7b"
  },
  "command-a": {
    "HuggingSpace": "command-a"
  },
  "qwen-2.5-coder-32b": {
    "Together": "qwen-2.5-coder-32b",
    "HuggingChat": "qwen-2.5-coder-32b",
    "HuggingFace": "Qwen/Qwen2.5-Coder-32B-Instruct"
  },
  "qwen-2.5-vl-72b": {
    "Together": "qwen-2.5-vl-72b",
    "HuggingFace": "Qwen/Qwen2.5-VL-72B-Instruct"
  },
  "qwen-3-235b": {
    "Together": "qwen-3-235b"
  },
  "qwen-3-32b": {
    "Together": "qwen-3-32b",
    "HuggingFace": "Qwen/Qwen3-32B"
  },
  "qwq-32b": {
    "Together": "qwq-32b",
    "HuggingChat": "qwq-32b"
  },
  "deepseek-v3": {
    "Together": "deepseek-v3"
  },
  "deepseek-r1": {
    "Pollinations": "deepseek-r1",
    "Together": "deepseek-r1"
  },
  "deepseek-r1-distill-llama-70b": {
    "Together": "deepseek-r1-distill-llama-70b"
  },
  "deepseek-r1-distill-qwen-1.5b": {
    "Together": "deepseek-r1-distill-qwen-1.5b"
  },
  "deepseek-r1-distill-qwen-14b": {
    "Together": "deepseek-r1-distill-qwen-14b"
  },
  "grok-2": {
    "Grok": "grok-2"
  },
  "grok-3": {
    "Grok": "grok-3"
  },
  "grok-3-r1": {
    "Grok": "grok-3-r1"
  },
  "kimi-k2": {
    "Groq": "moonshotai/Kimi-K2-Instruct"
  },
  "sonar": {},
  "sonar-pro": {},
  "sonar-reasoning": {},
  "sonar-reasoning-pro": {},
  "r1-1776": {
    "Together": "r1-1776",
    "Perplexity": "r1-1776"
  },
  "nemotron-70b": {
    "Together": "nemotron-70b",
    "HuggingChat": "nemotron-70b"
  },
  "aria": {
    "OperaAria": "aria"
  },
  "sdxl-turbo": {
    "HuggingFaceMedia": "sdxl-turbo",
    "PollinationsImage": "sdxl-turbo"
  },
  "sd-3.5-large": {
    "HuggingFaceMedia": "sd-3.5-large",
    "HuggingSpace": "sd-3.5-large"
  },
  "flux": {
    "HuggingFaceMedia": "flux",
    "PollinationsImage": "flux",
    "Together": "flux",
    "HuggingSpace": "flux"
  },
  "flux-pro": {
    "PollinationsImage": "flux-pro",
    "Together": "flux-pro"
  },
  "flux-kontext": {
    "Pollinations": "flux-kontext",
    "Together": "flux-kontext"
  },
  "hy4-flash": {
    "DeepInfra": "tencent/Hy4-flash",
    "HuggingFace": "tencent/Hy4-flash"
  },
  "hy4": {
    "DeepInfra": "tencent/Hy4",
    "HuggingFace": "tencent/Hy4"
  },
  "kimi-k3.1-pro": {
    "DeepInfra": "moonshotai/Kimi-K3.1-Pro",
    "HuggingFace": "moonshotai/Kimi-K3.1-Pro"
  },
  "deepseek-v4.1-flash-1017": {
    "DeepInfra": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "HuggingFace": "deepseek-ai/DeepSeek-V4.1-Flash-1017"
  },
  "nemotron-3.5-super-vl": {
    "DeepInfra": "nvidia/NVIDIA-Nemotron-3.5-Super-VL"
  },
  "mimo-v2.6-flash": {
    "DeepInfra": "XiaomiMiMo/MiMo-V2.6-Flash"
  },
  "mimo-v2.6-pro": {
    "DeepInfra": "XiaomiMiMo/MiMo-V2.6-Pro"
  },
  "deepseek-v4.1-flash": {
    "DeepInfra": "deepseek-ai/DeepSeek-V4.1-Flash",
    "HuggingFace": "deepseek-ai/DeepSeek-V4.1-Flash"
  },
  "glm-5.3": {
    "DeepInfra": "zai-org/GLM-5.3",
    "HuggingFace": "zai-org/GLM-5.3"
  },
  "glm-5.3-flash": {
    "DeepInfra": "zai-org/GLM-5.3-Flash",
    "HuggingFace": "zai-org/GLM-5.3-Flash"
  },
  "kimi-k3": {
    "DeepInfra": "moonshotai/Kimi-K3",
    "HuggingFace": "moonshotai/Kimi-K3"
  },
  "deepseek-v4-pro-0813": {
    "DeepInfra": "deepseek-ai/DeepSeek-V4-Pro-0813",
    "HuggingFace": "deepseek-ai/DeepSeek-V4-Pro-0813"
  },
  "deepseek-v4-flash-0731": {
    "DeepInfra": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "HuggingFace": "deepseek-ai/DeepSeek-V4-Flash-0731"
  },
  "glm-5.2": {
    "DeepInfra": "zai-org/GLM-5.2",
    "HuggingFace": "zai-org/GLM-5.2"
  },
  "nemotron-3-ultra-550b-a55b": {
    "DeepInfra": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "HuggingFace": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
  },
  "deepseek-v4-flash": {
    "DeepInfra": "deepseek-ai/DeepSeek-V4-Flash",
    "HuggingFace": "deepseek-ai/DeepSeek-V4-Flash"
  },
  "deepseek-v4-pro": {
    "DeepInfra": "deepseek-ai/DeepSeek-V4-Pro",
    "HuggingFace": "deepseek-ai/DeepSeek-V4-Pro"
  },
  "kimi-k2.6": {
    "DeepInfra": "moonshotai/Kimi-K2.6",
    "HuggingFace": "moonshotai/Kimi-K2.6"
  },
  "qwen-3.6-35b-a3b": {
    "DeepInfra": "Qwen/Qwen3.6-35B-A3B",
    "HuggingFace": "Qwen/Qwen3.6-35B-A3B"
  },
  "qwen-3.5-397b-a17b": {
    "DeepInfra": "Qwen/Qwen3.5-397B-A17B",
    "HuggingFace": "Qwen/Qwen3.5-397B-A17B"
  },
  "gemma-4-26b-a4b-it": {
    "DeepInfra": "google/gemma-4-26B-A4B-it",
    "HuggingFace": "google/gemma-4-26B-A4B-it"
  },
  "gemma-4-31b-it": {
    "DeepInfra": "google/gemma-4-31B-it",
    "HuggingFace": "google/gemma-4-31B-it"
  },
  "nemotron-3-super-120b-a12b": {
    "DeepInfra": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
    "HuggingFace": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
  },
  "deepseek-v3.2": {
    "DeepInfra": "deepseek-ai/DeepSeek-V3.2",
    "HuggingFace": "deepseek-ai/DeepSeek-V3.2"
  },
  "flux-2-klein-4b": {
    "DeepInfra": "black-forest-labs/FLUX-2-klein-4b"
  },
  "flux-2-klein-9b": {
    "DeepInfra": "black-forest-labs/FLUX-2-klein-9b"
  },
  "inkling-2-mini": {
    "HuggingFace": "thinkingmachines/Inkling-2-Mini"
  },
  "qwen-3.8-27b": {
    "HuggingFace": "Qwen/Qwen3.8-27B"
  },
  "daisyrose-27b": {
    "HuggingFace": "RadixArk/DaisyRose-27B"
  },
  "inkling-2-small": {
    "HuggingFace": "thinkingmachines/Inkling-2-Small"
  },
  "qwen-3.8-flash-next": {
    "HuggingFace": "Qwen/Qwen3.8-Flash-Next"
  },
  "lfm3-1.7b": {
    "HuggingFace": "LiquidAI/LFM3-1.7B"
  },
  "inkling": {
    "HuggingFace": "thinkingmachines/Inkling"
  },
  "gemma-3-4b-it": {
    "HuggingFace": "google/gemma-3-4b-it"
  },
  "mistral-large-4-1t-a52b": {
    "HuggingFace": "mistralai/Mistral-Large-4-1T-A52B"
  },
  "gpt-oss-20b": {
    "HuggingFace": "openai/gpt-oss-20b"
  },
  "muse-glimmer-30b": {
    "HuggingFace": "meta-models/Muse-Glimmer-30B"
  },
  "qwen-3.5-9b": {
    "HuggingFace": "Qwen/Qwen3.5-9B"
  },
  "qwen-3.8-2.4t-a95b": {
    "HuggingFace": "Qwen/Qwen3.8-2.4T-A95B"
  },
  "nemotron-3.5-super-vl-120b-a12b": {
    "HuggingFace": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
  },
  "glm-4.7-flash": {
    "HuggingFace": "zai-org/GLM-4.7-Flash"
  },
  "qwen-3-coder-next": {
    "HuggingFace": "Qwen/Qwen3-Coder-Next"
  },
  "qwen-3-coder-30b-a3b": {
    "HuggingFace": "Qwen/Qwen3-Coder-30B-A3B-Instruct"
  },
  "mimo-v2.6-pro-rl": {
    "HuggingFace": "XiaomiMiMo/MiMo-V2.6-Pro-RL"
  },
  "lfm2.5-24b-a2b": {
    "HuggingFace": "LiquidAI/LFM2.5-24B-A2B"
  },
  "qwen-3.6-27b": {
    "HuggingFace": "Qwen/Qwen3.6-27B"
  },
  "inkling-small": {
    "HuggingFace": "thinkingmachines/Inkling-Small"
  },
  "mimo-v2.6-flash-mopd": {
    "HuggingFace": "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
  },
  "mythomax-l2-13b": {
    "HuggingFace": "Gryphe/MythoMax-L2-13b"
  },
  "llama-3.3-70b": {
    "HuggingFace": "meta-llama/Llama-3.3-70B-Instruct"
  },
  "deepseek-r1-0528": {
    "HuggingFace": "deepseek-ai/DeepSeek-R1-0528"
  },
  "qwen-3-vl-8b": {
    "HuggingFace": "Qwen/Qwen3-VL-8B-Instruct"
  },
  "nemotron-3.5-lightning-30b-a3b": {
    "HuggingFace": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
  },
  "mimo-v2.6-flash-rl": {
    "HuggingFace": "XiaomiMiMo/MiMo-V2.6-Flash-RL"
  },
  "beam-1.0-501b-a23b": {
    "HuggingFace": "reflection/Beam-1.0-501B-A23B"
  },
  "minimax-m3.1-flash": {
    "HuggingFace": "MiniMaxAI/MiniMax-M3.1-Flash"
  },
  "dolphin-mistral-24b-venice-edition": {
    "HuggingFace": "dphn/Dolphin-Mistral-24B-Venice-Edition"
  },
  "ling-3.0-flash-vl": {
    "HuggingFace": "inclusionAI/Ling-3.0-flash-VL"
  },
  "gemma-3-27b-it": {
    "HuggingFace": "google/gemma-3-27b-it"
  },
  "llama-4-maverick-17b-128e": {
    "HuggingFace": "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
  },
  "moondream3": {
    "HuggingFace": "moondream/moondream3-preview"
  },
  "minimax-m2.5": {
    "HuggingFace": "MiniMaxAI/MiniMax-M2.5"
  },
  "granite-4.2-8b": {
    "HuggingFace": "ibm-granite/granite-4.2-8b"
  },
  "l3-8b-lunaris": {
    "HuggingFace": "Sao10K/L3-8B-Lunaris-v1"
  },
  "qwen-2.5-72b": {
    "HuggingFace": "Qwen/Qwen2.5-72B-Instruct"
  },
  "qwen-3-14b": {
    "HuggingFace": "Qwen/Qwen3-14B"
  },
  "deepseek-tng-r1t2-chimera": {
    "HuggingFace": "tngtech/DeepSeek-TNG-R1T2-Chimera"
  },
  "qwen-3-next-80b-a3b": {
    "HuggingFace": "Qwen/Qwen3-Next-80B-A3B-Instruct"
  },
  "qwen-3.5-35b-a3b": {
    "HuggingFace": "Qwen/Qwen3.5-35B-A3B"
  },
  "qwen-3.5-27b": {
    "HuggingFace": "Qwen/Qwen3.5-27B"
  },
  "minimax-m3": {
    "HuggingFace": "MiniMaxAI/MiniMax-M3"
  },
  "deepseek-v4-flash-vision-exp": {
    "HuggingFace": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
  },
  "muse-spark-open-816b-a42b": {
    "HuggingFace": "meta-models/Muse-Spark-Open-816B-A42B"
  },
  "deepseek": {
    "HuggingFace": "deepseek-ai/DeepSeek-V3"
  },
  "llama-4-scout-17b-16e": {
    "HuggingFace": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
  },
  "moondream3.1-9b-a2b": {
    "HuggingFace": "moondream/moondream3.1-9B-A2B"
  },
  "gemma-3-12b-it": {
    "HuggingFace": "google/gemma-3-12b-it"
  },
  "ui-tars-1.5-7b": {
    "HuggingFace": "ByteDance-Seed/UI-TARS-1.5-7B"
  },
  "qwen-3-30b-a3b": {
    "HuggingFace": "Qwen/Qwen3-30B-A3B"
  },
  "qwen-3-235b-a22b-2507": {
    "HuggingFace": "Qwen/Qwen3-235B-A22B-Instruct-2507"
  },
  "qwen-3-235b-a22b-thinking-2507": {
    "HuggingFace": "Qwen/Qwen3-235B-A22B-Thinking-2507"
  },
  "qwen-3-30b-a3b-2507": {
    "HuggingFace": "Qwen/Qwen3-30B-A3B-Instruct-2507"
  },
  "deepseek-v3.1": {
    "HuggingFace": "deepseek-ai/DeepSeek-V3.1"
  },
  "gpt-oss-safeguard-20b": {
    "HuggingFace": "openai/gpt-oss-safeguard-20b"
  },
  "qwen-3-vl-30b-a3b": {
    "HuggingFace": "Qwen/Qwen3-VL-30B-A3B-Instruct"
  },
  "kimi-k2.5": {
    "HuggingFace": "moonshotai/Kimi-K2.5"
  },
  "glm-5": {
    "HuggingFace": "zai-org/GLM-5"
  },
  "command-a-plus26": {
    "HuggingFace": "CohereLabs/command-a-plus-05-2026-bf16"
  },
  "kimi-k2.7-code": {
    "HuggingFace": "moonshotai/Kimi-K2.7-Code"
  },
  "ling-3.0-flash": {
    "HuggingFace": "inclusionAI/Ling-3.0-flash"
  },
  "magnum-v4-72b": {
    "HuggingFace": "anthracite-org/magnum-v4-72b"
  },
  "minimax-m2.7": {
    "HuggingFace": "MiniMaxAI/MiniMax-M2.7"
  },
  "remm-slerp-l2-13b": {
    "HuggingFace": "Undi95/ReMM-SLERP-L2-13B"
  },
  "gemma-2-27b-it": {
    "HuggingFace": "google/gemma-2-27b-it"
  },
  "phi-4": {
    "HuggingFace": "microsoft/phi-4"
  },
  "deepseek-v3-0324": {
    "HuggingFace": "deepseek-ai/DeepSeek-V3-0324"
  },
  "llama-guard-4-12b": {
    "HuggingFace": "meta-llama/Llama-Guard-4-12B"
  },
  "hermes-4-405b": {
    "HuggingFace": "NousResearch/Hermes-4-405B"
  },
  "qwen-3-vl-235b-a22b": {
    "HuggingFace": "Qwen/Qwen3-VL-235B-A22B-Instruct"
  },
  "glm-4.6": {
    "HuggingFace": "zai-org/GLM-4.6"
  },
  "nemotron-3-nano-30b-a3b": {
    "HuggingFace": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
  },
  "glm-4.7": {
    "HuggingFace": "zai-org/GLM-4.7"
  },
  "reka-edge-2603": {
    "HuggingFace": "RekaAI/reka-edge-2603"
  },
  "perplexity": {
    "Perplexity": "perplexity"
  },
  "pplx_pro": {
    "Perplexity": "pplx_pro"
  },
  "video": {
    "Video": "video"
  }
}
models_count = {
  "default": 16,
  "gpt-4": 2,
  "gpt-4o": 8,
  "gpt-4o-mini": 7,
  "gpt-4o-mini-tts": 2,
  "o1": 6,
  "o1-mini": 4,
  "o3-mini": 5,
  "o3-mini-high": 2,
  "o4-mini": 4,
  "o4-mini-high": 4,
  "gpt-4.1": 6,
  "gpt-4.1-mini": 5,
  "gpt-4.1-nano": 4,
  "gpt-4.5": 2,
  "gpt-oss-120b": 4,
  "dall-e-3": 3,
  "gpt-image": 2,
  "meta-ai": 2,
  "llama-2-70b": 4,
  "llama-3-8b": 5,
  "llama-3-70b": 3,
  "llama-3.1-8b": 2,
  "llama-3.1-70b": 2,
  "llama-3.1-405b": 3,
  "llama-3.2-3b": 2,
  "llama-3.2-90b": 2,
  "llama-4-scout": 2,
  "llama-4-maverick": 6,
  "mixtral-8x7b": 2,
  "mistral-small-3.1-24b": 4,
  "hermes-2-dpo": 2,
  "gemini-2.5-flash": 3,
  "gemini-2.5-pro": 3,
  "gemini-3-pro-preview": 2,
  "gemini-3.1-pro": 8,
  "gemini-3.1-flash-lite": 9,
  "gemini-3.5-flash": 8,
  "gemini-3.5-flash-thinking": 2,
  "gemini-auto": 2,
  "gemini-3.5-flash-thinking-lite": 2,
  "gemini-flash-lite": 5,
  "command-r7b": 3,
  "command-a": 5,
  "qwen-2.5-coder-32b": 3,
  "qwen-2.5-vl-72b": 2,
  "qwen-3-235b": 2,
  "qwen-3-32b": 2,
  "qwq-32b": 2,
  "deepseek-v3": 3,
  "deepseek-r1": 2,
  "deepseek-r1-distill-llama-70b": 3,
  "deepseek-r1-distill-qwen-1.5b": 2,
  "deepseek-r1-distill-qwen-14b": 4,
  "grok-2": 2,
  "grok-3": 2,
  "kimi-k2": 6,
  "sonar": 4,
  "sonar-pro": 3,
  "sonar-reasoning": 2,
  "sonar-reasoning-pro": 3,
  "r1-1776": 2,
  "nemotron-70b": 2,
  "sdxl-turbo": 2,
  "sd-3.5-large": 2,
  "flux": 4,
  "flux-pro": 2,
  "flux-kontext": 2,
  "glm-5.2": 2,
  "kimi-k2.7-code": 9,
  "nemotron-3-ultra-550b-a55b": 2,
  "deepseek-v4-flash": 2,
  "deepseek-v4-pro": 2,
  "kimi-k2.6": 2,
  "mimo-v2.5-pro": 6,
  "qwen-3.6-35b-a3b": 2,
  "glm-5.1": 6,
  "qwen-3.5-397b-a17b": 2,
  "gemma-4-26b-a4b-it": 2,
  "gemma-4-31b-it": 2,
  "glm-5": 5,
  "minimax-m2.5": 7,
  "qwen-3-max": 7,
  "qwen-3-max-thinking": 5,
  "kimi-k2.5": 6,
  "glm-4.7-flash": 6,
  "deepseek-v3.2": 2,
  "inkling": 7,
  "ternary-bonsai-27b-gguf": 2,
  "qwen-3.6-27b": 7,
  "hy3": 7,
  "ornith-1.0-35b": 2,
  "apertus-v1.5-70b": 2,
  "minimax-m3": 10,
  "apertus-v1.5-8b": 2,
  "qwen-3.5-9b": 5,
  "qwen-3-8b": 4,
  "gpt-oss-20b": 12,
  "qwen-3.5-27b": 7,
  "qwen-3-coder-30b-a3b": 6,
  "mimo-v2.5": 6,
  "qwen-3-coder-next": 5,
  "gemma-3-4b-it": 7,
  "qwen-3.5-122b-a10b": 7,
  "ling-2.6-1t": 4,
  "qwen-3-4b-2507": 2,
  "step-3.7-flash": 6,
  "glm-4.5-air": 5,
  "qwen-3-14b": 5,
  "qwen-3-next-80b-a3b": 6,
  "qwen-2.5-coder-7b": 3,
  "nemotron-3-ultra-550b-a55b-nvfp4": 2,
  "qwen-3.5-35b-a3b": 7,
  "gemma-3-27b-it": 7,
  "gemma-3-12b-it": 7,
  "phi-4": 5,
  "qwen-3-vl-30b-a3b": 6,
  "ternary-bonsai-27b-awq-4bit": 2,
  "kimi-k2-0905": 4,
  "qwen-2.5-7b": 5,
  "qwen-3-vl-235b-a22b": 6,
  "llama-3.3-70b": 8,
  "qwen-3-coder-480b-a35b": 5,
  "gpt-oss-safeguard-20b": 7,
  "command-r7b24": 7,
  "qwen-3-235b-a22b-2507": 6,
  "llama-4-scout-17b-16e": 2,
  "step-3.5-flash": 7,
  "qwen-2.5-coder-3b": 2,
  "l3-8b-stheno-v3.2": 2,
  "deepseek-r1-distill-qwen-7b": 2,
  "llama-guard-4-12b": 7,
  "qwen-3-4b-thinking-2507": 2,
  "glm-4.7": 5,
  "gemma-sea-lion-v4-27b-it": 3,
  "command-a25": 5,
  "deepseek-r1-distill-llama-8b": 3,
  "apertus-8b-2509": 2,
  "tiny-aya-global": 2,
  "qwen-3-235b-a22b": 5,
  "bielik-11b-v3.0": 2,
  "tiny-aya-earth": 2,
  "minimax-m2.7": 8,
  "llama-4-maverick-17b-128e": 2,
  "deepseek-v3-0324": 3,
  "ernie-4.5-vl-424b-a47b-base-pt": 2,
  "glm-4-32b-0414": 2,
  "deepseek-r1-0528": 5,
  "qwen-3-235b-a22b-thinking-2507": 6,
  "deepseek": 4,
  "glm-4.5": 5,
  "glm-4.5v": 5,
  "wizardlm-2-8x22b": 3,
  "qwen-2.5-72b": 5,
  "glm-4.6": 5,
  "glm-4.6v-flash": 4,
  "l3-8b-lunaris": 2,
  "deepseek-v3.1": 4,
  "glm-4.6v": 6,
  "minimax-m1-80k": 2,
  "deepseek-v3.1-terminus": 5,
  "command-r24": 7,
  "deepseek-v3.2-exp": 6,
  "aya-expanse-32b": 2,
  "aya-vision-32b": 2,
  "minimax-m2": 7,
  "command-r7b-arabic25": 4,
  "command-a-vision25": 2,
  "minimax-m2.1": 7,
  "command-a-reasoning25": 2,
  "gemma-3n-e4b-it": 7,
  "command-a-translate25": 2,
  "qwen-3-vl-235b-a22b-thinking": 6,
  "gemma-4-31b-it-pearl": 3,
  "tiny-aya-water": 2,
  "apertus-70b-2509": 2,
  "autoglm-phone-9b-multilingual": 2,
  "tiny-aya-fire": 2,
  "qwen-sea-lion-v4-32b-it": 2,
  "olmo-3-7b": 2,
  "eurollm-22b-2512": 2,
  "command-r": 3,
  "command-r-plus": 3,
  "command-r-plus24": 5,
  "flux-dev": 4,
  "flux-kontext-dev": 2,
  "perplexity": 2,
  "gemini-3.6-flash": 8,
  "gemini-3.7-flash": 8,
  "gemini-3.8-flash": 9,
  "gemini-3.8-pro": 2,
  "gemini-3.5-flash-lite": 8,
  "deepseek-v4.1-flash": 2,
  "glm-5.3": 2,
  "glm-5.3-flash": 2,
  "kimi-k3": 2,
  "qwen-3.8-2.4t-a95b": 8,
  "deepseek-v4-pro-0813": 2,
  "deepseek-v4-flash-0731": 2,
  "nemotron-3-super-120b-a12b": 2,
  "qwen-3.8-27b": 9,
  "deepseek-v4-flash-vision-exp": 7,
  "muse-glimmer-30b": 8,
  "ling-3.0-flash-vl": 6,
  "ling-3.0-flash-fin": 2,
  "nemotron-3.5-lightning-30b-a3b": 4,
  "ling-3.0-flash": 5,
  "granite-4.2-3b": 2,
  "granite-4.2-8b": 5,
  "inkling-small": 8,
  "qwen-3-30b-a3b": 7,
  "granite-4.2-30b": 2,
  "hermes-3-llama-3.1-70b": 5,
  "auto": 12,
  "deepseek-v4.1-flash-1017": 2,
  "nemotron-3.5-super-vl": 6,
  "hy4": 2,
  "mimo-v2.6-flash": 5,
  "mimo-v2.6-pro": 5,
  "minimax-m3.1-flash": 9,
  "beam-1.0-501b-a23b": 5,
  "muse-spark-open-816b-a42b": 3,
  "nemotron-3.5-super-vl-120b-a12b-nvfp4": 2,
  "mimo-v2.6-pro-rl": 2,
  "qwen-3.8-flash-next": 2,
  "mimo-v2.6-flash-rl": 2,
  "ling-3.1-flash": 6,
  "nemotron-sea-lion-v4.8-120b-a12b": 2,
  "qwen-sea-lion-v4.5-27b-it": 2,
  "alia-40b-2606": 2,
  "hy4-flash": 2,
  "kimi-k3.1-pro": 2
}
parents = {
  "HuggingSpace": [
    "BlackForestLabs_Flux1Dev",
    "BlackForestLabs_Flux1KontextDev",
    "CohereForAI_C4AI_Command",
    "StabilityAI_SD35Large"
  ],
  "HuggingFace": [
    "HuggingFaceMedia"
  ],
  "MetaAI": [
    "MetaAIAccount"
  ],
  "Pollinations": [
    "PollinationsAudio",
    "PollinationsImage"
  ]
}
model_aliases = {
  "": "auto",
  "chat": "gpt-4o-mini",
  "Copilot": "gpt-4o",
  "openai/gpt-4": "gpt-4",
  "openrouter:openai/gpt-4": "gpt-4",
  "openai/gpt-4o-2024-05-13": "gpt-4o",
  "openrouter:openai/gpt-4o-2024-11-20": "gpt-4o",
  "openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini",
  "openrouter:openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini",
  "coral": "gpt-4o-mini-tts",
  "Think Deeper": "o1",
  "reasoning": "grok-3-r1",
  "openai/o1": "o1",
  "openai:openai/o1": "o1",
  "openai:openai/o1-mini": "o1-mini",
  "o3-mini-2025-01-31": "o3-mini",
  "openai/o3-mini": "o3-mini",
  "openai:openai/o3-mini": "o3-mini",
  "openai/o3-mini-high": "o3-mini-high",
  "openrouter:openai/o3-mini-high": "o3-mini-high",
  "o4-mini-2025-04-16": "o4-mini",
  "openai/o4-mini": "o4-mini",
  "openai:openai/o4-mini": "o4-mini",
  "openai/o4-mini-high": "o4-mini-high",
  "openrouter:openai/o4-mini-high": "o4-mini-high",
  "gpt-4-1": "gpt-4.1",
  "gpt-4.1-2025-04-14": "gpt-4.1",
  "openai/gpt-4.1": "gpt-4.1",
  "openai:openai/gpt-4.1": "gpt-4.1",
  "gpt-4-1-mini": "gpt-4.1-mini",
  "gpt-4.1-mini-2025-04-14": "gpt-4.1-mini",
  "openai/gpt-4.1-mini": "gpt-4.1-mini",
  "openai:openai/gpt-4.1-mini": "gpt-4.1-mini",
  "openai-fast": "gpt-4.1-nano",
  "openai/gpt-4.1-nano": "gpt-4.1-nano",
  "openai:openai/gpt-4.1-nano": "gpt-4.1-nano",
  "gpt-4-5": "gpt-4.5",
  "openai:openai/gpt-4.5-preview": "gpt-4.5",
  "openai/gpt-oss-120b": "gpt-oss-120b",
  "@cf/openai/gpt-oss-120b": "gpt-oss-120b",
  "gpt-oss:120b": "gpt-oss-120b",
  "Catniti/gpt-oss-120b": "gpt-oss-120b",
  "togetherai:openai/gpt-oss-120b": "gpt-oss-120b",
  "gptimage": "gpt-image",
  "meta/llama2-70b": "llama-2-70b",
  "openrouter:meta-llama/llama-2-70b-chat": "llama-2-70b",
  "@hf/meta-llama/meta-llama-3-8b-instruct": "llama-3-8b",
  "llama-3-8b-instruct": "llama-3-8b",
  "openrouter:meta-llama/llama-3-8b-instruct": "llama-3-8b",
  "llama-3-70b-instruct": "llama-3-70b",
  "openrouter:meta-llama/llama-3-70b-instruct": "llama-3-70b",
  "meta/meta-llama-3-70b-instruct": "llama-3-70b",
  "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
  "llama3.1-8b": "llama-3.1-8b",
  "@cf/meta/llama-3.1-8b-instruct-fp8": "llama-3.1-8b",
  "hf:meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
  "meta/llama-3.1-8b-instruct": "llama-3.1-8b",
  "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
  "llama3.1-70b": "llama-3.1-70b",
  "hf:meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b",
  "meta/llama-3.1-70b-instruct": "llama-3.1-70b",
  "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b",
  "openrouter:meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b",
  "hf:meta-llama/Llama-3.1-405B-Instruct": "llama-3.1-405b",
  "@cf/meta/llama-3.2-3b-instruct": "llama-3.2-3b",
  "hf:meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
  "meta/llama-3.2-3b-instruct": "llama-3.2-3b",
  "meta-llama/llama-3.2-3b-instruct": "llama-3.2-3b",
  "openrouter:meta-llama/llama-3.2-90b-vision-instruct": "llama-3.2-90b",
  "llama-scout": "llama-4-scout",
  "@cf/meta/llama-4-scout-17b-16e-instruct": "llama-4-scout",
  "meta-llama/llama-4-scout": "llama-4-scout",
  "meta-llama/llama-4-maverick": "llama-4-maverick",
  "llama-maverick": "llama-4-maverick",
  "mixtral-8x7b-32768": "mixtral-8x7b",
  "mistral-small": "mistral-small-3.1-24b",
  "mistral-small-3.1-24b-instruct": "mistral-small-3.1-24b",
  "@cf/mistralai/mistral-small-3.1-24b-instruct": "mistral-small-3.1-24b",
  "mistralai/mistral-small-3.1-24b-instruct": "mistral-small-3.1-24b",
  "openrouter:mistralai/mistral-small-3.1-24b-instruct": "mistral-small-3.1-24b",
  "openrouter:nousresearch/nous-hermes-2-mixtral-8x7b-dpo": "hermes-2-dpo",
  "google/gemini-2.5-flash": "gemini-2.5-flash",
  "openrouter:google/gemini-2.5-flash-preview": "gemini-2.5-flash",
  "google/gemini-2.5-pro-preview-05-06": "gemini-2.5-pro",
  "gemini-large": "gemini-3.1-pro",
  "gemini-3.1-pro-preview": "gemini-3.1-pro",
  "google/gemini-3.1-pro-preview": "gemini-3.1-pro",
  "openrouter:google/gemini-3.1-pro-preview": "gemini-3.1-pro",
  "google/gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
  "gemini-flash-lite-3.5": "gemini-flash-lite",
  "openrouter:google/gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
  "google/gemini-3.5-flash": "gemini-3.5-flash",
  "gemini": "gemini-3.5-flash",
  "openrouter:google/gemini-3.5-flash": "gemini-3.5-flash",
  "openrouter:cohere/command-r7b-12-2024": "command-r7b24",
  "command-a-03-2025": "command-a25",
  "cohere/command-a": "command-a",
  "openrouter:cohere/command-a": "command-a",
  "Qwen/Qwen2.5-Coder-32B-Instruct": "qwen-2.5-coder-32b",
  "@cf/qwen/qwen2.5-coder-32b-instruct": "qwen-2.5-coder-32b",
  "hf:Qwen/Qwen2.5-Coder-32B-Instruct": "qwen-2.5-coder-32b",
  "qwen/qwen-2.5-coder-32b-instruct": "qwen-2.5-coder-32b",
  "qwen-3-coder": "qwen-2.5-coder-32b",
  "Qwen/Qwen2.5-VL-72B-Instruct": "qwen-2.5-vl-72b",
  "qwen/qwen2.5-vl-72b-instruct": "qwen-2.5-vl-72b",
  "Qwen/Qwen3-32B": "qwen-3-32b",
  "qwen3-32b": "qwen-3-32b",
  "qwen/qwen3-32b": "qwen-3-32b",
  "@cf/qwen/qwq-32b": "qwq-32b",
  "hf:Qwen/QwQ-32B-Preview": "qwq-32b",
  "openrouter:deepseek/deepseek-v3-base:free": "deepseek-v3",
  "deepseek-reasoning": "deepseek-r1",
  "deepseek-ai/DeepSeek-R1": "deepseek-r1",
  "deepseek/deepseek-r1": "deepseek-r1",
  "v3": "deepseek",
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b",
  "deepseek/deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
  "openrouter:deepseek/deepseek-r1-distill-qwen-1.5b": "deepseek-r1-distill-qwen-1.5b",
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek-r1-distill-qwen-14b",
  "x-ai:x-ai/grok-3": "grok-3",
  "grok-3-reasoning": "grok-3-r1",
  "moonshotai/Kimi-K2-Instruct": "kimi-k2",
  "moonshotai/kimi-k2": "kimi-k2",
  "openrouter:moonshotai/kimi-k2": "kimi-k2",
  "openrouter:perplexity/sonar": "sonar",
  "perplexity/sonar": "perplexity",
  "perplexity-fast": "sonar",
  "openrouter:perplexity/sonar-pro": "sonar-pro",
  "perplexity/sonar-pro": "sonar-pro",
  "openrouter:perplexity/sonar-reasoning": "sonar-reasoning",
  "perplexity-reasoning": "sonar-reasoning-pro",
  "openrouter:perplexity/sonar-reasoning-pro": "sonar-reasoning-pro",
  "perplexity/sonar-reasoning-pro": "sonar-reasoning-pro",
  "openrouter:perplexity/r1-1776": "r1-1776",
  "openrouter:nvidia/llama-3.1-nemotron-70b-instruct": "nemotron-70b",
  "stabilityai/sdxl-turbo": "sdxl-turbo",
  "turbo": "sdxl-turbo",
  "stabilityai/stable-diffusion-3.5-large": "sd-3.5-large",
  "stabilityai-stable-diffusion-3-5-large": "sd-3.5-large",
  "black-forest-labs/FLUX.1-dev": "flux",
  "black-forest-labs-flux-1-dev": "flux",
  "kontext": "flux-kontext",
  "zai-org/GLM-5.2": "glm-5.2",
  "zai-org/GLM-5.2-FP8": "glm-5.2",
  "@cf/zai-org/glm-5.2": "glm-5.2",
  "GLM-5.2": "glm-5.2",
  "z-ai/glm-5.2": "glm-5.2",
  "glm": "glm-5.2",
  "z-ai:z-ai/glm-5.2": "glm-5.2",
  "moonshotai/Kimi-K2.7-Code": "kimi-k2.7-code",
  "@cf/moonshotai/kimi-k2.7-code": "kimi-k2.7-code",
  "moonshotai/kimi-k2.7-code": "kimi-k2.7-code",
  "kimi-code": "kimi-k2.7-code",
  "togetherai:moonshotai/kimi-k2.7-code": "kimi-k2.7-code",
  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": "nemotron-3-ultra-550b-a55b",
  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "nemotron-3-ultra-550b-a55b",
  "deepseek-ai/DeepSeek-V4-Flash": "deepseek-v4-flash",
  "deepseek-ai/deepseek-v4-flash": "deepseek-v4-flash",
  "deepseek/deepseek-v4-flash": "deepseek",
  "deepseek:deepseek/deepseek-v4-flash": "deepseek-v4-flash",
  "deepseek-ai/DeepSeek-V4-Pro": "deepseek-v4-pro",
  "deepseek-ai/deepseek-v4-pro": "deepseek-v4-pro",
  "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
  "deepseek-pro": "deepseek-v4-pro",
  "deepseek:deepseek/deepseek-v4-pro": "deepseek-v4-pro",
  "moonshotai/Kimi-K2.6": "kimi-k2.6",
  "@cf/moonshotai/kimi-k2.6": "kimi-k2.6",
  "moonshotai/kimi-k2.6": "kimi-k2.6",
  "kimi": "kimi-k2.6",
  "moonshotai:moonshotai/kimi-k2.6": "kimi-k2.6",
  "XiaomiMiMo/MiMo-V2.5-Pro": "mimo-v2.5-pro",
  "xiaomi/mimo-v2.5-pro": "mimo-v2.5-pro",
  "openrouter:xiaomi/mimo-v2.5-pro": "mimo-v2.5-pro",
  "Qwen/Qwen3.6-35B-A3B": "qwen-3.6-35b-a3b",
  "qwen3.6-35b-a3b": "qwen-3.6-35b-a3b",
  "qwen/qwen3.6-35b-a3b": "qwen-3.6-35b-a3b",
  "alibaba:qwen/qwen3.6-35b-a3b": "qwen-3.6-35b-a3b",
  "zai-org/GLM-5.1": "glm-5.1",
  "zai-org/GLM-5.1-FP8": "glm-5.1",
  "GLM-5.1": "glm-5.1",
  "z-ai/glm-5.1": "glm-5.1",
  "z-ai:z-ai/glm-5.1": "glm-5.1",
  "Qwen/Qwen3.5-397B-A17B": "qwen-3.5-397b-a17b",
  "qwen3.5-397b-a17b": "qwen-3.5-397b-a17b",
  "qwen/qwen3.5-397b-a17b": "qwen-3.5-397b-a17b",
  "alibaba:qwen/qwen3.5-397b-a17b": "qwen-3.5-397b-a17b",
  "google/gemma-4-26B-A4B-it": "gemma-4-26b-a4b-it",
  "@cf/google/gemma-4-26b-a4b-it": "gemma-4-26b-a4b-it",
  "google/gemma-4-26b-a4b-it:free": "gemma-4-26b-a4b-it",
  "gemma": "gemma-4-26b-a4b-it",
  "openrouter:google/gemma-4-26b-a4b-it:free": "gemma-4-26b-a4b-it",
  "google/gemma-4-31B-it": "gemma-4-31b-it",
  "google/gemma-4-31b-it": "gemma-4-31b-it",
  "google/gemma-4-31b-it:free": "gemma-4-31b-it",
  "gemma-4-31b": "gemma-4-31b-it",
  "togetherai:google/gemma-4-31b-it": "gemma-4-31b-it",
  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": "nemotron-3-super-120b-a12b",
  "zai-org/GLM-5": "glm-5",
  "z-ai/glm-5": "glm-5",
  "z-ai:z-ai/glm-5": "glm-5",
  "MiniMaxAI/MiniMax-M2.5": "minimax-m2.5",
  "minimax/minimax-m2.5": "minimax-m2.5",
  "minimax:minimax/minimax-m2.5": "minimax-m2.5",
  "Qwen/Qwen3-Max": "qwen-3-max",
  "qwen3-max": "qwen-3-max",
  "qwen3-max-2025-09-26": "qwen-3-max",
  "qwen/qwen3-max": "qwen-3-max",
  "alibaba:qwen/qwen3-max": "qwen-3-max",
  "qwen3-max-2026-01-23": "qwen-3-max",
  "Qwen/Qwen3-Max-Thinking": "qwen-3-max-thinking",
  "qwen3-max-thinking": "qwen-3-max-thinking",
  "qwen/qwen3-max-thinking": "qwen-3-max-thinking",
  "openrouter:qwen/qwen3-max-thinking": "qwen-3-max-thinking",
  "moonshotai/Kimi-K2.5": "kimi-k2.5",
  "moonshotai/kimi-k2.5": "kimi-k2.5",
  "moonshotai:moonshotai/kimi-k2.5": "kimi-k2.5",
  "zai-org/GLM-4.7-Flash": "glm-4.7-flash",
  "@cf/zai-org/glm-4.7-flash": "glm-4.7-flash",
  "z-ai/glm-4.7-flash": "glm-4.7-flash",
  "Catniti/glm-4.7-flash": "glm-4.7-flash",
  "z-ai:z-ai/glm-4.7-flash": "glm-4.7-flash",
  "deepseek-ai/DeepSeek-V3.2": "deepseek-v3.2",
  "deepseek/deepseek-v3.2": "deepseek-v3.2",
  "Minor-fun/deepseek-v3.2": "deepseek-v3.2",
  "openrouter:deepseek/deepseek-v3.2": "deepseek-v3.2",
  "black-forest-labs/FLUX-2-klein-4b": "flux-2-klein-4b",
  "black-forest-labs/FLUX-2-klein-9b": "flux-2-klein-9b",
  "thinkingmachines/Inkling": "inkling",
  "thinkingmachines/inkling": "inkling",
  "togetherai:thinkingmachines/inkling": "inkling",
  "prism-ml/Ternary-Bonsai-27B-gguf": "ternary-bonsai-27b-gguf",
  "Qwen/Qwen3.6-27B": "qwen-3.6-27b",
  "qwen/qwen3.6-27b": "qwen-3.6-27b",
  "qwen3.6-27b": "qwen-3.6-27b",
  "alibaba:qwen/qwen3.6-27b": "qwen-3.6-27b",
  "tencent/Hy3": "hy3",
  "tencent/hy3-preview": "hy3",
  "openrouter:tencent/hy3-preview": "hy3",
  "deepreinforce-ai/Ornith-1.0-35B-FP8": "ornith-1.0-35b",
  "swiss-ai/Apertus-v1.5-70B": "apertus-v1.5-70b",
  "MiniMaxAI/MiniMax-M3": "minimax-m3",
  "MiniMax-M3": "minimax-m3",
  "minimaxai/minimax-m3": "minimax-m3",
  "minimax/minimax-m3": "minimax-m3",
  "minimax": "minimax-m3",
  "minimax:minimax/minimax-m3": "minimax-m3",
  "swiss-ai/Apertus-v1.5-8B": "apertus-v1.5-8b",
  "Qwen/Qwen3.5-9B": "qwen-3.5-9b",
  "qwen/qwen3.5-9b": "qwen-3.5-9b",
  "togetherai:qwen/qwen3.5-9b": "qwen-3.5-9b",
  "Qwen/Qwen3-8B": "qwen-3-8b",
  "qwen3-8b": "qwen-3-8b",
  "qwen/qwen3-8b": "qwen-3-8b",
  "openai/gpt-oss-20b": "gpt-oss-20b",
  "@cf/openai/gpt-oss-20b": "gpt-oss-20b",
  "gpt-oss:20b": "gpt-oss-20b",
  "openai/gpt-oss-20b:free": "gpt-oss-20b",
  "gpt-oss": "gpt-oss-20b",
  "togetherai:openai/gpt-oss-20b": "gpt-oss-20b",
  "Qwen/Qwen3.5-27B": "qwen-3.5-27b",
  "qwen3.5-27b": "qwen-3.5-27b",
  "qwen/qwen3.5-27b": "qwen-3.5-27b",
  "alibaba:qwen/qwen3.5-27b": "qwen-3.5-27b",
  "Qwen/Qwen3-Coder-30B-A3B-Instruct": "qwen-3-coder-30b-a3b",
  "qwen3-coder-30b-a3b": "qwen-3-coder-30b-a3b",
  "qwen/qwen3-coder-30b-a3b-instruct": "qwen-2.5-coder-32b",
  "alibaba:qwen/qwen3-coder-30b-a3b-instruct": "qwen-3-coder-30b-a3b",
  "XiaomiMiMo/MiMo-V2.5": "mimo-v2.5",
  "xiaomi/mimo-v2.5": "mimo-v2.5",
  "MarcosFRG/mimo-v2.5": "mimo-v2.5",
  "openrouter:xiaomi/mimo-v2.5": "mimo-v2.5",
  "Qwen/Qwen3-Coder-Next": "qwen-3-coder-next",
  "qwen/qwen3-coder-next": "qwen-3-coder-next",
  "openrouter:qwen/qwen3-coder-next": "qwen-3-coder-next",
  "google/gemma-3-4b-it": "gemma-3-4b-it",
  "openrouter:google/gemma-3-4b-it": "gemma-3-4b-it",
  "Qwen/Qwen3.5-122B-A10B": "qwen-3.5-122b-a10b",
  "qwen3.5-122b-a10b": "qwen-3.5-122b-a10b",
  "qwen/qwen3.5-122b-a10b": "qwen-3.5-122b-a10b",
  "alibaba:qwen/qwen3.5-122b-a10b": "qwen-3.5-122b-a10b",
  "inclusionAI/Ling-2.6-1T": "ling-2.6-1t",
  "inclusionai/ling-2.6-1t": "ling-2.6-1t",
  "openrouter:inclusionai/ling-2.6-1t": "ling-2.6-1t",
  "Qwen/Qwen3-4B-Instruct-2507": "qwen-3-4b-2507",
  "stepfun-ai/Step-3.7-Flash": "step-3.7-flash",
  "stepfun-ai/step-3.7-flash": "step-3.7-flash",
  "stepfun/step-3.7-flash": "step-3.7-flash",
  "step-flash": "step-3.7-flash",
  "openrouter:stepfun/step-3.7-flash": "step-3.7-flash",
  "zai-org/GLM-4.5-Air": "glm-4.5-air",
  "GLM-4.5-Air": "glm-4.5-air",
  "z-ai/glm-4.5-air": "glm-4.5-air",
  "z-ai:z-ai/glm-4.5-air": "glm-4.5-air",
  "Qwen/Qwen3-14B": "qwen-3-14b",
  "qwen3-14b": "qwen-3-14b",
  "qwen/qwen3-14b": "qwen-3-14b",
  "Qwen/Qwen3-Next-80B-A3B-Instruct": "qwen-3-next-80b-a3b",
  "qwen3-next-80b-a3b": "qwen-3-next-80b-a3b",
  "qwen3-next-80b-a3b-instruct": "qwen-3-next-80b-a3b",
  "qwen/qwen3-next-80b-a3b-instruct": "qwen-3-next-80b-a3b",
  "alibaba:qwen/qwen3-next-80b-a3b-instruct": "qwen-3-next-80b-a3b",
  "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen-2.5-coder-7b",
  "openrouter:qwen/qwen2.5-coder-7b-instruct": "qwen-2.5-coder-7b",
  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4": "nemotron-3-ultra-550b-a55b-nvfp4",
  "Qwen/Qwen3.5-35B-A3B": "qwen-3.5-35b-a3b",
  "qwen3.5-35b-a3b": "qwen-3.5-35b-a3b",
  "qwen/qwen3.5-35b-a3b": "qwen-3.5-35b-a3b",
  "alibaba:qwen/qwen3.5-35b-a3b": "qwen-3.5-35b-a3b",
  "google/gemma-3-27b-it": "gemma-3-27b-it",
  "openrouter:google/gemma-3-27b-it": "gemma-3-27b-it",
  "google/gemma-3-12b-it": "gemma-3-12b-it",
  "openrouter:google/gemma-3-12b-it": "gemma-3-12b-it",
  "microsoft/phi-4": "phi-4",
  "openrouter:microsoft/phi-4": "phi-4",
  "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen-3-vl-30b-a3b",
  "qwen3-vl-30b-a3b": "qwen-3-vl-30b-a3b",
  "qwen/qwen3-vl-30b-a3b-instruct": "qwen-3-vl-30b-a3b",
  "openrouter:qwen/qwen3-vl-30b-a3b-instruct": "qwen-3-vl-30b-a3b",
  "prism-ml/Ternary-Bonsai-27B-AWQ-4bit": "ternary-bonsai-27b-awq-4bit",
  "moonshotai/Kimi-K2-Instruct-0905": "kimi-k2-0905",
  "kimi-k2-0905-preview": "kimi-k2-0905",
  "moonshotai/kimi-k2-0905": "kimi-k2-0905",
  "openrouter:moonshotai/kimi-k2-0905": "kimi-k2-0905",
  "Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-7b",
  "hf:Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-7b",
  "qwen/qwen-2.5-7b-instruct": "qwen-2.5-7b",
  "Qwen/Qwen3-VL-235B-A22B-Instruct": "qwen-3-vl-235b-a22b",
  "qwen3-vl-235b-a22b": "qwen-3-vl-235b-a22b",
  "qwen3-vl-235b-a22b-instruct": "qwen-3-vl-235b-a22b",
  "qwen/qwen3-vl-235b-a22b-instruct": "qwen-3-vl-235b-a22b",
  "openrouter:qwen/qwen3-vl-235b-a22b-instruct": "qwen-3-vl-235b-a22b",
  "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
  "@cf/meta/llama-3.3-70b-instruct-fp8-fast": "llama-3.3-70b",
  "hf:meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
  "meta/llama-3.3-70b-instruct": "llama-3.3-70b",
  "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b",
  "llama": "llama-3.3-70b",
  "Qwen/Qwen3-Coder-480B-A35B-Instruct": "qwen-3-coder-480b-a35b",
  "qwen3-coder-480b-a35b": "qwen-3-coder-480b-a35b",
  "qwen3-coder-480b-a35b-instruct": "qwen-3-coder-480b-a35b",
  "alibaba:qwen/qwen3-coder-480b-a35b-instruct": "qwen-3-coder-480b-a35b",
  "openai/gpt-oss-safeguard-20b": "gpt-oss-safeguard-20b",
  "openrouter:openai/gpt-oss-safeguard-20b": "gpt-oss-safeguard-20b",
  "CohereLabs/c4ai-command-r7b-12-2024": "command-r7b24",
  "command-r7b-12-2024": "command-r7b24",
  "cohere/command-r7b-12-2024": "command-r7b24",
  "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen-3-235b-a22b-2507",
  "qwen3-235b-a22b-instruct-2507": "qwen-3-235b-a22b-2507",
  "qwen/qwen3-235b-a22b-2507": "qwen-3-235b-a22b-2507",
  "openrouter:qwen/qwen3-235b-a22b-2507": "qwen-3-235b-a22b-2507",
  "meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama-4-scout-17b-16e",
  "llama-4-scout-17b-16e-instruct": "llama-4-scout-17b-16e",
  "stepfun-ai/Step-3.5-Flash": "step-3.5-flash",
  "stepfun-ai/step-3.5-flash": "step-3.5-flash",
  "stepfun/step-3.5-flash": "step-3.5-flash",
  "openrouter:stepfun/step-3.5-flash": "step-3.5-flash",
  "Qwen/Qwen2.5-Coder-3B-Instruct": "qwen-2.5-coder-3b",
  "Sao10K/L3-8B-Stheno-v3.2": "l3-8b-stheno-v3.2",
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek-r1-distill-qwen-7b",
  "meta-llama/Llama-Guard-4-12B": "llama-guard-4-12b",
  "meta/llama-guard-4-12b": "llama-guard-4-12b",
  "meta-llama/llama-guard-4-12b": "llama-guard-4-12b",
  "togetherai:meta-llama/llama-guard-4-12b": "llama-guard-4-12b",
  "Qwen/Qwen3-4B-Thinking-2507": "qwen-3-4b-thinking-2507",
  "zai-org/GLM-4.7": "glm-4.7",
  "GLM-4.7": "glm-4.7",
  "z-ai/glm-4.7": "glm-4.7",
  "Catniti/glm-4.7": "glm-4.7",
  "z-ai:z-ai/glm-4.7": "glm-4.7",
  "aisingapore/Gemma-SEA-LION-v4-27B-IT": "gemma-sea-lion-v4-27b-it",
  "CohereLabs/c4ai-command-a-03-2025": "command-a25",
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-r1-distill-llama-8b",
  "openrouter:deepseek/deepseek-r1-distill-llama-8b": "deepseek-r1-distill-llama-8b",
  "swiss-ai/Apertus-8B-Instruct-2509": "apertus-8b-2509",
  "CohereLabs/tiny-aya-global": "tiny-aya-global",
  "Qwen/Qwen3-235B-A22B": "qwen-3-235b-a22b",
  "qwen3-235b-a22b": "qwen-3-235b-a22b",
  "qwen/qwen3-235b-a22b": "qwen-3-235b-a22b",
  "alibaba:qwen/qwen3-235b-a22b": "qwen-3-235b-a22b",
  "speakleash/Bielik-11B-v3.0-Instruct": "bielik-11b-v3.0",
  "CohereLabs/tiny-aya-earth": "tiny-aya-earth",
  "MiniMaxAI/MiniMax-M2.7": "minimax-m2.7",
  "MiniMax-M2.7": "minimax-m2.7",
  "minimaxai/minimax-m2.7": "minimax-m2.7",
  "minimax/minimax-m2.7": "minimax-m2.7",
  "minimax:minimax/minimax-m2.7": "minimax-m2.7",
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "llama-4-maverick-17b-128e",
  "meta/llama-4-maverick-17b-128e-instruct": "llama-4-maverick-17b-128e",
  "deepseek-ai/DeepSeek-V3-0324": "deepseek-v3-0324",
  "baidu/ERNIE-4.5-VL-424B-A47B-Base-PT": "ernie-4.5-vl-424b-a47b-base-pt",
  "zai-org/GLM-4-32B-0414": "glm-4-32b-0414",
  "deepseek-ai/DeepSeek-R1-0528": "deepseek-r1-0528",
  "deepseek/deepseek-r1-0528": "deepseek-r1-0528",
  "openrouter:deepseek/deepseek-r1-0528": "deepseek-r1-0528",
  "Qwen/Qwen3-235B-A22B-Thinking-2507": "qwen-3-235b-a22b-thinking-2507",
  "qwen3-235b-a22b-thinking-2507": "qwen-3-235b-a22b-thinking-2507",
  "qwen/qwen3-235b-a22b-thinking-2507": "qwen-3-235b-a22b-thinking-2507",
  "openrouter:qwen/qwen3-235b-a22b-thinking-2507": "qwen-3-235b-a22b-thinking-2507",
  "deepseek-ai/DeepSeek-V3": "deepseek",
  "hf:deepseek-ai/DeepSeek-V3": "deepseek",
  "zai-org/GLM-4.5": "glm-4.5",
  "GLM-4.5": "glm-4.5",
  "z-ai/glm-4.5": "glm-4.5",
  "z-ai:z-ai/glm-4.5": "glm-4.5",
  "zai-org/GLM-4.5V": "glm-4.5v",
  "zai-org/GLM-4.5V-FP8": "glm-4.5v",
  "z-ai/glm-4.5v": "glm-4.5v",
  "z-ai:z-ai/glm-4.5v": "glm-4.5v",
  "alpindale/WizardLM-2-8x22B": "wizardlm-2-8x22b",
  "microsoft/wizardlm-2-8x22b": "wizardlm-2-8x22b",
  "openrouter:microsoft/wizardlm-2-8x22b": "wizardlm-2-8x22b",
  "Qwen/Qwen2.5-72B-Instruct": "qwen-2.5-72b",
  "hf:Qwen/Qwen2.5-72B-Instruct": "qwen-2.5-72b",
  "qwen/qwen-2.5-72b-instruct": "qwen-2.5-72b",
  "zai-org/GLM-4.6": "glm-4.6",
  "z-ai/glm-4.6": "glm-4.6",
  "z-ai:z-ai/glm-4.6": "glm-4.6",
  "zai-org/GLM-4.6V-Flash": "glm-4.6v-flash",
  "MarcosFRG/glm-4.6v-flash": "glm-4.6v-flash",
  "z-ai:z-ai/glm-4.6v-flash": "glm-4.6v-flash",
  "Sao10K/L3-8B-Lunaris-v1": "l3-8b-lunaris",
  "deepseek-ai/DeepSeek-V3.1": "deepseek-v3.1",
  "zai-org/GLM-4.6V-FP8": "glm-4.6v",
  "GLM-4.6V": "glm-4.6v",
  "z-ai/glm-4.6v": "glm-4.6v",
  "z-ai:z-ai/glm-4.6v": "glm-4.6v",
  "MiniMaxAI/MiniMax-M1-80k": "minimax-m1-80k",
  "deepseek-ai/DeepSeek-V3.1-Terminus": "deepseek-v3.1-terminus",
  "deepseek/deepseek-v3.1-terminus": "deepseek-v3.1-terminus",
  "openrouter:deepseek/deepseek-v3.1-terminus": "deepseek-v3.1-terminus",
  "CohereLabs/c4ai-command-r-08-2024": "command-r24",
  "command-r-08-2024": "command-r24",
  "cohere/command-r-08-2024": "command-r24",
  "openrouter:cohere/command-r-08-2024": "command-r24",
  "deepseek-ai/DeepSeek-V3.2-Exp": "deepseek-v3.2-exp",
  "deepseek/deepseek-v3.2-exp": "deepseek-v3.2-exp",
  "openrouter:deepseek/deepseek-v3.2-exp": "deepseek-v3.2-exp",
  "CohereLabs/aya-expanse-32b": "aya-expanse-32b",
  "c4ai-aya-expanse-32b": "aya-expanse-32b",
  "CohereLabs/aya-vision-32b": "aya-vision-32b",
  "c4ai-aya-vision-32b": "aya-vision-32b",
  "MiniMaxAI/MiniMax-M2": "minimax-m2",
  "minimax-m2-preview": "minimax-m2",
  "minimax/minimax-m2": "minimax-m2",
  "minimax:minimax/minimax-m2": "minimax-m2",
  "CohereLabs/c4ai-command-r7b-arabic-02-2025": "command-r7b-arabic25",
  "command-r7b-arabic-02-2025": "command-r7b-arabic25",
  "CohereLabs/command-a-vision-07-2025": "command-a-vision25",
  "command-a-vision-07-2025": "command-a-vision25",
  "MiniMaxAI/MiniMax-M2.1": "minimax-m2.1",
  "minimax-m2.1-preview": "minimax-m2.1",
  "minimax/minimax-m2.1": "minimax-m2.1",
  "minimax:minimax/minimax-m2.1": "minimax-m2.1",
  "CohereLabs/command-a-reasoning-08-2025": "command-a-reasoning25",
  "command-a-reasoning-08-2025": "command-a-reasoning25",
  "google/gemma-3n-E4B-it": "gemma-3n-e4b-it",
  "google/gemma-3n-e4b-it": "gemma-3n-e4b-it",
  "togetherai:google/gemma-3n-e4b-it": "gemma-3n-e4b-it",
  "CohereLabs/command-a-translate-08-2025": "command-a-translate25",
  "command-a-translate-08-2025": "command-a-translate25",
  "Qwen/Qwen3-VL-235B-A22B-Thinking": "qwen-3-vl-235b-a22b-thinking",
  "qwen3-vl-235b-a22b-thinking": "qwen-3-vl-235b-a22b-thinking",
  "qwen/qwen3-vl-235b-a22b-thinking": "qwen-3-vl-235b-a22b-thinking",
  "openrouter:qwen/qwen3-vl-235b-a22b-thinking": "qwen-3-vl-235b-a22b-thinking",
  "pearl-ai/Gemma-4-31B-it-pearl": "gemma-4-31b-it-pearl",
  "CohereLabs/tiny-aya-water": "tiny-aya-water",
  "swiss-ai/Apertus-70B-Instruct-2509": "apertus-70b-2509",
  "zai-org/AutoGLM-Phone-9B-Multilingual": "autoglm-phone-9b-multilingual",
  "CohereLabs/tiny-aya-fire": "tiny-aya-fire",
  "aisingapore/Qwen-SEA-LION-v4-32B-IT": "qwen-sea-lion-v4-32b-it",
  "allenai/Olmo-3-7B-Instruct": "olmo-3-7b",
  "utter-project/EuroLLM-22B-Instruct-2512": "eurollm-22b-2512",
  "command-r-plus-08-2024": "command-r-plus24",
  "cohere/command-r-plus-08-2024": "command-r-plus24",
  "openrouter:cohere/command-r-plus-08-2024": "command-r-plus24",
  "default:": "default",
  "Agent Tools (MCP)": "agent-tools",
  "openai/gpt-4o:batch": "gpt-4o",
  "openrouter:openai/gpt-4o:batch": "gpt-4o",
  "openai/gpt-4o-mini:batch": "gpt-4o-mini",
  "openrouter:openai/gpt-4o-mini:batch": "gpt-4o-mini",
  "openrouter:openai/o1": "o1",
  "infron:openai/o1-mini": "o1-mini",
  "openai/o3-mini:batch": "o3-mini",
  "openrouter:openai/o3-mini:batch": "o3-mini",
  "openai/o4-mini:batch": "o4-mini",
  "openrouter:openai/o4-mini:batch": "o4-mini",
  "openai/gpt-4.1:batch": "gpt-4.1",
  "openrouter:openai/gpt-4.1:batch": "gpt-4.1",
  "openai/gpt-4.1-mini:batch": "gpt-4.1-mini",
  "openrouter:openai/gpt-4.1-mini:batch": "gpt-4.1-mini",
  "openai/gpt-4.1-nano:batch": "gpt-4.1-nano",
  "openrouter:openai/gpt-4.1-nano:batch": "gpt-4.1-nano",
  "gpt-4.5-preview": "gpt-4.5",
  "openai/gpt-oss-120b:batch": "gpt-oss-120b",
  "llama-3.1-405b-instruct": "llama-3.1-405b",
  "llamascout": "llama-4-scout",
  "google/gemini-3.1-pro-preview:batch": "gemini-3.1-pro",
  "openrouter:google/gemini-3.1-pro-preview:batch": "gemini-3.1-pro",
  "openrouter:google/gemini-3.1-flash-lite:batch": "gemini-3.1-flash-lite",
  "srv_mrgy0nmbc8a86c407f17:models/gemini-3.6-flash": "gemini-3.6-flash",
  "google/gemini-3.6-flash": "gemini-3.6-flash",
  "google/gemini-3.6-flash:batch": "gemini-3.6-flash",
  "openrouter:google/gemini-3.6-flash:batch": "gemini-3.6-flash",
  "models/gemini-3.7-flash": "gemini-3.7-flash",
  "google/gemini-3.7-flash": "gemini-3.5-flash",
  "google/gemini-3.7-flash:batch": "gemini-3.7-flash",
  "openrouter:google/gemini-3.7-flash:batch": "gemini-3.7-flash",
  "models/gemini-3.8-flash": "gemini-3.8-flash",
  "google/gemini-3.8-flash": "gemini-3.8-flash",
  "google/gemini-3.8-flash:batch": "gemini-3.8-flash",
  "openrouter:google/gemini-3.8-flash:batch": "gemini-3.8-flash",
  "srv_mrgy0nmbc8a86c407f17:models/gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
  "google/gemini-3.5-flash-lite": "gemini-flash-lite",
  "google/gemini-3.5-flash-lite:batch": "gemini-3.5-flash-lite",
  "openrouter:google/gemini-3.5-flash-lite:batch": "gemini-3.5-flash-lite",
  "gemini-3.5-flash:batch": "gemini-3.5-flash",
  "openrouter:google/gemini-3.5-flash:batch": "gemini-3.5-flash",
  "gemini-flash-lite-latest": "gemini-flash-lite",
  "models/gemini-flash-lite-latest": "gemini-flash-lite",
  "qwq-32b-preview": "qwq-32b",
  "deepseek-chat": "deepseek-v3",
  "auto": "grok-3",
  "kimi-k2-instruct": "kimi-k2",
  "deepseek-ai/DeepSeek-V4.1-Flash": "deepseek-v4.1-flash",
  "deepseek/deepseek-v4.1-flash": "deepseek-v4.1-flash",
  "togetherai:deepseek-ai/deepseek-v4.1-flash": "deepseek-v4.1-flash",
  "zai-org/GLM-5.3": "glm-5.3",
  "zai-org/GLM-5.3-BF16": "glm-5.3",
  "GLM-5.3": "glm-5.3",
  "z-ai/glm-5.3": "glm-5.3",
  "z-ai/glm-5.3:batch": "glm-5.3",
  "z-ai:z-ai/glm-5.3": "glm-5.3",
  "zai-org/GLM-5.3-Flash": "glm-5.3-flash",
  "zai-org/GLM-5.3-Flash-BF16": "glm-5.3-flash",
  "z-ai/glm-5.3-flash": "glm-5.3-flash",
  "GLM-5.3-Flash": "glm-5.3-flash",
  "z-ai/glm-5.3-flash:batch": "glm-5.3-flash",
  "z-ai:z-ai/glm-5.3-flash": "glm-5.3-flash",
  "moonshotai/Kimi-K3": "kimi-k3",
  "moonshotai/kimi-k3": "kimi-k3",
  "moonshotai/kimi-k3:batch": "kimi-k3",
  "togetherai:moonshotai/kimi-k3": "kimi-k3",
  "Qwen/Qwen3.8-2.4T-A95B": "qwen-3.8-2.4t-a95b",
  "qwen/qwen3.8-2.4t-a95b": "qwen-3.8-2.4t-a95b",
  "qwen/qwen3.8-2.4t-a95b:batch": "qwen-3.8-2.4t-a95b",
  "togetherai:qwen/qwen3.8-2.4t-a95b": "qwen-3.8-2.4t-a95b",
  "deepseek-ai/DeepSeek-V4-Pro-0813": "deepseek-v4-pro-0813",
  "HuggingChat:deepseek-ai/DeepSeek-V4-Pro-0813": "deepseek-v4-pro-0813",
  "deepseek/deepseek-v4-pro-0813": "deepseek-v4-pro-0813",
  "deepseek/deepseek-v4-pro-0813:batch": "deepseek-v4-pro-0813",
  "togetherai:deepseek-ai/deepseek-v4-pro-0813": "deepseek-v4-pro-0813",
  "deepseek-ai/DeepSeek-V4-Flash-0731": "deepseek-v4-flash-0731",
  "deepseek-ai/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
  "deepseek/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
  "deepseek/deepseek-v4-flash-0731:batch": "deepseek-v4-flash-0731",
  "togetherai:deepseek-ai/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
  "z-ai/glm-5.2:free": "glm-5.2",
  "openrouter:moonshotai/kimi-k2.7-code": "kimi-k2.7-code",
  "community/NamanSoni78/nemotron-3-ultra-550b-a55b": "nemotron-3-ultra-550b-a55b",
  "nvidia/nemotron-3-ultra-550b-a55b": "nemotron-3-ultra-550b-a55b",
  "nvidia/nemotron-3-ultra-550b-a55b:free": "nemotron-3-ultra-550b-a55b",
  "deepseek-z/deepseek-v4-flash": "deepseek-v4-flash",
  "deepseek-v4-flash:0731": "deepseek-v4-flash",
  "deepseek-v4-flash-free": "deepseek-v4-flash",
  "deepseek/deepseek-v4-flash-free": "deepseek-v4-flash",
  "deepseek/deepseek-v4-flash-latest": "deepseek-v4-flash",
  "deepseek-v4-pro:0813": "deepseek-v4-pro",
  "openrouter:deepseek/deepseek-v4-pro": "deepseek-v4-pro",
  "openrouter:moonshotai/kimi-k2.6": "kimi-k2.6",
  "HuggingChat:XiaomiMiMo/MiMo-V2.5-Pro": "mimo-v2.5-pro",
  "openrouter:qwen/qwen3.6-35b-a3b": "qwen-3.6-35b-a3b",
  "openrouter:qwen/qwen3.5-397b-a17b": "qwen-3.5-397b-a17b",
  "google/gemma-4-26b-a4b-it": "gemma-4-26b-a4b-it",
  "openrouter:google/gemma-4-31b-it:free": "gemma-4-31b-it",
  "nvidia/nemotron-3-super-120b-a12b": "nemotron-3-super-120b-a12b",
  "nvidia/nemotron-3-super-120b-a12b:free": "nemotron-3-super-120b-a12b",
  "openrouter:nvidia/nemotron-3-super-120b-a12b:free": "nemotron-3-super-120b-a12b",
  "qwen3-max-preview": "qwen-3-max",
  "openrouter:qwen/qwen3-max": "qwen-3-max",
  "Qwen/Qwen3.8-27B": "qwen-3.8-27b",
  "qwen3.8-27b": "qwen-3.8-27b",
  "openrouter-z/qwen3.8-27b": "qwen-3.8-27b",
  "qwen/qwen3.8-27b": "qwen-3.8-27b",
  "qwen/qwen3.8-27b:free": "qwen-3.8-27b",
  "qwen/qwen3.8-27b-free": "qwen-3.8-27b",
  "openrouter:qwen/qwen3.8-27b:free": "qwen-3.8-27b",
  "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp": "deepseek-v4-flash-vision-exp",
  "deepseek/deepseek-v4-flash-vision-exp": "deepseek-v4-flash-vision-exp",
  "deepseek/deepseek-v4-flash-vision-exp:batch": "deepseek-v4-flash-vision-exp",
  "openrouter:deepseek/deepseek-v4-flash-vision-exp:batch": "deepseek-v4-flash-vision-exp",
  "HuggingChat:Qwen/Qwen3.5-9B": "qwen-3.5-9b",
  "qwen/qwen3.5-9b:batch": "qwen-3.5-9b",
  "meta-models/Muse-Glimmer-30B": "muse-glimmer-30b",
  "srv_mp5miql908c8738d71be:community/YoannDev90/muse-glimmer-30b:free": "muse-glimmer-30b",
  "meta/muse-glimmer-30b": "muse-glimmer-30b",
  "meta/muse-glimmer-30b:batch": "muse-glimmer-30b",
  "community/YoannDev90/muse-glimmer-30b:free": "muse-glimmer-30b",
  "togetherai:meta-models/muse-glimmer-30b": "muse-glimmer-30b",
  "srv_mwvnwe1m0ddf970d82ac:openai/gpt-oss-20b": "gpt-oss-20b",
  "openrouter:openai/gpt-oss-20b": "gpt-oss-20b",
  "inclusionAI/Ling-3.0-flash-VL": "ling-3.0-flash-vl",
  "srv_monk1pkz433a519ff2be:inclusionai/ling-3.0-flash-vl:free": "ling-3.0-flash-vl",
  "inclusionai/ling-3.0-flash-vl": "ling-3.0-flash-vl",
  "inclusionai/ling-3.0-flash-vl:free": "ling-3.0-flash-vl",
  "openrouter:inclusionai/ling-3.0-flash-vl:free": "ling-3.0-flash-vl",
  "inclusionAI/Ling-3.0-flash-Fin": "ling-3.0-flash-fin",
  "inclusionai/ling-3.0-flash-fin:free": "ling-3.0-flash-fin",
  "ling-3.0-flash-fin-free": "ling-3.0-flash-fin",
  "openrouter:inclusionai/ling-3.0-flash-fin:free": "ling-3.0-flash-fin",
  "srv_mtsj8uzo97d3c0d49960:minimax-z/minimax-m3": "minimax-m3",
  "minimax/minimax-m3:batch": "minimax-m3",
  "togetherai:minimaxai/minimax-m3": "minimax-m3",
  "openrouter:qwen/qwen3.6-27b": "qwen-3.6-27b",
  "openrouter:qwen/qwen3-coder-30b-a3b-instruct": "qwen-3-coder-30b-a3b",
  "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "nemotron-3.5-lightning-30b-a3b",
  "srv_mwvnwe1m0ddf970d82ac:nvidia/nemotron-3.5-lightning-30b-a3b": "nemotron-3.5-lightning-30b-a3b",
  "nvidia/nemotron-3.5-lightning-30b-a3b": "nemotron-3.5-lightning-30b-a3b",
  "inclusionAI/Ling-3.0-flash": "ling-3.0-flash",
  "inclusionai/ling-3.0-flash": "ling-3.0-flash",
  "openrouter:inclusionai/ling-3.0-flash": "ling-3.0-flash",
  "openrouter:qwen/qwen3.5-35b-a3b": "qwen-3.5-35b-a3b",
  "ibm-granite/granite-4.2-3b": "granite-4.2-3b",
  "ibm-granite/granite-4.2-8b": "granite-4.2-8b",
  "openrouter:ibm-granite/granite-4.2-8b": "granite-4.2-8b",
  "thinkingmachines/inkling:free": "inkling",
  "srv_mtkdrhxt45fe276b15c0:@cf/zai-org/glm-4.7-flash": "glm-4.7-flash",
  "openrouter:z-ai/glm-4.7-flash": "glm-4.7-flash",
  "thinkingmachines/Inkling-Small": "inkling-small",
  "thinkingmachines/inkling-small:free": "inkling-small",
  "openrouter:thinkingmachines/inkling-small:free": "inkling-small",
  "srv_mkom688d57c76d8a3542:openai/gpt-oss-safeguard-20b": "gpt-oss-safeguard-20b",
  "openrouter:meta-llama/llama-guard-4-12b": "llama-guard-4-12b",
  "openrouter:moonshotai/kimi-k2.5": "kimi-k2.5",
  "Qwen/Qwen3-30B-A3B": "qwen-3-30b-a3b",
  "qwen3-30b-a3b": "qwen-3-30b-a3b",
  "qwen/qwen3-30b-a3b": "qwen-3-30b-a3b",
  "openrouter:qwen/qwen3-30b-a3b": "qwen-3-30b-a3b",
  "mimo-v2.5-free": "mimo-v2.5",
  "ibm-granite/granite-4.2-30b": "granite-4.2-30b",
  "openrouter:qwen/qwen3.5-122b-a10b": "qwen-3.5-122b-a10b",
  "stepfun/step-3.7-flash:free": "step-3.7-flash",
  "openrouter:qwen/qwen3.5-27b": "qwen-3.5-27b",
  "openrouter:minimax/minimax-m2.7": "minimax-m2.7",
  "tencent/hy3-free": "hy3",
  "openrouter:qwen/qwen3-235b-a22b": "qwen-3-235b-a22b",
  "infron:qwen/qwen3-coder-480b-a35b:priority": "qwen-3-coder-480b-a35b",
  "zai-org/GLM-4.6-FP8": "glm-4.6",
  "openrouter:minimax/minimax-m2.5": "minimax-m2.5",
  "NousResearch/Hermes-3-Llama-3.1-70B": "hermes-3-llama-3.1-70b",
  "nousresearch/hermes-3-llama-3.1-70b": "hermes-3-llama-3.1-70b",
  "openrouter:nousresearch/hermes-3-llama-3.1-70b": "hermes-3-llama-3.1-70b",
  "srv_msu3vtvl920c386aa8a4:deepseek-ai/DeepSeek-V3.1": "deepseek-v3.1",
  "infron:deepseek/deepseek-v3.1": "deepseek-v3.1",
  "openrouter:z-ai/glm-4.5v": "glm-4.5v",
  "openrouter:qwen/qwen3-next-80b-a3b-instruct": "qwen-3-next-80b-a3b",
  "srv_mp5miql908c8738d71be:community/MarcosFRG/glm-4.6v-flash": "glm-4.6v-flash",
  "community/MarcosFRG/glm-4.6v-flash": "glm-4.6v-flash",
  "infron:z-ai/glm-4.6v-flash": "glm-4.6v-flash",
  "openrouter:minimax/minimax-m2": "minimax-m2",
  "openrouter:minimax/minimax-m2.1": "minimax-m2.1",
  "qwen3-vl-30b-a3b-instruct": "qwen-3-vl-30b-a3b",
  "infron:cohere/command-a-03-2025": "command-a25",
  "openrouter:z-ai/glm-4.6v": "glm-4.6v",
  "openrouter/auto": "auto",
  "infron:infron/auto": "auto",
  "infron:openai/o1": "o1",
  "infron:openai/o3-mini": "o3-mini",
  "infron:openai/o4-mini": "o4-mini",
  "openrouter:openai/gpt-4.1": "gpt-4.1",
  "openrouter:openai/gpt-4.1-mini": "gpt-4.1-mini",
  "openrouter:openai/gpt-4.1-nano": "gpt-4.1-nano",
  "gpt-4.5-preview-2025-02-27": "gpt-4.5",
  "openai/gpt-image-1-mini": "gpt-image",
  "llama3-8b": "llama-3-8b",
  "llama-3.1-8b-instruct": "llama-3.1-8b",
  "llama-3.1-70b-instruct": "llama-3.1-70b",
  "llama-3.2-3b-instruct": "llama-3.2-3b",
  "meta/llama-4-scout": "llama-4-scout",
  "srv_mp5miql908c8738d71be:meta/llama-4-scout": "llama-4-scout",
  "srv_mp5miql908c8738d71be:meta/llama-4-maverick": "llama-4-maverick",
  "meta/llama-4-maverick": "llama-4-maverick",
  "gemini-2.5-pro-preview-06-05": "gemini-2.5-pro",
  "srv_mrgy0nmbc8a86c407f17:models/gemini-2.5-pro": "gemini-2.5-pro",
  "srv_mp5miql908c8738d71be:google/gemini-3.1-pro-preview": "gemini-3.1-pro",
  "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
  "srv_mrgy0nmbc8a86c407f17:models/gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
  "openrouter:google/gemini-3.6-flash": "gemini-3.6-flash",
  "openrouter:google/gemini-3.7-flash": "gemini-3.7-flash",
  "openrouter:google/gemini-3.8-flash": "gemini-3.8-flash",
  "openrouter:google/gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
  "qwen-2.5-coder-32b-instruct": "qwen-2.5-coder-32b",
  "qwen2.5-vl-72b-instruct": "qwen-2.5-vl-72b",
  "srv_mp5miql908c8738d71be:perplexity/sonar": "sonar",
  "black-forest-labs/flux.1-schnell": "flux",
  "deepseek-ai/DeepSeek-V4.1-Flash-1017": "deepseek-v4.1-flash-1017",
  "deepseek-v4.1-flash-1017:free": "deepseek-v4.1-flash-1017",
  "deepseek/deepseek-v4.1-flash-1017": "deepseek-v4.1-flash-1017",
  "togetherai:deepseek-ai/deepseek-v4.1-flash-1017": "deepseek-v4.1-flash-1017",
  "nvidia/NVIDIA-Nemotron-3.5-Super-VL": "nemotron-3.5-super-vl",
  "nvidia/nemotron-3.5-super-vl": "nemotron-3.5-super-vl",
  "togetherai:nvidia/nvidia-nemotron-3.5-super-vl": "nemotron-3.5-super-vl",
  "tencent/Hy4-preview": "hy4",
  "hy4-preview": "hy4",
  "tencent/hy4-preview": "hy4",
  "openrouter:tencent/hy4-preview": "hy4",
  "XiaomiMiMo/MiMo-V2.6-Flash": "mimo-v2.6-flash",
  "xiaomi/mimo-v2.6-flash": "mimo-v2.6-flash",
  "openrouter:xiaomi/mimo-v2.6-flash": "mimo-v2.6-flash",
  "XiaomiMiMo/MiMo-V2.6-Pro": "mimo-v2.6-pro",
  "xiaomi/mimo-v2.6-pro": "mimo-v2.6-pro",
  "openrouter:xiaomi/mimo-v2.6-pro": "mimo-v2.6-pro",
  "srv_mr9dda21bf67b4c62086:deepseek/deepseek-v4.1-flash": "deepseek-v4.1-flash",
  "deepseek-ai/deepseek-v4.1-flash": "deepseek-v4.1-flash",
  "deepseek-v4.1-flash:1017": "deepseek-v4.1-flash",
  "srv_mp5miql908c8738d71be:z-ai/glm-5.3": "glm-5.3",
  "zhipu/glm-5.3": "glm-5.3",
  "srv_mp5miql908c8738d71be:community/scriptsnsenses-sys/glm-5.3-flash-free": "glm-5.3-flash",
  "openrouter/kimi-k3": "kimi-k3",
  "openrouter:deepseek/deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
  "srv_mr9dda21bf67b4c62086:z-ai/glm-5.2": "glm-5.2",
  "zhipu/glm-5.2": "glm-5.2",
  "nvidia/nemotron-3-ultra": "nemotron-3-ultra-550b-a55b",
  "srv_mp5miql908c8738d71be:deepseek/deepseek-v4-pro": "deepseek-v4-pro",
  "srv_mp5miql908c8738d71be:moonshotai/kimi-k2.6": "kimi-k2.6",
  "srv_mvi8x0lo7e51b06cf96c:ovh-free/Qwen3.5-397B-A17B": "qwen-3.5-397b-a17b",
  "srv_mp5miql908c8738d71be:google/gemma-4-26b-a4b-it": "gemma-4-26b-a4b-it",
  "srv_mp5miql908c8738d71be:google/gemma-4-31b-it": "gemma-4-31b-it",
  "openrouter:nvidia/nemotron-3-super-120b-a12b": "nemotron-3-super-120b-a12b",
  "srv_mp5miql908c8738d71be:community/vendouple/deepseek-v3.2": "deepseek-v3.2",
  "MiniMaxAI/MiniMax-M3.1-Flash": "minimax-m3.1-flash",
  "srv_mwvnwe1m0ddf970d82ac:minimaxai/minimax-m3.1-flash": "minimax-m3.1-flash",
  "minimaxai/minimax-m3.1-flash": "minimax-m3.1-flash",
  "minimax/minimax-m3.1-flash": "minimax-m3.1-flash",
  "togetherai:minimaxai/minimax-m3.1-flash": "minimax-m3.1-flash",
  "srv_mvi8x0lo7e51b06cf96c:ovh-free/Qwen3.8-27B": "qwen-3.8-27b",
  "openrouter:qwen/qwen3.8-27b": "qwen-3.8-27b",
  "reflection/Beam-1.0-501B-A23B": "beam-1.0-501b-a23b",
  "reflection/beam-1.0-501b-a23b": "beam-1.0-501b-a23b",
  "togetherai:reflection/beam-1.0-501b-a23b": "beam-1.0-501b-a23b",
  "meta-models/Muse-Spark-Open-816B-A42B": "muse-spark-open-816b-a42b",
  "togetherai:meta-models/muse-spark-open-816b-a42b": "muse-spark-open-816b-a42b",
  "qwen3.5-9b": "qwen-3.5-9b",
  "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4": "nemotron-3.5-super-vl-120b-a12b-nvfp4",
  "srv_mwvnwe1m0ddf970d82ac:meta/muse-glimmer-30b": "muse-glimmer-30b",
  "XiaomiMiMo/MiMo-V2.6-Pro-RL": "mimo-v2.6-pro-rl",
  "qwen3.8-2.4t-a95b": "qwen-3.8-2.4t-a95b",
  "srv_mp5miql908c8738d71be:qwen/qwen3.8-2.4t-a95b": "qwen-3.8-2.4t-a95b",
  "Qwen/Qwen3.8-Flash-Next-FP8": "qwen-3.8-flash-next",
  "qwen2.5-72b-instruct": "qwen-2.5-72b",
  "XiaomiMiMo/MiMo-V2.6-Flash-RL": "mimo-v2.6-flash-rl",
  "inclusionAI/Ling-3.1-flash": "ling-3.1-flash",
  "inclusionai/ling-3.1-flash": "ling-3.1-flash",
  "openrouter:inclusionai/ling-3.1-flash": "ling-3.1-flash",
  "thinkingmachines/inkling-small": "inkling-small",
  "srv_mp5miql908c8738d71be:thinkingmachines/inkling-small": "inkling-small",
  "infron:qwen/qwen3-235b-a22b": "qwen-3-235b-a22b",
  "zai-org/GLM-4.7-FP8": "glm-4.7",
  "openrouter:inclusionai/ling-3.0-flash-vl": "ling-3.0-flash-vl",
  "tencent/hy3": "hy3",
  "infron:z-ai/glm-4.5v": "glm-4.5v",
  "srv_mwvnwe1m0ddf970d82ac:meta/llama-guard-4-12b": "llama-guard-4-12b",
  "srv_mp1v9cyha31b95fa8c9a:z-ai/glm-4.6v-flash": "glm-4.6v-flash",
  "infron:nousresearch/hermes-3-llama-3.1-70b": "hermes-3-llama-3.1-70b",
  "openrouter:deepseek/deepseek-v4-flash-vision-exp": "deepseek-v4-flash-vision-exp",
  "aisingapore/Nemotron-SEA-LION-v4.8-120B-A12B": "nemotron-sea-lion-v4.8-120b-a12b",
  "aisingapore/Qwen-SEA-LION-v4.5-27B-IT": "qwen-sea-lion-v4.5-27b-it",
  "BSC-LT/ALIA-40b-instruct-2606": "alia-40b-2606",
  "meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b",
  "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
  "tencent/Hy4-flash": "hy4-flash",
  "tencent/Hy4": "hy4",
  "moonshotai/Kimi-K3.1-Pro": "kimi-k3.1-pro",
  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": "nemotron-3-super-120b-a12b",
  "thinkingmachines/Inkling-2-Mini": "inkling-2-mini",
  "RadixArk/DaisyRose-27B": "daisyrose-27b",
  "thinkingmachines/Inkling-2-Small": "inkling-2-small",
  "Qwen/Qwen3.8-Flash-Next": "qwen-3.8-flash-next",
  "LiquidAI/LFM3-1.7B": "lfm3-1.7b",
  "mistralai/Mistral-Large-4-1T-A52B": "mistral-large-4-1t-a52b",
  "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16": "nemotron-3.5-super-vl-120b-a12b",
  "LiquidAI/LFM2.5-24B-A2B": "lfm2.5-24b-a2b",
  "XiaomiMiMo/MiMo-V2.6-Flash-MOPD": "mimo-v2.6-flash-mopd",
  "Gryphe/MythoMax-L2-13b": "mythomax-l2-13b",
  "Qwen/Qwen3-VL-8B-Instruct": "qwen-3-vl-8b",
  "dphn/Dolphin-Mistral-24B-Venice-Edition": "dolphin-mistral-24b-venice-edition",
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct": "llama-4-maverick-17b-128e",
  "moondream/moondream3-preview": "moondream3",
  "tngtech/DeepSeek-TNG-R1T2-Chimera": "deepseek-tng-r1t2-chimera",
  "moondream/moondream3.1-9B-A2B": "moondream3.1-9b-a2b",
  "ByteDance-Seed/UI-TARS-1.5-7B": "ui-tars-1.5-7b",
  "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen-3-30b-a3b-2507",
  "CohereLabs/command-a-plus-05-2026-bf16": "command-a-plus26",
  "anthracite-org/magnum-v4-72b": "magnum-v4-72b",
  "Undi95/ReMM-SLERP-L2-13B": "remm-slerp-l2-13b",
  "google/gemma-2-27b-it": "gemma-2-27b-it",
  "NousResearch/Hermes-4-405B": "hermes-4-405b",
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nemotron-3-nano-30b-a3b",
  "RekaAI/reka-edge-2603": "reka-edge-2603"
}
