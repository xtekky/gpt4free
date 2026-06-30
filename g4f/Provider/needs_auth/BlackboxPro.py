from __future__ import annotations

from aiohttp import ClientSession
import re
import json
import random
import string
from pathlib import Path
from typing import Optional
from datetime import datetime

from ...typing import AsyncResult, Messages, MediaListType
from ...requests.raise_for_status import raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..openai.har_file import get_har_files
from ...image import to_data_uri
from ...cookies import get_cookies_dir
from ..helper import format_media_prompt, render_messages
from ...providers.response import JsonConversation, ImageResponse
from ...tools.media import merge_media
from ...cookies import get_cookies
from ...errors import RateLimitError, NoValidHarFileError
from ... import debug

class Conversation(JsonConversation):
    validated_value: str = None
    chat_id: str = None
    message_history: Messages = []

    def __init__(self, model: str):
        self.model = model

models = [{
    "value": "openai/gpt-5",
    "label": "GPT-5",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-5 Chat is designed for advanced, natural, multimodal, and context-aware conversations for enterprise applications."
}, {
    "value": "x-ai/grok-4",
    "label": "Grok 4",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": "Grok 4 is xAI's latest reasoning model with a 256k context window. It supports parallel tool calling, structured outputs, and both image and text inputs. Note that reasoning is not exposed, reasoning cannot be disabled, and the reasoning effort cannot be specified."
}, {
    "value": "anthropic/claude-sonnet-4.5",
    "label": "Claude Sonnet 4.5",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude Sonnet 4.5 is Anthropic’s most advanced Sonnet model to date, optimized for real-world agents and coding workflows. It delivers state-of-the-art performance on coding benchmarks such as SWE-bench Verified, with improvements across system design, code security, and specification adherence. The model is designed for extended autonomous operation, maintaining task continuity across sessions and providing fact-based progress tracking."
}, {
    "value": "deepseek/deepseek-chat-v3.1",
    "label": "DeepSeek V3.1",
    "cover": "/images/icons/DeepSeek.png",
    "description": "DeepSeek V3.1 is here: A major upgrade over previous DeepSeek releases, delivering stronger reasoning, faster inference, and improved alignment.\n\nFully open-source model with [technical report](https://api-docs.deepseek.com/news/news250821).\n\nMIT licensed: Distill & commercialize freely!"
}, {
    "value": "google/gemini-2.5-pro-preview-03-25",
    "label": "Gemini 2.5 Pro",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks. It employs “thinking” capabilities, enabling it to reason through responses with enhanced accuracy and nuanced context handling. Gemini 2.5 Pro achieves top-tier performance on multiple benchmarks, including first-place positioning on the LMArena leaderboard, reflecting superior human-preference alignment and complex problem-solving abilities."
}, {
    "value": "deepseek/deepseek-r1-0528",
    "label": "DeepSeek R1 0528",
    "cover": "/images/icons/DeepSeek.png",
    "description": "May 28th update to the original DeepSeek R1 Performance on par with OpenAI o1, but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass. Fully open-source model."
}, {
    "value": "x-ai/grok-code-fast-1",
    "label": "Grok Code Fast 1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": "Grok Code Fast 1 is a speedy and economical reasoning model that excels at agentic coding. With reasoning traces visible in the response, developers can steer Grok Code for high-quality work flows."
}, {
    "value": "anthropic/claude-sonnet-4",
    "label": "Claude Sonnet 4",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude Sonnet 4 significantly enhances the capabilities of its predecessor, Sonnet 3.7, excelling in both coding and reasoning tasks with improved precision and controllability. Achieving state-of-the-art performance on SWE-bench (72.7%), Sonnet 4 balances capability and computational efficiency, making it suitable for a broad range of applications from routine coding tasks to complex software development projects. Key enhancements include improved autonomous codebase navigation, reduced error rates in agent-driven workflows, and increased reliability in following intricate instructions. Sonnet 4 is optimized for practical everyday use, providing advanced reasoning capabilities while maintaining efficiency and responsiveness in diverse internal and external scenarios."
}, {
    "value": "anthropic/claude-opus-4.1",
    "label": "Claude Opus 4.1",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude Opus 4.1 is an updated version of Anthropic’s flagship model, offering improved performance in coding, reasoning, and agentic tasks. It achieves 74.5% on SWE-bench Verified and shows notable gains in multi-file code refactoring, debugging precision, and detail-oriented reasoning. The model supports extended thinking up to 64K tokens and is optimized for tasks involving research, data analysis, and tool-assisted reasoning."
}, {
    "value": "openai/gpt-oss-120b",
    "label": "OpenAI: GPT OSS 120B",
    "cover": "/images/icons/OpenAI.svg",
    "description": "gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language model from OpenAI designed for high-reasoning, agentic, and general-purpose production use cases. It activates 5.1B parameters per forward pass and is optimized to run on a single H100 GPU with native MXFP4 quantization. The model supports configurable reasoning depth, full chain-of-thought access, and native tool use, including function calling, browsing, and structured output generation."
}, {
    "value": "x-ai/grok-3-beta",
    "label": "Grok 3",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": "Grok 3 is the latest model from xAI. It's their flagship model that excels at enterprise use cases like data extraction, coding, and text summarization. Possesses deep domain knowledge in finance, healthcare, law, and science.\n\nExcels in structured tasks and benchmarks like GPQA, LCB, and MMLU-Pro where it outperforms Grok 3 Mini even on high thinking. \n\nNote: That there are two xAI endpoints for this model. By default when using this model we will always route you to the base endpoint. If you want the fast endpoint you can add `provider: { sort: throughput}`, to sort by throughput instead. \n"
}, {
    "value": "anthropic/claude-3.7-sonnet",
    "label": "Claude 3.7 Sonnet",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. It introduces a hybrid reasoning approach, allowing users to choose between rapid responses and extended, step-by-step processing for complex tasks. The model demonstrates notable improvements in coding, particularly in front-end development and full-stack updates, and excels in agentic workflows, where it can autonomously navigate multi-step processes. \n\nClaude 3.7 Sonnet maintains performance parity with its predecessor in standard mode while offering an extended reasoning mode for enhanced accuracy in math, coding, and instruction-following tasks.\n\nRead more at the [blog post here](https://www.anthropic.com/news/claude-3-7-sonnet)"
}, {
    "value": "deepseek/deepseek-r1",
    "label": "DeepSeek R1",
    "cover": "/images/icons/DeepSeek.png",
    "description": "DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.\n\nFully open-source model & [technical report](https://api-docs.deepseek.com/news/news250120).\n\nMIT licensed: Distill & commercialize freely!"
}, {
    "value": "meta-llama/llama-4-maverick:free",
    "label": "Llama 4 Maverick",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://ai.meta.com/&size=256",
    "description": "Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forward pass (400B total). It supports multilingual text and image input, and produces multilingual text and code output across 12 supported languages. Optimized for vision-language tasks, Maverick is instruction-tuned for assistant-like behavior, image reasoning, and general-purpose multimodal interaction.\n\nMaverick features early fusion for native multimodality and a 1 million token context window. It was trained on a curated mixture of public, licensed, and Meta-platform data, covering ~22 trillion tokens, with a knowledge cutoff in August 2024. Released on April 5, 2025 under the Llama 4 Community License, Maverick is suited for research and commercial applications requiring advanced multimodal understanding and high model throughput."
}, {
    "value": "mistralai/mistral-large",
    "label": "Mistral Large",
    "cover": "/images/icons/Mistral.png",
    "description": "This is Mistral AI's flagship model, Mistral Large 2 (version `mistral-large-2407`). It's a proprietary weights-available model and excels at reasoning, code, JSON, chat, and more. Read the launch announcement [here](https://mistral.ai/news/mistral-large-2407/).\n\nIt supports dozens of languages including French, German, Spanish, Italian, Portuguese, Arabic, Hindi, Russian, Chinese, Japanese, and Korean, along with 80+ coding languages including Python, Java, C, C++, JavaScript, and Bash. Its long context window allows precise information recall from large documents."
}, {
    "value": "qwen/qwen-2.5-coder-32b-instruct",
    "label": "Qwen2.5 Coder 32B Instruct",
    "cover": "/images/icons/Qwen.png",
    "description": "Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). Qwen2.5-Coder brings the following improvements upon CodeQwen1.5:\n\n- Significantly improvements in **code generation**, **code reasoning** and **code fixing**. \n- A more comprehensive foundation for real-world applications such as **Code Agents**. Not only enhancing coding capabilities but also maintaining its strengths in mathematics and general competencies.\n\nTo read more about its evaluation results, check out [Qwen 2.5 Coder's blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/)."
}, {
    "value": "openai/gpt-4.1-mini",
    "label": "GPT-4.1 Mini",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-4.1 Mini is a mid-sized model delivering performance competitive with GPT-4o at substantially lower latency and cost. It retains a 1 million token context window and scores 45.1% on hard instruction evals, 35.8% on MultiChallenge, and 84.1% on IFEval. Mini also shows strong coding ability (e.g., 31.6% on Aider’s polyglot diff benchmark) and vision understanding, making it suitable for interactive applications with tight performance constraints."
}, {
    "value": "openai/gpt-4.1-nano",
    "label": "GPT-4.1 Nano",
    "cover": "/images/icons/OpenAI.svg",
    "description": "For tasks that demand low latency, GPT‑4.1 nano is the fastest and cheapest model in the GPT-4.1 series. It delivers exceptional performance at a small size with its 1 million token context window, and scores 80.1% on MMLU, 50.3% on GPQA, and 9.8% on Aider polyglot coding – even higher than GPT‑4o mini. It’s ideal for tasks like classification or autocompletion."
}, {
    "value": "anthropic/claude-3.7-sonnet:thinking",
    "label": "Claude 3.7 Sonnet (thinking)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. It introduces a hybrid reasoning approach, allowing users to choose between rapid responses and extended, step-by-step processing for complex tasks. The model demonstrates notable improvements in coding, particularly in front-end development and full-stack updates, and excels in agentic workflows, where it can autonomously navigate multi-step processes. \n\nClaude 3.7 Sonnet maintains performance parity with its predecessor in standard mode while offering an extended reasoning mode for enhanced accuracy in math, coding, and instruction-following tasks.\n\nRead more at the [blog post here](https://www.anthropic.com/news/claude-3-7-sonnet)"
}, {
    "value": "anthropic/claude-3.7-sonnet:beta",
    "label": "Claude 3.7 Sonnet (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. It introduces a hybrid reasoning approach, allowing users to choose between rapid responses and extended, step-by-step processing for complex tasks. The model demonstrates notable improvements in coding, particularly in front-end development and full-stack updates, and excels in agentic workflows, where it can autonomously navigate multi-step processes. \n\nClaude 3.7 Sonnet maintains performance parity with its predecessor in standard mode while offering an extended reasoning mode for enhanced accuracy in math, coding, and instruction-following tasks.\n\nRead more at the [blog post here](https://www.anthropic.com/news/claude-3-7-sonnet)"
}, {
    "value": "anthropic/claude-3.5-haiku:beta",
    "label": "Claude 3.5 Haiku (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3.5 Haiku features offers enhanced capabilities in speed, coding accuracy, and tool use. Engineered to excel in real-time applications, it delivers quick response times that are essential for dynamic tasks such as chat interactions and immediate coding suggestions.\n\nThis makes it highly suitable for environments that demand both speed and precision, such as software development, customer service bots, and data management systems.\n\nThis model is currently pointing to [Claude 3.5 Haiku (2024-10-22)](/anthropic/claude-3-5-haiku-20241022)."
}, {
    "value": "anthropic/claude-3.5-haiku",
    "label": "Claude 3.5 Haiku",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3.5 Haiku features offers enhanced capabilities in speed, coding accuracy, and tool use. Engineered to excel in real-time applications, it delivers quick response times that are essential for dynamic tasks such as chat interactions and immediate coding suggestions.\n\nThis makes it highly suitable for environments that demand both speed and precision, such as software development, customer service bots, and data management systems.\n\nThis model is currently pointing to [Claude 3.5 Haiku (2024-10-22)](/anthropic/claude-3-5-haiku-20241022)."
}, {
    "value": "anthropic/claude-3.5-haiku-20241022:beta",
    "label": "Claude 3.5 Haiku (2024-10-22) (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3.5 Haiku features enhancements across all skill sets including coding, tool use, and reasoning. As the fastest model in the Anthropic lineup, it offers rapid response times suitable for applications that require high interactivity and low latency, such as user-facing chatbots and on-the-fly code completions. It also excels in specialized tasks like data extraction and real-time content moderation, making it a versatile tool for a broad range of industries.\n\nIt does not support image inputs.\n\nSee the launch announcement and benchmark results [here](https://www.anthropic.com/news/3-5-models-and-computer-use)"
}, {
    "value": "anthropic/claude-3.5-haiku-20241022",
    "label": "Claude 3.5 Haiku (2024-10-22)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3.5 Haiku features enhancements across all skill sets including coding, tool use, and reasoning. As the fastest model in the Anthropic lineup, it offers rapid response times suitable for applications that require high interactivity and low latency, such as user-facing chatbots and on-the-fly code completions. It also excels in specialized tasks like data extraction and real-time content moderation, making it a versatile tool for a broad range of industries.\n\nIt does not support image inputs.\n\nSee the launch announcement and benchmark results [here](https://www.anthropic.com/news/3-5-models-and-computer-use)"
}, {
    "value": "anthropic/claude-3.5-sonnet:beta",
    "label": "Claude 3.5 Sonnet (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:\n\n- Coding: Scores ~49% on SWE-Bench Verified, higher than the last best score, and without any fancy prompt scaffolding\n- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights\n- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone\n- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)\n\n#multimodal"
}, {
    "value": "anthropic/claude-3.5-sonnet",
    "label": "Claude 3.5 Sonnet",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:\n\n- Coding: Scores ~49% on SWE-Bench Verified, higher than the last best score, and without any fancy prompt scaffolding\n- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights\n- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone\n- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)\n\n#multimodal"
}, {
    "value": "x-ai/grok-3-mini-beta",
    "label": "Grok 3 Mini Beta",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": 'Grok 3 Mini is a lightweight, smaller thinking model. Unlike traditional models that generate answers immediately, Grok 3 Mini thinks before responding. It’s ideal for reasoning-heavy tasks that don’t demand extensive domain knowledge, and shines in math-specific and quantitative use cases, such as solving challenging puzzles or math problems.\n\nTransparent "thinking" traces accessible. Defaults to low reasoning, can boost with setting `reasoning: { effort: "high" }`\n\nNote: That there are two xAI endpoints for this model. By default when using this model we will always route you to the base endpoint. If you want the fast endpoint you can add `provider: { sort: throughput}`, to sort by throughput instead. \n'
}, {
    "value": "google/gemini-2.0-flash-lite-001",
    "label": "Gemini 2.0 Flash Lite",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemini 2.0 Flash Lite offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1.5](/google/gemini-flash-1.5), while maintaining quality on par with larger models like [Gemini Pro 1.5](/google/gemini-pro-1.5), all at extremely economical token prices."
}, {
    "value": "meta-llama/llama-4-maverick",
    "label": "Llama 4 Maverick",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://ai.meta.com/&size=256",
    "description": "Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built on a mixture-of-experts (MoE) architecture with 128 experts and 17 billion active parameters per forward pass (400B total). It supports multilingual text and image input, and produces multilingual text and code output across 12 supported languages. Optimized for vision-language tasks, Maverick is instruction-tuned for assistant-like behavior, image reasoning, and general-purpose multimodal interaction.\n\nMaverick features early fusion for native multimodality and a 1 million token context window. It was trained on a curated mixture of public, licensed, and Meta-platform data, covering ~22 trillion tokens, with a knowledge cutoff in August 2024. Released on April 5, 2025 under the Llama 4 Community License, Maverick is suited for research and commercial applications requiring advanced multimodal understanding and high model throughput."
}, {
    "value": "meta-llama/llama-4-scout:free",
    "label": "Llama 4 Scout",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://ai.meta.com/&size=256",
    "description": "Llama 4 Scout 17B Instruct (16E) is a mixture-of-experts (MoE) language model developed by Meta, activating 17 billion parameters out of a total of 109B. It supports native multimodal input (text and image) and multilingual output (text and code) across 12 supported languages. Designed for assistant-style interaction and visual reasoning, Scout uses 16 experts per forward pass and features a context length of 10 million tokens, with a training corpus of ~40 trillion tokens.\n\nBuilt for high efficiency and local or commercial deployment, Llama 4 Scout incorporates early fusion for seamless modality integration. It is instruction-tuned for use in multilingual chat, captioning, and image understanding tasks. Released under the Llama 4 Community License, it was last trained on data up to August 2024 and launched publicly on April 5, 2025."
}, {
    "value": "meta-llama/llama-4-scout",
    "label": "Llama 4 Scout",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://ai.meta.com/&size=256",
    "description": "Llama 4 Scout 17B Instruct (16E) is a mixture-of-experts (MoE) language model developed by Meta, activating 17 billion parameters out of a total of 109B. It supports native multimodal input (text and image) and multilingual output (text and code) across 12 supported languages. Designed for assistant-style interaction and visual reasoning, Scout uses 16 experts per forward pass and features a context length of 10 million tokens, with a training corpus of ~40 trillion tokens.\n\nBuilt for high efficiency and local or commercial deployment, Llama 4 Scout incorporates early fusion for seamless modality integration. It is instruction-tuned for use in multilingual chat, captioning, and image understanding tasks. Released under the Llama 4 Community License, it was last trained on data up to August 2024 and launched publicly on April 5, 2025."
}, {
    "value": "nvidia/llama-3.1-nemotron-70b-instruct:free",
    "label": "Llama 3.1 Nemotron 70B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://targon.com/&size=256",
    "description": "NVIDIA's Llama 3.1 Nemotron 70B is a language model designed for generating precise and useful responses. Leveraging [Llama 3.1 70B](/models/meta-llama/llama-3.1-70b-instruct) architecture and Reinforcement Learning from Human Feedback (RLHF), it excels in automatic alignment benchmarks. This model is tailored for applications requiring high accuracy in helpfulness and response generation, suitable for diverse user queries across multiple domains.\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/)."
}, {
    "value": "nvidia/llama-3.1-nemotron-70b-instruct",
    "label": "Llama 3.1 Nemotron 70B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://lambdalabs.com/&size=256",
    "description": "NVIDIA's Llama 3.1 Nemotron 70B is a language model designed for generating precise and useful responses. Leveraging [Llama 3.1 70B](/models/meta-llama/llama-3.1-70b-instruct) architecture and Reinforcement Learning from Human Feedback (RLHF), it excels in automatic alignment benchmarks. This model is tailored for applications requiring high accuracy in helpfulness and response generation, suitable for diverse user queries across multiple domains.\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/)."
}, {
    "value": "x-ai/grok-2-vision-1212",
    "label": "Grok 2 Vision 1212",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": "Grok 2 Vision 1212 advances image-based AI with stronger visual comprehension, refined instruction-following, and multilingual support. From object recognition to style analysis, it empowers developers to build more intuitive, visually aware applications. Its enhanced steerability and reasoning establish a robust foundation for next-generation image solutions.\n\nTo read more about this model, check out [xAI's announcement](https://x.ai/blog/grok-1212)."
}, {
    "value": "x-ai/grok-2-1212",
    "label": "Grok 2 1212",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": "Grok 2 1212 introduces significant enhancements to accuracy, instruction adherence, and multilingual support, making it a powerful and flexible choice for developers seeking a highly steerable, intelligent model."
}, {
    "value": "eleutherai/llemma_7b",
    "label": "Llemma 7b",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": "Llemma 7B is a language model for mathematics. It was initialized with Code Llama 7B weights, and trained on the Proof-Pile-2 for 200B tokens. Llemma models are particularly strong at chain-of-thought mathematical reasoning and using computational tools for mathematics, such as Python and formal theorem provers."
}, {
    "value": "alfredpros/codellama-7b-instruct-solidity",
    "label": "CodeLLaMa 7B Instruct Solidity",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": "A finetuned 7 billion parameters Code LLaMA - Instruct model to generate Solidity smart contract using 4-bit QLoRA finetuning provided by PEFT library."
}, {
    "value": "arliai/qwq-32b-arliai-rpr-v1:free",
    "label": "QwQ 32B RpR v1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "QwQ-32B-ArliAI-RpR-v1 is a 32B parameter model fine-tuned from Qwen/QwQ-32B using a curated creative writing and roleplay dataset originally developed for the RPMax series. It is designed to maintain coherence and reasoning across long multi-turn conversations by introducing explicit reasoning steps per dialogue turn, generated and refined using the base model itself.\n\nThe model was trained using RS-QLORA+ on 8K sequence lengths and supports up to 128K context windows (with practical performance around 32K). It is optimized for creative roleplay and dialogue generation, with an emphasis on minimizing cross-context repetition while preserving stylistic diversity."
}, {
    "value": "agentica-org/deepcoder-14b-preview:free",
    "label": "Deepcoder 14B Preview",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "DeepCoder-14B-Preview is a 14B parameter code generation model fine-tuned from DeepSeek-R1-Distill-Qwen-14B using reinforcement learning with GRPO+ and iterative context lengthening. It is optimized for long-context program synthesis and achieves strong performance across coding benchmarks, including 60.6% on LiveCodeBench v5, competitive with models like o3-Mini"
}, {
    "value": "moonshotai/kimi-vl-a3b-thinking:free",
    "label": "Kimi VL A3B Thinking",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Kimi-VL is a lightweight Mixture-of-Experts vision-language model that activates only 2.8B parameters per step while delivering strong performance on multimodal reasoning and long-context tasks. The Kimi-VL-A3B-Thinking variant, fine-tuned with chain-of-thought and reinforcement learning, excels in math and visual reasoning benchmarks like MathVision, MMMU, and MathVista, rivaling much larger models such as Qwen2.5-VL-7B and Gemma-3-12B. It supports 128K context and high-resolution input via its MoonViT encoder."
}, {
    "value": "openrouter/optimus-alpha",
    "label": "Optimus Alpha",
    "cover": "/images/icons/OpenRouter.svg",
    "description": "This is a cloaked model provided to the community to gather feedback. It's geared toward real world use cases, including programming.\n\n**Note:** All prompts and completions for this model are logged by the provider and may be used to improve the model."
}, {
    "value": "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
    "label": "Llama 3.1 Nemotron Nano 8B v1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Llama-3.1-Nemotron-Nano-8B-v1 is a compact large language model (LLM) derived from Meta's Llama-3.1-8B-Instruct, specifically optimized for reasoning tasks, conversational interactions, retrieval-augmented generation (RAG), and tool-calling applications. It balances accuracy and efficiency, fitting comfortably onto a single consumer-grade RTX GPU for local deployment. The model supports extended context lengths of up to 128K tokens.\n\nNote: you must include `detailed thinking on` in the system prompt to enable reasoning. Please see [Usage Recommendations](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1#quick-start-and-usage-recommendations) for more."
}, {
    "value": "nvidia/llama-3.3-nemotron-super-49b-v1:free",
    "label": "Llama 3.3 Nemotron Super 49B v1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Llama-3.3-Nemotron-Super-49B-v1 is a large language model (LLM) optimized for advanced reasoning, conversational interactions, retrieval-augmented generation (RAG), and tool-calling tasks. Derived from Meta's Llama-3.3-70B-Instruct, it employs a Neural Architecture Search (NAS) approach, significantly enhancing efficiency and reducing memory requirements. This allows the model to support a context length of up to 128K tokens and fit efficiently on single high-performance GPUs, such as NVIDIA H200.\n\nNote: you must include `detailed thinking on` in the system prompt to enable reasoning. Please see [Usage Recommendations](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1#quick-start-and-usage-recommendations) for more."
}, {
    "value": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "label": "Llama 3.1 Nemotron Ultra 253B v1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Llama-3.1-Nemotron-Ultra-253B-v1 is a large language model (LLM) optimized for advanced reasoning, human-interactive chat, retrieval-augmented generation (RAG), and tool-calling tasks. Derived from Meta’s Llama-3.1-405B-Instruct, it has been significantly customized using Neural Architecture Search (NAS), resulting in enhanced efficiency, reduced memory usage, and improved inference latency. The model supports a context length of up to 128K tokens and can operate efficiently on an 8x NVIDIA H100 node.\n\nNote: you must include `detailed thinking on` in the system prompt to enable reasoning. Please see [Usage Recommendations](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1#quick-start-and-usage-recommendations) for more."
}, {
    "value": "tokyotech-llm/llama-3.1-swallow-8b-instruct-v0.3",
    "label": "Llama 3.1 Swallow 8B Instruct V0.3",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://sambanova.ai/&size=256",
    "description": "Llama 3.1 Swallow 8B is a large language model that was built by continual pre-training on the Meta Llama 3.1 8B. Llama 3.1 Swallow enhanced the Japanese language capabilities of the original Llama 3.1 while retaining the English language capabilities. \nSwallow used approximately 200 billion tokens that were sampled from a large Japanese web corpus (Swallow Corpus Version 2), Japanese and English Wikipedia articles, and mathematical and coding contents, etc (see the Training Datasets section of the base model) for continual pre-training. The instruction-tuned models (Instruct) were built by supervised fine-tuning (SFT) on the synthetic data specially built for Japanese.\n"
}, {
    "value": "openrouter/quasar-alpha",
    "label": "Quasar Alpha",
    "cover": "/images/icons/OpenRouter.svg",
    "description": "This is a cloaked model provided to the community to gather feedback. It’s a powerful, all-purpose model supporting long-context tasks, including code generation.\n\n**Note:** All prompts and completions for this model are logged by the provider  and may be used to improve the model."
}, {
    "value": "all-hands/openhands-lm-32b-v0.1",
    "label": "OpenHands LM 32B V0.1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": "OpenHands LM v0.1 is a 32B open-source coding model fine-tuned from Qwen2.5-Coder-32B-Instruct using reinforcement learning techniques outlined in SWE-Gym. It is optimized for autonomous software development agents and achieves strong performance on SWE-Bench Verified, with a 37.2% resolve rate. The model supports a 128K token context window, making it well-suited for long-horizon code reasoning and large codebase tasks.\n\nOpenHands LM is designed for local deployment and runs on consumer-grade GPUs such as a single 3090. It enables fully offline agent workflows without dependency on proprietary APIs. This release is intended as a research preview, and future updates aim to improve generalizability, reduce repetition, and offer smaller variants."
}, {
    "value": "mistral/ministral-8b",
    "label": "Ministral 8B",
    "cover": "/images/icons/Mistral.png",
    "description": "Ministral 8B is a state-of-the-art language model optimized for on-device and edge computing. Designed for efficiency in knowledge-intensive tasks, commonsense reasoning, and function-calling, it features a specialized interleaved sliding-window attention mechanism, enabling faster and more memory-efficient inference. Ministral 8B excels in local, low-latency applications such as offline translation, smart assistants, autonomous robotics, and local analytics.\n\nThe model supports up to 128k context length and can function as a performant intermediary in multi-step agentic workflows, efficiently handling tasks like input parsing, API calls, and task routing. It consistently outperforms comparable models like Mistral 7B across benchmarks, making it particularly suitable for compute-efficient, privacy-focused scenarios."
}, {
    "value": "deepseek/deepseek-v3-base:free",
    "label": "DeepSeek V3 Base",
    "cover": "/images/icons/DeepSeek.png",
    "description": "Note that this is a base model mostly meant for testing, you need to provide detailed prompts for the model to return useful responses. \n\nDeepSeek-V3 Base is a 671B parameter open Mixture-of-Experts (MoE) language model with 37B active parameters per forward pass and a context length of 128K tokens. Trained on 14.8T tokens using FP8 mixed precision, it achieves high training efficiency and stability, with strong performance across language, reasoning, math, and coding tasks. \n\nDeepSeek-V3 Base is the pre-trained model behind [DeepSeek V3](/deepseek/deepseek-chat-v3)"
}, {
    "value": "scb10x/llama3.1-typhoon2-8b-instruct",
    "label": "Typhoon2 8B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "Llama3.1-Typhoon2-8B-Instruct is a Thai-English instruction-tuned model with 8 billion parameters, built on Llama 3.1. It significantly improves over its base model in Thai reasoning, instruction-following, and function-calling tasks, while maintaining competitive English performance. The model is optimized for bilingual interaction and performs well on Thai-English code-switching, MT-Bench, IFEval, and tool-use benchmarks.\n\nDespite its smaller size, it demonstrates strong generalization across math, coding, and multilingual benchmarks, outperforming comparable 8B models across most Thai-specific tasks. Full benchmark results and methodology are available in the [technical report.](https://arxiv.org/abs/2412.13702)"
}, {
    "value": "scb10x/llama3.1-typhoon2-70b-instruct",
    "label": "Typhoon2 70B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "Llama3.1-Typhoon2-70B-Instruct is a Thai-English instruction-tuned language model with 70 billion parameters, built on Llama 3.1. It demonstrates strong performance across general instruction-following, math, coding, and tool-use tasks, with state-of-the-art results in Thai-specific benchmarks such as IFEval, MT-Bench, and Thai-English code-switching.\n\nThe model excels in bilingual reasoning and function-calling scenarios, offering high accuracy across diverse domains. Comparative evaluations show consistent improvements over prior Thai LLMs and other Llama-based baselines. Full results and methodology are available in the [technical report.](https://arxiv.org/abs/2412.13702)"
}, {
    "value": "allenai/molmo-7b-d:free",
    "label": "Molmo 7B D",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Molmo is a family of open vision-language models developed by the Allen Institute for AI. Molmo models are trained on PixMo, a dataset of 1 million, highly-curated image-text pairs. It has state-of-the-art performance among multimodal models with a similar size while being fully open-source. You can find all models in the Molmo family [here](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19). Learn more about the Molmo family [in the announcement blog post](https://molmo.allenai.org/blog) or the [paper](https://huggingface.co/papers/2409.17146).\n\nMolmo 7B-D is based on [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) and uses [OpenAI CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336) as vision backbone. It performs comfortably between GPT-4V and GPT-4o on both academic benchmarks and human evaluation.\n\nThis checkpoint is a preview of the Molmo release. All artifacts used in creating Molmo (PixMo dataset, training code, evaluations, intermediate checkpoints) will be made available at a later date, furthering our commitment to open-source AI development and reproducibility."
}, {
    "value": "bytedance-research/ui-tars-72b:free",
    "label": "UI-TARS 72B ",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "UI-TARS 72B is an open-source multimodal AI model designed specifically for automating browser and desktop tasks through visual interaction and control. The model is built with a specialized vision architecture enabling accurate interpretation and manipulation of on-screen visual data. It supports automation tasks within web browsers as well as desktop applications, including Microsoft Office and VS Code.\n\nCore capabilities include intelligent screen detection, predictive action modeling, and efficient handling of repetitive interactions. UI-TARS employs supervised fine-tuning (SFT) tailored explicitly for computer control scenarios. It can be deployed locally or accessed via Hugging Face for demonstration purposes. Intended use cases encompass workflow automation, task scripting, and interactive desktop control applications."
}, {
    "value": "qwen/qwen2.5-vl-3b-instruct:free",
    "label": "Qwen2.5 VL 3B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Qwen2.5 VL 3B is a multimodal LLM from the Qwen Team with the following key enhancements:\n\n- SoTA understanding of images of various resolution & ratio: Qwen2.5-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.\n\n- Agent that can operate your mobiles, robots, etc.: with the abilities of complex reasoning and decision making, Qwen2.5-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.\n\n- Multilingual Support: to serve global users, besides English and Chinese, Qwen2.5-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen2-vl/) and [GitHub repo](https://github.com/QwenLM/Qwen2-VL).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen2.5-vl-32b-instruct:free",
    "label": "Qwen2.5 VL 32B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.alibabacloud.com/&size=256",
    "description": "Qwen2.5-VL-32B is a multimodal vision-language model fine-tuned through reinforcement learning for enhanced mathematical reasoning, structured outputs, and visual problem-solving capabilities. It excels at visual analysis tasks, including object recognition, textual interpretation within images, and precise event localization in extended videos. Qwen2.5-VL-32B demonstrates state-of-the-art performance across multimodal benchmarks such as MMMU, MathVista, and VideoMME, while maintaining strong reasoning and clarity in text-based tasks like MMLU, mathematical problem-solving, and code generation."
}, {
    "value": "qwen/qwen2.5-vl-32b-instruct",
    "label": "Qwen2.5 VL 32B Instruct",
    "cover": "/images/icons/Fireworks.png",
    "description": "Qwen2.5-VL-32B is a multimodal vision-language model fine-tuned through reinforcement learning for enhanced mathematical reasoning, structured outputs, and visual problem-solving capabilities. It excels at visual analysis tasks, including object recognition, textual interpretation within images, and precise event localization in extended videos. Qwen2.5-VL-32B demonstrates state-of-the-art performance across multimodal benchmarks such as MMMU, MathVista, and VideoMME, while maintaining strong reasoning and clarity in text-based tasks like MMLU, mathematical problem-solving, and code generation."
}, {
    "value": "deepseek/deepseek-chat-v3-0324:free",
    "label": "DeepSeek V3 0324",
    "cover": "/images/icons/DeepSeek.png",
    "description": "DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.\n\nIt succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) model and performs really well on a variety of tasks."
}, {
    "value": "deepseek/deepseek-chat-v3-0324",
    "label": "DeepSeek V3 0324",
    "cover": "/images/icons/DeepSeek.png",
    "description": "DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.\n\nIt succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) model and performs really well on a variety of tasks."
}, {
    "value": "featherless/qwerky-72b:free",
    "label": "Qwerky 72B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": "Qwerky-72B is a linear-attention RWKV variant of the Qwen 2.5 72B model, optimized to significantly reduce computational cost at scale. Leveraging linear attention, it achieves substantial inference speedups (>1000x) while retaining competitive accuracy on common benchmarks like ARC, HellaSwag, Lambada, and MMLU. It inherits knowledge and language support from Qwen 2.5, supporting approximately 30 languages, making it suitable for efficient inference in large-context applications."
}, {
    "value": "openai/o1-pro",
    "label": "o1-pro",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The o1 series of models are trained with reinforcement learning to think before they answer and perform complex reasoning. The o1-pro model uses more compute to think harder and provide consistently better answers."
}, {
    "value": "mistralai/mistral-small-3.1-24b-instruct:free",
    "label": "Mistral Small 3.1 24B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. It provides state-of-the-art performance in text-based reasoning and vision tasks, including image analysis, programming, mathematical reasoning, and multilingual support across dozens of languages. Equipped with an extensive 128k token context window and optimized for efficient local inference, it supports use cases such as conversational agents, function calling, long-document comprehension, and privacy-sensitive deployments."
}, {
    "value": "mistralai/mistral-small-3.1-24b-instruct",
    "label": "Mistral Small 3.1 24B",
    "cover": "/images/icons/Mistral.png",
    "description": "Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billion parameters with advanced multimodal capabilities. It provides state-of-the-art performance in text-based reasoning and vision tasks, including image analysis, programming, mathematical reasoning, and multilingual support across dozens of languages. Equipped with an extensive 128k token context window and optimized for efficient local inference, it supports use cases such as conversational agents, function calling, long-document comprehension, and privacy-sensitive deployments."
}, {
    "value": "open-r1/olympiccoder-7b:free",
    "label": "OlympicCoder 7B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "OlympicCoder-7B is an open-source language model fine-tuned on the CodeForces-CoTs dataset, consisting of nearly 100,000 high-quality chain-of-thought examples from competitive programming contexts. Optimized specifically for olympiad-level coding problems, this model demonstrates strong chain-of-thought reasoning and competitive code generation capabilities, achieving performance competitive with frontier closed-source models on tasks from the IOI 2024 and similar coding contests."
}, {
    "value": "open-r1/olympiccoder-32b:free",
    "label": "OlympicCoder 32B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "OlympicCoder-32B is a high-performing open-source model fine-tuned using the CodeForces-CoTs dataset, containing approximately 100,000 chain-of-thought programming samples. It excels at complex competitive programming benchmarks, such as IOI 2024 and Codeforces-style challenges, frequently surpassing state-of-the-art closed-source models. OlympicCoder-32B provides advanced reasoning, coherent multi-step problem-solving, and robust code generation capabilities, demonstrating significant potential for olympiad-level competitive programming applications."
}, {
    "value": "steelskull/l3.3-electra-r1-70b",
    "label": "L3.3 Electra R1 70B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.parasail.io/&size=256",
    "description": "L3.3-Electra-R1-70 is the newest release of the Unnamed series. Built on a DeepSeek R1 Distill base, Electra-R1 integrates various models together to provide an intelligent and coherent model capable of providing deep character insights. Through proper prompting, the model demonstrates advanced reasoning capabilities and unprompted exploration of character inner thoughts and motivations. Read more about the model and [prompting here](https://huggingface.co/Steelskull/L3.3-Electra-R1-70b)"
}, {
    "value": "allenai/olmo-2-0325-32b-instruct",
    "label": "Olmo 2 32B Instruct",
    "cover": "",
    "description": "OLMo-2 32B Instruct is a supervised instruction-finetuned variant of the OLMo-2 32B March 2025 base model. It excels in complex reasoning and instruction-following tasks across diverse benchmarks such as GSM8K, MATH, IFEval, and general NLP evaluation. Developed by AI2, OLMo-2 32B is part of an open, research-oriented initiative, trained primarily on English-language datasets to advance the understanding and development of open-source language models."
}, {
    "value": "google/gemma-3-1b-it:free",
    "label": "Gemma 3 1B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Gemma 3 1B is the smallest of the new Gemma 3 family. It handles context windows up to 32k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Note: Gemma 3 1B is not multimodal. For the smallest multimodal Gemma 3 model, please see [Gemma 3 4B](google/gemma-3-4b-it)"
}, {
    "value": "google/gemma-3-4b-it:free",
    "label": "Gemma 3 4B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling."
}, {
    "value": "google/gemma-3-4b-it",
    "label": "Gemma 3 4B",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling."
}, {
    "value": "ai21/jamba-1.6-large",
    "label": "Jamba 1.6 Large",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://ai21.com/&size=256",
    "description": "AI21 Jamba Large 1.6 is a high-performance hybrid foundation model combining State Space Models (Mamba) with Transformer attention mechanisms. Developed by AI21, it excels in extremely long-context handling (256K tokens), demonstrates superior inference efficiency (up to 2.5x faster than comparable models), and supports structured JSON output and tool-use capabilities. It has 94 billion active parameters (398 billion total), optimized quantization support (ExpertsInt8), and multilingual proficiency in languages such as English, Spanish, French, Portuguese, Italian, Dutch, German, Arabic, and Hebrew.\n\nUsage of this model is subject to the [Jamba Open Model License](https://www.ai21.com/licenses/jamba-open-model-license)."
}, {
    "value": "ai21/jamba-1.6-mini",
    "label": "Jamba Mini 1.6",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://ai21.com/&size=256",
    "description": "AI21 Jamba Mini 1.6 is a hybrid foundation model combining State Space Models (Mamba) with Transformer attention mechanisms. With 12 billion active parameters (52 billion total), this model excels in extremely long-context tasks (up to 256K tokens) and achieves superior inference efficiency, outperforming comparable open models on tasks such as retrieval-augmented generation (RAG) and grounded question answering. Jamba Mini 1.6 supports multilingual tasks across English, Spanish, French, Portuguese, Italian, Dutch, German, Arabic, and Hebrew, along with structured JSON output and tool-use capabilities.\n\nUsage of this model is subject to the [Jamba Open Model License](https://www.ai21.com/licenses/jamba-open-model-license)."
}, {
    "value": "google/gemma-3-12b-it:free",
    "label": "Gemma 3 12B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 12B is the second largest in the family of Gemma 3 models after [Gemma 3 27B](google/gemma-3-27b-it)"
}, {
    "value": "google/gemma-3-12b-it",
    "label": "Gemma 3 12B",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 12B is the second largest in the family of Gemma 3 models after [Gemma 3 27B](google/gemma-3-27b-it)"
}, {
    "value": "cohere/command-a",
    "label": "Command A",
    "cover": "/images/icons/Cohere.png",
    "description": "Command A is an open-weights 111B parameter model with a 256k context window focused on delivering great performance across agentic, multilingual, and coding use cases.\nCompared to other leading proprietary and open-weights models Command A delivers maximum performance with minimum hardware costs, excelling on business-critical agentic and multilingual tasks."
}, {
    "value": "openai/gpt-4.1-mini-search-preview",
    "label": "gpt-4.1-mini Search Preview",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-4o mini Search Preview is a specialized model for web search in Chat Completions. It is trained to understand and execute web search queries."
}, {
    "value": "tokyotech-llm/llama-3.1-swallow-70b-instruct-v0.3",
    "label": "Llama 3.1 Swallow 70B Instruct V0.3",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://sambanova.ai/&size=256",
    "description": "Llama 3.1 Swallow 70B is a large language model that was built by continual pre-training on the Meta Llama 3.1 70B. Llama 3.1 Swallow enhanced the Japanese language capabilities of the original Llama 3.1 while retaining the English language capabilities. Swallow used approximately 200 billion tokens that were sampled from a large Japanese web corpus (Swallow Corpus Version 2), Japanese and English Wikipedia articles, and mathematical and coding contents, etc (see the Training Datasets section of the base model) for continual pre-training. The instruction-tuned models (Instruct) were built by supervised fine-tuning (SFT) on the synthetic data specially built for Japanese. "
}, {
    "value": "rekaai/reka-flash-3:free",
    "label": "Flash 3",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": 'Reka Flash 3 is a general-purpose, instruction-tuned large language model with 21 billion parameters, developed by Reka. It excels at general chat, coding tasks, instruction-following, and function calling. Featuring a 32K context length and optimized through reinforcement learning (RLOO), it provides competitive performance comparable to proprietary models within a smaller parameter footprint. Ideal for low-latency, local, or on-device deployments, Reka Flash 3 is compact, supports efficient quantization (down to 11GB at 4-bit precision), and employs explicit reasoning tags ("<reasoning>") to indicate its internal thought process.\n\nReka Flash 3 is primarily an English model with limited multilingual understanding capabilities. The model weights are released under the Apache 2.0 license.'
}, {
    "value": "google/gemma-3-27b-it:free",
    "label": "Gemma 3 27B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 27B is Google's latest open source model, successor to [Gemma 2](google/gemma-2-27b-it)"
}, {
    "value": "google/gemma-3-27b-it",
    "label": "Gemma 3 27B",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including structured outputs and function calling. Gemma 3 27B is Google's latest open source model, successor to [Gemma 2](google/gemma-2-27b-it)"
}, {
    "value": "thedrummer/anubis-pro-105b-v1",
    "label": "Anubis Pro 105B V1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.parasail.io/&size=256",
    "description": "Anubis Pro 105B v1 is an expanded and refined variant of Meta’s Llama 3.3 70B, featuring 50% additional layers and further fine-tuning to leverage its increased capacity. Designed for advanced narrative, roleplay, and instructional tasks, it demonstrates enhanced emotional intelligence, creativity, nuanced character portrayal, and superior prompt adherence compared to smaller models. Its larger parameter count allows for deeper contextual understanding and extended reasoning capabilities, optimized for engaging, intelligent, and coherent interactions."
}, {
    "value": "latitudegames/wayfarer-large-70b-llama-3.3",
    "label": "Wayfarer Large 70B Llama 3.3",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.parasail.io/&size=256",
    "description": "Wayfarer Large 70B is a roleplay and text-adventure model fine-tuned from Meta’s Llama-3.3-70B-Instruct. Specifically optimized for narrative-driven, challenging scenarios, it introduces realistic stakes, conflicts, and consequences often avoided by standard RLHF-aligned models. Trained using a curated blend of adventure, roleplay, and instructive fiction datasets, Wayfarer emphasizes tense storytelling, authentic player failure scenarios, and robust narrative immersion, making it uniquely suited for interactive fiction and gaming experiences."
}, {
    "value": "thedrummer/skyfall-36b-v2",
    "label": "Skyfall 36B V2",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.parasail.io/&size=256",
    "description": "Skyfall 36B v2 is an enhanced iteration of Mistral Small 2501, specifically fine-tuned for improved creativity, nuanced writing, role-playing, and coherent storytelling."
}, {
    "value": "microsoft/phi-4-multimodal-instruct",
    "label": "Phi 4 Multimodal Instruct",
    "cover": "/images/icons/DeepInfra.webp",
    "description": "Phi-4 Multimodal Instruct is a versatile 5.6B parameter foundation model that combines advanced reasoning and instruction-following capabilities across both text and visual inputs, providing accurate text outputs. The unified architecture enables efficient, low-latency inference, suitable for edge and mobile deployments. Phi-4 Multimodal Instruct supports text inputs in multiple languages including Arabic, Chinese, English, French, German, Japanese, Spanish, and more, with visual input optimized primarily for English. It delivers impressive performance on multimodal tasks involving mathematical, scientific, and document reasoning, providing developers and enterprises a powerful yet compact model for sophisticated interactive applications. For more information, see the [Phi-4 Multimodal blog post](https://azure.microsoft.com/en-us/blog/empowering-innovation-the-next-generation-of-the-phi-family/).\n"
}, {
    "value": "deepseek/deepseek-r1-zero:free",
    "label": "DeepSeek R1 Zero",
    "cover": "/images/icons/DeepSeek.png",
    "description": "DeepSeek-R1-Zero is a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step. It's 671B parameters in size, with 37B active in an inference pass.\n\nIt demonstrates remarkable performance on reasoning. With RL, DeepSeek-R1-Zero naturally emerged with numerous powerful and interesting reasoning behaviors.\n\nDeepSeek-R1-Zero encounters challenges such as endless repetition, poor readability, and language mixing. See [DeepSeek R1](/deepseek/deepseek-r1) for the SFT model.\n\n"
}, {
    "value": "qwen/qwq-32b:free",
    "label": "QwQ 32B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://nineteen.ai/&size=256",
    "description": "QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models, QwQ, which is capable of thinking and reasoning, can achieve significantly enhanced performance in downstream tasks, especially hard problems. QwQ-32B is the medium-sized reasoning model, which is capable of achieving competitive performance against state-of-the-art reasoning models, e.g., DeepSeek-R1, o1-mini."
}, {
    "value": "qwen/qwq-32b",
    "label": "QwQ 32B",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41a073403f9e2b7806f05_qwen-logo.webp",
    "description": "QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models, QwQ, which is capable of thinking and reasoning, can achieve significantly enhanced performance in downstream tasks, especially hard problems. QwQ-32B is the medium-sized reasoning model, which is capable of achieving competitive performance against state-of-the-art reasoning models, e.g., DeepSeek-R1, o1-mini."
}, {
    "value": "qwen/qwen2.5-32b-instruct",
    "label": "Qwen2.5 32B Instruct",
    "cover": "",
    "description": "Qwen2.5 32B Instruct is the instruction-tuned variant of the latest Qwen large language model series. It provides enhanced instruction-following capabilities, improved proficiency in coding and mathematical reasoning, and robust handling of structured data and outputs such as JSON. It supports long-context processing up to 128K tokens and multilingual tasks across 29+ languages. The model has 32.5 billion parameters, 64 layers, and utilizes an advanced transformer architecture with RoPE, SwiGLU, RMSNorm, and Attention QKV bias.\n\nFor more details, please refer to the [Qwen2.5 Blog](https://qwenlm.github.io/blog/qwen2.5/) ."
}, {
    "value": "moonshotai/moonlight-16b-a3b-instruct:free",
    "label": "Moonlight 16B A3B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Moonlight-16B-A3B-Instruct is a 16B-parameter Mixture-of-Experts (MoE) language model developed by Moonshot AI. It is optimized for instruction-following tasks with 3B activated parameters per inference. The model advances the Pareto frontier in performance per FLOP across English, coding, math, and Chinese benchmarks. It outperforms comparable models like Llama3-3B and Deepseek-v2-Lite while maintaining efficient deployment capabilities through Hugging Face integration and compatibility with popular inference engines like vLLM12."
}, {
    "value": "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "label": "DeepHermes 3 Llama 3 8B Preview",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": 'DeepHermes 3 Preview is the latest version of our flagship Hermes series of LLMs by Nous Research, and one of the first models in the world to unify Reasoning (long chains of thought that improve answer accuracy) and normal LLM response modes into one model. We have also improved LLM annotation, judgement, and function calling.\n\nDeepHermes 3 Preview is one of the first LLM models to unify both "intuitive", traditional mode responses and long chain of thought reasoning responses into a single model, toggled by a system prompt.'
}, {
    "value": "openai/gpt-4.5-preview",
    "label": "GPT-4.5 (Preview)",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-4.5 (Preview) is a research preview of OpenAI’s latest language model, designed to advance capabilities in reasoning, creativity, and multi-turn conversation. It builds on previous iterations with improvements in world knowledge, contextual coherence, and the ability to follow user intent more effectively.\n\nThe model demonstrates enhanced performance in tasks that require open-ended thinking, problem-solving, and communication. Early testing suggests it is better at generating nuanced responses, maintaining long-context coherence, and reducing hallucinations compared to earlier versions.\n\nThis research preview is intended to help evaluate GPT-4.5’s strengths and limitations in real-world use cases as OpenAI continues to refine and develop future models. Read more at the [blog post here.](https://openai.com/index/introducing-gpt-4-5/)"
}, {
    "value": "mistralai/mistral-saba",
    "label": "Saba",
    "cover": "/images/icons/Mistral.png",
    "description": "Mistral Saba is a 24B-parameter language model specifically designed for the Middle East and South Asia, delivering accurate and contextually relevant responses while maintaining efficient performance. Trained on curated regional datasets, it supports multiple Indian-origin languages—including Tamil and Malayalam—alongside Arabic. This makes it a versatile option for a range of regional and multilingual applications. Read more at the blog post [here](https://mistral.ai/en/news/mistral-saba)"
}, {
    "value": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "label": "Dolphin3.0 R1 Mistral 24B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Dolphin 3.0 R1 is the next generation of the Dolphin series of instruct-tuned models.  Designed to be the ultimate general purpose local model, enabling coding, math, agentic, function calling, and general use cases.\n\nThe R1 version has been trained for 3 epochs to reason using 800k reasoning traces from the Dolphin-R1 dataset.\n\nDolphin aims to be a general purpose reasoning instruct model, similar to the models behind ChatGPT, Claude, Gemini.\n\nPart of the [Dolphin 3.0 Collection](https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3) Curated and trained by [Eric Hartford](https://huggingface.co/ehartford), [Ben Gitter](https://huggingface.co/bigstorm), [BlouseJury](https://huggingface.co/BlouseJury) and [Cognitive Computations](https://huggingface.co/cognitivecomputations)"
}, {
    "value": "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "label": "Dolphin3.0 Mistral 24B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Dolphin 3.0 is the next generation of the Dolphin series of instruct-tuned models.  Designed to be the ultimate general purpose local model, enabling coding, math, agentic, function calling, and general use cases.\n\nDolphin aims to be a general purpose instruct model, similar to the models behind ChatGPT, Claude, Gemini. \n\nPart of the [Dolphin 3.0 Collection](https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3) Curated and trained by [Eric Hartford](https://huggingface.co/ehartford), [Ben Gitter](https://huggingface.co/bigstorm), [BlouseJury](https://huggingface.co/BlouseJury) and [Cognitive Computations](https://huggingface.co/cognitivecomputations)"
}, {
    "value": "meta-llama/llama-guard-3-8b",
    "label": "Llama Guard 3 8B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "Llama Guard 3 is a Llama-3.1-8B pretrained model, fine-tuned for content safety classification. Similar to previous versions, it can be used to classify content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM – it generates text in its output that indicates whether a given prompt or response is safe or unsafe, and if unsafe, it also lists the content categories violated.\n\nLlama Guard 3 was aligned to safeguard against the MLCommons standardized hazards taxonomy and designed to support Llama 3.1 capabilities. Specifically, it provides content moderation in 8 languages, and was optimized to support safety and security for search and code interpreter tool calls.\n"
}, {
    "value": "openai/o3-mini-high",
    "label": "o3 Mini High",
    "cover": "/images/icons/OpenAI.svg",
    "description": "OpenAI o3-mini-high is the same model as [o3-mini](/openai/o3-mini) with reasoning_effort set to high. \n\no3-mini is a cost-efficient language model optimized for STEM reasoning tasks, particularly excelling in science, mathematics, and coding. The model features three adjustable reasoning effort levels and supports key developer capabilities including function calling, structured outputs, and streaming, though it does not include vision processing capabilities.\n\nThe model demonstrates significant improvements over its predecessor, with expert testers preferring its responses 56% of the time and noting a 39% reduction in major errors on complex questions. With medium reasoning effort settings, o3-mini matches the performance of the larger o1 model on challenging reasoning evaluations like AIME and GPQA, while maintaining lower latency and cost."
}, {
    "value": "allenai/llama-3.1-tulu-3-405b",
    "label": "Llama 3.1 Tulu 3 405B",
    "cover": "",
    "description": "T\xfclu 3 405B is the largest model in the T\xfclu 3 family, applying fully open post-training recipes at a 405B parameter scale. Built on the Llama 3.1 405B base, it leverages Reinforcement Learning with Verifiable Rewards (RLVR) to enhance instruction following, MATH, GSM8K, and IFEval performance. As part of T\xfclu 3’s fully open-source approach, it offers state-of-the-art capabilities while surpassing prior open-weight models like Llama 3.1 405B Instruct and Nous Hermes 3 405B on multiple benchmarks. To read more, [click here.](https://allenai.org/blog/tulu-3-405B)"
}, {
    "value": "deepseek/deepseek-r1-distill-llama-8b",
    "label": "R1 Distill Llama 8B",
    "cover": "/images/icons/DeepSeek.png",
    "description": "DeepSeek R1 Distill Llama 8B is a distilled large language model based on [Llama-3.1-8B-Instruct](/meta-llama/llama-3.1-8b-instruct), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). The model combines advanced distillation techniques to achieve high performance across multiple benchmarks, including:\n\n- AIME 2024 pass@1: 50.4\n- MATH-500 pass@1: 89.1\n- CodeForces Rating: 1205\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models.\n\nHugging Face: \n- [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) \n- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)   |"
}, {
    "value": "google/gemini-2.0-flash-001",
    "label": "Gemini 2.0 Flash",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1.5](/google/gemini-flash-1.5), while maintaining quality on par with larger models like [Gemini Pro 1.5](/google/gemini-pro-1.5). It introduces notable enhancements in multimodal understanding, coding capabilities, complex instruction following, and function calling. These advancements come together to deliver more seamless and robust agentic experiences."
}, {
    "value": "qwen/qwen-vl-plus",
    "label": "Qwen VL Plus",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.alibabacloud.com/&size=256",
    "description": "Qwen's Enhanced Large Visual Language Model. Significantly upgraded for detailed recognition capabilities and text recognition abilities, supporting ultra-high pixel resolutions up to millions of pixels and extreme aspect ratios for image input. It delivers significant performance across a broad range of visual tasks.\n"
}, {
    "value": "aion-labs/aion-1.0",
    "label": "Aion-1.0",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.aionlabs.ai/&size=256",
    "description": "Aion-1.0 is a multi-model system designed for high performance across various tasks, including reasoning and coding. It is built on DeepSeek-R1, augmented with additional models and techniques such as Tree of Thoughts (ToT) and Mixture of Experts (MoE). It is Aion Lab's most powerful reasoning model."
}, {
    "value": "aion-labs/aion-1.0-mini",
    "label": "Aion-1.0-Mini",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.aionlabs.ai/&size=256",
    "description": "Aion-1.0-Mini 32B parameter model is a distilled version of the DeepSeek-R1 model, designed for strong performance in reasoning domains such as mathematics, coding, and logic. It is a modified variant of a FuseAI model that outperforms R1-Distill-Qwen-32B and R1-Distill-Llama-70B, with benchmark results available on its [Hugging Face page](https://huggingface.co/FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview), independently replicated for verification."
}, {
    "value": "aion-labs/aion-rp-llama-3.1-8b",
    "label": "Aion-RP 1.0 (8B)",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.aionlabs.ai/&size=256",
    "description": "Aion-RP-Llama-3.1-8B ranks the highest in the character evaluation portion of the RPBench-Auto benchmark, a roleplaying-specific variant of Arena-Hard-Auto, where LLMs evaluate each other’s responses. It is a fine-tuned base model rather than an instruct model, designed to produce more natural and varied writing."
}, {
    "value": "qwen/qwen-vl-max",
    "label": "Qwen VL Max",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.alibabacloud.com/&size=256",
    "description": "Qwen VL Max is a visual understanding model with 7500 tokens context length. It excels in delivering optimal performance for a broader spectrum of complex tasks.\n"
}, {
    "value": "qwen/qwen-turbo",
    "label": "Qwen-Turbo",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.alibabacloud.com/&size=256",
    "description": "Qwen-Turbo, based on Qwen2.5, is a 1M context model that provides fast speed and low cost, suitable for simple tasks."
}, {
    "value": "qwen/qwen2.5-vl-72b-instruct:free",
    "label": "Qwen2.5 VL 72B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.alibabacloud.com/&size=256",
    "description": "Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, fish, and insects. It is also highly capable of analyzing texts, charts, icons, graphics, and layouts within images."
}, {
    "value": "qwen/qwen2.5-vl-72b-instruct",
    "label": "Qwen2.5 VL 72B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.parasail.io/&size=256",
    "description": "Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, fish, and insects. It is also highly capable of analyzing texts, charts, icons, graphics, and layouts within images."
}, {
    "value": "qwen/qwen-plus",
    "label": "Qwen-Plus",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.alibabacloud.com/&size=256",
    "description": "Qwen-Plus, based on the Qwen2.5 foundation model, is a 131K context model with a balanced performance, speed, and cost combination."
}, {
    "value": "qwen/qwen-max",
    "label": "Qwen-Max ",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.alibabacloud.com/&size=256",
    "description": "Qwen-Max, based on Qwen2.5, provides the best inference performance among [Qwen models](/qwen), especially for complex multi-step tasks. It's a large-scale MoE model that has been pretrained on over 20 trillion tokens and further post-trained with curated Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) methodologies. The parameter count is unknown."
}, {
    "value": "openai/o3-mini",
    "label": "o3 Mini",
    "cover": "/images/icons/OpenAI.svg",
    "description": 'OpenAI o3-mini is a cost-efficient language model optimized for STEM reasoning tasks, particularly excelling in science, mathematics, and coding.\n\nThis model supports the `reasoning_effort` parameter, which can be set to "high", "medium", or "low" to control the thinking time of the model. The default is "medium". OpenRouter also offers the model slug `openai/o3-mini-high` to default the parameter to "high".\n\nThe model features three adjustable reasoning effort levels and supports key developer capabilities including function calling, structured outputs, and streaming, though it does not include vision processing capabilities.\n\nThe model demonstrates significant improvements over its predecessor, with expert testers preferring its responses 56% of the time and noting a 39% reduction in major errors on complex questions. With medium reasoning effort settings, o3-mini matches the performance of the larger o1 model on challenging reasoning evaluations like AIME and GPQA, while maintaining lower latency and cost.'
}, {
    "value": "deepseek/deepseek-r1-distill-qwen-1.5b",
    "label": "R1 Distill Qwen 1.5B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "DeepSeek R1 Distill Qwen 1.5B is a distilled large language model based on  [Qwen 2.5 Math 1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It's a very small and efficient model which outperforms [GPT 4o 0513](/openai/gpt-4o-2024-05-13) on Math Benchmarks.\n\nOther benchmark results include:\n\n- AIME 2024 pass@1: 28.9\n- AIME 2024 cons@64: 52.7\n- MATH-500 pass@1: 83.9\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models."
}, {
    "value": "mistralai/mistral-small-24b-instruct-2501:free",
    "label": "Mistral Small 3",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Mistral Small 3 is a 24B-parameter language model optimized for low-latency performance across common AI tasks. Released under the Apache 2.0 license, it features both pre-trained and instruction-tuned versions designed for efficient local deployment.\n\nThe model achieves 81% accuracy on the MMLU benchmark and performs competitively with larger models like Llama 3.3 70B and Qwen 32B, while operating at three times the speed on equivalent hardware. [Read the blog post about the model here.](https://mistral.ai/news/mistral-small-3/)"
}, {
    "value": "mistralai/mistral-small-24b-instruct-2501",
    "label": "Mistral Small 3",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f4197d83b94e540f009dc3_mistral-logo.webp",
    "description": "Mistral Small 3 is a 24B-parameter language model optimized for low-latency performance across common AI tasks. Released under the Apache 2.0 license, it features both pre-trained and instruction-tuned versions designed for efficient local deployment.\n\nThe model achieves 81% accuracy on the MMLU benchmark and performs competitively with larger models like Llama 3.3 70B and Qwen 32B, while operating at three times the speed on equivalent hardware. [Read the blog post about the model here.](https://mistral.ai/news/mistral-small-3/)"
}, {
    "value": "deepseek/deepseek-r1-distill-qwen-32b:free",
    "label": "R1 Distill Qwen 32B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://nineteen.ai/&size=256",
    "description": "DeepSeek R1 Distill Qwen 32B is a distilled large language model based on [Qwen 2.5 32B](https://huggingface.co/Qwen/Qwen2.5-32B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.\n\nOther benchmark results include:\n\n- AIME 2024 pass@1: 72.6\n- MATH-500 pass@1: 94.3\n- CodeForces Rating: 1691\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models."
}, {
    "value": "deepseek/deepseek-r1-distill-qwen-32b",
    "label": "R1 Distill Qwen 32B",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41a324f1d713df2cbfbf4_deepseek-logo.webp",
    "description": "DeepSeek R1 Distill Qwen 32B is a distilled large language model based on [Qwen 2.5 32B](https://huggingface.co/Qwen/Qwen2.5-32B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.\n\nOther benchmark results include:\n\n- AIME 2024 pass@1: 72.6\n- MATH-500 pass@1: 94.3\n- CodeForces Rating: 1691\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models."
}, {
    "value": "deepseek/deepseek-r1-distill-qwen-14b:free",
    "label": "R1 Distill Qwen 14B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "DeepSeek R1 Distill Qwen 14B is a distilled large language model based on [Qwen 2.5 14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.\n\nOther benchmark results include:\n\n- AIME 2024 pass@1: 69.7\n- MATH-500 pass@1: 93.9\n- CodeForces Rating: 1481\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models."
}, {
    "value": "deepseek/deepseek-r1-distill-qwen-14b",
    "label": "R1 Distill Qwen 14B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://novita.ai/&size=256",
    "description": "DeepSeek R1 Distill Qwen 14B is a distilled large language model based on [Qwen 2.5 14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). It outperforms OpenAI's o1-mini across various benchmarks, achieving new state-of-the-art results for dense models.\n\nOther benchmark results include:\n\n- AIME 2024 pass@1: 69.7\n- MATH-500 pass@1: 93.9\n- CodeForces Rating: 1481\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models."
}, {
    "value": "liquid/lfm-7b",
    "label": "LFM 7B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.liquid.ai/&size=256",
    "description": "LFM-7B, a new best-in-class language model. LFM-7B is designed for exceptional chat capabilities, including languages like Arabic and Japanese. Powered by the Liquid Foundation Model (LFM) architecture, it exhibits unique features like low memory footprint and fast inference speed. \n\nLFM-7B is the world’s best-in-class multilingual language model in English, Arabic, and Japanese.\n\nSee the [launch announcement](https://www.liquid.ai/lfm-7b) for benchmarks and more info."
}, {
    "value": "liquid/lfm-3b",
    "label": "LFM 3B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.liquid.ai/&size=256",
    "description": "Liquid's LFM 3B delivers incredible performance for its size. It positions itself as first place among 3B parameter transformers, hybrids, and RNN models It is also on par with Phi-3.5-mini on multiple benchmarks, while being 18.4% smaller.\n\nLFM-3B is the ideal choice for mobile and other edge text-based applications.\n\nSee the [launch announcement](https://www.liquid.ai/liquid-foundation-models) for benchmarks and more info."
}, {
    "value": "deepseek/deepseek-r1-distill-llama-70b:free",
    "label": "R1 Distill Llama 70B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://targon.com/&size=256",
    "description": "DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](/meta-llama/llama-3.3-70b-instruct), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). The model combines advanced distillation techniques to achieve high performance across multiple benchmarks, including:\n\n- AIME 2024 pass@1: 70.0\n- MATH-500 pass@1: 94.5\n- CodeForces Rating: 1633\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models."
}, {
    "value": "deepseek/deepseek-r1-distill-llama-70b",
    "label": "R1 Distill Llama 70B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://inference.net/&size=256",
    "description": "DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](/meta-llama/llama-3.3-70b-instruct), using outputs from [DeepSeek R1](/deepseek/deepseek-r1). The model combines advanced distillation techniques to achieve high performance across multiple benchmarks, including:\n\n- AIME 2024 pass@1: 70.0\n- MATH-500 pass@1: 94.5\n- CodeForces Rating: 1633\n\nThe model leverages fine-tuning from DeepSeek R1's outputs, enabling competitive performance comparable to larger frontier models."
}, {
    "value": "google/gemini-2.0-flash-thinking-exp:free",
    "label": "Gemini 2.0 Flash Thinking Experimental 01-21",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": 'Gemini 2.0 Flash Thinking Experimental (01-21) is a snapshot of Gemini 2.0 Flash Thinking Experimental.\n\nGemini 2.0 Flash Thinking Mode is an experimental model that\'s trained to generate the "thinking process" the model goes through as part of its response. As a result, Thinking Mode is capable of stronger reasoning capabilities in its responses than the [base Gemini 2.0 Flash model](/google/gemini-2.0-flash-exp).'
}, {
    "value": "deepseek/deepseek-r1:free",
    "label": "R1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.\n\nFully open-source model & [technical report](https://api-docs.deepseek.com/news/news250120).\n\nMIT licensed: Distill & commercialize freely!"
}, {
    "value": "deepseek/deepseek-r1",
    "label": "DeepSeek R1",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41a324f1d713df2cbfbf4_deepseek-logo.webp",
    "description": "DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It's 671B parameters in size, with 37B active in an inference pass.\n\nFully open-source model & [technical report](https://api-docs.deepseek.com/news/news250120).\n\nMIT licensed: Distill & commercialize freely!"
}, {
    "value": "sophosympatheia/rogue-rose-103b-v0.2:free",
    "label": "Rogue Rose 103B v0.2",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://nineteen.ai/&size=256",
    "description": "Rogue Rose demonstrates strong capabilities in roleplaying and storytelling applications, potentially surpassing other models in the 103-120B parameter range. While it occasionally exhibits inconsistencies with scene logic, the overall interaction quality represents an advancement in natural language processing for creative applications.\n\nIt is a 120-layer frankenmerge model combining two custom 70B architectures from November 2023, derived from the [xwin-stellarbright-erp-70b-v2](https://huggingface.co/sophosympatheia/xwin-stellarbright-erp-70b-v2) base.\n"
}, {
    "value": "minimax/minimax-01",
    "label": "MiniMax-01",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://minimaxi.com/&size=256",
    "description": "MiniMax-01 is a combines MiniMax-Text-01 for text generation and MiniMax-VL-01 for image understanding. It has 456 billion parameters, with 45.9 billion parameters activated per inference, and can handle a context of up to 4 million tokens.\n\nThe text model adopts a hybrid architecture that combines Lightning Attention, Softmax Attention, and Mixture-of-Experts (MoE). The image model adopts the “ViT-MLP-LLM” framework and is trained on top of the text model.\n\nTo read more about the release, see: https://www.minimaxi.com/en/news/minimax-01-series-2"
}, {
    "value": "mistralai/codestral-2501",
    "label": "Codestral 2501",
    "cover": "/images/icons/Mistral.png",
    "description": "[Mistral](/mistralai)'s cutting-edge language model for coding. Codestral specializes in low-latency, high-frequency tasks such as fill-in-the-middle (FIM), code correction and test generation. \n\nLearn more on their blog post: https://mistral.ai/news/codestral-2501/"
}, {
    "value": "microsoft/phi-4",
    "label": "Phi 4",
    "cover": "/images/icons/DeepInfra.webp",
    "description": "[Microsoft Research](/microsoft) Phi-4 is designed to perform well in complex reasoning tasks and can operate efficiently in situations with limited memory or where quick responses are needed. \n\nAt 14 billion parameters, it was trained on a mix of high-quality synthetic datasets, data from curated websites, and academic materials. It has undergone careful improvement to follow instructions accurately and maintain strong safety standards. It works best with English language inputs.\n\nFor more information, please see [Phi-4 Technical Report](https://arxiv.org/pdf/2412.08905)\n"
}, {
    "value": "sao10k/l3.1-70b-hanami-x1",
    "label": "Llama 3.1 70B Hanami x1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://infermatic.ai/&size=256",
    "description": "This is [Sao10K](/sao10k)'s experiment over [Euryale v2.2](/sao10k/l3.1-euryale-70b)."
}, {
    "value": "deepseek/deepseek-chat:free",
    "label": "DeepSeek V3",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens, the reported evaluations reveal that the model outperforms other open-source models and rivals leading closed-source models.\n\nFor model details, please visit [the DeepSeek-V3 repo](https://github.com/deepseek-ai/DeepSeek-V3) for more information, or see the [launch announcement](https://api-docs.deepseek.com/news/news1226)."
}, {
    "value": "deepseek/deepseek-chat",
    "label": "DeepSeek V3",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41a324f1d713df2cbfbf4_deepseek-logo.webp",
    "description": "DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens, the reported evaluations reveal that the model outperforms other open-source models and rivals leading closed-source models.\n\nFor model details, please visit [the DeepSeek-V3 repo](https://github.com/deepseek-ai/DeepSeek-V3) for more information, or see the [launch announcement](https://api-docs.deepseek.com/news/news1226)."
}, {
    "value": "google/gemini-2.0-flash-thinking-exp-1219:free",
    "label": "Gemini 2.0 Flash Thinking Experimental",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": 'Gemini 2.0 Flash Thinking Mode is an experimental model that\'s trained to generate the "thinking process" the model goes through as part of its response. As a result, Thinking Mode is capable of stronger reasoning capabilities in its responses than the [base Gemini 2.0 Flash model](/google/gemini-2.0-flash-exp).'
}, {
    "value": "sao10k/l3.3-euryale-70b",
    "label": "Llama 3.3 Euryale 70B",
    "cover": "/images/icons/DeepInfra.webp",
    "description": "Euryale L3.3 70B is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). It is the successor of [Euryale L3 70B v2.2](/models/sao10k/l3-euryale-70b)."
}, {
    "value": "inflatebot/mn-mag-mell-r1",
    "label": "Mag Mell R1 12B",
    "cover": "",
    "description": 'Mag Mell is a merge of pre-trained language models created using mergekit, based on [Mistral Nemo](/mistralai/mistral-nemo). It is a great roleplay and storytelling model which combines the best parts of many other models to be a general purpose solution for many usecases.\n\nIntended to be a general purpose "Best of Nemo" model for any fictional, creative use case. \n\nMag Mell is composed of 3 intermediate parts:\n- Hero (RP, trope coverage)\n- Monk (Intelligence, groundedness)\n- Deity (Prose, flair)'
}, {
    "value": "openai/o1",
    "label": "o1",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding. The o1 model series is trained with large-scale reinforcement learning to reason using chain of thought. \n\nThe o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).\n"
}, {
    "value": "eva-unit-01/eva-llama-3.33-70b",
    "label": "EVA Llama 3.33 70B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": 'EVA Llama 3.33 70b is a roleplay and storywriting specialist model. It is a full-parameter finetune of [Llama-3.3-70B-Instruct](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct) on mixture of synthetic and natural data.\n\nIt uses Celeste 70B 0.1 data mixture, greatly expanding it to improve versatility, creativity and "flavor" of the resulting model\n\nThis model was built with Llama by Meta.\n'
}, {
    "value": "cohere/command-r7b-12-2024",
    "label": "Command R7B (12-2024)",
    "cover": "/images/icons/Cohere.png",
    "description": "Command R7B (12-2024) is a small, fast update of the Command R+ model, delivered in December 2024. It excels at RAG, tool use, agents, and similar tasks requiring complex reasoning and multiple steps.\n\nUse of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement)."
}, {
    "value": "google/gemini-2.0-flash-exp:free",
    "label": "Gemini 2.0 Flash Experimental",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash 1.5](/google/gemini-flash-1.5), while maintaining quality on par with larger models like [Gemini Pro 1.5](/google/gemini-pro-1.5). It introduces notable enhancements in multimodal understanding, coding capabilities, complex instruction following, and function calling. These advancements come together to deliver more seamless and robust agentic experiences."
}, {
    "value": "meta-llama/llama-3.3-70b-instruct:free",
    "label": "Llama 3.3 70B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://crusoe.ai/&size=256",
    "description": "The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperforms many of the available open source and closed chat models on common industry benchmarks.\n\nSupported languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.\n\n[Model Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md)"
}, {
    "value": "meta-llama/llama-3.3-70b-instruct",
    "label": "Llama 3.3 70B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://inference.net/&size=256",
    "description": "The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperforms many of the available open source and closed chat models on common industry benchmarks.\n\nSupported languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.\n\n[Model Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md)"
}, {
    "value": "amazon/nova-lite-v1",
    "label": "Nova Lite 1.0",
    "cover": "/images/icons/Bedrock.svg",
    "description": "Amazon Nova Lite 1.0 is a very low-cost multimodal model from Amazon that focused on fast processing of image, video, and text inputs to generate text output. Amazon Nova Lite can handle real-time customer interactions, document analysis, and visual question-answering tasks with high accuracy.\n\nWith an input context of 300K tokens, it can analyze multiple images or up to 30 minutes of video in a single input."
}, {
    "value": "amazon/nova-micro-v1",
    "label": "Nova Micro 1.0",
    "cover": "/images/icons/Bedrock.svg",
    "description": "Amazon Nova Micro 1.0 is a text-only model that delivers the lowest latency responses in the Amazon Nova family of models at a very low cost. With a context length of 128K tokens and optimized for speed and cost, Amazon Nova Micro excels at tasks such as text summarization, translation, content classification, interactive chat, and brainstorming. It has  simple mathematical reasoning and coding abilities."
}, {
    "value": "amazon/nova-pro-v1",
    "label": "Nova Pro 1.0",
    "cover": "/images/icons/Bedrock.svg",
    "description": "Amazon Nova Pro 1.0 is a capable multimodal model from Amazon focused on providing a combination of accuracy, speed, and cost for a wide range of tasks. As of December 2024, it achieves state-of-the-art performance on key benchmarks including visual question answering (TextVQA) and video understanding (VATEX).\n\nAmazon Nova Pro demonstrates strong capabilities in processing both visual and textual information and at analyzing financial documents.\n\n**NOTE**: Video input is not supported at this time."
}, {
    "value": "qwen/qwq-32b-preview:free",
    "label": "QwQ 32B Preview",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "QwQ-32B-Preview is an experimental research model focused on AI reasoning capabilities developed by the Qwen Team. As a preview release, it demonstrates promising analytical abilities while having several important limitations:\n\n1. **Language Mixing and Code-Switching**: The model may mix languages or switch between them unexpectedly, affecting response clarity.\n2. **Recursive Reasoning Loops**: The model may enter circular reasoning patterns, leading to lengthy responses without a conclusive answer.\n3. **Safety and Ethical Considerations**: The model requires enhanced safety measures to ensure reliable and secure performance, and users should exercise caution when deploying it.\n4. **Performance and Benchmark Limitations**: The model excels in math and coding but has room for improvement in other areas, such as common sense reasoning and nuanced language understanding.\n\n"
}, {
    "value": "qwen/qwq-32b-preview",
    "label": "QwQ 32B Preview",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://hyperbolic.xyz/&size=256",
    "description": "QwQ-32B-Preview is an experimental research model focused on AI reasoning capabilities developed by the Qwen Team. As a preview release, it demonstrates promising analytical abilities while having several important limitations:\n\n1. **Language Mixing and Code-Switching**: The model may mix languages or switch between them unexpectedly, affecting response clarity.\n2. **Recursive Reasoning Loops**: The model may enter circular reasoning patterns, leading to lengthy responses without a conclusive answer.\n3. **Safety and Ethical Considerations**: The model requires enhanced safety measures to ensure reliable and secure performance, and users should exercise caution when deploying it.\n4. **Performance and Benchmark Limitations**: The model excels in math and coding but has room for improvement in other areas, such as common sense reasoning and nuanced language understanding.\n\n"
}, {
    "value": "google/gemini-exp-1121",
    "label": "Gemini Experimental 1121",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Experimental release (November 21st, 2024) of Gemini."
}, {
    "value": "google/learnlm-1.5-pro-experimental:free",
    "label": "LearnLM 1.5 Pro Experimental",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "An experimental version of [Gemini 1.5 Pro](/google/gemini-pro-1.5) from Google."
}, {
    "value": "eva-unit-01/eva-qwen-2.5-72b",
    "label": "EVA Qwen2.5 72B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.parasail.io/&size=256",
    "description": 'EVA Qwen2.5 72B is a roleplay and storywriting specialist model. It\'s a full-parameter finetune of Qwen2.5-72B on mixture of synthetic and natural data.\n\nIt uses Celeste 70B 0.1 data mixture, greatly expanding it to improve versatility, creativity and "flavor" of the resulting model.'
}, {
    "value": "openai/gpt-4o-2024-11-20",
    "label": "GPT-4o (2024-11-20)",
    "cover": "/images/icons/OpenAI.svg",
    "description": 'The 2024-11-20 version of GPT-4o offers a leveled-up creative writing ability with more natural, engaging, and tailored writing to improve relevance & readability. It’s also better at working with uploaded files, providing deeper insights & more thorough responses.\n\nGPT-4o ("o" for "omni") is OpenAI\'s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.'
}, {
    "value": "mistralai/mistral-large-2411",
    "label": "Mistral Large 2411",
    "cover": "/images/icons/Mistral.png",
    "description": "Mistral Large 2 2411 is an update of [Mistral Large 2](/mistralai/mistral-large) released together with [Pixtral Large 2411](/mistralai/pixtral-large-2411)\n\nIt provides a significant upgrade on the previous [Mistral Large 24.07](/mistralai/mistral-large-2407), with notable improvements in long context understanding, a new system prompt, and more accurate function calling."
}, {
    "value": "mistralai/mistral-large-2407",
    "label": "Mistral Large 2407",
    "cover": "/images/icons/Mistral.png",
    "description": "This is Mistral AI's flagship model, Mistral Large 2 (version mistral-large-2407). It's a proprietary weights-available model and excels at reasoning, code, JSON, chat, and more. Read the launch announcement [here](https://mistral.ai/news/mistral-large-2407/).\n\nIt supports dozens of languages including French, German, Spanish, Italian, Portuguese, Arabic, Hindi, Russian, Chinese, Japanese, and Korean, along with 80+ coding languages including Python, Java, C, C++, JavaScript, and Bash. Its long context window allows precise information recall from large documents.\n"
}, {
    "value": "mistralai/pixtral-large-2411",
    "label": "Pixtral Large 2411",
    "cover": "/images/icons/Mistral.png",
    "description": "Pixtral Large is a 124B parameter, open-weight, multimodal model built on top of [Mistral Large 2](/mistralai/mistral-large-2411). The model is able to understand documents, charts and natural images.\n\nThe model is available under the Mistral Research License (MRL) for research and educational use, and the Mistral Commercial License for experimentation, testing, and production for commercial purposes.\n\n"
}, {
    "value": "x-ai/grok-vision-beta",
    "label": "Grok Vision Beta",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": "Grok Vision Beta is xAI's experimental language model with vision capability.\n\n"
}, {
    "value": "google/gemini-exp-1114",
    "label": "Gemini Experimental 1114",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": 'Gemini 11-14 (2024) experimental model features "quality" improvements.'
}, {
    "value": "infermatic/mn-inferor-12b",
    "label": "Mistral Nemo Inferor 12B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": "Inferor 12B is a merge of top roleplay models, expert on immersive narratives and storytelling.\n\nThis model was merged using the [Model Stock](https://arxiv.org/abs/2403.19522) merge method using [anthracite-org/magnum-v4-12b](https://openrouter.ai/anthracite-org/magnum-v4-72b) as a base.\n"
}, {
    "value": "qwen/qwen-2.5-coder-32b-instruct:free",
    "label": "Qwen2.5 Coder 32B Instruct",
    "cover": "/images/icons/Qwen.png",
    "description": "Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). Qwen2.5-Coder brings the following improvements upon CodeQwen1.5:\n\n- Significantly improvements in **code generation**, **code reasoning** and **code fixing**. \n- A more comprehensive foundation for real-world applications such as **Code Agents**. Not only enhancing coding capabilities but also maintaining its strengths in mathematics and general competencies.\n\nTo read more about its evaluation results, check out [Qwen 2.5 Coder's blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/)."
}, {
    "value": "raifle/sorcererlm-8x22b",
    "label": "SorcererLM 8x22B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://infermatic.ai/&size=256",
    "description": "SorcererLM is an advanced RP and storytelling model, built as a Low-rank 16-bit LoRA fine-tuned on [WizardLM-2 8x22B](/microsoft/wizardlm-2-8x22b).\n\n- Advanced reasoning and emotional intelligence for engaging and immersive interactions\n- Vivid writing capabilities enriched with spatial and contextual awareness\n- Enhanced narrative depth, promoting creative and dynamic storytelling"
}, {
    "value": "eva-unit-01/eva-qwen-2.5-32b",
    "label": "EVA Qwen2.5 32B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": 'EVA Qwen2.5 32B is a roleplaying/storywriting specialist model. It\'s a full-parameter finetune of Qwen2.5-32B on mixture of synthetic and natural data.\n\nIt uses Celeste 70B 0.1 data mixture, greatly expanding it to improve versatility, creativity and "flavor" of the resulting model.'
}, {
    "value": "thedrummer/unslopnemo-12b",
    "label": "Unslopnemo 12B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://infermatic.ai/&size=256",
    "description": "UnslopNemo v4.1 is the latest addition from the creator of Rocinante, designed for adventure writing and role-play scenarios."
}, {
    "value": "anthracite-org/magnum-v4-72b",
    "label": "Magnum v4 72B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "This is a series of models designed to replicate the prose quality of the Claude 3 models, specifically Sonnet(https://openrouter.ai/anthropic/claude-3.5-sonnet) and Opus(https://openrouter.ai/anthropic/claude-3-opus).\n\nThe model is fine-tuned on top of [Qwen2.5 72B](https://openrouter.ai/qwen/qwen-2.5-72b-instruct)."
}, {
    "value": "neversleep/llama-3.1-lumimaid-70b",
    "label": "Lumimaid v0.2 70B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": 'Lumimaid v0.2 70B is a finetune of [Llama 3.1 70B](/meta-llama/llama-3.1-70b-instruct) with a "HUGE step up dataset wise" compared to Lumimaid v0.1. Sloppy chats output were purged.\n\nUsage of this model is subject to [Meta\'s Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).'
}, {
    "value": "x-ai/grok-beta",
    "label": "Grok Beta",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": "Grok Beta is xAI's experimental language model with state-of-the-art reasoning capabilities, best for complex and multi-step use cases.\n\nIt is the successor of [Grok 2](https://x.ai/blog/grok-2) with enhanced context length."
}, {
    "value": "mistralai/ministral-8b",
    "label": "Ministral 8B",
    "cover": "/images/icons/Mistral.png",
    "description": "Ministral 8B is an 8B parameter model featuring a unique interleaved sliding-window attention pattern for faster, memory-efficient inference. Designed for edge use cases, it supports up to 128k context length and excels in knowledge and reasoning tasks. It outperforms peers in the sub-10B category, making it perfect for low-latency, privacy-first applications."
}, {
    "value": "mistralai/ministral-3b",
    "label": "Ministral 3B",
    "cover": "/images/icons/Mistral.png",
    "description": "Ministral 3B is a 3B parameter model optimized for on-device and edge computing. It excels in knowledge, commonsense reasoning, and function-calling, outperforming larger models like Mistral 7B on most benchmarks. Supporting up to 128k context length, it’s ideal for orchestrating agentic workflows and specialist tasks with efficient inference."
}, {
    "value": "qwen/qwen-2.5-7b-instruct:free",
    "label": "Qwen2.5 7B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://nineteen.ai/&size=256",
    "description": "Qwen2.5 7B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:\n\n- Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.\n\n- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.\n\n- Long-context Support up to 128K tokens and can generate up to 8K tokens.\n\n- Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen-2.5-7b-instruct",
    "label": "Qwen2.5 7B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41a073403f9e2b7806f05_qwen-logo.webp",
    "description": "Qwen2.5 7B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:\n\n- Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.\n\n- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.\n\n- Long-context Support up to 128K tokens and can generate up to 8K tokens.\n\n- Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "x-ai/grok-2-mini",
    "label": "Grok 2 mini",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": "Grok 2 Mini is xAI's fast, lightweight language model that offers a balance between speed and answer quality.\n\nTo use the stronger model, see [Grok Beta](/x-ai/grok-beta).\n\nFor more information, see the [launch announcement](https://x.ai/blog/grok-2)."
}, {
    "value": "x-ai/grok-2",
    "label": "Grok 2",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://x.ai/&size=256",
    "description": "Grok 2 is xAI's frontier language model with state-of-the-art reasoning capabilities, best for complex and multi-step use cases.\n\nTo use a faster version, see [Grok 2 Mini](/x-ai/grok-2-mini).\n\nFor more information, see the [launch announcement](https://x.ai/blog/grok-2)."
}, {
    "value": "inflection/inflection-3-pi",
    "label": "Inflection 3 Pi",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://inflection.ai/&size=256",
    "description": "Inflection 3 Pi powers Inflection's [Pi](https://pi.ai) chatbot, including backstory, emotional intelligence, productivity, and safety. It has access to recent news, and excels in scenarios like customer support and roleplay.\n\nPi has been trained to mirror your tone and style, if you use more emojis, so will Pi! Try experimenting with various prompts and conversation styles."
}, {
    "value": "inflection/inflection-3-productivity",
    "label": "Inflection 3 Productivity",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://inflection.ai/&size=256",
    "description": "Inflection 3 Productivity is optimized for following instructions. It is better for tasks requiring JSON output or precise adherence to provided guidelines. It has access to recent news.\n\nFor emotional intelligence similar to Pi, see [Inflect 3 Pi](/inflection/inflection-3-pi)\n\nSee [Inflection's announcement](https://inflection.ai/blog/enterprise) for more details."
}, {
    "value": "google/gemini-flash-1.5-8b",
    "label": "Gemini 1.5 Flash 8B",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemini Flash 1.5 8B is optimized for speed and efficiency, offering enhanced performance in small prompt tasks like chat, transcription, and translation. With reduced latency, it is highly effective for real-time and large-scale operations. This model focuses on cost-effective solutions while maintaining high-quality results.\n\n[Click here to learn more about this model](https://developers.googleblog.com/en/gemini-15-flash-8b-is-now-generally-available-for-use/).\n\nUsage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms)."
}, {
    "value": "liquid/lfm-40b",
    "label": "LFM 40B MoE",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.liquid.ai/&size=256",
    "description": "Liquid's 40.3B Mixture of Experts (MoE) model. Liquid Foundation Models (LFMs) are large neural networks built with computational units rooted in dynamic systems.\n\nLFMs are general-purpose AI models that can be used to model any kind of sequential data, including video, audio, text, time series, and signals.\n\nSee the [launch announcement](https://www.liquid.ai/liquid-foundation-models) for benchmarks and more info."
}, {
    "value": "eva-unit-01/eva-qwen-2.5-14b",
    "label": "EVA Qwen2.5 14B",
    "cover": "",
    "description": "A model specializing in RP and creative writing, this model is based on Qwen2.5-14B, fine-tuned with a mixture of synthetic and natural data.\n\nIt is trained on 1.5M tokens of role-play data, and fine-tuned on 1.5M tokens of synthetic data."
}, {
    "value": "thedrummer/rocinante-12b",
    "label": "Rocinante 12B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://infermatic.ai/&size=256",
    "description": "Rocinante 12B is designed for engaging storytelling and rich prose.\n\nEarly testers have reported:\n- Expanded vocabulary with unique and expressive word choices\n- Enhanced creativity for vivid narratives\n- Adventure-filled and captivating stories"
}, {
    "value": "anthracite-org/magnum-v2-72b",
    "label": "Magnum v2 72B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://infermatic.ai/&size=256",
    "description": "From the maker of [Goliath](https://openrouter.ai/models/alpindale/goliath-120b), Magnum 72B is the seventh in a family of models designed to achieve the prose quality of the Claude 3 models, notably Opus & Sonnet.\n\nThe model is based on [Qwen2 72B](https://openrouter.ai/models/qwen/qwen-2-72b-instruct) and trained with 55 million tokens of highly curated roleplay (RP) data."
}, {
    "value": "meta-llama/llama-3.2-3b-instruct:free",
    "label": "Llama 3.2 3B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://nineteen.ai/&size=256",
    "description": "Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimized for advanced natural language processing tasks like dialogue generation, reasoning, and summarization. Designed with the latest transformer architecture, it supports eight languages, including English, Spanish, and Hindi, and is adaptable for additional languages.\n\nTrained on 9 trillion tokens, the Llama 3.2 3B model excels in instruction-following, complex reasoning, and tool use. Its balanced performance makes it ideal for applications needing accuracy and efficiency in text generation across multilingual settings.\n\nClick here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3.2-3b-instruct",
    "label": "Llama 3.2 3B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://lambdalabs.com/&size=256",
    "description": "Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimized for advanced natural language processing tasks like dialogue generation, reasoning, and summarization. Designed with the latest transformer architecture, it supports eight languages, including English, Spanish, and Hindi, and is adaptable for additional languages.\n\nTrained on 9 trillion tokens, the Llama 3.2 3B model excels in instruction-following, complex reasoning, and tool use. Its balanced performance makes it ideal for applications needing accuracy and efficiency in text generation across multilingual settings.\n\nClick here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3.2-90b-vision-instruct",
    "label": "Llama 3.2 90B Vision Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://sambanova.ai/&size=256",
    "description": "The Llama 90B Vision model is a top-tier, 90-billion-parameter multimodal model designed for the most challenging visual reasoning and language tasks. It offers unparalleled accuracy in image captioning, visual question answering, and advanced image-text comprehension. Pre-trained on vast multimodal datasets and fine-tuned with human feedback, the Llama 90B Vision is engineered to handle the most demanding image-based AI tasks.\n\nThis model is perfect for industries requiring cutting-edge multimodal AI capabilities, particularly those dealing with complex, real-time visual and textual analysis.\n\nClick here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md).\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3.2-1b-instruct:free",
    "label": "Llama 3.2 1B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://crusoe.ai/&size=256",
    "description": "Llama 3.2 1B is a 1-billion-parameter language model focused on efficiently performing natural language tasks, such as summarization, dialogue, and multilingual text analysis. Its smaller size allows it to operate efficiently in low-resource environments while maintaining strong task performance.\n\nSupporting eight core languages and fine-tunable for more, Llama 1.3B is ideal for businesses or developers seeking lightweight yet powerful AI solutions that can operate in diverse multilingual settings without the high computational demand of larger models.\n\nClick here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3.2-1b-instruct",
    "label": "Llama 3.2 1B Instruct",
    "cover": "/images/icons/Lepton.png",
    "description": "Llama 3.2 1B is a 1-billion-parameter language model focused on efficiently performing natural language tasks, such as summarization, dialogue, and multilingual text analysis. Its smaller size allows it to operate efficiently in low-resource environments while maintaining strong task performance.\n\nSupporting eight core languages and fine-tunable for more, Llama 1.3B is ideal for businesses or developers seeking lightweight yet powerful AI solutions that can operate in diverse multilingual settings without the high computational demand of larger models.\n\nClick here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3.2-11b-vision-instruct:free",
    "label": "Llama 3.2 11B Vision Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "Llama 3.2 11B Vision is a multimodal model with 11 billion parameters, designed to handle tasks combining visual and textual data. It excels in tasks such as image captioning and visual question answering, bridging the gap between language generation and visual reasoning. Pre-trained on a massive dataset of image-text pairs, it performs well in complex, high-accuracy image analysis.\n\nIts ability to integrate visual understanding with language processing makes it an ideal solution for industries requiring comprehensive visual-linguistic AI applications, such as content creation, AI-driven customer service, and research.\n\nClick here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md).\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3.2-11b-vision-instruct",
    "label": "Llama 3.2 11B Vision Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41918314a4184b51788ed_meta-logo.png",
    "description": "Llama 3.2 11B Vision is a multimodal model with 11 billion parameters, designed to handle tasks combining visual and textual data. It excels in tasks such as image captioning and visual question answering, bridging the gap between language generation and visual reasoning. Pre-trained on a massive dataset of image-text pairs, it performs well in complex, high-accuracy image analysis.\n\nIts ability to integrate visual understanding with language processing makes it an ideal solution for industries requiring comprehensive visual-linguistic AI applications, such as content creation, AI-driven customer service, and research.\n\nClick here for the [original model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md).\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://www.llama.com/llama3/use-policy/)."
}, {
    "value": "qwen/qwen-2.5-72b-instruct:free",
    "label": "Qwen2.5 72B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Qwen2.5 72B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:\n\n- Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.\n\n- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.\n\n- Long-context Support up to 128K tokens and can generate up to 8K tokens.\n\n- Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen-2.5-72b-instruct",
    "label": "Qwen2.5 72B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41a073403f9e2b7806f05_qwen-logo.webp",
    "description": "Qwen2.5 72B is the latest series of Qwen large language models. Qwen2.5 brings the following improvements upon Qwen2:\n\n- Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.\n\n- Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.\n\n- Long-context Support up to 128K tokens and can generate up to 8K tokens.\n\n- Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen-2.5-vl-72b-instruct",
    "label": "Qwen2.5-VL 72B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://hyperbolic.xyz/&size=256",
    "description": "Qwen2.5 VL 72B is a multimodal LLM from the Qwen Team with the following key enhancements:\n\n- SoTA understanding of images of various resolution & ratio: Qwen2.5-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.\n\n- Understanding videos of 20min+: Qwen2.5-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.\n\n- Agent that can operate your mobiles, robots, etc.: with the abilities of complex reasoning and decision making, Qwen2.5-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.\n\n- Multilingual Support: to serve global users, besides English and Chinese, Qwen2.5-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen2-vl/) and [GitHub repo](https://github.com/QwenLM/Qwen2-VL).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "neversleep/llama-3.1-lumimaid-8b",
    "label": "Lumimaid v0.2 8B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": 'Lumimaid v0.2 8B is a finetune of [Llama 3.1 8B](/models/meta-llama/llama-3.1-8b-instruct) with a "HUGE step up dataset wise" compared to Lumimaid v0.1. Sloppy chats output were purged.\n\nUsage of this model is subject to [Meta\'s Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/).'
}, {
    "value": "openai/o1-mini",
    "label": "o1-mini",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.\n\nThe o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).\n\nNote: This model is currently experimental and not suitable for production use-cases, and may be heavily rate-limited."
}, {
    "value": "openai/o1-preview",
    "label": "o1-preview",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.\n\nThe o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).\n\nNote: This model is currently experimental and not suitable for production use-cases, and may be heavily rate-limited."
}, {
    "value": "openai/o1-preview-2024-09-12",
    "label": "o1-preview (2024-09-12)",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.\n\nThe o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).\n\nNote: This model is currently experimental and not suitable for production use-cases, and may be heavily rate-limited."
}, {
    "value": "openai/o1-mini-2024-09-12",
    "label": "o1-mini (2024-09-12)",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before responding.\n\nThe o1 models are optimized for math, science, programming, and other STEM-related tasks. They consistently exhibit PhD-level accuracy on benchmarks in physics, chemistry, and biology. Learn more in the [launch announcement](https://openai.com/o1).\n\nNote: This model is currently experimental and not suitable for production use-cases, and may be heavily rate-limited."
}, {
    "value": "mistralai/pixtral-12b",
    "label": "Pixtral 12B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://hyperbolic.xyz/&size=256",
    "description": "The first multi-modal, text+image-to-text model from Mistral AI. Its weights were launched via torrent: https://x.com/mistralai/status/1833758285167722836."
}, {
    "value": "mattshumer/reflection-70b",
    "label": "Reflection 70B",
    "cover": "",
    "description": "Reflection Llama-3.1 70B is trained with a new technique called Reflection-Tuning that teaches a LLM to detect mistakes in its reasoning and correct course.\n\nThe model was trained on synthetic data."
}, {
    "value": "cohere/command-r-plus-08-2024",
    "label": "Command R+ (08-2024)",
    "cover": "/images/icons/Cohere.png",
    "description": "command-r-plus-08-2024 is an update of the [Command R+](/models/cohere/command-r-plus) with roughly 50% higher throughput and 25% lower latencies as compared to the previous Command R+ version, while keeping the hardware footprint the same.\n\nRead the launch post [here](https://docs.cohere.com/changelog/command-gets-refreshed).\n\nUse of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement)."
}, {
    "value": "cohere/command-r-08-2024",
    "label": "Command R (08-2024)",
    "cover": "/images/icons/Cohere.png",
    "description": "command-r-08-2024 is an update of the [Command R](/models/cohere/command-r) with improved performance for multilingual retrieval-augmented generation (RAG) and tool use. More broadly, it is better at math, code and reasoning and is competitive with the previous version of the larger Command R+ model.\n\nRead the launch post [here](https://docs.cohere.com/changelog/command-gets-refreshed).\n\nUse of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement)."
}, {
    "value": "sao10k/l3.1-euryale-70b",
    "label": "Llama 3.1 Euryale 70B v2.2",
    "cover": "/images/icons/DeepInfra.webp",
    "description": "Euryale L3.1 70B v2.2 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). It is the successor of [Euryale L3 70B v2.1](/models/sao10k/l3-euryale-70b)."
}, {
    "value": "qwen/qwen-2.5-vl-7b-instruct:free",
    "label": "Qwen2.5-VL 7B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Qwen2.5 VL 7B is a multimodal LLM from the Qwen Team with the following key enhancements:\n\n- SoTA understanding of images of various resolution & ratio: Qwen2.5-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.\n\n- Understanding videos of 20min+: Qwen2.5-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.\n\n- Agent that can operate your mobiles, robots, etc.: with the abilities of complex reasoning and decision making, Qwen2.5-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.\n\n- Multilingual Support: to serve global users, besides English and Chinese, Qwen2.5-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen2-vl/) and [GitHub repo](https://github.com/QwenLM/Qwen2-VL).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen-2.5-vl-7b-instruct",
    "label": "Qwen2.5-VL 7B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://hyperbolic.xyz/&size=256",
    "description": "Qwen2.5 VL 7B is a multimodal LLM from the Qwen Team with the following key enhancements:\n\n- SoTA understanding of images of various resolution & ratio: Qwen2.5-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.\n\n- Understanding videos of 20min+: Qwen2.5-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.\n\n- Agent that can operate your mobiles, robots, etc.: with the abilities of complex reasoning and decision making, Qwen2.5-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.\n\n- Multilingual Support: to serve global users, besides English and Chinese, Qwen2.5-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen2-vl/) and [GitHub repo](https://github.com/QwenLM/Qwen2-VL).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "google/gemini-flash-1.5-exp",
    "label": "Gemini 1.5 Flash Experimental",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemini 1.5 Flash Experimental is an experimental version of the [Gemini 1.5 Flash](/models/google/gemini-flash-1.5) model.\n\nUsage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms).\n\n#multimodal\n\nNote: This model is experimental and not suited for production use-cases. It may be removed or redirected to another model in the future."
}, {
    "value": "google/gemini-flash-1.5-8b-exp",
    "label": "Gemini 1.5 Flash 8B Experimental",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemini Flash 1.5 8B Experimental is an experimental, 8B parameter version of the [Gemini Flash 1.5](/models/google/gemini-flash-1.5) model.\n\nUsage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms).\n\n#multimodal\n\nNote: This model is currently experimental and not suitable for production use-cases, and may be heavily rate-limited."
}, {
    "value": "lynn/soliloquy-v3",
    "label": "Llama 3 Soliloquy 7B v3 32K",
    "cover": "",
    "description": "Soliloquy v3 is a highly capable roleplaying model designed for immersive, dynamic experiences. Trained on over 2 billion tokens of roleplaying data, Soliloquy v3 boasts a vast knowledge base and rich literary expression, supporting up to 32k context length. It outperforms existing models of comparable size, delivering enhanced roleplaying capabilities.\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "01-ai/yi-1.5-34b-chat",
    "label": "Yi 1.5 34B Chat",
    "cover": "",
    "description": "The Yi series models are large language models trained from scratch by developers at [01.AI](https://01.ai/). This is a predecessor to the Yi 34B model. It is continuously pre-trained on Yi with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse fine-tuning samples.."
}, {
    "value": "ai21/jamba-1-5-mini",
    "label": "Jamba 1.5 Mini",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://ai21.com/&size=256",
    "description": "Jamba 1.5 Mini is the world's first production-grade Mamba-based model, combining SSM and Transformer architectures for a 256K context window and high efficiency.\n\nIt works with 9 languages and can handle various writing and analysis tasks as well as or better than similar small models.\n\nThis model uses less computer memory and works faster with longer texts than previous designs.\n\nRead their [announcement](https://www.ai21.com/blog/announcing-jamba-model-family) to learn more."
}, {
    "value": "ai21/jamba-1-5-large",
    "label": "Jamba 1.5 Large",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://ai21.com/&size=256",
    "description": "Jamba 1.5 Large is part of AI21's new family of open models, offering superior speed, efficiency, and quality.\n\nIt features a 256K effective context window, the longest among open models, enabling improved performance on tasks like document summarization and analysis.\n\nBuilt on a novel SSM-Transformer architecture, it outperforms larger models like Llama 3.1 70B on benchmarks while maintaining resource efficiency.\n\nRead their [announcement](https://www.ai21.com/blog/announcing-jamba-model-family) to learn more."
}, {
    "value": "microsoft/phi-3.5-mini-128k-instruct",
    "label": "Phi-3.5 Mini 128K Instruct",
    "cover": "/images/icons/Azure.svg",
    "description": "Phi-3.5 models are lightweight, state-of-the-art open models. These models were trained with Phi-3 datasets that include both synthetic data and the filtered, publicly available websites data, with a focus on high quality and reasoning-dense properties. Phi-3.5 Mini uses 3.8B parameters, and is a dense decoder-only transformer model using the same tokenizer as [Phi-3 Mini](/models/microsoft/phi-3-mini-128k-instruct).\n\nThe models underwent a rigorous enhancement process, incorporating both supervised fine-tuning, proximal policy optimization, and direct preference optimization to ensure precise instruction adherence and robust safety measures. When assessed against benchmarks that test common sense, language understanding, math, code, long context and logical reasoning, Phi-3.5 models showcased robust and state-of-the-art performance among models with less than 13 billion parameters."
}, {
    "value": "nousresearch/hermes-3-llama-3.1-70b",
    "label": "Hermes 3 70B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://lambdalabs.com/&size=256",
    "description": "Hermes 3 is a generalist language model with many improvements over [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo), including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.\n\nHermes 3 70B is a competitive, if not superior finetune of the [Llama-3.1 70B foundation model](/models/meta-llama/llama-3.1-70b-instruct), focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.\n\nThe Hermes 3 series builds and expands on the Hermes 2 set of capabilities, including more powerful and reliable function calling and structured output capabilities, generalist assistant capabilities, and improved code generation skills."
}, {
    "value": "nousresearch/hermes-3-llama-3.1-405b",
    "label": "Hermes 3 405B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://lambdalabs.com/&size=256",
    "description": "Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coherence, and improvements across the board.\n\nHermes 3 405B is a frontier-level, full-parameter finetune of the Llama-3.1 405B foundation model, focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user.\n\nThe Hermes 3 series builds and expands on the Hermes 2 set of capabilities, including more powerful and reliable function calling and structured output capabilities, generalist assistant capabilities, and improved code generation skills.\n\nHermes 3 is competitive, if not superior, to Llama-3.1 Instruct models at general capabilities, with varying strengths and weaknesses attributable between the two."
}, {
    "value": "aetherwiing/mn-starcannon-12b",
    "label": "Starcannon 12B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": "Starcannon 12B v2 is a creative roleplay and story writing model, based on Mistral Nemo, using [nothingiisreal/mn-celeste-12b](/nothingiisreal/mn-celeste-12b) as a base, with [intervitens/mini-magnum-12b-v1.1](https://huggingface.co/intervitens/mini-magnum-12b-v1.1) merged in using the [TIES](https://arxiv.org/abs/2306.01708) method.\n\nAlthough more similar to Magnum overall, the model remains very creative, with a pleasant writing style. It is recommended for people wanting more variety than Magnum, and yet more verbose prose than Celeste."
}, {
    "value": "sao10k/l3-lunaris-8b",
    "label": "Llama 3 8B Lunaris",
    "cover": "/images/icons/DeepInfra.webp",
    "description": "Lunaris 8B is a versatile generalist and roleplaying model based on Llama 3. It's a strategic merge of multiple models, designed to balance creativity with improved logic and general knowledge.\n\nCreated by [Sao10k](https://huggingface.co/Sao10k), this model aims to offer an improved experience over Stheno v3.2, with enhanced creativity and logical reasoning.\n\nFor best results, use with Llama 3 Instruct context template, temperature 1.4, and min_p 0.1."
}, {
    "value": "openai/gpt-4o-2024-08-06",
    "label": "GPT-4o (2024-08-06)",
    "cover": "/images/icons/OpenAI.svg",
    "description": 'The 2024-08-06 version of GPT-4o offers improved performance in structured outputs, with the ability to supply a JSON schema in the respone_format. Read more [here](https://openai.com/index/introducing-structured-outputs-in-the-api/).\n\nGPT-4o ("o" for "omni") is OpenAI\'s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.\n\nFor benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963966370209)'
}, {
    "value": "01-ai/yi-large-fc",
    "label": "Yi Large FC",
    "cover": "",
    "description": "The Yi Large Function Calling (FC) is a specialized model with capability of tool use. The model can decide whether to call the tool based on the tool definition passed in by the user, and the calling method will be generate in the specified format.\n\nIt's applicable to various production scenarios that require building agents or workflows."
}, {
    "value": "meta-llama/llama-3.1-405b",
    "label": "Llama 3.1 405B (base)",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://hyperbolic.xyz/&size=256",
    "description": "Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This is the base 405B pre-trained version.\n\nIt has demonstrated strong performance compared to leading closed-source models in human evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "nothingiisreal/mn-celeste-12b",
    "label": "Mistral Nemo 12B Celeste",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": "A specialized story writing and roleplaying model based on Mistral's NeMo 12B Instruct. Fine-tuned on curated datasets including Reddit Writing Prompts and Opus Instruct 25K.\n\nThis model excels at creative writing, offering improved NSFW capabilities, with smarter and more active narration. It demonstrates remarkable versatility in both SFW and NSFW scenarios, with strong Out of Character (OOC) steering capabilities, allowing fine-tuned control over narrative direction and character behavior.\n\nCheck out the model's [HuggingFace page](https://huggingface.co/nothingiisreal/MN-12B-Celeste-V1.9) for details on what parameters and prompts work best!"
}, {
    "value": "01-ai/yi-vision",
    "label": "Yi Vision",
    "cover": "",
    "description": "The Yi Vision is a complex visual task models provide high-performance understanding and analysis capabilities based on multiple images.\n\nIt's ideal for scenarios that require analysis and interpretation of images and charts, such as image question answering, chart understanding, OCR, visual reasoning, education, research report understanding, or multilingual document reading."
}, {
    "value": "01-ai/yi-large-turbo",
    "label": "Yi Large Turbo",
    "cover": "",
    "description": "The Yi Large Turbo model is a High Performance and Cost-Effectiveness model offering powerful capabilities at a competitive price.\n\nIt's ideal for a wide range of scenarios, including complex inference and high-quality text generation.\n\nCheck out the [launch announcement](https://01-ai.github.io/blog/01.ai-yi-large-llm-launch) to learn more."
}, {
    "value": "google/gemini-pro-1.5-exp",
    "label": "Gemini 1.5 Pro Experimental",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemini 1.5 Pro Experimental is a bleeding-edge version of the [Gemini 1.5 Pro](/models/google/gemini-pro-1.5) model. Because it's currently experimental, it will be **heavily rate-limited** by Google.\n\nUsage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms).\n\n#multimodal"
}, {
    "value": "meta-llama/llama-3.1-70b-instruct",
    "label": "Llama 3.1 70B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41918314a4184b51788ed_meta-logo.png",
    "description": "Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 70B instruct-tuned version is optimized for high quality dialogue usecases.\n\nIt has demonstrated strong performance compared to leading closed-source models in human evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3.1-8b-instruct:free",
    "label": "Llama 3.1 8B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://crusoe.ai/&size=256",
    "description": "Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 8B instruct-tuned version is fast and efficient.\n\nIt has demonstrated strong performance compared to leading closed-source models in human evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3.1-8b-instruct",
    "label": "Llama 3.1 8B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41918314a4184b51788ed_meta-logo.png",
    "description": "Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 8B instruct-tuned version is fast and efficient.\n\nIt has demonstrated strong performance compared to leading closed-source models in human evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3.1-405b-instruct",
    "label": "Llama 3.1 405B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41918314a4184b51788ed_meta-logo.png",
    "description": "The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, the Meta AI team continues to push the frontier of open-source LLMs.\n\nMeta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 405B instruct-tuned version is optimized for high quality dialogue usecases.\n\nIt has demonstrated strong performance compared to leading closed-source models including GPT-4o and Claude 3.5 Sonnet in evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3-1/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "cognitivecomputations/dolphin-llama-3-70b",
    "label": "Dolphin Llama 3 70B \uD83D\uDC2C",
    "cover": "",
    "description": "Dolphin 2.9 is designed for instruction following, conversational, and coding. This model is a fine-tune of [Llama 3 70B](/models/meta-llama/llama-3-70b-instruct). It demonstrates improvements in instruction, conversation, coding, and function calling abilities, when compared to the original.\n\nUncensored and is stripped of alignment and bias, it requires an external alignment layer for ethical use. Users are cautioned to use this highly compliant model responsibly, as detailed in a blog post about uncensored models at [erichartford.com/uncensored-models](https://erichartford.com/uncensored-models).\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "mistralai/mistral-nemo:free",
    "label": "Mistral Nemo",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA.\n\nThe model is multilingual, supporting English, French, German, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, and Hindi.\n\nIt supports function calling and is released under the Apache 2.0 license."
}, {
    "value": "mistralai/mistral-nemo",
    "label": "Mistral Nemo",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f4197d83b94e540f009dc3_mistral-logo.webp",
    "description": "A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA.\n\nThe model is multilingual, supporting English, French, German, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, and Hindi.\n\nIt supports function calling and is released under the Apache 2.0 license."
}, {
    "value": "mistralai/codestral-mamba",
    "label": "Codestral Mamba",
    "cover": "/images/icons/Mistral.png",
    "description": "A 7.3B parameter Mamba-based model designed for code and reasoning tasks.\n\n- Linear time inference, allowing for theoretically infinite sequence lengths\n- 256k token context window\n- Optimized for quick responses, especially beneficial for code productivity\n- Performs comparably to state-of-the-art transformer models in code and reasoning tasks\n- Available under the Apache 2.0 license for free use, modification, and distribution"
}, {
    "value": "openai/gpt-4.1-mini-2024-07-18",
    "label": "gpt-4.1-mini (2024-07-18)",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-4o mini is OpenAI's newest model after [GPT-4 Omni](/models/openai/gpt-4o), supporting both text and image inputs with text outputs.\n\nAs their most advanced small model, it is many multiples more affordable than other recent frontier models, and more than 60% cheaper than [GPT-3.5 Turbo](/models/openai/gpt-3.5-turbo). It maintains SOTA intelligence, while being significantly more cost-effective.\n\nGPT-4o mini achieves an 82% score on MMLU and presently ranks higher than GPT-4 on chat preferences [common leaderboards](https://arena.lmsys.org/).\n\nCheck out the [launch announcement](https://openai.com/index/gpt-4.1-mini-advancing-cost-efficient-intelligence/) to learn more.\n\n#multimodal"
}, {
    "value": "openai/gpt-4.1-mini",
    "label": "gpt-4.1-mini",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-4o mini is OpenAI's newest model after [GPT-4 Omni](/models/openai/gpt-4o), supporting both text and image inputs with text outputs.\n\nAs their most advanced small model, it is many multiples more affordable than other recent frontier models, and more than 60% cheaper than [GPT-3.5 Turbo](/models/openai/gpt-3.5-turbo). It maintains SOTA intelligence, while being significantly more cost-effective.\n\nGPT-4o mini achieves an 82% score on MMLU and presently ranks higher than GPT-4 on chat preferences [common leaderboards](https://arena.lmsys.org/).\n\nCheck out the [launch announcement](https://openai.com/index/gpt-4.1-mini-advancing-cost-efficient-intelligence/) to learn more.\n\n#multimodal"
}, {
    "value": "qwen/qwen-2-7b-instruct",
    "label": "Qwen 2 7B Instruct",
    "cover": "",
    "description": "Qwen2 7B is a transformer-based model that excels in language understanding, multilingual capabilities, coding, mathematics, and reasoning.\n\nIt features SwiGLU activation, attention QKV bias, and group query attention. It is pretrained on extensive data with supervised finetuning and direct preference optimization.\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen2/) and [GitHub repo](https://github.com/QwenLM/Qwen2).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "google/gemma-2-27b-it",
    "label": "Gemma 2 27B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "Gemma 2 27B by Google is an open model built from the same research and technology used to create the [Gemini models](/models?q=gemini).\n\nGemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning.\n\nSee the [launch announcement](https://blog.google/technology/developers/google-gemma-2/) for more details. Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms)."
}, {
    "value": "alpindale/magnum-72b",
    "label": "Magnum 72B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "From the maker of [Goliath](https://openrouter.ai/models/alpindale/goliath-120b), Magnum 72B is the first in a new family of models designed to achieve the prose quality of the Claude 3 models, notably Opus & Sonnet.\n\nThe model is based on [Qwen2 72B](https://openrouter.ai/models/qwen/qwen-2-72b-instruct) and trained with 55 million tokens of highly curated roleplay (RP) data."
}, {
    "value": "nousresearch/hermes-2-theta-llama-3-8b",
    "label": "Hermes 2 Theta 8B",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "An experimental merge model based on Llama 3, exhibiting a very distinctive style of writing. It combines the the best of [Meta's Llama 3 8B](https://openrouter.ai/models/meta-llama/llama-3-8b-instruct) and Nous Research's [Hermes 2 Pro](https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b).\n\nHermes-2 Θ (theta) was specifically designed with a few capabilities in mind: executing function calls, generating JSON output, and most remarkably, demonstrating metacognitive abilities (contemplating the nature of thought and recognizing the diversity of cognitive processes among individuals)."
}, {
    "value": "google/gemma-2-9b-it:free",
    "label": "Gemma 2 9B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://chutes.ai/&size=256",
    "description": "Gemma 2 9B by Google is an advanced, open-source language model that sets a new standard for efficiency and performance in its size class.\n\nDesigned for a wide variety of tasks, it empowers developers and researchers to build innovative applications, while maintaining accessibility, safety, and cost-effectiveness.\n\nSee the [launch announcement](https://blog.google/technology/developers/google-gemma-2/) for more details. Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms)."
}, {
    "value": "google/gemma-2-9b-it",
    "label": "Gemma 2 9B",
    "cover": "/images/icons/Lepton.png",
    "description": "Gemma 2 9B by Google is an advanced, open-source language model that sets a new standard for efficiency and performance in its size class.\n\nDesigned for a wide variety of tasks, it empowers developers and researchers to build innovative applications, while maintaining accessibility, safety, and cost-effectiveness.\n\nSee the [launch announcement](https://blog.google/technology/developers/google-gemma-2/) for more details. Usage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms)."
}, {
    "value": "sao10k/l3-stheno-8b",
    "label": "Llama 3 Stheno 8B v3.3 32K",
    "cover": "",
    "description": "Stheno 8B 32K is a creative writing/roleplay model from [Sao10k](https://ko-fi.com/sao10k). It was trained at 8K context, then expanded to 32K context.\n\nCompared to older Stheno version, this model is trained on:\n- 2x the amount of creative writing samples\n- Cleaned up roleplaying samples\n- Fewer low quality samples"
}, {
    "value": "ai21/jamba-instruct",
    "label": "Jamba Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://ai21.com/&size=256",
    "description": "The Jamba-Instruct model, introduced by AI21 Labs, is an instruction-tuned variant of their hybrid SSM-Transformer Jamba model, specifically optimized for enterprise applications.\n\n- 256K Context Window: It can process extensive information, equivalent to a 400-page novel, which is beneficial for tasks involving large documents such as financial reports or legal documents\n- Safety and Accuracy: Jamba-Instruct is designed with enhanced safety features to ensure secure deployment in enterprise environments, reducing the risk and cost of implementation\n\nRead their [announcement](https://www.ai21.com/blog/announcing-jamba) to learn more.\n\nJamba has a knowledge cutoff of February 2024."
}, {
    "value": "01-ai/yi-large",
    "label": "Yi Large",
    "cover": "/images/icons/Fireworks.png",
    "description": "The Yi Large model was designed by 01.AI with the following usecases in mind: knowledge search, data classification, human-like chat bots, and customer service.\n\nIt stands out for its multilingual proficiency, particularly in Spanish, Chinese, Japanese, German, and French.\n\nCheck out the [launch announcement](https://01-ai.github.io/blog/01.ai-yi-large-llm-launch) to learn more."
}, {
    "value": "nvidia/nemotron-4-340b-instruct",
    "label": "Nemotron-4 340B Instruct",
    "cover": "",
    "description": "Nemotron-4-340B-Instruct is an English-language chat model optimized for synthetic data generation. This large language model (LLM) is a fine-tuned version of Nemotron-4-340B-Base, designed for single and multi-turn chat use-cases with a 4,096 token context length.\n\nThe base model was pre-trained on 9 trillion tokens from diverse English texts, 50+ natural languages, and 40+ coding languages. The instruct model underwent additional alignment steps:\n\n1. Supervised Fine-tuning (SFT)\n2. Direct Preference Optimization (DPO)\n3. Reward-aware Preference Optimization (RPO)\n\nThe alignment process used approximately 20K human-annotated samples, while 98% of the data for fine-tuning was synthetically generated. Detailed information about the synthetic data generation pipeline is available in the [technical report](https://arxiv.org/html/2406.11704v1)."
}, {
    "value": "anthropic/claude-3.5-sonnet-20240620:beta",
    "label": "Claude 3.5 Sonnet (2024-06-20) (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:\n\n- Coding: Autonomously writes, edits, and runs code with reasoning and troubleshooting\n- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights\n- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone\n- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)\n\nFor the latest version (2024-10-23), check out [Claude 3.5 Sonnet](/anthropic/claude-3.5-sonnet).\n\n#multimodal"
}, {
    "value": "anthropic/claude-3.5-sonnet-20240620",
    "label": "Claude 3.5 Sonnet (2024-06-20)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Sonnet is particularly good at:\n\n- Coding: Autonomously writes, edits, and runs code with reasoning and troubleshooting\n- Data science: Augments human data science expertise; navigates unstructured data while using multiple tools for insights\n- Visual processing: excelling at interpreting charts, graphs, and images, accurately transcribing text to derive insights beyond just the text alone\n- Agentic tasks: exceptional tool use, making it great at agentic tasks (i.e. complex, multi-step problem solving tasks that require engaging with other systems)\n\nFor the latest version (2024-10-23), check out [Claude 3.5 Sonnet](/anthropic/claude-3.5-sonnet).\n\n#multimodal"
}, {
    "value": "sao10k/l3-euryale-70b",
    "label": "Llama 3 Euryale 70B v2.1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://novita.ai/&size=256",
    "description": "Euryale 70B v2.1 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k).\n\n- Better prompt adherence.\n- Better anatomy / spatial awareness.\n- Adapts much better to unique and custom formatting / reply formats.\n- Very creative, lots of unique swipes.\n- Is not restrictive during roleplays."
}, {
    "value": "microsoft/phi-3-medium-4k-instruct",
    "label": "Phi-3 Medium 4K Instruct",
    "cover": "",
    "description": "Phi-3 4K Medium is a powerful 14-billion parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjustments, it excels in tasks involving common sense, mathematics, logical reasoning, and code processing.\n\nAt time of release, Phi-3 Medium demonstrated state-of-the-art performance among lightweight models. In the MMLU-Pro eval, the model even comes close to a Llama3 70B level of performance.\n\nFor 128k context length, try [Phi-3 Medium 128K](/models/microsoft/phi-3-medium-128k-instruct)."
}, {
    "value": "bigcode/starcoder2-15b-instruct",
    "label": "StarCoder2 15B Instruct",
    "cover": "",
    "description": "StarCoder2 15B Instruct excels in coding-related tasks, primarily in Python. It is the first self-aligned open-source LLM developed by BigCode. This model was fine-tuned without any human annotations or distilled data from proprietary LLMs.\n\nThe base model uses [Grouped Query Attention](https://arxiv.org/abs/2305.13245) and was trained using the [Fill-in-the-Middle objective](https://arxiv.org/abs/2207.14255) objective on 4+ trillion tokens."
}, {
    "value": "cognitivecomputations/dolphin-mixtral-8x22b",
    "label": "Dolphin 2.9.2 Mixtral 8x22B \uD83D\uDC2C",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://novita.ai/&size=256",
    "description": "Dolphin 2.9 is designed for instruction following, conversational, and coding. This model is a finetune of [Mixtral 8x22B Instruct](/models/mistralai/mixtral-8x22b-instruct). It features a 64k context length and was fine-tuned with a 16k sequence length using ChatML templates.\n\nThis model is a successor to [Dolphin Mixtral 8x7B](/models/cognitivecomputations/dolphin-mixtral-8x7b).\n\nThe model is uncensored and is stripped of alignment and bias. It requires an external alignment layer for ethical use. Users are cautioned to use this highly compliant model responsibly, as detailed in a blog post about uncensored models at [erichartford.com/uncensored-models](https://erichartford.com/uncensored-models).\n\n#moe #uncensored"
}, {
    "value": "qwen/qwen-2-72b-instruct",
    "label": "Qwen 2 72B Instruct",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "Qwen2 72B is a transformer-based model that excels in language understanding, multilingual capabilities, coding, mathematics, and reasoning.\n\nIt features SwiGLU activation, attention QKV bias, and group query attention. It is pretrained on extensive data with supervised finetuning and direct preference optimization.\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen2/) and [GitHub repo](https://github.com/QwenLM/Qwen2).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "openchat/openchat-8b",
    "label": "OpenChat 3.6 8B",
    "cover": "",
    "description": 'OpenChat 8B is a library of open-source language models, fine-tuned with "C-RLFT (Conditioned Reinforcement Learning Fine-Tuning)" - a strategy inspired by offline reinforcement learning. It has been trained on mixed-quality data without preference labels.\n\nIt outperforms many similarly sized models including [Llama 3 8B Instruct](/models/meta-llama/llama-3-8b-instruct) and various fine-tuned models. It excels in general conversation, coding assistance, and mathematical reasoning.\n\n- For OpenChat fine-tuned on Mistral 7B, check out [OpenChat 7B](/models/openchat/openchat-7b).\n- For OpenChat fine-tuned on Llama 8B, check out [OpenChat 8B](/models/openchat/openchat-8b).\n\n#open-source'
}, {
    "value": "mistralai/mistral-7b-instruct-v0.3",
    "label": "Mistral 7B Instruct v0.3",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f4197d83b94e540f009dc3_mistral-logo.webp",
    "description": "A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.\n\nAn improved version of [Mistral 7B Instruct v0.2](/models/mistralai/mistral-7b-instruct-v0.2), with the following changes:\n\n- Extended vocabulary to 32768\n- Supports v3 Tokenizer\n- Supports function calling\n\nNOTE: Support for function calling depends on the provider."
}, {
    "value": "nousresearch/hermes-2-pro-llama-3-8b",
    "label": "Hermes 2 Pro - Llama-3 8B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://lambdalabs.com/&size=256",
    "description": "Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and JSON Mode dataset developed in-house."
}, {
    "value": "mistralai/mistral-7b-instruct:free",
    "label": "Mistral 7B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f4197d83b94e540f009dc3_mistral-logo.webp",
    "description": "A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.\n\n*Mistral 7B Instruct has multiple version variants, and this is intended to be the latest version.*"
}, {
    "value": "mistralai/mistral-7b-instruct",
    "label": "Mistral 7B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f4197d83b94e540f009dc3_mistral-logo.webp",
    "description": "A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.\n\n*Mistral 7B Instruct has multiple version variants, and this is intended to be the latest version.*"
}, {
    "value": "microsoft/phi-3-mini-128k-instruct",
    "label": "Phi-3 Mini 128K Instruct",
    "cover": "/images/icons/Azure.svg",
    "description": "Phi-3 Mini is a powerful 3.8B parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjustments, it excels in tasks involving common sense, mathematics, logical reasoning, and code processing.\n\nAt time of release, Phi-3 Medium demonstrated state-of-the-art performance among lightweight models. This model is static, trained on an offline dataset with an October 2023 cutoff date."
}, {
    "value": "microsoft/phi-3-medium-128k-instruct",
    "label": "Phi-3 Medium 128K Instruct",
    "cover": "/images/icons/Azure.svg",
    "description": "Phi-3 128K Medium is a powerful 14-billion parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjustments, it excels in tasks involving common sense, mathematics, logical reasoning, and code processing.\n\nAt time of release, Phi-3 Medium demonstrated state-of-the-art performance among lightweight models. In the MMLU-Pro eval, the model even comes close to a Llama3 70B level of performance.\n\nFor 4k context length, try [Phi-3 Medium 4K](/models/microsoft/phi-3-medium-4k-instruct)."
}, {
    "value": "neversleep/llama-3-lumimaid-70b",
    "label": "Llama 3 Lumimaid 70B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "The NeverSleep team is back, with a Llama 3 70B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessary.\n\nTo enhance it's overall intelligence and chat capability, roughly 40% of the training data was not roleplay. This provides a breadth of knowledge to access, while still keeping roleplay as the primary strength.\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "deepseek/deepseek-chat-v2.5",
    "label": "DeepSeek V2.5",
    "cover": "/images/icons/DeepSeek.png",
    "description": "DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. The new model integrates the general and coding abilities of the two previous versions. For model details, please visit [DeepSeek-V2 page](https://github.com/deepseek-ai/DeepSeek-V2) for more information."
}, {
    "value": "google/gemini-flash-1.5",
    "label": "Gemini 1.5 Flash ",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Gemini 1.5 Flash is a foundation model that performs well at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video. It's adept at processing visual and text inputs such as photographs, documents, infographics, and screenshots.\n\nGemini 1.5 Flash is designed for high-volume, high-frequency tasks where cost and latency matter. On most common tasks, Flash achieves comparable quality to other Gemini Pro models at a significantly reduced cost. Flash is well-suited for applications like chat assistants and on-demand content generation where speed and scale matter.\n\nUsage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms).\n\n#multimodal"
}, {
    "value": "deepseek/deepseek-coder",
    "label": "DeepSeek-Coder-V2",
    "cover": "/images/icons/DeepSeek.png",
    "description": "DeepSeek-Coder-V2, an open-source Mixture-of-Experts (MoE) code language model. It is further pre-trained from an intermediate checkpoint of DeepSeek-V2 with additional 6 trillion tokens.\n\nThe original V1 model was trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. It was pre-trained on project-level code corpus by employing a extra fill-in-the-blank task."
}, {
    "value": "meta-llama/llama-3-70b",
    "label": "Llama 3 70B (Base)",
    "cover": "",
    "description": "Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This is the base 70B pre-trained version.\n\nIt has demonstrated strong performance compared to leading closed-source models in human evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "openai/gpt-4o-2024-05-13",
    "label": "GPT-4o (2024-05-13)",
    "cover": "/images/icons/OpenAI.svg",
    "description": 'GPT-4o ("o" for "omni") is OpenAI\'s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.\n\nFor benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963966370209)\n\n#multimodal'
}, {
    "value": "meta-llama/llama-guard-2-8b",
    "label": "LlamaGuard 2 8B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "This safeguard model has 8B parameters and is based on the Llama 3 family. Just like is predecessor, [LlamaGuard 1](https://huggingface.co/meta-llama/LlamaGuard-7b), it can do both prompt and response classification.\n\nLlamaGuard 2 acts as a normal LLM would, generating text that indicates whether the given input/output is safe/unsafe. If deemed unsafe, it will also share the content categories violated.\n\nFor best results, please use raw prompt input or the `/completions` endpoint, instead of the chat API.\n\nIt has demonstrated strong performance compared to leading closed-source models in human evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "openai/gpt-4o",
    "label": "GPT-4o",
    "cover": "/images/icons/OpenAI.svg",
    "description": 'GPT-4o ("o" for "omni") is OpenAI\'s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.\n\nFor benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963966370209)\n\n#multimodal'
}, {
    "value": "openai/gpt-4o:extended",
    "label": "GPT-4o (extended)",
    "cover": "/images/icons/OpenAI.svg",
    "description": 'GPT-4o ("o" for "omni") is OpenAI\'s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/openai/gpt-4-turbo) while being twice as fast and 50% more cost-effective. GPT-4o also offers improved performance in processing non-English languages and enhanced visual capabilities.\n\nFor benchmarking against other models, it was briefly called ["im-also-a-good-gpt2-chatbot"](https://twitter.com/LiamFedus/status/1790064963966370209)\n\n#multimodal'
}, {
    "value": "meta-llama/llama-3-8b",
    "label": "Llama 3 8B (Base)",
    "cover": "",
    "description": "Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This is the base 8B pre-trained version.\n\nIt has demonstrated strong performance compared to leading closed-source models in human evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "liuhaotian/llava-yi-34b",
    "label": "LLaVA v1.6 34B",
    "cover": "",
    "description": "LLaVA Yi 34B is an open-source model trained by fine-tuning LLM on multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture. Base LLM: [NousResearch/Nous-Hermes-2-Yi-34B](/models/nousresearch/nous-hermes-yi-34b)\n\nIt was trained in December 2023."
}, {
    "value": "allenai/olmo-7b-instruct",
    "label": "OLMo 7B Instruct",
    "cover": "",
    "description": "OLMo 7B Instruct by the Allen Institute for AI is a model finetuned for question answering. It demonstrates **notable performance** across multiple benchmarks including TruthfulQA and ToxiGen.\n\n**Open Source**: The model, its code, checkpoints, logs are released under the [Apache 2.0 license](https://choosealicense.com/licenses/apache-2.0).\n\n- [Core repo (training, inference, fine-tuning etc.)](https://github.com/allenai/OLMo)\n- [Evaluation code](https://github.com/allenai/OLMo-Eval)\n- [Further fine-tuning code](https://github.com/allenai/open-instruct)\n- [Paper](https://arxiv.org/abs/2402.00838)\n- [Technical blog post](https://blog.allenai.org/olmo-open-language-model-87ccfc95f580)\n- [W&B Logs](https://wandb.ai/ai2-llm/OLMo-7B/reports/OLMo-7B--Vmlldzo2NzQyMzk5)"
}, {
    "value": "qwen/qwen-7b-chat",
    "label": "Qwen 1.5 7B Chat",
    "cover": "",
    "description": "Qwen1.5 7B is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include:\n\n- Significant performance improvement in human preference for chat models\n- Multilingual support of both base and chat models\n- Stable support of 32K context length for models of all sizes\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen1.5/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen-4b-chat",
    "label": "Qwen 1.5 4B Chat",
    "cover": "",
    "description": "Qwen1.5 4B is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include:\n\n- Significant performance improvement in human preference for chat models\n- Multilingual support of both base and chat models\n- Stable support of 32K context length for models of all sizes\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen1.5/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen-32b-chat",
    "label": "Qwen 1.5 32B Chat",
    "cover": "",
    "description": "Qwen1.5 32B is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include:\n\n- Significant performance improvement in human preference for chat models\n- Multilingual support of both base and chat models\n- Stable support of 32K context length for models of all sizes\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen1.5/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen-72b-chat",
    "label": "Qwen 1.5 72B Chat",
    "cover": "",
    "description": "Qwen1.5 72B is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include:\n\n- Significant performance improvement in human preference for chat models\n- Multilingual support of both base and chat models\n- Stable support of 32K context length for models of all sizes\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen1.5/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen-110b-chat",
    "label": "Qwen 1.5 110B Chat",
    "cover": "",
    "description": "Qwen1.5 110B is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include:\n\n- Significant performance improvement in human preference for chat models\n- Multilingual support of both base and chat models\n- Stable support of 32K context length for models of all sizes\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen1.5/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "qwen/qwen-14b-chat",
    "label": "Qwen 1.5 14B Chat",
    "cover": "",
    "description": "Qwen1.5 14B is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data. In comparison with the previous released Qwen, the improvements include:\n\n- Significant performance improvement in human preference for chat models\n- Multilingual support of both base and chat models\n- Stable support of 32K context length for models of all sizes\n\nFor more details, see this [blog post](https://qwenlm.github.io/blog/qwen1.5/) and [GitHub repo](https://github.com/QwenLM/Qwen1.5).\n\nUsage of this model is subject to [Tongyi Qianwen LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen1.5-110B-Chat/blob/main/LICENSE)."
}, {
    "value": "neversleep/llama-3-lumimaid-8b:extended",
    "label": "Llama 3 Lumimaid 8B (extended)",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "The NeverSleep team is back, with a Llama 3 8B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessary.\n\nTo enhance it's overall intelligence and chat capability, roughly 40% of the training data was not roleplay. This provides a breadth of knowledge to access, while still keeping roleplay as the primary strength.\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "neversleep/llama-3-lumimaid-8b",
    "label": "Llama 3 Lumimaid 8B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "The NeverSleep team is back, with a Llama 3 8B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessary.\n\nTo enhance it's overall intelligence and chat capability, roughly 40% of the training data was not roleplay. This provides a breadth of knowledge to access, while still keeping roleplay as the primary strength.\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "snowflake/snowflake-arctic-instruct",
    "label": "Arctic Instruct",
    "cover": "",
    "description": "Arctic is a dense-MoE Hybrid transformer architecture pre-trained from scratch by the Snowflake AI Research Team. Arctic combines a 10B dense transformer model with a residual 128x3.66B MoE MLP resulting in 480B total and 17B active parameters chosen using a top-2 gating.\n\nTo read more about this model's release, [click here](https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/)."
}, {
    "value": "fireworks/firellava-13b",
    "label": "FireLLaVA 13B",
    "cover": "",
    "description": "A blazing fast vision-language model, FireLLaVA quickly understands both text and images. It achieves impressive chat skills in tests, and was designed to mimic multimodal GPT-4.\n\nThe first commercially permissive open source LLaVA model, trained entirely on open source LLM generated instruction following data."
}, {
    "value": "lynn/soliloquy-l3",
    "label": "Llama 3 Soliloquy 8B v2",
    "cover": "",
    "description": "Soliloquy-L3 v2 is a fast, highly capable roleplaying model designed for immersive, dynamic experiences. Trained on over 250 million tokens of roleplaying data, Soliloquy-L3 has a vast knowledge base, rich literary expression, and support for up to 24k context length. It outperforms existing ~13B models, delivering enhanced roleplaying capabilities.\n\nUsage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "sao10k/fimbulvetr-11b-v2",
    "label": "Fimbulvetr 11B v2",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://featherless.ai/&size=256",
    "description": "Creative writing model, routed with permission. It's fast, it keeps the conversation going, and it stays in character.\n\nIf you submit a raw prompt, you can use Alpaca or Vicuna formats."
}, {
    "value": "meta-llama/llama-3-8b-instruct",
    "label": "Llama 3 8B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41918314a4184b51788ed_meta-logo.png",
    "description": "Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This 8B instruct-tuned version was optimized for high quality dialogue usecases.\n\nIt has demonstrated strong performance compared to leading closed-source models in human evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "meta-llama/llama-3-70b-instruct",
    "label": "Llama 3 70B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41918314a4184b51788ed_meta-logo.png",
    "description": "Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This 70B instruct-tuned version was optimized for high quality dialogue usecases.\n\nIt has demonstrated strong performance compared to leading closed-source models in human evaluations.\n\nTo read more about the model release, [click here](https://ai.meta.com/blog/meta-llama-3/). Usage of this model is subject to [Meta's Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/)."
}, {
    "value": "mistralai/mixtral-8x22b-instruct",
    "label": "Mixtral 8x22B Instruct",
    "cover": "/images/icons/Fireworks.png",
    "description": "Mistral's official instruct fine-tuned version of [Mixtral 8x22B](/models/mistralai/mixtral-8x22b). It uses 39B active parameters out of 141B, offering unparalleled cost efficiency for its size. Its strengths include:\n- strong math, coding, and reasoning\n- large context length (64k)\n- fluency in English, French, Italian, German, and Spanish\n\nSee benchmarks on the launch announcement [here](https://mistral.ai/news/mixtral-8x22b/).\n#moe"
}, {
    "value": "microsoft/wizardlm-2-8x22b",
    "label": "WizardLM-2 8x22B",
    "cover": "/images/icons/DeepInfra.webp",
    "description": "WizardLM-2 8x22B is Microsoft AI's most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing state-of-the-art opensource models.\n\nIt is an instruct finetune of [Mixtral 8x22B](/models/mistralai/mixtral-8x22b).\n\nTo read more about the model release, [click here](https://wizardlm.github.io/WizardLM2/).\n\n#moe"
}, {
    "value": "microsoft/wizardlm-2-7b",
    "label": "WizardLM-2 7B",
    "cover": "/images/icons/Lepton.png",
    "description": "WizardLM-2 7B is the smaller variant of Microsoft AI's latest Wizard model. It is the fastest and achieves comparable performance with existing 10x larger opensource leading models\n\nIt is a finetune of [Mistral 7B Instruct](/models/mistralai/mistral-7b-instruct), using the same technique as [WizardLM-2 8x22B](/models/microsoft/wizardlm-2-8x22b).\n\nTo read more about the model release, [click here](https://wizardlm.github.io/WizardLM2/).\n\n#moe"
}, {
    "value": "huggingfaceh4/zephyr-orpo-141b-a35b",
    "label": "Zephyr 141B-A35B",
    "cover": "",
    "description": "Zephyr 141B-A35B is A Mixture of Experts (MoE) model with 141B total parameters and 35B active parameters. Fine-tuned on a mix of publicly available, synthetic datasets.\n\nIt is an instruct finetune of [Mixtral 8x22B](/models/mistralai/mixtral-8x22b).\n\n#moe"
}, {
    "value": "mistralai/mixtral-8x22b",
    "label": "Mixtral 8x22B (base)",
    "cover": "",
    "description": "Mixtral 8x22B is a large-scale language model from Mistral AI. It consists of 8 experts, each 22 billion parameters, with each token using 2 experts at a time.\n\nIt was released via [X](https://twitter.com/MistralAI/status/1777869263778291896).\n\n#moe"
}, {
    "value": "google/gemini-pro-1.5",
    "label": "Gemini 1.5 Pro",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Google's latest multimodal model, supports image and video[0] in text or chat prompts.\n\nOptimized for language tasks including:\n\n- Code generation\n- Text generation\n- Text editing\n- Problem solving\n- Recommendations\n- Information extraction\n- Data extraction or generation\n- AI agents\n\nUsage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms).\n\n* [0]: Video input is not available through OpenRouter at this time."
}, {
    "value": "openai/gpt-4-turbo",
    "label": "GPT-4 Turbo",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling.\n\nTraining data: up to December 2023."
}, {
    "value": "cohere/command-r-plus",
    "label": "Command R+",
    "cover": "/images/icons/Cohere.png",
    "description": "Command R+ is a new, 104B-parameter LLM from Cohere. It's useful for roleplay, general consumer usecases, and Retrieval Augmented Generation (RAG).\n\nIt offers multilingual support for ten key languages to facilitate global business operations. See benchmarks and the launch post [here](https://txt.cohere.com/command-r-plus-microsoft-azure/).\n\nUse of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement)."
}, {
    "value": "cohere/command-r-plus-04-2024",
    "label": "Command R+ (04-2024)",
    "cover": "/images/icons/Cohere.png",
    "description": "Command R+ is a new, 104B-parameter LLM from Cohere. It's useful for roleplay, general consumer usecases, and Retrieval Augmented Generation (RAG).\n\nIt offers multilingual support for ten key languages to facilitate global business operations. See benchmarks and the launch post [here](https://txt.cohere.com/command-r-plus-microsoft-azure/).\n\nUse of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement)."
}, {
    "value": "databricks/dbrx-instruct",
    "label": "DBRX 132B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f41a22506fc209003d5722_databricks-logo.webp",
    "description": "DBRX is a new open source large language model developed by Databricks. At 132B, it outperforms existing open source LLMs like Llama 2 70B and [Mixtral-8x7b](/models/mistralai/mixtral-8x7b) on standard industry benchmarks for language understanding, programming, math, and logic.\n\nIt uses a fine-grained mixture-of-experts (MoE) architecture. 36B parameters are active on any input. It was pre-trained on 12T tokens of text and code data. Compared to other open MoE models like Mixtral-8x7B and Grok-1, DBRX is fine-grained, meaning it uses a larger number of smaller experts.\n\nSee the launch announcement and benchmark results [here](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm).\n\n#moe"
}, {
    "value": "sophosympatheia/midnight-rose-70b",
    "label": "Midnight Rose 70B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://novita.ai/&size=256",
    "description": "A merge with a complex family tree, this model was crafted for roleplaying and storytelling. Midnight Rose is a successor to Rogue Rose and Aurora Nights and improves upon them both. It wants to produce lengthy output by default and is the best creative writing merge produced so far by sophosympatheia.\n\nDescending from earlier versions of Midnight Rose and [Wizard Tulu Dolphin 70B](https://huggingface.co/sophosympatheia/Wizard-Tulu-Dolphin-70B-v1.0), it inherits the best qualities of each."
}, {
    "value": "cohere/command",
    "label": "Command",
    "cover": "/images/icons/Cohere.png",
    "description": "Command is an instruction-following conversational model that performs language tasks with high quality, more reliably and with a longer context than our base generative models.\n\nUse of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement)."
}, {
    "value": "cohere/command-r",
    "label": "Command R",
    "cover": "/images/icons/Cohere.png",
    "description": "Command-R is a 35B parameter model that performs conversational language tasks at a higher quality, more reliably, and with a longer context than previous models. It can be used for complex workflows like code generation, retrieval augmented generation (RAG), tool use, and agents.\n\nRead the launch post [here](https://txt.cohere.com/command-r/).\n\nUse of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement)."
}, {
    "value": "anthropic/claude-3-haiku:beta",
    "label": "Claude 3 Haiku (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3 Haiku is Anthropic's fastest and most compact model for\nnear-instant responsiveness. Quick and accurate targeted performance.\n\nSee the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-haiku)\n\n#multimodal"
}, {
    "value": "anthropic/claude-3-haiku",
    "label": "Claude 3 Haiku",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3 Haiku is Anthropic's fastest and most compact model for\nnear-instant responsiveness. Quick and accurate targeted performance.\n\nSee the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-haiku)\n\n#multimodal"
}, {
    "value": "anthropic/claude-3-opus:beta",
    "label": "Claude 3 Opus (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3 Opus is Anthropic's most powerful model for highly complex tasks. It boasts top-level performance, intelligence, fluency, and understanding.\n\nSee the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-family)\n\n#multimodal"
}, {
    "value": "anthropic/claude-3-opus",
    "label": "Claude 3 Opus",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3 Opus is Anthropic's most powerful model for highly complex tasks. It boasts top-level performance, intelligence, fluency, and understanding.\n\nSee the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-family)\n\n#multimodal"
}, {
    "value": "anthropic/claude-3-sonnet:beta",
    "label": "Claude 3 Sonnet (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3 Sonnet is an ideal balance of intelligence and speed for enterprise workloads. Maximum utility at a lower price, dependable, balanced for scaled deployments.\n\nSee the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-family)\n\n#multimodal"
}, {
    "value": "anthropic/claude-3-sonnet",
    "label": "Claude 3 Sonnet",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 3 Sonnet is an ideal balance of intelligence and speed for enterprise workloads. Maximum utility at a lower price, dependable, balanced for scaled deployments.\n\nSee the launch announcement and benchmark results [here](https://www.anthropic.com/news/claude-3-family)\n\n#multimodal"
}, {
    "value": "cohere/command-r-03-2024",
    "label": "Command R (03-2024)",
    "cover": "/images/icons/Cohere.png",
    "description": "Command-R is a 35B parameter model that performs conversational language tasks at a higher quality, more reliably, and with a longer context than previous models. It can be used for complex workflows like code generation, retrieval augmented generation (RAG), tool use, and agents.\n\nRead the launch post [here](https://txt.cohere.com/command-r/).\n\nUse of this model is subject to Cohere's [Usage Policy](https://docs.cohere.com/docs/usage-policy) and [SaaS Agreement](https://cohere.com/saas-agreement)."
}, {
    "value": "google/gemma-7b-it",
    "label": "Gemma 7B",
    "cover": "",
    "description": "Gemma by Google is an advanced, open-source language model family, leveraging the latest in decoder-only, text-to-text technology. It offers English language capabilities across text generation tasks like question answering, summarization, and reasoning. The Gemma 7B variant is comparable in performance to leading open source models.\n\nUsage of Gemma is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms)."
}, {
    "value": "nousresearch/nous-hermes-2-mistral-7b-dpo",
    "label": "Hermes 2 Mistral 7B DPO",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "This is the flagship 7B Hermes model, a Direct Preference Optimization (DPO) of [Teknium/OpenHermes-2.5-Mistral-7B](/models/teknium/openhermes-2.5-mistral-7b). It shows improvement across the board on all benchmarks tested - AGIEval, BigBench Reasoning, GPT4All, and TruthfulQA.\n\nThe model prior to DPO was trained on 1,000,000 instructions/chats of GPT-4 quality or better, primarily synthetic data as well as other high quality datasets."
}, {
    "value": "meta-llama/codellama-70b-instruct",
    "label": "CodeLlama 70B Instruct",
    "cover": "",
    "description": "Code Llama is a family of large language models for code. This one is based on [Llama 2 70B](/models/meta-llama/llama-2-70b-chat) and provides zero-shot instruction-following ability for programming tasks."
}, {
    "value": "recursal/eagle-7b",
    "label": "Eagle 7B",
    "cover": "",
    "description": 'Eagle 7B is trained on 1.1 Trillion Tokens across 100+ world languages (70% English, 15% multilang, 15% code).\n\n- Built on the [RWKV-v5](/models?q=rwkv) architecture (a linear transformer with 10-100x+ lower inference cost)\n- Ranks as the world\'s greenest 7B model (per token)\n- Outperforms all 7B class models in multi-lingual benchmarks\n- Approaches Falcon (1.5T), LLaMA2 (2T), Mistral (>2T?) level of performance in English evals\n- Trade blows with MPT-7B (1T) in English evals\n- All while being an ["Attention-Free Transformer"](https://www.isattentionallyouneed.com/)\n\nEagle 7B models are provided for free, by [Recursal.AI](https://recursal.ai), for the beta period till end of March 2024\n\nFind out more [here](https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers)\n\n[rnn](/models?q=rwkv)'
}, {
    "value": "openai/gpt-3.5-turbo-0613",
    "label": "GPT-3.5 Turbo (older v0613)",
    "cover": "/images/icons/Azure.svg",
    "description": "GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.\n\nTraining data up to Sep 2021."
}, {
    "value": "openai/gpt-4-turbo-preview",
    "label": "GPT-4 Turbo Preview",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The preview GPT-4 model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Training data: up to Dec 2023.\n\n**Note:** heavily rate limited by OpenAI while in preview."
}, {
    "value": "01-ai/yi-34b-200k",
    "label": "Yi 34B 200K",
    "cover": "",
    "description": "The Yi series models are large language models trained from scratch by developers at [01.AI](https://01.ai/). This version was trained on a large context length, allowing ~200k words (1000 paragraphs) of combined input and output."
}, {
    "value": "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
    "label": "Hermes 2 Mixtral 8x7B DPO",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "Nous Hermes 2 Mixtral 8x7B DPO is the new flagship Nous Research model trained over the [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b).\n\nThe model was trained on over 1,000,000 entries of primarily [GPT-4](/models/openai/gpt-4) generated data, as well as other high quality data from open datasets across the AI landscape, achieving state of the art performance on a variety of tasks.\n\n#moe"
}, {
    "value": "nousresearch/nous-hermes-2-mixtral-8x7b-sft",
    "label": "Hermes 2 Mixtral 8x7B SFT",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "Nous Hermes 2 Mixtral 8x7B SFT is the supervised finetune only version of [the Nous Research model](/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo) trained over the [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b).\n\nThe model was trained on over 1,000,000 entries of primarily GPT-4 generated data, as well as other high quality data from open datasets across the AI landscape, achieving state of the art performance on a variety of tasks.\n\n#moe"
}, {
    "value": "mistralai/mistral-medium",
    "label": "Mistral Medium",
    "cover": "/images/icons/Mistral.png",
    "description": "This is Mistral AI's closed-source, medium-sided model. It's powered by a closed-source prototype and excels at reasoning, code, JSON, chat, and more. In benchmarks, it compares with many of the flagship models of other companies."
}, {
    "value": "mistralai/mistral-tiny",
    "label": "Mistral Tiny",
    "cover": "/images/icons/Mistral.png",
    "description": 'Note: This model is being deprecated. Recommended replacement is the newer [Ministral 8B](/mistral/ministral-8b)\n\nThis model is currently powered by Mistral-7B-v0.2, and incorporates a "better" fine-tuning than [Mistral 7B](/models/mistralai/mistral-7b-instruct-v0.1), inspired by community work. It\'s best used for large batch processing tasks where cost is a significant factor but reasoning capabilities are not crucial.'
}, {
    "value": "mistralai/mistral-small",
    "label": "Mistral Small",
    "cover": "/images/icons/Mistral.png",
    "description": "With 22 billion parameters, Mistral Small v24.09 offers a convenient mid-point between (Mistral NeMo 12B)[/mistralai/mistral-nemo] and (Mistral Large 2)[/mistralai/mistral-large], providing a cost-effective solution that can be deployed across various platforms and environments. It has better reasoning, exhibits more capabilities, can produce and reason about code, and is multiligual, supporting English, French, German, Italian, and Spanish."
}, {
    "value": "austism/chronos-hermes-13b",
    "label": "Chronos Hermes 13B v2",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "A 75/25 merge of [Chronos 13b v2](https://huggingface.co/elinas/chronos-13b-v2) and [Nous Hermes Llama2 13b](/models/nousresearch/nous-hermes-llama2-13b). This offers the imaginative writing style of Chronos while retaining coherency. Outputs are long and use exceptional prose. #merge"
}, {
    "value": "jondurbin/bagel-34b",
    "label": "Bagel 34B v0.2",
    "cover": "",
    "description": "An experimental fine-tune of [Yi 34b 200k](/models/01-ai/yi-34b-200k) using [bagel](https://github.com/jondurbin/bagel). This is the version of the fine-tune before direct preference optimization (DPO) has been applied. DPO performs better on benchmarks, but this version is likely better for creative writing, roleplay, etc."
}, {
    "value": "nousresearch/nous-hermes-yi-34b",
    "label": "Hermes 2 Yi 34B",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "Nous Hermes 2 Yi 34B was trained on 1,000,000 entries of primarily GPT-4 generated data, as well as other high quality data from open datasets across the AI landscape.\n\nNous-Hermes 2 on Yi 34B outperforms all Nous-Hermes & Open-Hermes models of the past, achieving new heights in all benchmarks for a Nous Research LLM as well as surpassing many popular finetunes."
}, {
    "value": "neversleep/noromaid-mixtral-8x7b-instruct",
    "label": "Noromaid Mixtral 8x7B Instruct",
    "cover": "",
    "description": "This model was trained for 8h(v1) + 8h(v2) + 12h(v3) on customized modified datasets, focusing on RP, uncensoring, and a modified version of the Alpaca prompting (that was already used in LimaRP), which should be at the same conversational level as ChatLM or Llama2-Chat without adding any additional special tokens."
}, {
    "value": "mistralai/mistral-7b-instruct-v0.2",
    "label": "Mistral 7B Instruct v0.2",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context length.\n\nAn improved version of [Mistral 7B Instruct](/modelsmistralai/mistral-7b-instruct-v0.1), with the following changes:\n\n- 32k context window (vs 8k context in v0.1)\n- Rope-theta = 1e6\n- No Sliding-Window Attention"
}, {
    "value": "cognitivecomputations/dolphin-mixtral-8x7b",
    "label": "Dolphin 2.6 Mixtral 8x7B \uD83D\uDC2C",
    "cover": "/images/icons/Lepton.png",
    "description": "This is a 16k context fine-tune of [Mixtral-8x7b](/models/mistralai/mixtral-8x7b). It excels in coding tasks due to extensive training with coding data and is known for its obedience, although it lacks DPO tuning.\n\nThe model is uncensored and is stripped of alignment and bias. It requires an external alignment layer for ethical use. Users are cautioned to use this highly compliant model responsibly, as detailed in a blog post about uncensored models at [erichartford.com/uncensored-models](https://erichartford.com/uncensored-models).\n\n#moe #uncensored"
}, {
    "value": "google/gemini-pro",
    "label": "Gemini Pro 1.0",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Google's flagship text generation model. Designed to handle natural language tasks, multiturn text and code chat, and code generation.\n\nSee the benchmarks and prompting guidelines from [Deepmind](https://deepmind.google/technologies/gemini/).\n\nUsage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms)."
}, {
    "value": "google/gemini-pro-vision",
    "label": "Gemini Pro Vision 1.0",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "Google's flagship multimodal model, supporting image and video in text or chat prompts for a text or code response.\n\nSee the benchmarks and prompting guidelines from [Deepmind](https://deepmind.google/technologies/gemini/).\n\nUsage of Gemini is subject to Google's [Gemini Terms of Use](https://ai.google.dev/terms).\n\n#multimodal"
}, {
    "value": "recursal/rwkv-5-3b-ai-town",
    "label": "RWKV v5 3B AI Town",
    "cover": "",
    "description": 'This is an [RWKV 3B model](/models/rwkv/rwkv-5-world-3b) finetuned specifically for the [AI Town](https://github.com/a16z-infra/ai-town) project.\n\n[RWKV](https://wiki.rwkv.com) is an RNN (recurrent neural network) with transformer-level performance. It aims to combine the best of RNNs and transformers - great performance, fast inference, low VRAM, fast training, "infinite" context length, and free sentence embedding.\n\nRWKV 3B models are provided for free, by Recursal.AI, for the beta period. More details [here](https://substack.recursal.ai/p/public-rwkv-3b-model-via-openrouter).\n\n#rnn'
}, {
    "value": "rwkv/rwkv-5-world-3b",
    "label": "RWKV v5 World 3B",
    "cover": "",
    "description": '[RWKV](https://wiki.rwkv.com) is an RNN (recurrent neural network) with transformer-level performance. It aims to combine the best of RNNs and transformers - great performance, fast inference, low VRAM, fast training, "infinite" context length, and free sentence embedding.\n\nRWKV-5 is trained on 100+ world languages (70% English, 15% multilang, 15% code).\n\nRWKV 3B models are provided for free, by Recursal.AI, for the beta period. More details [here](https://substack.recursal.ai/p/public-rwkv-3b-model-via-openrouter).\n\n#rnn'
}, {
    "value": "mistralai/mixtral-8x7b-instruct",
    "label": "Mixtral 8x7B Instruct",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f4197d83b94e540f009dc3_mistral-logo.webp",
    "description": "Mixtral 8x7B Instruct is a pretrained generative Sparse Mixture of Experts, by Mistral AI, for chat and instruction use. Incorporates 8 experts (feed-forward networks) for a total of 47 billion parameters.\n\nInstruct model fine-tuned by Mistral. #moe"
}, {
    "value": "togethercomputer/stripedhyena-nous-7b",
    "label": "StripedHyena Nous 7B",
    "cover": "",
    "description": "This is the chat model variant of the [StripedHyena series](/models?q=stripedhyena) developed by Together in collaboration with Nous Research.\n\nStripedHyena uses a new architecture that competes with traditional Transformers, particularly in long-context data processing. It combines attention mechanisms with gated convolutions for improved speed, efficiency, and scaling. This model marks a significant advancement in AI architecture for sequence modeling tasks."
}, {
    "value": "togethercomputer/stripedhyena-hessian-7b",
    "label": "StripedHyena Hessian 7B (base)",
    "cover": "",
    "description": "This is the base model variant of the [StripedHyena series](/models?q=stripedhyena), developed by Together.\n\nStripedHyena uses a new architecture that competes with traditional Transformers, particularly in long-context data processing. It combines attention mechanisms with gated convolutions for improved speed, efficiency, and scaling. This model marks an advancement in AI architecture for sequence modeling tasks."
}, {
    "value": "koboldai/psyfighter-13b-2",
    "label": "Psyfighter v2 13B",
    "cover": "",
    "description": "The v2 of [Psyfighter](/models/jebcarter/psyfighter-13b) - a merged model created by the KoboldAI community members Jeb Carter and TwistedShadows, made possible thanks to the KoboldAI merge request service.\n\nThe intent was to add medical data to supplement the model's fictional ability with more details on anatomy and mental states. This model should not be used for medical advice or therapy because of its high likelihood of pulling in fictional data.\n\nIt's a merge between:\n\n- [KoboldAI/LLaMA2-13B-Tiefighter](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter)\n- [Doctor-Shotgun/cat-v1.0-13b](https://huggingface.co/Doctor-Shotgun/cat-v1.0-13b)\n- [Doctor-Shotgun/llama-2-13b-chat-limarp-v2-merged](https://huggingface.co/Doctor-Shotgun/llama-2-13b-chat-limarp-v2-merged).\n\n#merge"
}, {
    "value": "gryphe/mythomist-7b",
    "label": "MythoMist 7B",
    "cover": "",
    "description": "From the creator of [MythoMax](/models/gryphe/mythomax-l2-13b), merges a suite of models to reduce word anticipation, ministrations, and other undesirable words in ChatGPT roleplaying data.\n\nIt combines [Neural Chat 7B](/models/intel/neural-chat-7b), Airoboros 7b, [Toppy M 7B](/models/undi95/toppy-m-7b), [Zepher 7b beta](/models/huggingfaceh4/zephyr-7b-beta), [Nous Capybara 34B](/models/nousresearch/nous-capybara-34b), [OpenHeremes 2.5](/models/teknium/openhermes-2.5-mistral-7b), and many others.\n\n#merge"
}, {
    "value": "01-ai/yi-34b",
    "label": "Yi 34B (base)",
    "cover": "",
    "description": "The Yi series models are large language models trained from scratch by developers at [01.AI](https://01.ai/). This is the base 34B parameter model."
}, {
    "value": "01-ai/yi-6b",
    "label": "Yi 6B (base)",
    "cover": "",
    "description": "The Yi series models are large language models trained from scratch by developers at [01.AI](https://01.ai/). This is the base 6B parameter model."
}, {
    "value": "01-ai/yi-34b-chat",
    "label": "Yi 34B Chat",
    "cover": "",
    "description": "The Yi series models are large language models trained from scratch by developers at [01.AI](https://01.ai/). This 34B parameter model has been instruct-tuned for chat."
}, {
    "value": "nousresearch/nous-hermes-2-vision-7b",
    "label": "Hermes 2 Vision 7B (alpha)",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "This vision-language model builds on innovations from the popular [OpenHermes-2.5](/models/teknium/openhermes-2.5-mistral-7b) model, by Teknium. It adds vision support, and is trained on a custom dataset enriched with function calling\n\nThis project is led by [qnguyen3](https://twitter.com/stablequan) and [teknium](https://twitter.com/Teknium1).\n\n#multimodal"
}, {
    "value": "openrouter/cinematika-7b",
    "label": "Cinematika 7B (alpha)",
    "cover": "",
    "description": "This model is under development. Check the [OpenRouter Discord](https://discord.gg/fVyRaUDgxW) for updates."
}, {
    "value": "nousresearch/nous-capybara-7b",
    "label": "Capybara 7B",
    "cover": "",
    "description": "The Capybara series is a collection of datasets and models made by fine-tuning on data created by Nous, mostly in-house.\n\nV1.9 uses unalignment techniques for more consistent and dynamic control. It also leverages a significantly better foundation model, [Mistral 7B](/models/mistralai/mistral-7b-instruct-v0.1)."
}, {
    "value": "jebcarter/psyfighter-13b",
    "label": "Psyfighter 13B",
    "cover": "",
    "description": "A merge model based on [Llama-2-13B](/models/meta-llama/llama-2-13b-chat) and made possible thanks to the compute provided by the KoboldAI community. It's a merge between:\n\n- [KoboldAI/LLaMA2-13B-Tiefighter](https://huggingface.co/KoboldAI/LLaMA2-13B-Tiefighter)\n- [chaoyi-wu/MedLLaMA_13B](https://huggingface.co/chaoyi-wu/MedLLaMA_13B)\n- [Doctor-Shotgun/llama-2-13b-chat-limarp-v2-merged](https://huggingface.co/Doctor-Shotgun/llama-2-13b-chat-limarp-v2-merged).\n\n#merge"
}, {
    "value": "openchat/openchat-7b",
    "label": "OpenChat 3.5 7B",
    "cover": "/images/icons/Lepton.png",
    "description": 'OpenChat 7B is a library of open-source language models, fine-tuned with "C-RLFT (Conditioned Reinforcement Learning Fine-Tuning)" - a strategy inspired by offline reinforcement learning. It has been trained on mixed-quality data without preference labels.\n\n- For OpenChat fine-tuned on Mistral 7B, check out [OpenChat 7B](/models/openchat/openchat-7b).\n- For OpenChat fine-tuned on Llama 8B, check out [OpenChat 8B](/models/openchat/openchat-8b).\n\n#open-source'
}, {
    "value": "neversleep/noromaid-20b",
    "label": "Noromaid 20B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "A collab between IkariDev and Undi. This merge is suitable for RP, ERP, and general knowledge.\n\n#merge #uncensored"
}, {
    "value": "intel/neural-chat-7b",
    "label": "Neural Chat 7B v3.1",
    "cover": "",
    "description": "A fine-tuned model based on [mistralai/Mistral-7B-v0.1](/models/mistralai/mistral-7b-instruct-v0.1) on the open source dataset [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca), aligned with DPO algorithm. For more details, refer to the blog: [The Practice of Supervised Fine-tuning and Direct Preference Optimization on Habana Gaudi2](https://medium.com/@NeuralCompressor/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3)."
}, {
    "value": "anthropic/claude-2:beta",
    "label": "Claude v2 (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 2 delivers advancements in key capabilities for enterprises—including an industry-leading 200K token context window, significant reductions in rates of model hallucination, system prompts and a new beta feature: tool use."
}, {
    "value": "anthropic/claude-2",
    "label": "Claude v2",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 2 delivers advancements in key capabilities for enterprises—including an industry-leading 200K token context window, significant reductions in rates of model hallucination, system prompts and a new beta feature: tool use."
}, {
    "value": "anthropic/claude-instant-1.1",
    "label": "Claude Instant v1.1",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Anthropic's model for low-latency, high throughput text generation. Supports hundreds of pages of text."
}, {
    "value": "anthropic/claude-2.1:beta",
    "label": "Claude v2.1 (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 2 delivers advancements in key capabilities for enterprises—including an industry-leading 200K token context window, significant reductions in rates of model hallucination, system prompts and a new beta feature: tool use."
}, {
    "value": "anthropic/claude-2.1",
    "label": "Claude v2.1",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Claude 2 delivers advancements in key capabilities for enterprises—including an industry-leading 200K token context window, significant reductions in rates of model hallucination, system prompts and a new beta feature: tool use."
}, {
    "value": "teknium/openhermes-2.5-mistral-7b",
    "label": "OpenHermes 2.5 Mistral 7B",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "A continuation of [OpenHermes 2 model](/models/teknium/openhermes-2-mistral-7b), trained on additional code datasets.\nPotentially the most interesting finding from training on a good ratio (est. of around 7-14% of the total dataset) of code instruction was that it has boosted several non-code benchmarks, including TruthfulQA, AGIEval, and GPT4All suite. It did however reduce BigBench benchmark score, but the net gain overall is significant."
}, {
    "value": "liuhaotian/llava-13b",
    "label": "LLaVA 13B",
    "cover": "",
    "description": "LLaVA is a large multimodal model that combines a vision encoder and Vicuna for general-purpose visual and language understanding, achieving impressive chat capabilities and setting a new state-of-the-art accuracy on Science QA.\n\n#multimodal"
}, {
    "value": "nousresearch/nous-capybara-34b",
    "label": "Capybara 34B",
    "cover": "",
    "description": "This model is trained on the Yi-34B model for 3 epochs on the Capybara dataset. It's the first 34B Nous model and first 200K context length Nous model."
}, {
    "value": "openai/gpt-4-vision-preview",
    "label": "GPT-4 Vision",
    "cover": "",
    "description": "Ability to understand images, in addition to all other [GPT-4 Turbo capabilties](/models/openai/gpt-4-turbo). Training data: up to Apr 2023.\n\n**Note:** heavily rate limited by OpenAI while in preview.\n\n#multimodal"
}, {
    "value": "lizpreciatior/lzlv-70b-fp16-hf",
    "label": "lzlv 70B",
    "cover": "",
    "description": "A Mythomax/MLewd_13B-style merge of selected 70B models.\nA multi-model merge of several LLaMA2 70B finetunes for roleplaying and creative work. The goal was to create a model that combines creativity with intelligence for an enhanced experience.\n\n#merge #uncensored"
}, {
    "value": "undi95/toppy-m-7b",
    "label": "Toppy M 7B",
    "cover": "/images/icons/Lepton.png",
    "description": "A wild 7B parameter model that merges several models using the new task_arithmetic merge method from mergekit.\nList of merged models:\n- NousResearch/Nous-Capybara-7B-V1.9\n- [HuggingFaceH4/zephyr-7b-beta](/models/huggingfaceh4/zephyr-7b-beta)\n- lemonilia/AshhLimaRP-Mistral-7B\n- Vulkane/120-Days-of-Sodom-LoRA-Mistral-7b\n- Undi95/Mistral-pippa-sharegpt-7b-qlora\n\n#merge #uncensored"
}, {
    "value": "alpindale/goliath-120b",
    "label": "Goliath 120B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "A large LLM created by combining two fine-tuned Llama 70B models into one 120B model. Combines Xwin and Euryale.\n\nCredits to\n- [@chargoddard](https://huggingface.co/chargoddard) for developing the framework used to merge the model - [mergekit](https://github.com/cg123/mergekit).\n- [@Undi95](https://huggingface.co/Undi95) for helping with the merge ratios.\n\n#merge"
}, {
    "value": "openrouter/auto",
    "label": "Auto Router",
    "cover": "",
    "description": "Your prompt will be processed by a meta-model and routed to one of dozens of models (see below), optimizing for the best possible output.\n\nTo see which model was used, visit [Activity](/activity), or read the `model` attribute of the response. Your response will be priced at the same rate as the routed model.\n\nThe meta-model is powered by [Not Diamond](https://docs.notdiamond.ai/docs/how-not-diamond-works). Learn more in our [docs](/docs/model-routing).\n\nRequests will be routed to the following models:\n- [openai/gpt-4o-2024-08-06](/openai/gpt-4o-2024-08-06)\n- [openai/gpt-4o-2024-05-13](/openai/gpt-4o-2024-05-13)\n- [openai/gpt-4.1-mini-2024-07-18](/openai/gpt-4.1-mini-2024-07-18)\n- [openai/chatgpt-4o-latest](/openai/chatgpt-4o-latest)\n- [openai/o1-preview-2024-09-12](/openai/o1-preview-2024-09-12)\n- [openai/o1-mini-2024-09-12](/openai/o1-mini-2024-09-12)\n- [anthropic/claude-3.5-sonnet](/anthropic/claude-3.5-sonnet)\n- [anthropic/claude-3.5-haiku](/anthropic/claude-3.5-haiku)\n- [anthropic/claude-3-opus](/anthropic/claude-3-opus)\n- [anthropic/claude-2.1](/anthropic/claude-2.1)\n- [google/gemini-pro-1.5](/google/gemini-pro-1.5)\n- [google/gemini-flash-1.5](/google/gemini-flash-1.5)\n- [mistralai/mistral-large-2407](/mistralai/mistral-large-2407)\n- [mistralai/mistral-nemo](/mistralai/mistral-nemo)\n- [deepseek/deepseek-r1](/deepseek/deepseek-r1)\n- [meta-llama/llama-3.1-70b-instruct](/meta-llama/llama-3.1-70b-instruct)\n- [meta-llama/llama-3.1-405b-instruct](/meta-llama/llama-3.1-405b-instruct)\n- [mistralai/mixtral-8x22b-instruct](/mistralai/mixtral-8x22b-instruct)\n- [cohere/command-r-plus](/cohere/command-r-plus)\n- [cohere/command-r](/cohere/command-r)"
}, {
    "value": "openai/gpt-4-1106-preview",
    "label": "GPT-4 Turbo (older v1106)",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling.\n\nTraining data: up to April 2023."
}, {
    "value": "openai/gpt-3.5-turbo-1106",
    "label": "GPT-3.5 Turbo 16k (older v1106)",
    "cover": "/images/icons/OpenAI.svg",
    "description": "An older GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Training data: up to Sep 2021."
}, {
    "value": "google/palm-2-codechat-bison-32k",
    "label": "PaLM 2 Code Chat 32k",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "PaLM 2 fine-tuned for chatbot conversations that help with code-related questions."
}, {
    "value": "google/palm-2-chat-bison-32k",
    "label": "PaLM 2 Chat 32k",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "PaLM 2 is a language model by Google with improved multilingual, reasoning and coding capabilities."
}, {
    "value": "teknium/openhermes-2-mistral-7b",
    "label": "OpenHermes 2 Mistral 7B",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "Trained on 900k instructions, surpasses all previous versions of Hermes 13B and below, and matches 70B on some benchmarks. Hermes 2 has strong multiturn chat skills and system prompt capabilities."
}, {
    "value": "open-orca/mistral-7b-openorca",
    "label": "Mistral OpenOrca 7B",
    "cover": "/images/icons/Mistral.png",
    "description": "A fine-tune of Mistral using the OpenOrca dataset. First 7B model to beat all other models <30B."
}, {
    "value": "jondurbin/airoboros-l2-70b",
    "label": "Airoboros 70B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://novita.ai/&size=256",
    "description": "A Llama 2 70B fine-tune using synthetic data (the Airoboros dataset).\n\nCurrently based on [jondurbin/airoboros-l2-70b](https://huggingface.co/jondurbin/airoboros-l2-70b-2.2.1), but might get updated in the future."
}, {
    "value": "nousresearch/nous-hermes-llama2-70b",
    "label": "Hermes 70B",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "A state-of-the-art language model fine-tuned on over 300k instructions by Nous Research, with Teknium and Emozilla leading the fine tuning process."
}, {
    "value": "xwin-lm/xwin-lm-70b",
    "label": "Xwin 70B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "Xwin-LM aims to develop and open-source alignment tech for LLMs. Our first release, built-upon on the [Llama2](/models/${Model.Llama_2_13B_Chat}) base models, ranked TOP-1 on AlpacaEval. Notably, it's the first to surpass [GPT-4](/models/${Model.GPT_4}) on this benchmark. The project will be continuously updated."
}, {
    "value": "mistralai/mistral-7b-instruct-v0.1",
    "label": "Mistral 7B Instruct v0.1",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "A 7.3B parameter model that outperforms Llama 2 13B on all benchmarks, with optimizations for speed and context length."
}, {
    "value": "openai/gpt-3.5-turbo-instruct",
    "label": "GPT-3.5 Turbo Instruct",
    "cover": "/images/icons/OpenAI.svg",
    "description": "This model is a variant of GPT-3.5 Turbo tuned for instructional prompts and omitting chat-related optimizations. Training data: up to Sep 2021."
}, {
    "value": "migtissera/synthia-70b",
    "label": "Synthia 70B",
    "cover": "",
    "description": "SynthIA (Synthetic Intelligent Agent) is a LLama-2 70B model trained on Orca style datasets. It has been fine-tuned for instruction following as well as having long-form conversations."
}, {
    "value": "pygmalionai/mythalion-13b",
    "label": "Mythalion 13B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "A blend of the new Pygmalion-13b and MythoMax. #merge"
}, {
    "value": "openai/gpt-4-32k-0314",
    "label": "GPT-4 32k (older v0314)",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-4-32k is an extended version of GPT-4, with the same capabilities but quadrupled context length, allowing for processing up to 40 pages of text in a single pass. This is particularly beneficial for handling longer content like interacting with PDFs without an external vector database. Training data: up to Sep 2021."
}, {
    "value": "openai/gpt-3.5-turbo-16k",
    "label": "GPT-3.5 Turbo 16k",
    "cover": "/images/icons/OpenAI.svg",
    "description": "This model offers four times the context length of gpt-3.5-turbo, allowing it to support approximately 20 pages of text in a single request at a higher cost. Training data: up to Sep 2021."
}, {
    "value": "openai/gpt-4-32k",
    "label": "GPT-4 32k",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-4-32k is an extended version of GPT-4, with the same capabilities but quadrupled context length, allowing for processing up to 40 pages of text in a single pass. This is particularly beneficial for handling longer content like interacting with PDFs without an external vector database. Training data: up to Sep 2021."
}, {
    "value": "meta-llama/codellama-34b-instruct",
    "label": "CodeLlama 34B Instruct",
    "cover": "",
    "description": "Code Llama is built upon Llama 2 and excels at filling in code, handling extensive input contexts, and following programming instructions without prior training for various programming tasks."
}, {
    "value": "phind/phind-codellama-34b",
    "label": "CodeLlama 34B v2",
    "cover": "",
    "description": "A fine-tune of CodeLlama-34B on an internal dataset that helps it exceed GPT-4 on some benchmarks, including HumanEval."
}, {
    "value": "nousresearch/nous-hermes-llama2-13b",
    "label": "Hermes 13B",
    "cover": "https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/66f419f202c64707fcabd6ac_nous-logo.webp",
    "description": "A state-of-the-art language model fine-tuned on over 300k instructions by Nous Research, with Teknium and Emozilla leading the fine tuning process."
}, {
    "value": "huggingfaceh4/zephyr-7b-beta:free",
    "label": "Zephyr 7B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://huggingface.co/&size=256",
    "description": "Zephyr is a series of language models that are trained to act as helpful assistants. Zephyr-7B-β is the second model in the series, and is a fine-tuned version of [mistralai/Mistral-7B-v0.1](/models/mistralai/mistral-7b-instruct-v0.1) that was trained on a mix of publicly available, synthetic datasets using Direct Preference Optimization (DPO)."
}, {
    "value": "mancer/weaver",
    "label": "Weaver (alpha)",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "An attempt to recreate Claude-style verbosity, but don't expect the same level of coherence or memory. Meant for use in roleplay/narrative situations."
}, {
    "value": "anthropic/claude-instant-1",
    "label": "Claude Instant v1",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Anthropic's model for low-latency, high throughput text generation. Supports hundreds of pages of text."
}, {
    "value": "anthropic/claude-1",
    "label": "Claude v1",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Anthropic's model for low-latency, high throughput text generation. Supports hundreds of pages of text."
}, {
    "value": "anthropic/claude-instant-1.0",
    "label": "Claude Instant v1.0",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Anthropic's model for low-latency, high throughput text generation. Supports hundreds of pages of text."
}, {
    "value": "anthropic/claude-1.2",
    "label": "Claude v1.2",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Anthropic's model for low-latency, high throughput text generation. Supports hundreds of pages of text."
}, {
    "value": "anthropic/claude-2.0:beta",
    "label": "Claude v2.0 (self-moderated)",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Anthropic's flagship model. Superior performance on tasks that require complex reasoning. Supports hundreds of pages of text."
}, {
    "value": "anthropic/claude-2.0",
    "label": "Claude v2.0",
    "cover": "https://cdn.prod.website-files.com/67ce28cfec624e2b733f8a52/67d31dd7aa394792257596c5_webclip.png",
    "description": "Anthropic's flagship model. Superior performance on tasks that require complex reasoning. Supports hundreds of pages of text."
}, {
    "value": "undi95/remm-slerp-l2-13b",
    "label": "ReMM SLERP 13B",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://mancer.tech/&size=256",
    "description": "A recreation trial of the original MythoMax-L2-B13 but with updated models. #merge"
}, {
    "value": "google/palm-2-codechat-bison",
    "label": "PaLM 2 Code Chat",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "PaLM 2 fine-tuned for chatbot conversations that help with code-related questions."
}, {
    "value": "google/palm-2-chat-bison",
    "label": "PaLM 2 Chat",
    "cover": "https://registry.npmmirror.com/@lobehub/icons-static-png/1.18.0/files/dark/google-color.png",
    "description": "PaLM 2 is a language model by Google with improved multilingual, reasoning and coding capabilities."
}, {
    "value": "gryphe/mythomax-l2-13b",
    "label": "MythoMax 13B",
    "cover": "/images/icons/DeepInfra.webp",
    "description": "One of the highest performing and most popular fine-tunes of Llama 2 13B, with rich descriptions and roleplay. #merge"
}, {
    "value": "meta-llama/llama-2-13b-chat",
    "label": "Llama 2 13B Chat",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "A 13 billion parameter language model from Meta, fine tuned for chat completions"
}, {
    "value": "meta-llama/llama-2-70b-chat",
    "label": "Llama 2 70B Chat",
    "cover": "https://t0.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://www.together.ai/&size=256",
    "description": "The flagship, 70 billion parameter language model from Meta, fine tuned for chat completions. Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety."
}, {
    "value": "openai/gpt-3.5-turbo-0125",
    "label": "GPT-3.5 Turbo 16k",
    "cover": "/images/icons/OpenAI.svg",
    "description": "The latest GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Training data: up to Sep 2021.\n\nThis version has a higher accuracy at responding in requested formats and a fix for a bug which caused a text encoding issue for non-English language function calls."
}, {
    "value": "openai/gpt-4",
    "label": "GPT-4",
    "cover": "/images/icons/OpenAI.svg",
    "description": "OpenAI's flagship model, GPT-4 is a large-scale multimodal language model capable of solving difficult problems with greater accuracy than previous models due to its broader general knowledge and advanced reasoning capabilities. Training data: up to Sep 2021."
}, {
    "value": "openai/gpt-3.5-turbo",
    "label": "GPT-3.5 Turbo",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.\n\nTraining data up to Sep 2021."
}, {
    "value": "openai/gpt-4-0314",
    "label": "GPT-4 (older v0314)",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-4-0314 is the first version of GPT-4 released, with a context length of 8,192 tokens, and was supported until June 14. Training data: up to Sep 2021."
}, {
    "value": "openai/gpt-3.5-turbo-0301",
    "label": "GPT-3.5 Turbo (older v0301)",
    "cover": "/images/icons/OpenAI.svg",
    "description": "GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.\n\nTraining data up to Sep 2021."
}]

class BlackboxPro(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Blackbox AI Pro"
    url = "https://www.blackbox.ai"
    cookie_domain = ".blackbox.ai"
    login_url = None
    api_endpoint = "https://www.blackbox.ai/api/chat"
    
    working = True
    needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = ""
    default_image_model = 'flux'
    image_models = [default_image_model]   
    vision_models = ['GPT-4o', 'o1', 'o3-mini', 'Gemini-PRO', 'Gemini Agent', 'llama-3.1-8b Agent', 'llama-3.1-70b Agent', 'llama-3.1-405 Agent', 'Gemini-Flash-2.0', 'DeepSeek-V3']
    fallback_models = [default_model]
    models = fallback_models + [model.get('value') for model in models]

    session_data = None  # Class variable to store session data
    
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
    def get_models(cls, **kwargs) -> list:
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
        max_tokens: int = 1024,
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
            cls.session_data = cls._find_session_in_har_files() or cls.session_data

            if not cls.session_data:
                async with session.get('https://www.blackbox.ai/api/auth/session', cookies=get_cookies(cls.cookie_domain, False)) as resp:
                    resp.raise_for_status()
                    cls.session_data = await resp.json()

            # Check if we have a valid session
            if not cls.session_data:
                # No valid session found, raise an error
                debug.log("BlackboxPro: No valid session found in HAR files")
                raise NoValidHarFileError("No valid Blackbox session found. Please log in to Blackbox AI in your browser first.")

            debug.log(f"BlackboxPro: Using session from cookies / HAR file (email: {cls.session_data['user'].get('email', 'unknown')})")

            # Check subscription status
            subscription_status = {"status": "FREE", "customerId": None, "isTrialSubscription": False, "lastChecked": None}
            if cls.session_data.get('user', {}).get('email'):
                subscription_status = await cls.check_subscription(cls.session_data['user']['email'])
                debug.log(f"BlackboxPro: Subscription status for {cls.session_data['user']['email']}: {subscription_status['status']}")

            # Determine if user has premium access based on subscription status
            is_premium = False
            if subscription_status['status'] == "PREMIUM":
                is_premium = True
            else:
                # For free accounts, check for requested model
                if model:
                    debug.log(f"BlackboxPro: Model {model} not available in free account, falling back to default model")
                    model = cls.default_model

            data = {
                "messages": current_messages,
                "id": conversation.chat_id,
                "previewToken": None,
                "userId": None,
                "codeModelMode": True,
                "trendingAgentMode": {},
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
                "userSelectedModel": model if model else None,
                "userSelectedAgent": "VscodeAgent",
                "validated": "a38f5889-8fef-46d4-8ede-bf4668b6a9bb",
                "imageGenerationMode": model == cls.default_image_model,
                "imageGenMode": "autoMode",
                "webSearchModePrompt": False,
                "deepSearchMode": False,
                "promptSelection": "",
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
                "webSearchModeOption": {
                    "autoMode": True,
                    "webMode": False,
                    "offlineMode": False
                },
                "session": cls.session_data,
                "isPremium": is_premium, 
                "subscriptionCache": {
                    "status": subscription_status['status'],
                    "customerId": subscription_status['customerId'],
                    "isTrialSubscription": subscription_status['isTrialSubscription'],
                    "lastChecked": int(datetime.now().timestamp() * 1000)
                },
                "beastMode": False,
                "reasoningMode": False,
                "designerMode": False,
                "workspaceId": "",
                "asyncMode": False,
                "integrations": {},
                "isTaskPersistent": False,
                "selectedElement": None
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
                    yield conversation
                # For image models that didn't produce an image, fall back to text response
                elif model == cls.default_image_model:
                    yield full_response_text
