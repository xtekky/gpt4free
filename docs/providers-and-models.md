
# G4F - Providers and Models

This document provides an overview of various AI providers and models, including text generation, image generation, and vision capabilities. It aims to help users navigate the diverse landscape of AI services and choose the most suitable option for their needs.

## Table of Contents
  - [Providers](#providers)
  - [Models](#models)
    - [Text Models](#text-models)
    - [Image Models](#image-models)
    - [Vision Models](#vision-models)
  - [Conclusion and Usage Tips](#conclusion-and-usage-tips)

---
## Providers
| Provider | Text Models | Image Models | Vision Models | Stream | Status | Auth |
|----------|-------------|--------------|---------------|--------|--------|------|
|[ai4chat.co](https://www.ai4chat.co)|`g4f.Provider.Ai4Chat`|`gpt-4`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[chat.ai365vip.com](https://chat.ai365vip.com)|`g4f.Provider.AI365VIP`|`gpt-3.5-turbo, gpt-4o`|❌|❌|?|![Cloudflare](https://img.shields.io/badge/Cloudflare-f48d37)|❌|
|[aichatfree.info](https://aichatfree.info)|`g4f.Provider.AIChatFree`|`gemini-pro`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[aichatonline.org](https://aichatonline.org)|`g4f.Provider.AiChatOnline`|`gpt-4o-mini`|❌|❌|?|![Cloudflare](https://img.shields.io/badge/Cloudflare-f48d37)|❌|
|[ai-chats.org](https://ai-chats.org)|`g4f.Provider.AiChats`|`gpt-4`|`dalle`|❌|?|![Captcha](https://img.shields.io/badge/Captcha-f48d37)|❌|
|[api.airforce](https://api.airforce)|`g4f.Provider.AiMathGPT`|`llama-3.1-70b`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[api.airforce](https://api.airforce)|`g4f.Provider.Airforce`|`gpt-4, gpt-4-turbo, gpt-4o-mini, gpt-3.5-turbo, gpt-4o, claude-3-haiku, claude-3-sonnet, claude-3-5-sonnet, claude-3-opus, llama-3-70b, llama-3-8b, llama-2-13b, llama-3.1-405b, llama-3.1-70b, llama-3.1-8b, llamaguard-2-8b, llamaguard-7b, llama-3.2-90b, mixtral-8x7b mixtral-8x22b, mistral-7b, qwen-1.5-7b, qwen-1.5-14b, qwen-1.5-72b, qwen-1.5-110b, qwen-2-72b, gemma-2b, gemma-2-9b, gemma-2-27b, gemini-flash, gemini-pro, deepseek, mixtral-8x7b-dpo, yi-34b, wizardlm-2-8x22b, solar-10.7b, mythomax-l2-13b, cosmosrp`|`flux, flux-realism', flux-anime, flux-3d, flux-disney, flux-pixel, flux-4o, any-dark, dalle-3`|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[aiuncensored.info](https://www.aiuncensored.info)|`g4f.Provider.AIUncensored`|✔|✔|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[allyfy.chat](https://allyfy.chat/)|`g4f.Provider.Allyfy`|`gpt-3.5-turbo`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[amigochat.io/chat](https://amigochat.io/chat/)|`g4f.Provider.AmigoChat`|`gpt-4o, gpt-4o-mini, o1, o1-mini, claude-3.5-sonnet, llama-3.2-90b, llama-3.1-405b, gemini-pro`|`flux-pro, flux-realism, dalle-3`|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[openchat.team](https://openchat.team/)|`g4f.Provider.Aura`|✔|❌|❌|?|![Disabled](https://img.shields.io/badge/Disabled-red)|❌|
|[bing.com](https://bing.com/chat)|`g4f.Provider.Bing`|`gpt-4`|✔|`gpt-4-vision`|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌+✔|
|[bing.com/images](https://www.bing.com/images/create)|`g4f.Provider.BingCreateImages`|`❌|✔|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|✔|
|[blackbox.ai](https://www.blackbox.ai)|`g4f.Provider.Blackbox`|`blackboxai, blackboxai-pro, gemini-flash, llama-3.1-8b, llama-3.1-70b, gpt-4o, gemini-pro, claude-3.5-sonnet`|`flux`|✔|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[chatgot.one](https://www.chatgot.one/)|`g4f.Provider.ChatGot`|`gemini-pro`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[chatgpt.com](https://chatgpt.com)|`g4f.Provider.ChatGpt`|`?`|`?`|`?`|?|![Unknown](https://img.shields.io/badge/Unknown-grey) |❌|
|[chatgpt.es](https://chatgpt.es)|`g4f.Provider.ChatGptEs`|`gpt-4o, gpt-4o-mini`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[chatgpt4online.org](https://chatgpt4online.org)|`g4f.Provider.Chatgpt4Online`|`gpt-4`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[chatgpt4o.one](https://chatgpt4o.one)|`g4f.Provider.Chatgpt4o`|✔|❌|❌|❌|![Cloudflare](https://img.shields.io/badge/Cloudflare-f48d37)|❌|
|[chatgptfree.ai](https://chatgptfree.ai)|`g4f.Provider.ChatgptFree`|`gpt-4o-mini`|❌|❌|?|![Cloudflare](https://img.shields.io/badge/Cloudflare-f48d37)|❌|
|[app.chathub.gg](https://app.chathub.gg)|`g4f.Provider.ChatHub`|`llama-3.1-8b, mixtral-8x7b, gemma-2, sonar-online`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[chatify-ai.vercel.app](https://chatify-ai.vercel.app)|`g4f.Provider.ChatifyAI`|`llama-3.1-8b`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[playground.ai.cloudflare.com](https://playground.ai.cloudflare.com)|`g4f.Provider.Cloudflare`|`german-7b, gemma-7b, llama-2-7b, llama-3-8b, llama-3.1-8b, llama-3.2-11b, llama-3.2-1b, llama-3.2-3b, mistral-7b, openchat-3.5, phi-2, qwen-1.5-0.5b, qwen-1.5-1.8b, qwen-1.5-14b, qwen-1.5-7b, tinyllama-1.1b, cybertron-7b`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[aiuncensored.info](https://www.aiuncensored.info)|`g4f.Provider.DarkAI`|`gpt-4o, gpt-3.5-turbo, llama-3-70b, llama-3-405b`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[duckduckgo.com](https://duckduckgo.com/duckchat/v1/chat)|`g4f.Provider.DDG`|`gpt-4o-mini, claude-3-haiku, llama-3.1-70b, mixtral-8x7b`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[deepinfra.com](https://deepinfra.com)|`g4f.Provider.DeepInfra`|✔|❌|❌|✔|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[deepinfra.com/chat](https://deepinfra.com/chat)|`g4f.Provider.DeepInfraChat`|`llama-3.1-405b, llama-3.1-70b, llama-3.1-8B, mixtral-8x22b, mixtral-8x7b, wizardlm-2-8x22b, wizardlm-2-7b, qwen-2-72b, phi-3-medium-4k, gemma-2b-27b, minicpm-llama-3-v2.5, mistral-7b, lzlv_70b, openchat-3.6-8b, phind-codellama-34b-v2, dolphin-2.9.1-llama-3-70b`|❌|`minicpm-llama-3-v2.5`|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[deepinfra.com](https://deepinfra.com)|`g4f.Provider.DeepInfraImage`|❌|✔|❌|❌|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[deepinfra.com](https://deepinfra.com)|`g4f.Provider.Editee`|`claude-3.5-sonnet, gpt-4o, gemini-pro, mistral-large`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[flowgpt.com](https://flowgpt.com/chat)|`g4f.Provider.FlowGpt`|✔||❌|✔|![Disabled](https://img.shields.io/badge/Disabled-red)|❌|
|[chat10.free2gpt.xyz](chat10.free2gpt.xyz)|`g4f.Provider.Free2GPT`|`llama-3.1-70b`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[chat.chatgpt.org.uk](https://chat.chatgpt.org.uk)|`g4f.Provider.FreeChatgpt`|`qwen-1.5-14b, sparkdesk-v1.1, qwen-2-7b, glm-4-9b, glm-3-6b, yi-1.5-9b`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[freegptsnav.aifree.site](https://freegptsnav.aifree.site)|`g4f.Provider.FreeGpt`|`llama-3.1-70b`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[free.netfly.top](https://free.netfly.top)|`g4f.Provider.FreeNetfly`|✔|❌|❌|?|![Cloudflare](https://img.shields.io/badge/Cloudflare-f48d37)|❌|
|[gemini.google.com](https://gemini.google.com)|`g4f.Provider.Gemini`|✔|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|✔|
|[ai.google.dev](https://ai.google.dev)|`g4f.Provider.GeminiPro`|✔|❌|✔|?|![Active](https://img.shields.io/badge/Active-brightgreen)|✔|
|[app.giz.ai](https://app.giz.ai/assistant/)|`g4f.Provider.GizAI`|`gemini-flash, gemini-pro, gpt-4o-mini, gpt-4o, claude-3.5-sonnet, claude-3-haiku, llama-3.1-70b, llama-3.1-8b, mistral-large`|`sdxl, sd-1.5, sd-3.5, dalle-3, flux-schnell, flux1-pro`|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[developers.sber.ru](https://developers.sber.ru/gigachat)|`g4f.Provider.GigaChat`|✔|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|✔|
|[gprochat.com](https://gprochat.com)|`g4f.Provider.GPROChat`|`gemini-pro`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[console.groq.com/playground](https://console.groq.com/playground)|`g4f.Provider.Groq`|✔|❌|❌|?|![Active](https://img.shields.io/badge/Active-brightgreen)|✔|
|[huggingface.co/chat](https://huggingface.co/chat)|`g4f.Provider.HuggingChat`|`llama-3.1-70b, command-r-plus, qwen-2-72b, llama-3.2-11b, hermes-3, mistral-nemo, phi-3.5-mini`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[huggingface.co](https://huggingface.co/chat)|`g4f.Provider.HuggingFace`|✔|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[koala.sh/chat](https://koala.sh/chat)|`g4f.Provider.Koala`|`gpt-4o-mini`|❌|❌|?|![Cloudflare](https://img.shields.io/badge/Cloudflare-f48d37)|❌|
|[liaobots.work](https://liaobots.work)|`g4f.Provider.Liaobots`|`gpt-3.5-turbo, gpt-4o-mini, gpt-4o, gpt-4-turbo, grok-2, grok-2-mini, claude-3-opus, claude-3-sonnet, claude-3-5-sonnet, claude-3-haiku, claude-2.1, gemini-flash, gemini-pro`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[magickpen.com](https://magickpen.com)|`g4f.Provider.MagickPen`|`gpt-4o-mini`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[meta.ai](https://www.meta.ai)|`g4f.Provider.MetaAI`|✔|✔|?|?|![Active](https://img.shields.io/badge/Active-brightgreen)|✔|
|[app.myshell.ai/chat](https://app.myshell.ai/chat)|`g4f.Provider.MyShell`|✔|❌|?|?|![Disabled](https://img.shields.io/badge/Disabled-red)|❌|
|[nexra.aryahcr.cc/bing](https://nexra.aryahcr.cc/documentation/bing/en)|`g4f.Provider.NexraBing`|✔|❌|❌|✔|![Disabled](https://img.shields.io/badge/Disabled-red)|❌|
|[nexra.aryahcr.cc/blackbox](https://nexra.aryahcr.cc/documentation/blackbox/en)|`g4f.Provider.NexraBlackbox`|`blackboxai` |❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/chatgpt](https://nexra.aryahcr.cc/documentation/chatgpt/en)|`g4f.Provider.NexraChatGPT`|`gpt-4, gpt-3.5-turbo, gpt-3, gpt-4o` |❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/dall-e](https://nexra.aryahcr.cc/documentation/dall-e/en)|`g4f.Provider.NexraDallE`|❌|`dalle`|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/dall-e](https://nexra.aryahcr.cc/documentation/dall-e/en)|`g4f.Provider.NexraDallE2`|❌|`dalle-2`|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/emi](https://nexra.aryahcr.cc/documentation/emi/en)|`g4f.Provider.NexraEmi`|❌|`emi`|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/flux-pro](https://nexra.aryahcr.cc/documentation/flux-pro/en)|`g4f.Provider.NexraFluxPro`|❌|`flux-pro`|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/gemini-pro](https://nexra.aryahcr.cc/documentation/gemini-pro/en)|`g4f.Provider.NexraGeminiPro`|`gemini-pro`|❌|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/midjourney](https://nexra.aryahcr.cc/documentation/midjourney/en)|`g4f.Provider.NexraMidjourney`|❌|`midjourney`|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/prodia](https://nexra.aryahcr.cc/documentation/prodia/en)|`g4f.Provider.NexraProdiaAI`|❌|✔|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/qwen](https://nexra.aryahcr.cc/documentation/qwen/en)|`g4f.Provider.NexraQwen`|`qwen`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[nexra.aryahcr.cc/stable-diffusion](https://nexra.aryahcr.cc/documentation/stable-diffusion/en)|`g4f.Provider.NexraSD15`|❌|`sd-1.5`|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌
|[nexra.aryahcr.cc/stable-diffusion](https://nexra.aryahcr.cc/documentation/stable-diffusion/en)|`g4f.Provider.NexraSDLora`|❌|`sdxl-lora`|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌
|[nexra.aryahcr.cc/stable-diffusion](https://nexra.aryahcr.cc/documentation/stable-diffusion/en)|`g4f.Provider.NexraSDTurbo`|❌|`sdxl-turbo`|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌
|[openrouter.ai](https://openrouter.ai)|`g4f.Provider.OpenRouter`|✔|❌|?|?|![Disabled](https://img.shields.io/badge/Disabled-red)|❌|
|[platform.openai.com](https://platform.openai.com/)|`g4f.Provider.Openai`|✔|❌|✔||![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[chatgpt.com](https://chatgpt.com/)|`g4f.Provider.OpenaiChat`|`gpt-4o, gpt-4o-mini, gpt-4`|❌|✔||![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[www.perplexity.ai)](https://www.perplexity.ai)|`g4f.Provider.PerplexityAi`|✔|❌|❌|?|![Disabled](https://img.shields.io/badge/Disabled-red)|❌|
|[perplexity.ai](https://www.perplexity.ai)|`g4f.Provider.PerplexityApi`|✔|❌|❌|?|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[labs.perplexity.ai](https://labs.perplexity.ai)|`g4f.Provider.PerplexityLabs`|`sonar-online, sonar-chat, llama-3.1-8b, llama-3.1-70b`|❌|❌|?|![Cloudflare](https://img.shields.io/badge/Cloudflare-f48d37)|❌|
|[pi.ai/talk](https://pi.ai/talk)|`g4f.Provider.Pi`|`pi`|❌|❌|?|![Unknown](https://img.shields.io/badge/Unknown-grey)|❌|
|[]()|`g4f.Provider.Pizzagpt`|`gpt-4o-mini`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[poe.com](https://poe.com)|`g4f.Provider.Poe`|✔|❌|❌|?|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[app.prodia.com](https://app.prodia.com)|`g4f.Provider.Prodia`|❌|✔|❌|❌|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[raycast.com](https://raycast.com)|`g4f.Provider.Raycast`|✔|❌|❌|✔|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[chat.reka.ai](https://chat.reka.ai/)|`g4f.Provider.Reka`|✔|❌|✔|✔|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[replicate.com](https://replicate.com)|`g4f.Provider.Replicate`|✔|❌|❌|?|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[replicate.com](https://replicate.com)|`g4f.Provider.ReplicateHome`|`llama-3-70b, mixtral-8x7b, llava-13b`|`flux-schnell, sdxl, sdxl, playground-v2.5`|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[replicate.com](https://replicate.com)|`g4f.Provider.RubiksAI`|`llama-3.1-70b, gpt-4o-mini`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[talkai.info](https://talkai.info)|`g4f.Provider.TalkAi`|✔|❌|❌|✔|![Disabled](https://img.shields.io/badge/Disabled-red)|❌|
|[teach-anything.com](https://www.teach-anything.com)|`g4f.Provider.TeachAnything`|`llama-3.1-70b`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[beta.theb.ai](https://beta.theb.ai)|`g4f.Provider.Theb`|✔|❌|❌|✔|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[beta.theb.ai](https://beta.theb.ai)|`g4f.Provider.ThebApi`|✔|❌|❌|✔|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[console.upstage.ai/playground/chat](https://console.upstage.ai/playground/chat)|`g4f.Provider.Upstage`|`solar-pro, solar-mini`|❌|❌|✔|![Active](https://img.shields.io/badge/Active-brightgreen)|❌|
|[whiterabbitneo.com](https://www.whiterabbitneo.com)|`g4f.Provider.WhiteRabbitNeo`|✔|❌|❌|?|![Unknown](https://img.shields.io/badge/Unknown-grey)|✔|
|[you.com](https://you.com)|`g4f.Provider.You`|✔|✔|✔|✔|![Unknown](https://img.shields.io/badge/Unknown-grey)|❌+✔|

## Models

### Text Models
| Model | Base Provider | Providers | Website |
|-------|---------------|-----------|---------|
|gpt-3|OpenAI|1+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-base)|
|gpt-3.5-turbo|OpenAI|5+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-3-5-turbo)|
|gpt-4|OpenAI|7+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4)|
|gpt-4-turbo|OpenAI|3+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4)|
|gpt-4o|OpenAI|10+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-4o)|
|gpt-4o-mini|OpenAI|14+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-4o-mini)|
|o1|OpenAI|1+ Providers|[platform.openai.com](https://openai.com/index/introducing-openai-o1-preview/)|
|o1-mini|OpenAI|2+ Providers|[platform.openai.com](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/)|
|llama-2-7b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/meta-llama/Llama-2-7b)|
|llama-2-13b|Meta Llama|1+ Providers|[llama.com](https://www.llama.com/llama2/)|
|llama-3-8b|Meta Llama|4+ Providers|[ai.meta.com](https://ai.meta.com/blog/meta-llama-3/)|
|llama-3-70b|Meta Llama|4+ Providers|[ai.meta.com](https://ai.meta.com/blog/meta-llama-3/)|
|llama-3.1-8b|Meta Llama|7+ Providers|[ai.meta.com](https://ai.meta.com/blog/meta-llama-3-1/)|
|llama-3.1-70b|Meta Llama|14+ Providers|[ai.meta.com](https://ai.meta.com/blog/meta-llama-3-1/)|
|llama-3.1-405b|Meta Llama|5+ Providers|[ai.meta.com](https://ai.meta.com/blog/meta-llama-3-1/)|
|llama-3.2-1b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/meta-llama/Llama-3.2-1B)|
|llama-3.2-3b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/blog/llama32)|
|llama-3.2-11b|Meta Llama|3+ Providers|[ai.meta.com](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)|
|llama-3.2-90b|Meta Llama|2+ Providers|[ai.meta.com](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)|
|llamaguard-7b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/meta-llama/LlamaGuard-7b)|
|llamaguard-2-8b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B)|
|mistral-7b|Mistral AI|4+ Providers|[mistral.ai](https://mistral.ai/news/announcing-mistral-7b/)|
|mixtral-8x7b|Mistral AI|6+ Providers|[mistral.ai](https://mistral.ai/news/mixtral-of-experts/)|
|mixtral-8x22b|Mistral AI|3+ Providers|[mistral.ai](https://mistral.ai/news/mixtral-8x22b/)|
|mistral-nemo|Mistral AI|2+ Providers|[huggingface.co](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)|
|mistral-large|Mistral AI|2+ Providers|[mistral.ai](https://mistral.ai/news/mistral-large-2407/)|
|mixtral-8x7b-dpo|NousResearch|1+ Providers|[huggingface.co](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)|
|yi-34b|NousResearch|1+ Providers|[huggingface.co](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B)|
|hermes-3|NousResearch|2+ Providers|[huggingface.co](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)|
|gemini|Google DeepMind|1+ Providers|[deepmind.google](http://deepmind.google/technologies/gemini/)|
|gemini-flash|Google DeepMind|4+ Providers|[deepmind.google](https://deepmind.google/technologies/gemini/flash/)|
|gemini-pro|Google DeepMind|10+ Providers|[deepmind.google](https://deepmind.google/technologies/gemini/pro/)|
|gemma-2b|Google|5+ Providers|[huggingface.co](https://huggingface.co/google/gemma-2b)|
|gemma-2b-9b|Google|1+ Providers|[huggingface.co](https://huggingface.co/google/gemma-2-9b)|
|gemma-2b-27b|Google|2+ Providers|[huggingface.co](https://huggingface.co/google/gemma-2-27b)|
|gemma-7b|Google|1+ Providers|[huggingface.co](https://huggingface.co/google/gemma-7b)|
|gemma-2|Google|2+ Providers|[huggingface.co](https://huggingface.co/blog/gemma2)|
|gemma_2_27b|Google|1+ Providers|[huggingface.co](https://huggingface.co/blog/gemma2)|
|claude-2.1|Anthropic|1+ Providers|[anthropic.com](https://www.anthropic.com/news/claude-2)|
|claude-3-haiku|Anthropic|4+ Providers|[anthropic.com](https://www.anthropic.com/news/claude-3-haiku)|
|claude-3-sonnet|Anthropic|2+ Providers|[anthropic.com](https://www.anthropic.com/news/claude-3-family)|
|claude-3-opus|Anthropic|2+ Providers|[anthropic.com](https://www.anthropic.com/news/claude-3-family)|
|claude-3.5-sonnet|Anthropic|6+ Providers|[anthropic.com](https://www.anthropic.com/news/claude-3-5-sonnet)|
|blackboxai|Blackbox AI|2+ Providers|[docs.blackbox.chat](https://docs.blackbox.chat/blackbox-ai-1)|
|blackboxai-pro|Blackbox AI|1+ Providers|[docs.blackbox.chat](https://docs.blackbox.chat/blackbox-ai-1)|
|yi-1.5-9b|01-ai|1+ Providers|[huggingface.co](https://huggingface.co/01-ai/Yi-1.5-9B)|
|phi-2|Microsoft|1+ Providers|[huggingface.co](https://huggingface.co/microsoft/phi-2)|
|phi-3-medium-4k|Microsoft|1+ Providers|[huggingface.co](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)|
|phi-3.5-mini|Microsoft|2+ Providers|[huggingface.co](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)|
|dbrx-instruct|Databricks|1+ Providers|[huggingface.co](https://huggingface.co/databricks/dbrx-instruct)|
|command-r-plus|CohereForAI|1+ Providers|[docs.cohere.com](https://docs.cohere.com/docs/command-r-plus)|
|sparkdesk-v1.1|iFlytek|1+ Providers|[xfyun.cn](https://www.xfyun.cn/doc/spark/Guide.html)|
|qwen|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen)|
|qwen-1.5-0.5b|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen1.5-0.5B)|
|qwen-1.5-7b|Qwen|2+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen1.5-7B)|
|qwen-1.5-14b|Qwen|3+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen1.5-14B)|
|qwen-1.5-72b|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen1.5-72B)|
|qwen-1.5-110b|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen1.5-110B)|
|qwen-1.5-1.8b|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen1.5-1.8B)|
|qwen-2-72b|Qwen|4+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen2-72B)|
|glm-3-6b|Zhipu AI|1+ Providers|[github.com/THUDM/ChatGLM3](https://github.com/THUDM/ChatGLM3)|
|glm-4-9B|Zhipu AI|1+ Providers|[github.com/THUDM/GLM-4](https://github.com/THUDM/GLM-4)|
|solar-1-mini|Upstage|1+ Providers|[upstage.ai/](https://www.upstage.ai/feed/product/solarmini-performance-report)|
|solar-10-7b|Upstage|1+ Providers|[huggingface.co](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)|
|solar-pro|Upstage|1+ Providers|[huggingface.co](https://huggingface.co/upstage/solar-pro-preview-instruct)|
|pi|Inflection|1+ Providers|[inflection.ai](https://inflection.ai/blog/inflection-2-5)|
|deepseek|DeepSeek|1+ Providers|[deepseek.com](https://www.deepseek.com/)|
|wizardlm-2-7b|WizardLM|1+ Providers|[huggingface.co](https://huggingface.co/dreamgen/WizardLM-2-7B)|
|wizardlm-2-8x22b|WizardLM|2+ Providers|[huggingface.co](https://huggingface.co/alpindale/WizardLM-2-8x22B)|
|sh-n-7b|Together|1+ Providers|[huggingface.co](https://huggingface.co/togethercomputer/StripedHyena-Nous-7B)|
|llava-13b|Yorickvp|1+ Providers|[huggingface.co](https://huggingface.co/liuhaotian/llava-v1.5-13b)|
|minicpm-llama-3-v2.5|OpenBMB|1+ Providers|[huggingface.co](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)|
|lzlv-70b|Lzlv|1+ Providers|[huggingface.co](https://huggingface.co/lizpreciatior/lzlv_70b_fp16_hf)|
|openchat-3.5|OpenChat|1+ Providers|[huggingface.co](https://huggingface.co/openchat/openchat_3.5)|
|openchat-3.6-8b|OpenChat|1+ Providers|[huggingface.co](https://huggingface.co/openchat/openchat-3.6-8b-20240522)|
|phind-codellama-34b-v2|Phind|1+ Providers|[huggingface.co](https://huggingface.co/Phind/Phind-CodeLlama-34B-v2)|
|dolphin-2.9.1-llama-3-70b|Cognitive Computations|1+ Providers|[huggingface.co](https://huggingface.co/cognitivecomputations/dolphin-2.9.1-llama-3-70b)|
|grok-2-mini|x.ai|1+ Providers|[x.ai](https://x.ai/blog/grok-2)|
|grok-2|x.ai|1+ Providers|[x.ai](https://x.ai/blog/grok-2)|
|sonar-online|Perplexity AI|2+ Providers|[docs.perplexity.ai](https://docs.perplexity.ai/)|
|sonar-chat|Perplexity AI|1+ Providers|[docs.perplexity.ai](https://docs.perplexity.ai/)|
|mythomax-l2-13b|Gryphe|1+ Providers|[huggingface.co](https://huggingface.co/Gryphe/MythoMax-L2-13b)|
|cosmosrp|Gryphe|1+ Providers|[huggingface.co](https://huggingface.co/PawanKrd/CosmosRP-8k)|
|german-7b|TheBloke|1+ Providers|[huggingface.co](https://huggingface.co/TheBloke/DiscoLM_German_7b_v1-GGUF)|
|tinyllama-1.1b|TinyLlama|1+ Providers|[huggingface.co](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)|
|cybertron-7b|TheBloke|1+ Providers|[huggingface.co](https://huggingface.co/fblgit/una-cybertron-7b-v2-bf16)|

### Image Models
| Model | Base Provider | Providers | Website |
|-------|---------------|-----------|---------|
|sdxl|Stability AI|1+ Providers|[huggingface.co](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl)|
|sdxl-lora|Stability AI|1+ Providers|[huggingface.co](https://huggingface.co/blog/lcm_lora)|
|sdxl-turbo|Stability AI|1+ Providers|[huggingface.co](https://huggingface.co/stabilityai/sdxl-turbo)|
|sd-1.5|Stability AI|1+ Providers|[huggingface.co](https://huggingface.co/runwayml/stable-diffusion-v1-5)|
|sd-3|Stability AI|1+ Providers|[huggingface.co](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3)|
|sd-3.5|Stability AI|1+ Providers|[stability.ai](https://stability.ai/news/introducing-stable-diffusion-3-5)|
|playground-v2.5|Playground AI|1+ Providers|[huggingface.co](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)|
|flux|Black Forest Labs|2+ Providers|[github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)|
|flux-pro|Black Forest Labs|2+ Providers|[github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)|
|flux-realism|Flux AI|2+ Providers|[]()|
|flux-anime|Flux AI|1+ Providers|[]()|
|flux-3d|Flux AI|1+ Providers|[]()|
|flux-disney|Flux AI|1+ Providers|[]()|
|flux-pixel|Flux AI|1+ Providers|[]()|
|flux-4o|Flux AI|1+ Providers|[]()|
|flux-schnell|Black Forest Labs|2+ Providers|[huggingface.co](https://huggingface.co/black-forest-labs/FLUX.1-schnell)|
|dalle|OpenAI|1+ Providers|[openai.com](https://openai.com/index/dall-e/)|
|dalle-2|OpenAI|1+ Providers|[openai.com](https://openai.com/index/dall-e-2/)|
|emi||1+ Providers|[]()|
|any-dark||1+ Providers|[]()|
|midjourney|Midjourney|1+ Providers|[docs.midjourney.com](https://docs.midjourney.com/docs/model-versions)|

### Vision Models
| Model | Base Provider | Providers | Website |
|-------|---------------|-----------|---------|
|gpt-4-vision|OpenAI|1+ Providers|[openai.com](https://openai.com/research/gpt-4v-system-card)|
|gemini-pro-vision|Google DeepMind|1+ Providers | [deepmind.google](https://deepmind.google/technologies/gemini/)|
|blackboxai|Blackbox AI|1+ Providers|[docs.blackbox.chat](https://docs.blackbox.chat/blackbox-ai-1)|
|minicpm-llama-3-v2.5|OpenBMB|1+ Providers | [huggingface.co](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)|

## Conclusion and Usage Tips
This document provides a comprehensive overview of various AI providers and models available for text generation, image generation, and vision tasks. **When choosing a provider or model, consider the following factors:**
   1. **Availability**: Check the status of the provider to ensure it's currently active and accessible.
   2. **Model Capabilities**: Different models excel at different tasks. Choose a model that best fits your specific needs, whether it's text generation, image creation, or vision-related tasks.
   3. **Authentication**: Some providers require authentication, while others don't. Consider this when selecting a provider for your project.
   4. **Streaming Support**: If real-time responses are important for your application, prioritize providers that offer streaming capabilities.
   5. **Vision Models**: For tasks requiring image understanding or multimodal interactions, look for providers offering vision models.

Remember to stay updated with the latest developments in the AI field, as new models and providers are constantly emerging and evolving.

---

[Return to Home](/)
