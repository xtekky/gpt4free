


# G4F - Providers and Models

This document provides an overview of various AI providers and models, including text generation, image generation, and vision capabilities. It aims to help users navigate the diverse landscape of AI services and choose the most suitable option for their needs.

> **Note**: See our [Authentication Guide](authentication.md) for authentication instructions for the provider.


## Table of Contents
  - [Providers](#providers)
    - [No auth required](#providers-not-needs-auth)
    - [HuggingSpace](#providers-huggingspace)
    - [Needs auth](#providers-needs-auth)
  - [Models](#models)
    - [Text Models](#text-models)
    - [Image Models](#image-models)
  - [Conclusion and Usage Tips](#conclusion-and-usage-tips)

---
## Providers
**Authentication types:**
- **Get API key** - Requires an API key for authentication. You need to obtain an API key from the provider's website to use their services.
- **Manual cookies** - Requires manual browser cookies setup. You need to be logged in to the provider's website to use their services.
- **Automatic cookies** - Browser cookies authentication that is automatically fetched. No manual setup needed.
- **Optional API key** - Works without authentication, but you can provide an API key for better rate limits or additional features. The service is usable without an API key.
- **API key / Cookies** - Supports both authentication methods. You can use either an API key or browser cookies for authentication.
- **No auth required** - No authentication needed. The service is publicly available without any credentials.

**Symbols:**
- ✔ - Feature is supported
- ❌ - Feature is not supported
- ✔ _**(n+)**_ - Number of additional models supported by the provider but not publicly listed

---
### Providers No auth required
| Website | API Credentials | Provider | Text Models | Image Models | Vision (Image Upload) | Stream | Status |
|----------|-------------|--------------|---------------|--------|--------|------|------|
|[aichatfree.info](https://aichatfree.info)|No auth required|`g4f.Provider.AIChatFree`|`gemini-1.5-pro` _**(1+)**_|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[autonomous.ai](https://www.autonomous.ai/anon/)|No auth required|`g4f.Provider.AutonomousAI`|`llama-3.3-70b, qwen-2.5-coder-32b, hermes-3, llama-3.2-90b, llama-3.3-70b, llama-3-2-70b`|✔|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[blackbox.ai](https://www.blackbox.ai)|No auth required|`g4f.Provider.Blackbox`|`blackboxai, gpt-4, gpt-4o, gemini-1.5-flash, gemini-1.5-pro, claude-3.5-sonnet, blackboxai-pro, llama-3.1-8b, llama-3.1-70b, llama-3-1-405b, llama-3.3-70b, mixtral-7b, deepseek-chat, dbrx-instruct, qwq-32b, hermes-2-dpo, deepseek-r1` _**(+31)**_|`flux`|`blackboxai, gpt-4o, gemini-1.5-pro, gemini-1.5-flash, llama-3.1-8b, llama-3.1-70b, llama-3.1-405b`|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[cablyai.com](https://cablyai.com)|No auth required|`g4f.Provider.CablyAI`|`cably-80b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[chatglm.cn](https://chatglm.cn)|No auth required|`g4f.Provider.ChatGLM`|`glm-4`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[chatgpt.com](https://chatgpt.com)|No auth required|`g4f.Provider.ChatGpt`|✔ _**(+7)**_|❌|❌|✔|![Error](https://img.shields.io/badge/HTTPError-f48d37)|
|[chatgpt.es](https://chatgpt.es)|No auth required|`g4f.Provider.ChatGptEs`|`gpt-4, gpt-4o, gpt-4o-mini`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[chatgptt.me](https://chatgptt.me)|No auth required|`g4f.Provider.ChatGptt`|`gpt-4, gpt-4o, gpt-4o-mini`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[playground.ai.cloudflare.com](https://playground.ai.cloudflare.com)|[Automatic cookies](https://playground.ai.cloudflare.com)|`g4f.Provider.Cloudflare`|`llama-2-7b, llama-3-8b, llama-3.1-8b, llama-3.2-1b, qwen-1.5-7b`|❌|❌|✔|![Error](https://img.shields.io/badge/Active-brightgreen)|❌|
|[copilot.microsoft.com](https://copilot.microsoft.com)|Optional API key|`g4f.Provider.Copilot`|`gpt-4, gpt-4o`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[darkai.foundation](https://darkai.foundation)|No auth required|`g4f.Provider.DarkAI`|`gpt-3.5-turbo, gpt-4o, llama-3.1-70b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[duckduckgo.com/aichat](https://duckduckgo.com/aichat)|No auth required|`g4f.Provider.DDG`|`gpt-4, gpt-4o-mini, claude-3-haiku, llama-3.1-70b, mixtral-8x7b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[deepinfra.com/chat](https://deepinfra.com/chat)|No auth required|`g4f.Provider.DeepInfraChat`|`llama-3.1-8b, llama-3.1-70b, deepseek-chat, qwq-32b, wizardlm-2-8x22b, wizardlm-2-7b, qwen-2.5-72b, qwen-2.5-coder-32b, nemotron-70b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[chat10.free2gpt.xyz](https://chat10.free2gpt.xyz)|No auth required|`g4f.Provider.Free2GPT`|`mistral-7b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[freegptsnav.aifree.site](https://freegptsnav.aifree.site)|No auth required|`g4f.Provider.FreeGpt`|`gemini-1.5-pro`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[app.giz.ai/assistant](https://app.giz.ai/assistant)|No auth required|`g4f.Provider.GizAI`|`gemini-1.5-flash`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[glider.so](https://glider.so)|No auth required|`g4f.Provider.Glider`|`llama-3.1-70b, llama-3.1-8b, llama-3.2-3b, deepseek-r1`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[gprochat.com](https://gprochat.com)|No auth required|`g4f.Provider.GPROChat`|`gemini-1.5-pro`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[hailuo.ai](https://www.hailuo.ai)|No auth required|`g4f.Provider.HailuoAI`|`MiniMax` _**(1)**_|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[editor.imagelabs.net](https://editor.imagelabs.net)|No auth required|`g4f.Provider.ImageLabs`|`gemini-1.5-pro`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[editor.imagelabs.net](editor.imagelabs.net)|No auth required|`g4f.Provider.ImageLabs`|❌|`sdxl-turbo`|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[huggingface.co/spaces](https://huggingface.co/spaces)|Optional API key|`g4f.Provider.HuggingSpace`|`qvq-72b, qwen-2-72b, command-r, command-r-plus, command-r7b`|`flux-dev, flux-schnell, sd-3.5`|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[jmuz.me](https://jmuz.me)|Optional API key|`g4f.Provider.Jmuz`|`claude-3-haiku, claude-3-opus, claude-3-haiku, claude-3.5-sonnet, deepseek-r1, deepseek-chat, gemini-exp, gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash-thinking, gpt-4, gpt-4o, gpt-4o-mini, llama-3-70b, llama-3-8b, llama-3.1-405b, llama-3.1-70b, llama-3.1-8b, llama-3.2-11b, llama-3.2-90b, llama-3.3-70b, mixtral-8x7b, qwen-2.5-72b, qwen-2.5-coder-32b, qwq-32b, wizardlm-2-8x22b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[liaobots.work](https://liaobots.work)|[Automatic cookies](https://liaobots.work)|`g4f.Provider.Liaobots`|`grok-2, gpt-4o-mini, gpt-4o, gpt-4, o1-preview, o1-mini, claude-3-opus, claude-3.5-sonnet, claude-3-sonnet, gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash, gemini-2.0-flash-thinking`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[mhystical.cc](https://mhystical.cc)|[Optional API key](https://mhystical.cc/dashboard)|`g4f.Provider.Mhystical`|`gpt-4`|❌|❌|✔|![Error](https://img.shields.io/badge/Active-brightgreen)|
|[oi-vscode-server.onrender.com](https://oi-vscode-server.onrender.com)|No auth required|`g4f.Provider.OIVSCode`|`gpt-4o-mini`|❌|`gpt-4o-mini`|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[labs.perplexity.ai](https://labs.perplexity.ai)|No auth required|`g4f.Provider.PerplexityLabs`|`sonar-online, sonar-chat, llama-3.3-70b, llama-3.1-8b, llama-3.1-70b, lfm-40b`|❌|❌|✔|![Error](https://img.shields.io/badge/Active-brightgreen)|
|[pi.ai/talk](https://pi.ai/talk)|[Manual cookies](https://pi.ai/talk)|`g4f.Provider.Pi`|`pi`|❌|❌|✔|![Error](https://img.shields.io/badge/Active-brightgreen)|
|[pizzagpt.it](https://www.pizzagpt.it)|No auth required|`g4f.Provider.Pizzagpt`|`gpt-4o-mini`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[pollinations.ai](https://pollinations.ai)|No auth required|`g4f.Provider.PollinationsAI`|`gpt-4o-mini, gpt-4o, qwen-2.5-72b, qwen-2.5-coder-32b, llama-3.3-70b, mistral-nemo, deepseek-chat, llama-3.1-8b, deepseek-r1` _**(2+)**_|`flux, flux-pro, flux-realism, flux-cablyai, flux-anime, flux-3d, midjourney, dall-e-3, sdxl-turbo`|gpt-4o, gpt-4o-mini|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[app.prodia.com](https://app.prodia.com)|No auth required|`g4f.Provider.Prodia`|❌|✔ _**(46)**_|❌|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[teach-anything.com](https://www.teach-anything.com)|No auth required|`g4f.Provider.TeachAnything`|`llama-3.1-70b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[you.com](https://you.com)|[Manual cookies](https://you.com)|`g4f.Provider.You`|✔|✔|✔|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[chat9.yqcloud.top](https://chat9.yqcloud.top)|No auth required|`g4f.Provider.Yqcloud`|`gpt-4`|✔|✔|✔|![](https://img.shields.io/badge/Active-brightgreen)|

---
### Providers HuggingSpace
| Website | API Credentials | Provider | Text Models | Image Models | Vision (Image Upload) | Stream | Status | Auth |
|----------|-------------|--------------|---------------|--------|--------|------|------|------|
|[black-forest-labs-flux-1-dev.hf.space](https://black-forest-labs-flux-1-dev.hf.space)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.BlackForestLabsFlux1Dev`|❌|`flux-dev`|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[black-forest-labs-flux-1-schnell.hf.space](https://black-forest-labs-flux-1-schnell.hf.space)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.BlackForestLabsFlux1Schnell`|❌|`flux-schnell`|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[cohereforai-c4ai-command.hf.space](https://cohereforai-c4ai-command.hf.space)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.CohereForAI`|`command-r, command-r-plus, command-r7b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[huggingface.co/spaces/deepseek-ai/Janus-Pro-7B](https://huggingface.co/spaces/deepseek-ai/Janus-Pro-7B)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.Janus_Pro_7B`|✔|✔|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[qwen-qvq-72b-preview.hf.space](https://qwen-qvq-72b-preview.hf.space)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.Qwen_QVQ_72B`|`qvq-72b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[qwen-qwen2-5-1m-demo.hf.space](https://qwen-qwen2-5-1m-demo.hf.space)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.Qwen_Qwen_2_5M_Demo`|`qwen-2.5-1m-demo`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[qwen-qwen2-72b-instruct.hf.space](https://qwen-qwen2-72b-instruct.hf.space)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.Qwen_Qwen_2_72B_Instruct`|`qwen-2-72b`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[stabilityai-stable-diffusion-3-5-large.hf.space](https://stabilityai-stable-diffusion-3-5-large.hf.space)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.StableDiffusion35Large`|❌|`sd-3.5`|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[voodoohop-flux-1-schnell.hf.space](https://voodoohop-flux-1-schnell.hf.space)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.VoodoohopFlux1Schnell`|❌|`flux-schnell`|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|

---
### Providers Needs Auth
| Website | API Credentials | Provider | Text Models | Image Models | Vision (Image Upload) | Stream | Status |
|----------|-------------|--------------|---------------|--------|--------|------|------|
|[console.anthropic.com](https://console.anthropic.com)|[Get API key](https://console.anthropic.com/settings/keys)|`g4f.Provider.Anthropic`|✔ _**(8+)**_|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[bing.com/images/create](https://www.bing.com/images/create)|[Manual cookies](https://www.bing.com)|`g4f.Provider.BingCreateImages`|❌|`dall-e-3`|❌|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[inference.cerebras.ai](https://inference.cerebras.ai/)|[Get API key](https://cloud.cerebras.ai)|`g4f.Provider.Cerebras`|✔ _**(3+)**_|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[copilot.microsoft.com](https://copilot.microsoft.com)|[Manual cookies](https://copilot.microsoft.com)|`g4f.Provider.CopilotAccount`|✔ _**(1+)**_|✔ _**(1+)**_|✔ _**(1+)**_|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[deepinfra.com](https://deepinfra.com)|[Get API key](https://deepinfra.com/dash/api_keys)|`g4f.Provider.DeepInfra`|✔ _**(17+)**_|✔ _**(6+)**_|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[platform.deepseek.com](https://platform.deepseek.com)|[Get API key](https://platform.deepseek.com/api_keys)|`g4f.Provider.DeepSeek`|✔ _**(1+)**_|❌|❌|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[gemini.google.com](https://gemini.google.com)|[Manual cookies](https://gemini.google.com)|`g4f.Provider.Gemini`|`gemini, gemini-1.5-flash, gemini-1.5-pro`|`gemini`|`gemini`|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[ai.google.dev](https://ai.google.dev)|[Get API key](https://aistudio.google.com/u/0/apikey)|`g4f.Provider.GeminiPro`|`gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash`|❌|`gemini-1.5-pro`|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[developers.sber.ru/gigachat](https://developers.sber.ru/gigachat)|[Manual cookies](https://developers.sber.ru/gigachat)|`g4f.Provider.GigaChat`|✔ _**(3+)**_|❌|❌|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[github.com/copilot](https://github.com/copilot)|[Manual cookies](https://github.com/copilot)|`g4f.Provider.GithubCopilot`|✔ _**(4+)**_|❌|❌|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[glhf.chat](https://glhf.chat)|[Get API key](https://glhf.chat/user-settings/api)|`g4f.Provider.GlhfChat`|✔ _**(22+)**_|❌|❌|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[console.groq.com/playground](https://console.groq.com/playground)|[Get API key](https://console.groq.com/keys)|`g4f.Provider.Groq`|✔ _**(18+)**_|❌|✔|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[huggingface.co/chat](https://huggingface.co/chat)|[Manual cookies](https://huggingface.co/chat)|`g4f.Provider.HuggingChat`|`qwen-2.5-72b, llama-3.3-70b, command-r-plus, deepseek-r1, qwq-32b, nemotron-70b, llama-3.2-11b, mistral-nemo, phi-3.5-mini`|`flux-dev, flux-schnell`|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[huggingface.co/chat](https://huggingface.co/chat)|[API key / Cookies](https://huggingface.co/settings/tokens)|`g4f.Provider.HuggingFace`|✔ _**(47+)**_|✔ _**(9+)**_|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[api-inference.huggingface.co](https://api-inference.huggingface.co)|[Get API key](https://huggingface.co/settings/tokens)|`g4f.Provider.HuggingFaceAPI`|✔ _**(9+)**_|✔ _**(2+)**_|✔ _**(1+)**_|❌|![](https://img.shields.io/badge/Active-brightgreen)|✔|
|[meta.ai](https://www.meta.ai)|[Manual cookies](https://www.meta.ai)|`g4f.Provider.MetaAI`|`meta-ai`|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|✔|
|[meta.ai](https://www.meta.ai)|[Manual cookies](https://www.meta.ai)|`g4f.Provider.MetaAIAccount`|❌|`meta-ai`|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[designer.microsoft.com](https://designer.microsoft.com)|[Manual cookies](https://designer.microsoft.com)|`g4f.Provider.MicrosoftDesigner`|❌|`dall-e-3`|❌|❌|![](https://img.shields.io/badge/Active-brightgreen)|
|[hailuo.ai/chat](https://www.hailuo.ai/chat)|[Get API key](https://intl.minimaxi.com/user-center/basic-information/interface-key)|`g4f.Provider.MiniMax`|`MiniMax`  _**(1)**_|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[platform.openai.com](https://platform.openai.com)|[Get API key](https://platform.openai.com/settings/organization/api-keys)|`g4f.Provider.OpenaiAPI`|✔|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[chatgpt.com](https://chatgpt.com)|[Manual cookies](https://chatgpt.com)|`g4f.Provider.OpenaiChat`|`gpt-4o, gpt-4o-mini, gpt-4` _**(8+)**_|✔_**(1)**_|✔_**(8+)**_|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[perplexity.ai](https://www.perplexity.ai)|[Get API key](https://www.perplexity.ai/settings/api)|`g4f.Provider.PerplexityApi`|✔ _**(6+)**_|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[chat.reka.ai](https://chat.reka.ai)|[Manual cookies](https://chat.reka.ai)|`g4f.Provider.Reka`|`reka-core`|❌|✔|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[replicate.com](https://replicate.com)|[Get API key](https://replicate.com/account/api-tokens)|`g4f.Provider.Replicate`|✔ _**(1+)**_|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[beta.theb.ai](https://beta.theb.ai)|[Get API key](https://beta.theb.ai)|`g4f.Provider.ThebApi`|✔ _**(21+)**_|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[whiterabbitneo.com](https://www.whiterabbitneo.com)|[Manual cookies](https://www.whiterabbitneo.com)|`g4f.Provider.WhiteRabbitNeo`|✔|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|
|[console.x.ai](https://console.x.ai)|[Get API key](https://console.x.ai)|`g4f.Provider.xAI`|✔|❌|❌|✔|![](https://img.shields.io/badge/Active-brightgreen)|

---
## Models

### Text Models
| Model | Base Provider | Providers | Website |
|-------|---------------|-----------|---------|
|gpt-3|OpenAI|1+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-3-5-turbo)|
|gpt-3.5-turbo|OpenAI|1+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-3-5-turbo)|
|gpt-4|OpenAI|11+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4)|
|gpt-4o|OpenAI|9+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-4o)|
|gpt-4o-mini|OpenAI|8+ Providers|[platform.openai.com](https://platform.openai.com/docs/models/gpt-4o-mini)|
|o1|OpenAI|1+ Providers|[openai.com](https://openai.com/index/introducing-openai-o1-preview/)|
|o1-preview|OpenAI|1+ Providers|[openai.com](https://openai.com/index/introducing-openai-o1-preview/)|
|o1-mini|OpenAI|1+ Providers|[openai.com](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/)|
|gigachat|GigaChat|1+ Providers|[developers.sber.ru/gigachat](https://developers.sber.ru/gigachat)|
|meta-ai|Meta|1+ Providers|[ai.meta.com](https://ai.meta.com/)|
|llama-2-7b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/meta-llama/Llama-2-7b)|
|llama-3-8b|Meta Llama|2+ Providers|[ai.meta.com](https://ai.meta.com/blog/meta-llama-3/)|
|llama-3-70b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/meta-llama/Meta-Llama-3-70B)|
|llama-3.1-8b|Meta Llama|6+ Providers|[ai.meta.com](https://ai.meta.com/blog/meta-llama-3-1/)|
|llama-3.1-70b|Meta Llama|6+ Providers|[ai.meta.com](https://ai.meta.com/blog/meta-llama-3-1/)|
|llama-3.1-405b|Meta Llama|2+ Providers|[huggingface.co](https://huggingface.co/meta-llama/Llama-3.1-405B)|
|llama-3.2-1b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/meta-llama/Llama-3.2-1B)|
|llama-3.2-3b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/meta-llama/Llama-3.2-3B)|
|llama-3.2-11b|Meta Llama|3+ Providers|[ai.meta.com](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)|
|llama-3.2-90b|Meta Llama|1+ Providers|[huggingface.co](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision)|
|llama-3.3-70b|Meta Llama|6+ Providers|[ai.meta.com](https://ai.meta.com/blog/llama-3-3/)|
|mixtral-7b|Mistral|1+|[mistral.ai](https://mistral.ai/news/mixtral-of-experts/)|
|mixtral-8x7b|Mistral|2+ Providers|[mistral.ai](https://mistral.ai/news/mixtral-of-experts/)|
|mistral-nemo|Mistral|3+ Providers|[huggingface.co](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)|
|hermes-2-dpo|NousResearch|1+ Providers|[huggingface.co](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)|
|phi-3.5-mini|Microsoft|1+ Providers|[huggingface.co](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)|
|wizardlm-2-7b|Microsoft|1+ Providers|[wizardlm.github.io](https://wizardlm.github.io/WizardLM2/)|
|wizardlm-2-8x22b|Microsoft|2+ Providers|[wizardlm.github.io](https://wizardlm.github.io/WizardLM2/)|
|gemini|Google DeepMind|1+|[deepmind.google](http://deepmind.google/technologies/gemini/)|
|gemini-exp|Google DeepMind|1+ Providers|[blog.google](https://blog.google/feed/gemini-exp-1206/)|
|gemini-1.5-flash|Google DeepMind|5+ Providers|[deepmind.google](https://deepmind.google/technologies/gemini/flash/)|
|gemini-1.5-pro|Google DeepMind|6+ Providers|[deepmind.google](https://deepmind.google/technologies/gemini/pro/)|
|gemini-2.0-flash|Google DeepMind|2+ Providers|[deepmind.google](https://deepmind.google/technologies/gemini/flash/)|
|gemini-2.0-flash-thinking|Google DeepMind|1+ Providers|[ai.google.dev](https://ai.google.dev/gemini-api/docs/thinking-mode)|
|claude-3-haiku|Anthropic|2+ Providers|[anthropic.com](https://www.anthropic.com/news/claude-3-haiku)|
|claude-3-sonnet|Anthropic|1+ Providers|[anthropic.com](https://www.anthropic.com/news/claude-3-family)|
|claude-3-opus|Anthropic|2+ Providers|[anthropic.com](https://www.anthropic.com/news/claude-3-family)|
|claude-3.5-sonnet|Anthropic|3+ Providers|[anthropic.com](https://www.anthropic.com/news/claude-3-5-sonnet)|
|reka-core|Reka AI|1+ Providers|[reka.ai](https://www.reka.ai/ourmodels)|
|blackboxai|Blackbox AI|1+ Providers|[docs.blackbox.chat](https://docs.blackbox.chat/blackbox-ai-1)|
|blackboxai-pro|Blackbox AI|1+ Providers|[docs.blackbox.chat](https://docs.blackbox.chat/blackbox-ai-1)|
|command-r|CohereForAI|1+ Providers|[docs.cohere.com](https://docs.cohere.com/docs/command-r-plus)|
|command-r-plus|CohereForAI|2+ Providers|[docs.cohere.com](https://docs.cohere.com/docs/command-r-plus)|
|command-r7b|CohereForAI|1+ Providers|[huggingface.co](https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024)|
|qwen-1.5-7b|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen1.5-7B)|
|qwen-2-72b|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen2-72B)|
|qwen-2-vl-7b|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen2-VL-7B)|
|qwen-2.5-72b|Qwen|3+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)|
|qwen-2.5-coder-32b|Qwen|4+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen2.5-Coder-32B)|
|qwen-2.5-1m-demo|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen/Qwen2.5-1M-Demo)|
|qwq-32b|Qwen|4+ Providers|[huggingface.co](https://huggingface.co/Qwen/QwQ-32B-Preview)|
|qvq-72b|Qwen|1+ Providers|[huggingface.co](https://huggingface.co/Qwen/QVQ-72B-Preview)|
|pi|Inflection|1+ Providers|[inflection.ai](https://inflection.ai/blog/inflection-2-5)|
|deepseek-chat|DeepSeek|3+ Providers|[huggingface.co](https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat)|
|deepseek-v3|DeepSeek|2+ Providers|[api-docs.deepseek.com](https://api-docs.deepseek.com/news/news250120)|
|deepseek-r1|DeepSeek|6+ Providers|[api-docs.deepseek.com](https://api-docs.deepseek.com/news/news250120)|
|grok-2|x.ai|1+|[x.ai](https://x.ai/blog/grok-2)|
|sonar|Perplexity AI|1+ Providers|[docs.perplexity.ai](https://docs.perplexity.ai/)|
|sonar-pro|Perplexity AI|1+ Providers|[docs.perplexity.ai](https://docs.perplexity.ai/)|
|sonar-reasoning|Perplexity AI|1+ Providers|[docs.perplexity.ai](https://docs.perplexity.ai/)|
|nemotron-70b|Nvidia|3+ Providers|[build.nvidia.com](https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct)|
|dbrx-instruct|Databricks|1+ Providers|[huggingface.co](https://huggingface.co/databricks/dbrx-instruct)|
|p1|PollinationsAI|1+ Providers|[pollinations.ai](https://pollinations.ai/)|
|cably-80b|CablyAI|1+ Providers|[cablyai.com](https://cablyai.com)|
|glm-4|THUDM|1+ Providers|[github.com/THUDM](https://github.com/THUDM/GLM-4)|
|mini_max|MiniMax|1+ Providers|[hailuo.ai](https://www.hailuo.ai/)|
|evil|Evil Mode - Experimental|1+ Providers||

---

### Image Models
| Model | Base Provider | Providers | Website |
|-------|---------------|-----------|---------|
|sdxl-turbo|Stability AI|2+ Providers|[huggingface.co](https://huggingface.co/stabilityai/sdxl-turbo)|
|sd-3.5|Stability AI|1+ Providers|[huggingface.co](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)|
|flux|Black Forest Labs|3+ Providers|[github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)|
|flux-pro|Black Forest Labs|1+ Providers|[huggingface.co](https://huggingface.co/enhanceaiteam/FLUX.1-Pro)|
|flux-dev|Black Forest Labs|3+ Providers|[huggingface.co](https://huggingface.co/black-forest-labs/FLUX.1-dev)|
|flux-schnell|Black Forest Labs|3+ Providers|[huggingface.co](https://huggingface.co/black-forest-labs/FLUX.1-schnell)|
|dall-e-3|OpenAI|5+ Providers|[openai.com](https://openai.com/index/dall-e/)|
|midjourney|Midjourney|1+ Providers|[docs.midjourney.com](https://docs.midjourney.com/docs/model-versions)|


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
