## Providers

```py
from g4f.Provider import (
    AItianhu,
    Acytoo,
    Aichat,
    Ails,
    Bard,
    Bing,
    ChatgptAi,
    ChatgptLogin,
    DeepAi,
    DfeHub,
    EasyChat,
    GetGpt,
    H2o,
    Raycast,
    opchatgpts,
)
# Usage:
response = g4f.ChatCompletion.create(..., provider=ProviderName)
```

| Website| Provider| gpt-3.5 | gpt-4 | Streaming | Status | Auth |
| --- | --- | --- | --- | --- | --- | --- |
| [www.aitianhu.com](https://www.aitianhu.com/api/chat-process) | `g4f.Provider.AItianhu` | ✔️ | ❌ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat.acytoo.com](https://chat.acytoo.com/api/completions) | `g4f.Provider.Acytoo` | ✔️ | ❌ | ❌ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat-gpt.org](https://chat-gpt.org/chat) | `g4f.Provider.Aichat` | ✔️ | ❌ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [ai.ls](https://ai.ls) | `g4f.Provider.Ails` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [bard.google.com](https://bard.google.com) | `g4f.Provider.Bard` | ❌ | ❌ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [bing.com](https://bing.com/chat) | `g4f.Provider.Bing` | ❌ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chatgpt.ai](https://chatgpt.ai/gpt-4/) | `g4f.Provider.ChatgptAi` | ❌ | ✔️ | ❌ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chatgptlogin.ac](https://chatgptlogin.ac) | `g4f.Provider.ChatgptLogin` | ✔️ | ❌ | ❌ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [api.deepai.org](https://api.deepai.org/) | `g4f.Provider.DeepAi` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat.dfehub.com](https://chat.dfehub.com/api/chat) | `g4f.Provider.DfeHub` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [free.easychat.work](https://free.easychat.work) | `g4f.Provider.EasyChat` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat.getgpt.world](https://chat.getgpt.world/) | `g4f.Provider.GetGpt` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [gpt-gm.h2o.ai](https://gpt-gm.h2o.ai) | `g4f.Provider.H2o` | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [backend.raycast.com](https://backend.raycast.com/api/v1/ai/chat_completions) | `g4f.Provider.Raycast` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [opchatgpts.net](https://opchatgpts.net) | `g4f.Provider.opchatgpts` | ✔️ | ❌ | ❌ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [aiservice.vercel.app](https://aiservice.vercel.app/api/chat/answer) | `g4f.Provider.AiService` | ✔️ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [b.ai-huan.xyz](https://b.ai-huan.xyz) | `g4f.Provider.BingHuan` | ✔️ | ✔️ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [forefront.com](https://forefront.com) | `g4f.Provider.Forefront` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [liaobots.com](https://liaobots.com) | `g4f.Provider.Liaobots` | ✔️ | ✔️ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ✔️ |
| [supertest.lockchat.app](http://supertest.lockchat.app) | `g4f.Provider.Lockchat` | ✔️ | ✔️ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [theb.ai](https://theb.ai) | `g4f.Provider.Theb` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ✔️ |
| [play.vercel.ai](https://play.vercel.ai) | `g4f.Provider.Vercel` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [wewordle.org](https://wewordle.org/gptapi/v1/android/turbo) | `g4f.Provider.Wewordle` | ✔️ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [you.com](https://you.com) | `g4f.Provider.You` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ✔️ |
| [chat9.yqcloud.top](https://chat9.yqcloud.top/) | `g4f.Provider.Yqcloud` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
