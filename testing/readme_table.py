from g4f.Provider import (
    Ails,
    You,
    Bing,
    Yqcloud,
    Theb,
    Aichat,
    Bard,
    Vercel,
    Forefront,
    Lockchat,
    Liaobots,
    H2o,
    ChatgptLogin,
    DeepAi,
    GetGpt
)

from urllib.parse import urlparse

providers = [
    Ails,
    You,
    Bing,
    Yqcloud,
    Theb,
    Aichat,
    Bard,
    Vercel,
    Forefront,
    Lockchat,
    Liaobots,
    H2o,
    ChatgptLogin,
    DeepAi,
    GetGpt
]

# | Website| Provider| gpt-3.5-turbo | gpt-4 | Supports Stream | Status | Needs Auth |
print('| Website| Provider| gpt-3.5 | gpt-4 | Streaming | Status | Auth |')
print('| --- | --- | --- | --- | --- | --- | --- |')

for provider in providers:
    parsed_url = urlparse(provider.url)
    name = f"`g4f.Provider{provider.__name__.split('.')[-1]}`"
    url = f'[{parsed_url.netloc}]({provider.url})'
    has_gpt4 = '✔️' if 'gpt-4' in provider.model else '❌'
    has_gpt3_5 = '✔️' if 'gpt-3.5-turbo' in provider.model else '❌'
    streaming = '✔️' if provider.supports_stream else '❌'
    needs_auth = '✔️' if provider.needs_auth else '❌'
    
    print(f'| {url} | {name} | {has_gpt3_5} | {has_gpt4} | {streaming} | ![Active](https://img.shields.io/badge/Active-brightgreen) | {needs_auth} |')