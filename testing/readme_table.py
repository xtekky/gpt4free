import os, glob, importlib
from urllib.parse import urlparse
from g4f import Provider, ChatCompletion

def get_provider_names() -> iter:
    for provider in glob.glob("g4f/Provider/Providers/*.py"):
        if "__init__.py" not in provider:
            yield os.path.splitext(os.path.basename(provider))[0]

def validate_response(response):
    if not isinstance(response, str):
        raise RuntimeError("Response is not a string")
    elif not response.strip():
        raise RuntimeError("Empty response")
    elif response in (
        "Vercel is currently not working.",
        "Unable to fetch the response, Please try again."
    ) or response.startswith("{\"error\":{\"message\":"):
        raise RuntimeError("Response: {response}")
    else:
        return response
    
def test(provider: Provider):
    if provider.needs_auth:
        return
    model = provider.model if type(provider.model) == str else provider.model[0]
    try:
        response = ChatCompletion.create(
            model=model,
            provider=provider,
            messages=[{"role": "user", "content": "Hello"}],
        )
        return validate_response(response)
    except Exception:
        pass

providers = [
    importlib.import_module(f"g4f.Provider.Providers.{provider_name}")
    for provider_name in sorted(get_provider_names())
]
print("## Providers")
print()

print("```py")
print("from g4f.Provider import (")
for provider in providers:
    if provider.working:
        print(f"    {provider.__name__.split('.')[-1]},")
print(")")
print("# Usage:")
print("response = g4f.ChatCompletion.create(..., provider=ProviderName)")
print("```")
print()

# | Website| Provider| gpt-3.5-turbo | gpt-4 | Supports Stream | Status | Needs Auth |
print('| Website| Provider| gpt-3.5 | gpt-4 | Streaming | Status | Auth |')
print('| --- | --- | --- | --- | --- | --- | --- |')
for is_working in (True, False):
    for provider in providers:
        if is_working != provider.working:
            continue

        parsed_url = urlparse(provider.url)
        name = f"`g4f.Provider.{provider.__name__.split('.')[-1]}`"
        url = f'[{parsed_url.netloc}]({provider.url})'
        has_gpt4 = '✔️' if 'gpt-4' in provider.model else '❌'
        has_gpt3_5 = '✔️' if 'gpt-3.5-turbo' in provider.model else '❌'
        streaming = '✔️' if provider.supports_stream else '❌'
        needs_auth = '✔️' if provider.needs_auth else '❌'
        
        if provider.working:
            if test(provider):
                working = '![Active](https://img.shields.io/badge/Active-brightgreen)'
            else:
                working = '![Unknown](https://img.shields.io/badge/Unknown-grey)'
        else:
            working = '![Inactive](https://img.shields.io/badge/Inactive-red)'
        
        print(f'| {url} | {name} | {has_gpt3_5} | {has_gpt4} | {streaming} | {working} | {needs_auth} |')