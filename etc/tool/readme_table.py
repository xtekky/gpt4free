import re
from urllib.parse import urlparse
import asyncio

from g4f import models, ChatCompletion
from g4f.providers.types import BaseRetryProvider, ProviderType
from etc.testing._providers import get_providers
from g4f import debug

debug.logging = True

async def test_async(provider: ProviderType):
    if not provider.working:
        return False
    messages = [{"role": "user", "content": "Hello Assistant!"}]
    try:
        response = await asyncio.wait_for(ChatCompletion.create_async(
            model=models.default,
            messages=messages,
            provider=provider
        ), 30)
        return bool(response)
    except Exception as e:
        if debug.logging:
            print(f"{provider.__name__}: {e.__class__.__name__}: {e}")
        return False

def test_async_list(providers: list[ProviderType]):
    responses: list = [
        asyncio.run(test_async(_provider))
        for _provider in providers
    ]
    return responses

def print_providers():

    providers = get_providers()
    responses = test_async_list(providers)

    for type in ("GPT-4", "GPT-3.5", "Other"):
        lines = [
            "",
            f"### {type}",
            "",
            "| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |",
            "| ------  | -------  | ------- | ----- | ------ | ------ | ---- |",
        ]
        for is_working in (True, False):
            for idx, _provider in enumerate(providers):
                if is_working != _provider.working:
                    continue
                do_continue = False
                if type == "GPT-4" and _provider.supports_gpt_4:
                    do_continue = True
                elif type == "GPT-3.5" and not _provider.supports_gpt_4 and _provider.supports_gpt_35_turbo:
                    do_continue = True
                elif type == "Other" and not _provider.supports_gpt_4 and not _provider.supports_gpt_35_turbo:
                    do_continue = True
                if not do_continue:
                    continue
                netloc = urlparse(_provider.url).netloc.replace("www.", "")
                website = f"[{netloc}]({_provider.url})"

                provider_name = f"`g4f.Provider.{_provider.__name__}`"

                has_gpt_35 = "✔️" if _provider.supports_gpt_35_turbo else "❌"
                has_gpt_4 = "✔️" if _provider.supports_gpt_4 else "❌"
                stream = "✔️" if _provider.supports_stream else "❌"
                if _provider.working:
                    status = '![Active](https://img.shields.io/badge/Active-brightgreen)'
                    if responses[idx]:
                        status = '![Active](https://img.shields.io/badge/Active-brightgreen)'
                    else:
                        status = '![Unknown](https://img.shields.io/badge/Unknown-grey)'
                else:
                    status = '![Inactive](https://img.shields.io/badge/Inactive-red)'
                auth = "✔️" if _provider.needs_auth else "❌"

                lines.append(
                    f"| {website} | {provider_name} | {has_gpt_35} | {has_gpt_4} | {stream} | {status} | {auth} |"
                )
        print("\n".join(lines))

def print_models():
    base_provider_names = {
        "google": "Google",
        "openai": "OpenAI",
        "huggingface": "Huggingface",
        "anthropic": "Anthropic",
        "inflection": "Inflection",
        "meta": "Meta"
    }
    provider_urls = {
        "google": "https://gemini.google.com/",
        "openai": "https://openai.com/",
        "huggingface": "https://huggingface.co/",
        "anthropic": "https://www.anthropic.com/",
        "inflection": "https://inflection.ai/",
        "meta": "https://llama.meta.com/"
    }

    lines = [
        "| Model | Base Provider | Provider | Website |",
        "| ----- | ------------- | -------- | ------- |",
    ]
    for name, model in models.ModelUtils.convert.items():
        if name.startswith("gpt-3.5") or name.startswith("gpt-4"):
            if name not in ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"):
                continue
        name = re.split(r":|/", model.name)[-1]
        base_provider = base_provider_names[model.base_provider]
        if not isinstance(model.best_provider, BaseRetryProvider):
            provider_name = f"g4f.Provider.{model.best_provider.__name__}"
        else:
            provider_name = f"{len(model.best_provider.providers)}+ Providers"
        provider_url = provider_urls[model.base_provider]
        netloc = urlparse(provider_url).netloc.replace("www.", "")
        website = f"[{netloc}]({provider_url})"

        lines.append(f"| {name} | {base_provider} | {provider_name} | {website} |")

    print("\n".join(lines))

if __name__ == "__main__":
    print_providers()
    print("\n", "-" * 50, "\n")
    print_models()