import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).parent.parent.parent))

import asyncio
from g4f import models
from g4f import ChatCompletion
from g4f.Provider.base_provider import BaseProvider
from etc.testing._providers import get_providers

from g4f import debug

debug.logging = True


async def test_async(provider: type[BaseProvider]):
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


async def test_async_list(providers: list[type[BaseProvider]]):
    responses: list = [
        test_async(_provider)
        for _provider in providers
    ]
    return await asyncio.gather(*responses)


def print_providers():

    providers = get_providers()
    responses = asyncio.run(test_async_list(providers))

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
                netloc = urlparse(_provider.url).netloc
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
        "cohere": "Cohere",
        "google": "Google",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "replicate": "Replicate",
        "huggingface": "Huggingface",
    }
    provider_urls = {
        "Bard": "https://bard.google.com/",
        "H2o": "https://www.h2o.ai/",
        "Vercel": "https://sdk.vercel.ai/",
    }

    lines = [
        "| Model | Base Provider | Provider | Website |",
        "| ----- | ------------- | -------- | ------- |",
    ]

    _models = get_models()
    for model in _models:
        if not model.best_provider or model.best_provider.__name__ not in provider_urls:
            continue

        name = re.split(r":|/", model.name)[-1]
        base_provider = base_provider_names[model.base_provider]
        provider_name = f"g4f.provider.{model.best_provider.__name__}"
        provider_url = provider_urls[model.best_provider.__name__]
        netloc = urlparse(provider_url).netloc
        website = f"[{netloc}]({provider_url})"

        lines.append(f"| {name} | {base_provider} | {provider_name} | {website} |")

    print("\n".join(lines))


def get_models():
    _models = [item[1] for item in models.__dict__.items()]
    _models = [model for model in _models if type(model) is models.Model]
    return [model for model in _models if model.name not in ["gpt-3.5-turbo", "gpt-4"]]


if __name__ == "__main__":
    print_providers()
    print("\n", "-" * 50, "\n")
    print_models()