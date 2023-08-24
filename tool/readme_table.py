import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).parent.parent))

from g4f import models, Provider
from g4f.Provider.base_provider import BaseProvider
from testing.test_providers import test

def main():
    print_providers()
    print("\n", "-" * 50, "\n")
    print_models()


def print_providers():
    lines = [
        "| Website| Provider| gpt-3.5 | gpt-4 | Streaming | Status | Auth |",
        "| ------ | ------- | ------- | ----- | --------- | ------ | ---- |",
    ]
    providers = get_providers()
    for is_working in (True, False):
        for _provider in providers:
            if is_working != _provider.working:
                continue
            netloc = urlparse(_provider.url).netloc
            website = f"[{netloc}]({_provider.url})"

            provider_name = f"g4f.provider.{_provider.__name__}"

            has_gpt_35 = "✔️" if _provider.supports_gpt_35_turbo else "❌"
            has_gpt_4 = "✔️" if _provider.supports_gpt_4 else "❌"
            stream = "✔️" if _provider.supports_stream else "❌"
            if _provider.working:
                if test(_provider):
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


def get_providers() -> list[type[BaseProvider]]:
    provider_names = dir(Provider)
    ignore_names = [
        "base_provider",
        "BaseProvider",
    ]
    provider_names = [
        provider_name
        for provider_name in provider_names
        if not provider_name.startswith("__") and provider_name not in ignore_names
    ]
    return [getattr(Provider, provider_name) for provider_name in provider_names]


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
        if model.best_provider.__name__ not in provider_urls:
            continue
        split_name = re.split(r":|/", model.name)
        name = split_name[-1]

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
    main()
