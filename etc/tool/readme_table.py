import re
from urllib.parse import urlparse
import asyncio

from g4f import models, ChatCompletion
from g4f.providers.types import BaseRetryProvider, ProviderType
from g4f.providers.base_provider import ProviderModelMixin
from g4f.Provider import __providers__
from g4f.models import _all_models
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
    providers = [provider for provider in __providers__ if provider.working]
    responses = test_async_list(providers)
    lines = []
    for type in ("Free", "Auth"):
        lines += [
            "",
            f"## {type}",
            "",
        ]
        for idx, _provider in enumerate(providers):
            do_continue = False
            if type == "Auth" and _provider.needs_auth:
                do_continue = True
            elif type == "Free" and not _provider.needs_auth:
                do_continue = True
            if not do_continue:
                continue
            
            lines.append(
                f"### {getattr(_provider, 'label', _provider.__name__)}",
            )
            provider_name = f"`g4f.Provider.{_provider.__name__}`"
            lines.append(f"| Provider | {provider_name} |")
            lines.append("| -------- | ---- |")
            
            if _provider.url:
                netloc = urlparse(_provider.url).netloc.replace("www.", "")
                website = f"[{netloc}]({_provider.url})"
            else:
                website = "❌"

            message_history = "✔️" if _provider.supports_message_history else "❌"
            system = "✔️" if _provider.supports_system_message else "❌"
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

            lines.append(f"| **Website** | {website} | \n| **Status** | {status} |")

            if issubclass(_provider, ProviderModelMixin):
                try:
                    all_models = _provider.get_models()
                    models = [model for model in _all_models if model in all_models or model in _provider.model_aliases]
                    image_models = _provider.image_models
                    if image_models:
                        for alias, name in _provider.model_aliases.items():
                            if alias in _all_models and name in image_models:
                                image_models.append(alias)
                        image_models = [model for model in image_models if model in _all_models]
                        if image_models:
                            models = [model for model in models if model not in image_models]
                    if models:
                        lines.append(f"| **Models** | {', '.join(models)} ({len(all_models)})|")
                    if image_models:
                        lines.append(f"| **Image Models (Image Generation)** | {', '.join(image_models)} |")
                    if hasattr(_provider, "vision_models"):
                        lines.append(f"| **Vision (Image Upload)** | ✔️ |")
                except:
                    pass

            lines.append(f"| **Authentication** | {auth} | \n| **Streaming** | {stream} |")
            lines.append(f"| **System message** | {system} | \n| **Message history** | {message_history} |")
    return lines

def print_models():
    base_provider_names = {
        "google": "Google",
        "openai": "OpenAI",
        "huggingface": "Huggingface",
        "anthropic": "Anthropic",
        "inflection": "Inflection",
        "meta": "Meta",
    }
    provider_urls = {
        "google": "https://gemini.google.com/",
        "openai": "https://openai.com/",
        "huggingface": "https://huggingface.co/",
        "anthropic": "https://www.anthropic.com/",
        "inflection": "https://inflection.ai/",
        "meta": "https://llama.meta.com/",
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
        if model.base_provider not in base_provider_names:
            continue
        base_provider = base_provider_names[model.base_provider]
        if not isinstance(model.best_provider, BaseRetryProvider):
            provider_name = f"g4f.Provider.{model.best_provider.__name__}"
        else:
            provider_name = f"{len(model.best_provider.providers)}+ Providers"
        provider_url = provider_urls[model.base_provider]
        netloc = urlparse(provider_url).netloc.replace("www.", "")
        website = f"[{netloc}]({provider_url})"

        lines.append(f"| {name} | {base_provider} | {provider_name} | {website} |")

    return lines

def print_image_models():
    lines = [
        "| Label | Provider | Image Model | Vision Model | Website |",
        "| ----- | -------- | ----------- | ------------ | ------- |",
    ]
    for provider in [provider for provider in __providers__ if provider.working and getattr(provider, "image_models", None) or getattr(provider, "vision_models", None)]:
        provider_url = provider.url if provider.url else "❌"
        netloc = urlparse(provider_url).netloc.replace("www.", "")
        website = f"[{netloc}]({provider_url})"
        label = getattr(provider, "label", provider.__name__)
        if provider.image_models:
            image_models = ", ".join([model for model in provider.image_models if model in _all_models])
        else:
            image_models = "❌"
        if hasattr(provider, "vision_models"):
            vision_models = "✔️"
        else:
            vision_models = "❌"
        lines.append(f'| {label} | `g4f.Provider.{provider.__name__}` | {image_models}| {vision_models} | {website} |')

    return lines

if __name__ == "__main__":
    with open("docs/providers.md", "w") as f:
        f.write("\n".join(print_providers()))
        f.write(f"\n{'-' * 50} \n")
        #f.write("\n".join(print_models()))
        #f.write(f"\n{'-' * 50} \n")
        f.write("\n".join(print_image_models()))