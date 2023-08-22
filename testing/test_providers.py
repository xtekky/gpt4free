import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from g4f import BaseProvider, models, Provider

logging = False

def main():
    providers = get_providers()
    failed_providers = []

    for _provider in providers:
        if _provider.needs_auth:
            continue
        print("Provider:", _provider.__name__)
        result = judge(_provider)
        print("Result:", result)
        if _provider.working and not result:
            failed_providers.append([_provider, result])

    print("Failed providers:")
    for _provider, result in failed_providers:
       print(f"{_provider.__name__}: {result}")


def get_providers() -> list[type[BaseProvider]]:
    provider_names = dir(Provider)
    ignore_names = [
        "base_provider",
        "BaseProvider"
    ]
    provider_names = [
        provider_name
        for provider_name in provider_names
        if not provider_name.startswith("__") and provider_name not in ignore_names
    ]
    return [getattr(Provider, provider_name) for provider_name in provider_names]


def create_response(_provider: type[BaseProvider]) -> str:
    model = (
        models.gpt_35_turbo.name
        if _provider.supports_gpt_35_turbo
        else _provider.model
    )
    response = _provider.create_completion(
        model=model,
        messages=[{"role": "user", "content": "Hello world!"}],
        stream=False,
    )
    return "".join(response)


def judge(_provider: type[BaseProvider]) -> bool:
    if _provider.needs_auth:
        return _provider.working

    try:
        response = create_response(_provider)
        assert type(response) is str
        return response
    except Exception as e:
        if logging:
            print(e)
        return False


if __name__ == "__main__":
    main()
