import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from g4f import BaseProvider, models, provider


def main():
    providers = get_providers()
    results: list[list[str | bool]] = []

    for _provider in providers:
        print("start", _provider.__name__)
        actual_working = judge(_provider)
        expected_working = _provider.working
        match = actual_working == expected_working

        results.append([_provider.__name__, expected_working, actual_working, match])

    print("failed provider list")
    for result in results:
        if not result[3]:
            print(result)


def get_providers() -> list[type[BaseProvider]]:
    provider_names = dir(provider)
    ignore_names = [
        "base_provider",
        "BaseProvider",
    ]
    provider_names = [
        provider_name
        for provider_name in provider_names
        if not provider_name.startswith("__") and provider_name not in ignore_names
    ]
    return [getattr(provider, provider_name) for provider_name in provider_names]


def create_response(_provider: type[BaseProvider]) -> str:
    model = (
        models.gpt_35_turbo.name
        if _provider is not provider.H2o
        else models.falcon_7b.name
    )
    response = _provider.create_completion(
        model=model,
        messages=[{"role": "user", "content": "Hello world!, plz yourself"}],
        stream=False,
    )
    return "".join(response)


def judge(_provider: type[BaseProvider]) -> bool:
    if _provider.needs_auth:
        return _provider.working

    try:
        response = create_response(_provider)
        assert type(response) is str
        return len(response) > 1
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    main()
