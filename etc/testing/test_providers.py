import sys
from pathlib import Path
from colorama import Fore, Style

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
        result = test(_provider)
        print("Result:", result)
        if _provider.working and not result:
            failed_providers.append(_provider)

    print()

    if failed_providers:
        print(f"{Fore.RED + Style.BRIGHT}Failed providers:{Style.RESET_ALL}")
        for _provider in failed_providers:
            print(f"{Fore.RED}{_provider.__name__}")
    else:
        print(f"{Fore.GREEN + Style.BRIGHT}All providers are working")


def get_providers() -> list[type[BaseProvider]]:
    providers = dir(Provider)
    providers = [getattr(Provider, provider) for provider in providers if provider != "RetryProvider"]
    providers = [provider for provider in providers if isinstance(provider, type)]
    return [provider for provider in providers if issubclass(provider, BaseProvider)]


def create_response(_provider: type[BaseProvider]) -> str:
    model = models.gpt_35_turbo.name if _provider.supports_gpt_35_turbo else models.default.name
    response = _provider.create_completion(
        model=model,
        messages=[{"role": "user", "content": "Hello, who are you? Answer in detail much as possible."}],
        stream=False,
    )
    return "".join(response)

    
def test(_provider: type[BaseProvider]) -> bool:
    try:
        response = create_response(_provider)
        assert type(response) is str
        assert len(response) > 0
        return response
    except Exception as e:
        if logging:
            print(e)
        return False


if __name__ == "__main__":
    main()
    