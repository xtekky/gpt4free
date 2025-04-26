import sys
from pathlib import Path
from colorama import Fore, Style

sys.path.append(str(Path(__file__).parent.parent))

from g4f import Provider, ProviderType, models
from g4f.Provider import __providers__


def main():
    providers = get_providers()
    failed_providers = []

    for provider in providers:
        if provider.needs_auth:
            continue
        print("Provider:", provider.__name__)
        result = test(provider)
        print("Result:", result)
        if provider.working and not result:
            failed_providers.append(provider)
    print()

    if failed_providers:
        print(f"{Fore.RED + Style.BRIGHT}Failed providers:{Style.RESET_ALL}")
        for _provider in failed_providers:
            print(f"{Fore.RED}{_provider.__name__}")
    else:
        print(f"{Fore.GREEN + Style.BRIGHT}All providers are working")


def get_providers() -> list[ProviderType]:
    return [
        provider
        for provider in __providers__
        if provider.__name__ not in dir(Provider.deprecated)
        and provider.url is not None
    ]

def create_response(provider: ProviderType) -> str:
    response = provider.create_completion(
        model=models.default.name,
        messages=[{"role": "user", "content": "Hello, who are you? Answer in detail much as possible."}],
        stream=False,
    )
    return "".join(response)

def test(provider: ProviderType) -> bool:
    try:
        response = create_response(provider)
        assert type(response) is str
        assert len(response) > 0
        return response
    except Exception:
        return False


if __name__ == "__main__":
    main()
    
