from __future__ import annotations

import g4f
from g4f import BaseProvider


def get_provider(provider: str) -> BaseProvider | None:
    if not isinstance(provider, str):
        return None
    print(provider)
    if provider == 'g4f.Provider.Auto':
        return None

    return g4f.Provider.ProviderUtils.convert.get(provider)
