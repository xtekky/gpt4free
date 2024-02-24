#### Create Provider with AI Tool

Call in your terminal the `create_provider` script:
```bash
python -m etc.tool.create_provider
```
1. Enter your name for the new provider.
2. Copy and paste the `cURL` command from your browser developer tools.
3. Let the AI ​​create the provider for you.
4. Customize the provider according to your needs.

#### Create Provider

1. Check out the current [list of potential providers](https://github.com/zukixa/cool-ai-stuff#ai-chat-websites), or find your own provider source!
2. Create a new file in [g4f/Provider](/g4f/Provider) with the name of the Provider
3. Implement a class that extends [BaseProvider](/g4f/providers/base_provider.py).

```py
from __future__ import annotations

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

class HogeService(AsyncGeneratorProvider):
    url                   = "https://chat-gpt.com"
    working               = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        yield ""
```

4. Here, you can adjust the settings, for example, if the website does support streaming, set `supports_stream` to `True`...
5. Write code to request the provider in `create_async_generator` and `yield` the response, _even if_ it's a one-time response, do not hesitate to look at other providers for inspiration
6. Add the Provider Import in [`g4f/Provider/__init__.py`](./g4f/Provider/__init__.py)

```py
from .HogeService import HogeService

__all__ = [
  HogeService,
]
```

7. You are done !, test the provider by calling it:

```py
import g4f

response = g4f.ChatCompletion.create(model='gpt-3.5-turbo', provider=g4f.Provider.PROVIDERNAME,
                                    messages=[{"role": "user", "content": "test"}], stream=g4f.Provider.PROVIDERNAME.supports_stream)

for message in response:
    print(message, flush=True, end='')
```