import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).parent.parent))

import g4f
from g4f.Provider import AsyncProvider
from testing.test_providers import get_providers
from  testing.log_time import log_time_async

async def create_async(provider: AsyncProvider):
    model = g4f.models.gpt_35_turbo.name if provider.supports_gpt_35_turbo else g4f.models.default.name
    try:
        response =  await log_time_async(
            provider.create_async,
            model=model,
            messages=[{"role": "user", "content": "Hello Assistant!"}]
        )
        assert type(response) is str
        assert len(response) > 0
        return response
    except Exception as e:
        return e

async def run_async():
  _providers: list[AsyncProvider] = [
    _provider
    for _provider in get_providers()
    if _provider.working and hasattr(_provider, "create_async")
  ]
  responses = [create_async(_provider) for _provider in _providers]
  responses = await asyncio.gather(*responses)
  for idx, provider in enumerate(_providers):
      print(f"{provider.__name__}:", responses[idx])

print("Total:", asyncio.run(log_time_async(run_async)))