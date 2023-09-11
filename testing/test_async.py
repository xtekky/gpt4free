import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).parent.parent))

import g4f
from g4f.Provider import AsyncProvider
from testing.test_providers import get_providers
from testing.log_time import log_time_async

async def create_async(provider):
    model = g4f.models.gpt_35_turbo.name if provider.supports_gpt_35_turbo else g4f.models.default.name
    try:
        response = await log_time_async(
            provider.create_async,
            model=model,
            messages=[{"role": "user", "content": "Hello Assistant!"}]
        )
        print(f"{provider.__name__}:", response)
    except Exception as e:
        return f"{provider.__name__}: {e.__class__.__name__}: {e}"

async def run_async():
  responses: list = [
    create_async(_provider)
    for _provider in get_providers()
    if _provider.working and issubclass(_provider, AsyncProvider)
  ]
  responses = await asyncio.gather(*responses)
  for error in responses:
      if error:
        print(error)

print("Total:", asyncio.run(log_time_async(run_async)))