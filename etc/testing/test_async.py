import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import g4f
from testing.test_providers import get_providers
from testing.log_time import log_time_async

async def create_async(provider):
    try:
        response = await log_time_async(
            provider.create_async,
            model=g4f.models.default.name,
            messages=[{"role": "user", "content": "Hello, are you GPT 3.5?"}]
        )
        print(f"{provider.__name__}:", response)
    except Exception as e:
        print(f"{provider.__name__}: {e.__class__.__name__}: {e}")

async def run_async():
  responses: list = [
      create_async(provider)
      for provider in get_providers()
      if provider.working
  ]
  await asyncio.gather(*responses)

print("Total:", asyncio.run(log_time_async(run_async)))