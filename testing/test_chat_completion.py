import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import g4f, asyncio

print("create:", end=" ", flush=True)
for response in g4f.ChatCompletion.create(
    model=g4f.models.gpt_4,
    provider=g4f.Provider.Vercel,
    messages=[{"role": "user", "content": "hello!"}],
):
    print(response, end="", flush=True)
print()

async def run_async():
    response = await g4f.ChatCompletion.create_async(
        model=g4f.models.gpt_4_32k_0613,
        provider=g4f.Provider.Aivvm,
        messages=[{"role": "user", "content": "hello!"}],
        temperature=0.0
    )
    print("create_async:", response)

# asyncio.run(run_async())
