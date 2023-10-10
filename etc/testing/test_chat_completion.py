import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import g4f, asyncio

print("create:", end=" ", flush=True)
for response in g4f.ChatCompletion.create(
    model=g4f.models.gpt_4_32k_0613,
    provider=g4f.Provider.Aivvm,
    messages=[{"role": "user", "content": "write a poem about a tree"}],
    temperature=0.1,
    stream=True
):
    print(response, end="", flush=True)
print()

async def run_async():
    response = await g4f.ChatCompletion.create_async(
        model=g4f.models.gpt_35_turbo_16k_0613,
        provider=g4f.Provider.GptGod,
        messages=[{"role": "user", "content": "hello!"}],
    )
    print("create_async:", response)

# asyncio.run(run_async())
