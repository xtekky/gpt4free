import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import g4f


async def test(model: g4f.Model):
    try:
        try:
            for response in g4f.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": "write a poem about a tree"}],
                    temperature=0.1,
                    stream=True
            ):
                print(response, end="")

            print()
        except:
            for response in await g4f.ChatCompletion.create_async(
                    model=model,
                    messages=[{"role": "user", "content": "write a poem about a tree"}],
                    temperature=0.1,
                    stream=True
            ):
                print(response, end="")

            print()

        return True
    except Exception as e:
        print(model.name, "not working:", e)
        print(e.__traceback__.tb_next)
        return False


async def start_test():
    models_to_test = [
        # GPT-3.5 4K Context
        g4f.models.gpt_35_turbo,
        g4f.models.gpt_35_turbo_0613,

        # GPT-3.5 16K Context
        g4f.models.gpt_35_turbo_16k,
        g4f.models.gpt_35_turbo_16k_0613,

        # GPT-4 8K Context
        g4f.models.gpt_4,
        g4f.models.gpt_4_0613,

        # GPT-4 32K Context
        g4f.models.gpt_4_32k,
        g4f.models.gpt_4_32k_0613,
    ]

    models_working = []

    for model in models_to_test:
        if await test(model):
            models_working.append(model.name)

    print("working models:", models_working)


asyncio.run(start_test())
