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
        # GPT-3.5
        g4f.models.gpt_35_turbo,

        # GPT-4
        g4f.models.gpt_4,
    ]

    models_working = []

    for model in models_to_test:
        if await test(model):
            models_working.append(model.name)

    print("working models:", models_working)


asyncio.run(start_test())
