import asyncio
import g4f
from g4f.client import AsyncClient

async def main():
    client = AsyncClient(
        provider=g4f.Provider.Ecosia,
    )
    async for chunk in client.chat.completions.create(
        [{"role": "user", "content": "happy dogs on work. write some lines"}],
        g4f.models.default,
        stream=True,
        green=True,
    ):
        print(chunk.choices[0].delta.content or "", end="")
    print(f"\nwith {chunk.model}")

asyncio.run(main())