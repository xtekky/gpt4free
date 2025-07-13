import asyncio
from pathlib import Path

import g4f.debug
g4f.debug.logging = True

from g4f.Provider import PollinationsAI, Together, OpenaiAccount, CopilotAccount, BlackForestLabs_Flux1KontextDev
from g4f.client import AsyncClient

client = AsyncClient(provider=PollinationsAI)

# Example usage of the client to create image variations with different providers

## This examples needs a absolute URL to the source image:

async def main():
    result = await client.images.create_variation(
        image="https://g4f.dev/docs/images/strawberry.jpg",
        prompt="Remove background",
        model="gpt-image",
        response_format="url",
        transparent=True)
    print(result)

async def main_with_together():
    result = await client.images.create_variation(
        image="https://g4f.dev/docs/images/strawberry.jpg",
        model="flux-kontext-pro",
        provider=Together,
        prompt="Add nature background",
        response_format="url"
    )
    print(result)


async def main_with_copilot():
    result = await client.images.create_variation(
        image=Path("g4f.dev/docs/images/strawberry.jpg"),
        provider=CopilotAccount,
        prompt="Generate a variant of this image",
        response_format="url"
    )
    print(result)


async def main_with_copilot():
    result = await client.images.create_variation(
        image=Path("g4f.dev/docs/images/strawberry.jpg"),
        provider=BlackForestLabs_Flux1KontextDev,
        prompt="Generate a variant of this image",
        response_format="url"
    )
    print(result)

asyncio.run(main_with_copilot())