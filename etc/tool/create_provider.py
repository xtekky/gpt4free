
import sys, re
from pathlib import Path
from os import path

sys.path.append(str(Path(__file__).parent.parent.parent))

import g4f

g4f.debug.logging = True

def read_code(text):
    if match := re.search(r"```(python|py|)\n(?P<code>[\S\s]+?)\n```", text):
        return match.group("code")

def input_command():
    print("Enter/Paste the cURL command. Ctrl-D or Ctrl-Z ( windows ) to save it.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    return "\n".join(contents)

name = input("Name: ")
provider_path = f"g4f/Provider/{name}.py"

example = """
from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class {name}(AsyncGeneratorProvider, ProviderModelMixin):
    label = ""
    url = "https://example.com"
    api_endpoint = "https://example.com/api/completion"
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = ''
    models = ['', '']
    
    model_aliases = {
        "alias1": "model1",
    }

   @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {{
            "authority": "example.com",
            "accept": "application/json",
            "origin": cls.url,
            "referer": f"{{cls.url}}/chat",
        }}
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {{
                "prompt": prompt,
                "model": model,
            }}
            async with session.post(f"{{cls.url}}/api/chat", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode()
"""

if not path.isfile(provider_path):
    command = input_command()

    prompt = f"""
Create a provider from a cURL command. The command is:
```bash
{command}
```
A example for a provider:
```python
{example}
```
The name for the provider class:
{name}
Replace "hello" with `format_prompt(messages)`.
And replace "gpt-3.5-turbo" with `model`.
"""

    print("Create code...")
    response = []
    for chunk in g4f.ChatCompletion.create(
        model=g4f.models.gpt_4o,
        messages=[{"role": "user", "content": prompt}],
        timeout=300,
        stream=True,
    ):
        print(chunk, end="", flush=True)
        response.append(chunk)
    print()
    response = "".join(response)

    if code := read_code(response):
        with open(provider_path, "w") as file:
            file.write(code)
        print("Saved at:", provider_path)
        with open("g4f/Provider/__init__.py", "a") as file:
            file.write(f"\nfrom .{name} import {name}")
else:
    with open(provider_path, "r") as file:
        code = file.read()
