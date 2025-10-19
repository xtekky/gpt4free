#!/usr/bin/env python3
"""
Entry point for g4f CLI executable builds
This file is used as the main entry point for building executables with Nuitka
"""

import g4f.debug
g4f.debug.enable_logging()

from g4f.client import Client
from g4f.errors import ModelNotFoundError

import g4f.Provider

try:
    client = Client(provider=g4f.Provider.PollinationsAI)
    response = client.chat.completions.create(
        model="openai",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
        raw=True
    )
    for r in response:
        print(r)
except ModelNotFoundError as e:
    print(f"Successfully")
exit(0)

import g4f.cli

if __name__ == "__main__":
    g4f.cli.main()