import sys
from pathlib import Path
import asyncio
from time import time

sys.path.append(str(Path(__file__).parent.parent))

import g4f

providers = [g4f.Provider.OpenaiChat, g4f.Provider.Bard, g4f.Provider.Bing]

# Async support
async def log_time_async(method: callable, **kwargs):
    start = time()
    result = await method(**kwargs)
    secs = f"{round(time() - start, 2)} secs"
    if result:
        return " ".join([result, secs])
    return secs

def log_time_yield(method: callable, **kwargs):
    start = time()
    result = yield from method(**kwargs)
    yield f" {round(time() - start, 2)} secs"

def log_time(method: callable, **kwargs):
    start = time()
    result = method(**kwargs)
    secs = f"{round(time() - start, 2)} secs"
    if result:
        return " ".join([result, secs])
    return secs

async def run_async():
    responses = []
    for provider in providers:
        responses.append(log_time_async(
            provider.create_async, 
            model=None,
            messages=[{"role": "user", "content": "Hello"}],
            log_time=True
        ))
    responses = await asyncio.gather(*responses)
    for idx, provider in enumerate(providers):
        print(f"{provider.__name__}:", responses[idx])
print("Async Total:", asyncio.run(log_time_async(run_async)))

# Streaming support:
def run_stream():
    for provider in providers:
        print(f"{provider.__name__}: ", end="")
        for response in log_time_yield(
            provider.create_completion,
            model=None,
            messages=[{"role": "user", "content": "Hello"}],
        ):
            print(response, end="")
        print()
print("Stream Total:", log_time(run_stream))

# No streaming support:
def create_completion():
    for provider in providers:
        print(f"{provider.__name__}:", end=" ")
        for response in log_time_yield(
            g4f.Provider.Bard.create_completion,
            model=None,
            messages=[{"role": "user", "content": "Hello"}],
        ):
            print(response, end="")
        print()
print("No Stream Total:", log_time(create_completion))

for response in g4f.Provider.Hugchat.create_completion(
    model=None,
    messages=[{"role": "user", "content": "Hello, tell about you."}],
):
    print("Hugchat:", response)

"""
OpenaiChat: Hello! How can I assist you today? 2.0 secs
Bard: Hello! How can I help you today? 3.44 secs
Bing: Hello, this is Bing. How can I help? 😊 4.14 secs
Async Total: 4.25 secs

OpenaiChat: Hello! How can I assist you today? 1.85 secs
Bard: Hello! How can I help you today? 3.38 secs
Bing: Hello, this is Bing. How can I help? 😊 6.14 secs
Stream Total: 11.37 secs

OpenaiChat: Hello! How can I help you today? 3.28 secs
Bard: Hello there! How can I help you today? 3.58 secs
Bing: Hello! How can I help you today? 3.28 secs
No Stream Total: 10.14 secs
"""