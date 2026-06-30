import asyncio
from g4f.client import Client, AsyncClient

question = """
Hey! How can I recursively list all files in a directory in Python?
"""

# Synchronous streaming function
def sync_stream():
    client = Client()
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": question}
        ],
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content or "", end="")

# Asynchronous streaming function
async def async_stream():
    client = AsyncClient()
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": question}
        ],
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

# Main function to run both streams
def main():
    print("Synchronous Stream:")
    sync_stream()
    print("\n\nAsynchronous Stream:")
    asyncio.run(async_stream())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
