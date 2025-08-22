import asyncio
from pathlib import Path
from g4f.client import AsyncClient
from g4f.Provider import HuggingSpace, Azure
from g4f.cookies import read_cookie_files

# Load cookies and authentication environment variables needed by providers
read_cookie_files()

# Initialize asynchronous client to interact with providers
client = AsyncClient()

# Define an async function that creates an image variation using the HuggingSpace provider
async def main_with_hugging_space():
    # Call create_variation with an image path, provider, model, prompt and desired response format
    result = await client.images.create_variation(
        image=Path("g4f.dev/docs/images/strawberry.jpg"),  # Path to input image
        provider=HuggingSpace,                             # Provider to use
        model="flux-kontext-dev",                          # Model name for HuggingSpace
        prompt="Change color to black and white",         # Variation prompt
        response_format="url"                              # Return URL to generated image
    )
    print(result)  # Print the URL or result returned by the provider

# Define an async function that creates an image variation using the Azure provider
async def main_with_azure():
    result = await client.images.create_variation(
        image=Path("g4f.dev/docs/images/strawberry.jpg"),
        provider=Azure,
        model="flux-kontext",
        prompt="Add text 'Hello World' in the center",
        response_format="url"
    )
    print(result)  # Print the returned URL or response

# Run the Azure image variation example asynchronously
asyncio.run(main_with_azure())

# Import helper function to get directory used for cookies and related files
from g4f.cookies import get_cookies_dir

# Print the directory currently used for storing cookies
print(get_cookies_dir())