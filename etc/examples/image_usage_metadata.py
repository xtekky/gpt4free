"""
Example: Image Generation with Usage Metadata

This example demonstrates the new image metadata feature in usage statistics.
When generating images, the API now returns image dimensions and file size
instead of word/token counts.

Usage:
    python -m etc.examples.image_usage_metadata
"""

import asyncio
from g4f.client import AsyncClient


async def generate_image_with_metadata():
    """
    Generate an image and display usage statistics with metadata.
    
    The usage statistics for image generation will include:
    - images: Number of generated images
    - width: Width of the primary image in pixels
    - height: Height of the primary image in pixels
    - file_size: Total file size of all generated images in bytes
    """
    client = AsyncClient()
    
    print("Generating image...")
    print("-" * 60)
    
    try:
        response = await client.images.generate(
            model="flux",
            prompt="A serene mountain landscape at sunset",
            response_format="url"
        )
        
        if response and response.data:
            image = response.data[0]
            print(f"✓ Image generated successfully!")
            print(f"  URL: {image.url}")
            
            # Note: Usage metadata is available in the streaming API
            # When using the streaming backend API, you'll receive a usage event like:
            # {
            #   "type": "usage",
            #   "usage": {
            #     "images": 1,
            #     "width": 1024,
            #     "height": 1024,
            #     "file_size": 524288
            #   }
            # }
            
            print("\nUsage metadata (available in streaming API):")
            print("  - Shows image count, dimensions, and file size")
            print("  - No word/token counts for image generation")
            
        else:
            print("✗ Failed to generate image")
            
    except Exception as e:
        print(f"✗ Error: {e}")


async def generate_text_for_comparison():
    """
    Generate text for comparison to show traditional token-based usage.
    
    The usage statistics for text generation will include:
    - prompt_tokens: Number of tokens in the prompt
    - completion_tokens: Number of tokens in the completion
    - total_tokens: Total number of tokens used
    """
    client = AsyncClient()
    
    print("\nGenerating text for comparison...")
    print("-" * 60)
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}]
        )
        
        if response:
            print(f"✓ Text generated successfully!")
            print(f"  Content: {response.choices[0].message.content[:100]}...")
            
            # Token-based usage for text generation
            if hasattr(response, 'usage'):
                usage = response.usage
                print("\nUsage metadata:")
                print(f"  - Prompt tokens: {usage.prompt_tokens}")
                print(f"  - Completion tokens: {usage.completion_tokens}")
                print(f"  - Total tokens: {usage.total_tokens}")
            
        else:
            print("✗ Failed to generate text")
            
    except Exception as e:
        print(f"✗ Error: {e}")


async def main():
    """Run both examples to demonstrate the difference."""
    print("=" * 60)
    print("IMAGE GENERATION WITH USAGE METADATA - EXAMPLE")
    print("=" * 60)
    
    await generate_image_with_metadata()
    await generate_text_for_comparison()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The implementation provides context-appropriate usage statistics:

• Image Generation: Shows images, width, height, file_size
• Text Generation: Shows prompt_tokens, completion_tokens, total_tokens

This makes it easy to display meaningful statistics in the UI:
- For images: "1 image, 1024×1024 pixels, 512 KB"
- For text: "150 prompt tokens, 450 completion tokens"
    """)


if __name__ == "__main__":
    asyncio.run(main())
