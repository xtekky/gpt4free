### G4F - Media Documentation

This document outlines how to use the G4F (Generative Framework) library to generate and process various media types, including audio, images, and videos.

---

### 1. **Audio Generation and Transcription**

G4F supports audio generation through providers like PollinationsAI and audio transcription using providers like Microsoft_Phi_4.

#### **Generate Audio with PollinationsAI:**

```python
import asyncio
from g4f.client import AsyncClient
import g4f.Provider

async def main():
    client = AsyncClient(provider=g4f.Provider.PollinationsAI)

    response = await client.chat.completions.create(
        model="openai-audio",
        messages=[{"role": "user", "content": "Say good day to the world"}],
        audio={"voice": "alloy", "format": "mp3"},
    )
    response.choices[0].message.save("alloy.mp3")

asyncio.run(main())
```

#### **Transcribe an Audio File:**

```python
import asyncio
from g4f.client import AsyncClient
import g4f.Provider

async def main():
    client = AsyncClient(provider=g4f.Provider.Microsoft_Phi_4)

    with open("audio.wav", "rb") as audio_file:
        response = await client.chat.completions.create(
            messages="Transcribe this audio",
            provider=g4f.Provider.Microsoft_Phi_4,
            media=[[audio_file, "audio.wav"]],
            modalities=["text"],
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

---

### 2. **Image Generation**

G4F can generate images from text prompts and provides options to retrieve images as URLs or base64-encoded strings.

#### **Generate an Image:**

```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()

    response = await client.images.generate(
        prompt="a white siamese cat",
        model="flux",
        response_format="url",
    )

    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")

asyncio.run(main())
```

#### **Base64 Response Format:**

```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()

    response = await client.images.generate(
        prompt="a white siamese cat",
        model="flux",
        response_format="b64_json",
    )

    base64_text = response.data[0].b64_json
    print(base64_text)

asyncio.run(main())
```

#### **Image Parameters:**
- **`width`**: Defines the width of the generated image.
- **`height`**: Defines the height of the generated image.
- **`n`**: Specifies the number of images to generate.
- **`response_format`**: Specifies the format of the response:
  - `"url"`: Returns the URL of the image.
  - `"b64_json"`: Returns the image as a base64-encoded JSON string.
  - (Default): Saves the image locally and returns a local url.

#### **Example with Image Parameters:**

```python
import asyncio
from g4f.client import AsyncClient

async def main():
    client = AsyncClient()

    response = await client.images.generate(
        prompt="a white siamese cat",
        model="flux",
        response_format="url",
        width=512,
        height=512,
        n=2,
    )

    for image in response.data:
        print(f"Generated image URL: {image.url}")

asyncio.run(main())
```

---

### 3. **Creating Image Variations**

You can generate variations of an existing image using G4F.

#### **Create Image Variations:**

```python
import asyncio
from g4f.client import AsyncClient
from g4f.Provider import OpenaiChat

async def main():
    client = AsyncClient(image_provider=OpenaiChat)

    response = await client.images.create_variation(
        prompt="a white siamese cat",
        image=open("docs/images/cat.jpg", "rb"),
        model="dall-e-3",
    )

    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")

asyncio.run(main())
```

---

### 4. **Video Generation**

G4F supports video generation through providers like HuggingFaceMedia.

#### **Generate a Video:**

```python
import asyncio
from g4f.client import AsyncClient
from g4f.Provider import HuggingFaceMedia

async def main():
    client = AsyncClient(
        provider=HuggingFaceMedia,
        api_key=os.getenv("HF_TOKEN") # Your API key here
    )

    video_models = client.models.get_video()
    print("Available Video Models:", video_models)

    result = await client.media.generate(
        model=video_models[0],
        prompt="G4F AI technology is the best in the world.",
        response_format="url",
    )

    print("Generated Video URL:", result.data[0].url)

asyncio.run(main())
```

#### **Video Parameters:**
- **`resolution`**: Specifies the resolution of the generated video. Options include:
  - `"480p"` (default)
  - `"720p"`
- **`aspect_ratio`**: Defines the width-to-height ratio (e.g., `"16:9"`).
- **`n`**: Specifies the number of videos to generate.
- **`response_format`**: Specifies the format of the response:
  - `"url"`: Returns the URL of the video.
  - `"b64_json"`: Returns the video as a base64-encoded JSON string.
  - (Default): Saves the video locally and returns a local url.

#### **Example with Video Parameters:**

```python
import os
import asyncio
from g4f.client import AsyncClient
from g4f.Provider import HuggingFaceMedia

async def main():
    client = AsyncClient(
        provider=HuggingFaceMedia,
        api_key=os.getenv("HF_TOKEN")  # Your API key here
    )

    video_models = client.models.get_video()
    print("Available Video Models:", video_models)

    result = await client.media.generate(
        model=video_models[0],
        prompt="G4F AI technology is the best in the world.",
        resolution="720p",
        aspect_ratio="16:9",
        n=1,
        response_format="url",
    )

    print("Generated Video URL:", result.data[0].url)

asyncio.run(main())
```

---

**Key Points:**

- **Provider Selection**: Ensure the selected provider supports the desired media generation or processing task.
- **API Keys**: Some providers require API keys for authentication.
- **Response Formats**: Use `response_format` to control the output format (URL, base64, local file).
- **Parameter Usage**: Use parameters like `width`, `height`, `resolution`, `aspect_ratio`, and `n` to customize the generated media.
